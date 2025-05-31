import torch
import torch.nn as nn
import torch.nn.functional as F


# use safetensors
# use torch.compile when done
# use tensorboard
# use numbers that are powers of 2


class SelfAttn(nn.Module):
    def __init__(self, n_embd, n_heads, batch_size, max_seq_len=512):
        super().__init__()
        assert n_embd % n_heads == 0, "make sure num of embeddings is divisible by heads"
        self.n_embd = n_embd
        self.n_heads = n_heads
        self.head_dim = n_embd // n_heads

        # create projections for k, q, v
        self.query = nn.Linear(n_embd, n_embd)
        self.key = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)
        self.fc_out = nn.Linear(n_embd, n_embd)

        self.max_seq_len = max_seq_len
        self.register_buffer("freqs_cos", None)
        self.register_buffer("freqs_sin", None)

        # KV cache
        self.k_cache = None
        self.v_cache = None

        self.qk = torch.zeros(batch_size, max_seq_len, n_heads, self.head_dim)
        self.vk = torch.zeros(batch_size, max_seq_len, n_heads, self.head_dim)

    def init_rotary_embd(self, device):
        if self.freqs_cos is None or self.freqs_sin is None:
            freqs_cos, freqs_sin = precompute_freqs_cis(self.n_heads, self.max_seq_len)
            self.register_buffer("freqs_cos", freqs_cos)
            self.register_buffer("freqs_sin", freqs_sin)

        self.freqs_cos = self.freqs_cos.to(device)
        self.freqs_sin = self.freqs_sin.to(device)

    def forward(self, x, mask=None, use_cache=False, past_key_value=None):
        batch_size = x.shape[0]
        seq_len = x.shape[1]
        device = x.device

        self.init_rotary_embd(device)

        position_offset = 0
        if past_key_value is not None:
            # If using past key values, adjust the position offset
            position_offset = past_key_value[0].size(2)  # Size of cached sequence

        position_ids = torch.arange(position_offset, position_offset + seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        # Ensure position_ids are within the bounds of precomputed frequencies
        position_ids = torch.clamp(position_ids, max=self.max_seq_len - 1)

        # shape (batch size, seq_len, n_embd)
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        # split into "self.n_heads" pieces
        # shape (batch_size, seq_len, n_heads, head_dim)
        q = q.reshape(batch_size, seq_len, self.n_heads, self.head_dim)
        k = k.reshape(batch_size, seq_len, self.n_heads, self.head_dim)
        v = v.reshape(batch_size, seq_len, self.n_heads, self.head_dim)

        # make (batch_size, n_heads, seq_len, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        q = apply_RoPE(q, self.freqs_cos, self.freqs_sin, position_ids)
        k = apply_RoPE(k, self.freqs_cos, self.freqs_sin, position_ids)

        # Use cached KV if available
        if past_key_value is not None:
            # Concatenate past keys and values with current
            k_past, v_past = past_key_value
            k = torch.cat([k_past, k], dim=2)
            v = torch.cat([v_past, v], dim=2)

        # Save current KV for next iteration if using cache
        if use_cache:
            present_key_value = (k, v)
        else:
            present_key_value = None

        ### attention time!
        out = F.scaled_dot_product_attention(q, k, v, mask, dropout_p=0.0)  # so we use flashattention

        out = out.transpose(1, 2)
        out = out.reshape(batch_size, seq_len, self.n_embd)

        out = self.fc_out(out)
        
        if use_cache:
            return out, present_key_value
        return out


class FeedForward(nn.Module):
    def __init__(self, n_embd, ff_dim):
        super().__init__()
        self.fc1 = nn.Linear(n_embd, ff_dim)
        self.fc2 = nn.Linear(ff_dim, n_embd)
        self.silu = nn.SiLU()  # llama using swiglu? i think its the same ting

    def forward(self, x):
        x = self.silu(self.fc1(x))
        x = self.fc2(x)
        return x


class MultiHeadAttn(nn.Module):
    def __init__(self, n_embd, n_heads, ff_dim, max_len, batch_size):
        super().__init__()
        self.attention = SelfAttn(n_embd=n_embd, n_heads=n_heads, batch_size=batch_size, max_seq_len=max_len)
        self.n1 = nn.RMSNorm(n_embd)
        self.n2 = nn.RMSNorm(n_embd)  # use rmsnorm
        self.n3 = nn.RMSNorm(n_embd)
        self.feed = FeedForward(n_embd, ff_dim)

    def forward(self, x, mask, use_cache=False, past_key_value=None):
        x = self.n1(x)
        
        if use_cache:
            attn, present_key_value = self.attention(x, mask, use_cache=True, past_key_value=past_key_value)
        else:
            attn = self.attention(x, mask)
            present_key_value = None
            
        x = self.n2(attn + x)
        forw = self.feed(x)

        if use_cache:
            return forw, present_key_value
        return forw


## RoPE helper methods:
def precompute_freqs_cis(dim, end, theta=10000.0):
    # precompute the frequency tensor
    device = torch.device("cpu")  # Default to CPU, will be moved to the right device later
    freqs = 1.0 / (theta ** torch.arange(0, dim, 2, device=device)[: (dim // 2)].float() / dim)
    t = torch.arange(end, device=device)
    freqs = torch.outer(t, freqs)

    freqs_cos = torch.cos(freqs)
    freqs_sin = torch.sin(freqs)

    return freqs_cos, freqs_sin


def reshape_for_rotation(x):
    x_reshape = x.reshape(*x.shape[:-1], -1, 2)
    return x_reshape


def apply_RoPE(x, freqs_cos, freqs_sin, position_ids):
    # Ensure position_ids are within bounds
    max_pos = freqs_cos.size(0) - 1
    position_ids = torch.clamp(position_ids, max=max_pos)

    cos = freqs_cos[position_ids].unsqueeze(1)
    sin = freqs_sin[position_ids].unsqueeze(1)

    x_reshaped = reshape_for_rotation(x)
    x1, x2 = x_reshaped[..., 0], x_reshaped[..., 1]

    # Handle dimension mismatch - expand cos and sin if needed
    if x1.size(-1) > cos.size(-1):
        repeat_factor = x1.size(-1) // cos.size(-1)
        cos = torch.repeat_interleave(cos, repeat_factor, dim=-1)
        sin = torch.repeat_interleave(sin, repeat_factor, dim=-1)

    output = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    return output.flatten(-2)
