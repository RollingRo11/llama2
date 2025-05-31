# llama2
**Python implementation of "LLaMA 2: Open Foundation and Fine-Tuned Chat Models" (Touvron et al.)**

Read the paper [here!](https://arxiv.org/abs/2307.09288)

## Paper implementation overview
This implementation of the llama 2 architecture is done in a combination of both NumPy and PyTorch.
(I've implemented llama-specific code from scratch in NumPy, and the rest in PyTorch, for readability).

## Some key aspects of the implementation:

### RoPE (Rotary Positional Embedding)
```Python
def init_rotary_embd(self, device):
    if self.freqs_cos is None or self.freqs_sin is None:
        freqs_cos, freqs_sin = precompute_freqs_cis(self.n_heads, self.max_seq_len)
        self.register_buffer("freqs_cos", freqs_cos)
        self.register_buffer("freqs_sin", freqs_sin)

    self.freqs_cos = self.freqs_cos.to(device)
    self.freqs_sin = self.freqs_sin.to(device)

...

def precompute_freqs_cis(dim, end, theta=10000.0):
    device = torch.device("cpu")
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
    max_pos = freqs_cos.size(0) - 1
    position_ids = torch.clamp(position_ids, max=max_pos)

    cos = freqs_cos[position_ids].unsqueeze(1)
    sin = freqs_sin[position_ids].unsqueeze(1)

    x_reshaped = reshape_for_rotation(x)
    x1, x2 = x_reshaped[..., 0], x_reshaped[..., 1]

    if x1.size(-1) > cos.size(-1):
        repeat_factor = x1.size(-1) // cos.size(-1)
        cos = torch.repeat_interleave(cos, repeat_factor, dim=-1)
        sin = torch.repeat_interleave(sin, repeat_factor, dim=-1)

    output = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    return output.flatten(-2)
```

$$
\begin{pmatrix}
q_{m,2i} \\
q_{m,2i+1}
\end{pmatrix}
=
\begin{pmatrix}
\cos(m\theta_i) & -\sin(m\theta_i) \\
\sin(m\theta_i) & \cos(m\theta_i)
\end{pmatrix}
\begin{pmatrix}
x_{m,2i} \\
x_{m,2i+1}
\end{pmatrix}
$$

where $\theta_i = 10000^{-2i/d}$ and $d$ is the dimension of the head.

This can be written more compactly as:
$$
f_q(x_m, m) = (x_m \odot \cos(m\theta)) + (R_d x_m \odot \sin(m\theta))
$$


### KV Cache-ing
```Python
self.k_cache = None
self.v_cache = None

self.qk = torch.zeros(batch_size, max_seq_len, n_heads, self.head_dim)
self.vk = torch.zeros(batch_size, max_seq_len, n_heads, self.head_dim)

...

if past_key_value is not None:
    k_past, v_past = past_key_value
    k = torch.cat([k_past, k], dim=2)
    v = torch.cat([v_past, v], dim=2)

if use_cache:
    present_key_value = (k, v)
else:
    present_key_value = None
```

(plus **Flash Attention** Using PyTorch's `F.scaled_dot_product_attention` for kernel computation)
