import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from layers import SelfAttn, MultiHeadAttn, FeedForward

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"


class LlamaTransformer(nn.Module):
    def __init__(self, vocab_size, embd_size, num_layers, n_heads, ff_dim, max_len, batch_size):
        super().__init__()

        self.embd_size = embd_size
        self.token_embd = nn.Embedding(vocab_size, embd_size)

        self.layers = nn.ModuleList(
            [MultiHeadAttn(embd_size, n_heads, ff_dim, max_len, batch_size) for _ in range(num_layers)]
        )
        self.fc_out = nn.Linear(embd_size, vocab_size)
        self.max_length = max_len
        self.rmsnorm = nn.RMSNorm(embd_size)

    def forward(self, x, past_key_values=None, use_cache=False):
        batch_size, seq_len = x.shape

        if past_key_values is None:
            past_key_values = [None] * len(self.layers)

        if past_key_values[0] is not None:
            kv_seq_len = past_key_values[0][0].size(2) + seq_len
            mask = torch.tril(torch.ones((kv_seq_len, kv_seq_len))).unsqueeze(0).unsqueeze(0)
            mask = mask[:, :, -seq_len:, :].to(x.device)
        else:
            mask = torch.tril(torch.ones((seq_len, seq_len))).unsqueeze(0).unsqueeze(0)
            mask = mask.to(x.device)

        x = self.token_embd(x)

        present_key_values = []
        for i, layer in enumerate(self.layers):
            if use_cache:
                x, present_kv = layer(x, mask, use_cache=True, past_key_value=past_key_values[i])
                present_key_values.append(present_kv)
            else:
                x = layer(x, mask)

        x = self.rmsnorm(x)
        x = self.fc_out(x)

        if use_cache:
            return x, present_key_values
        return x
