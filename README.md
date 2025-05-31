# llama2
**Python implementation of "LLaMA 2: Open Foundation and Fine-Tuned Chat Models" (Touvron et al.)**

Read the paper [here!](https://arxiv.org/abs/2307.09288)

## Paper implementation overview
This implementation of the llama 2 architecture is done in a combination of both NumPy and PyTorch.
(I've implemented llama-specific code from scratch in NumPy, and the rest in PyTorch, for readability).

The custom implementation includes:

- **RoPE (Rotary Position Embedding)**:
```Python
def init_rotary_embd(self, device):
    if self.freqs_cos is None or self.freqs_sin is None:
        freqs_cos, freqs_sin = precompute_freqs_cis(self.n_heads, self.max_seq_len)
        self.register_buffer("freqs_cos", freqs_cos)
        self.register_buffer("freqs_sin", freqs_sin)

    self.freqs_cos = self.freqs_cos.to(device)
    self.freqs_sin = self.freqs_sin.to(device)
```

- **KV Cache-ing**:
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

- **Flash Attention**: Using PyTorch's `F.scaled_dot_product_attention` for kernel computation.
