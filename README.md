# llama2
**Python implementation of "LLaMA 2: Open Foundation and Fine-Tuned Chat Models" (Touvron et al.)**

Read the paper [here!](https://arxiv.org/abs/2307.09288)

## Paper implementation overview

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

```math
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
```

where $\theta_i = 10000^{-2i/d}$ and $d$ is the dimension of the head.

```math
f_q(x_m, m) = (x_m \odot \cos(m\theta)) + (R_d x_m \odot \sin(m\theta))
```

where $R_d$ rotates the vector by swapping and negating the alternate dimensions, and $\odot$ denotes element-wise mult.


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


# The fun part! Sample output:
The default prompt for the model is `ROMEO:`, of which some sample outputs are:
```
(.venv) rohan@Floppy ~/D/c/llama2 (main)> uv run main.py inference
using device mps
ROMEO:
Well, call the flattering- curses: but, deny,
And look thee from the search of dimmseyman.'
But let their innocent alack, I'll be with Englishman.

Nurse:
I'll have this come in this while one jot
As he that slew him; he will not find a flower.

JULIET:
Nurse?

Nurse:
Ah, well-day, I have mercy to back
```

With `SKIBIDI:` as the prompt.
```
(.venv) rohan@Floppy ~/D/c/llama2 (main)> uv run main.py inference
using device mps
SKIBIDI:
Stay: this news so!

PAULINA:
I say, the king: I know thy lord:
Beseech you, give me't;
As I must go to the queen'?

BENVOLIO:
Why, what I will do not suffer it,
Why I shall be a fellow.

BAPTISTA:
Well, this was husband.

HORTENSIO:
I'll waste my daughter Kate
```

With `ROHAN:` as the prompt:
```
(.venv) rohan@Floppy ~/D/c/llama2 (main)> uv run main.py inference
using device mps
ROHAN:
It is a banish'd from the wager.

CLIFFORD:
Plantagenet, I find the kingdom after my sword:
d my tongue.

LADY ANNE:
Dost thou not? will I swear to slay thyself?

KING EDWARD IV:
Seize on the act of Clarence sweeps,
And says aught I am a subject.

GLOUCESTER:
Why,ator, my
```
