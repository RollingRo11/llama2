# llama2
**Python implementation of "LLaMA 2: Open Foundation and Fine-Tuned Chat Models" (Touvron et al.)**

Read the paper [here!](https://arxiv.org/abs/2307.09288)

## Paper implementation overview
My implementation of LLaMA 2 is done in a combination of NumPy and PyTorch. I've implemented key LLaMA-specific components from scratch while leveraging PyTorch for standard operations (Linear layers, embeddings, etc.)

The custom implementation includes several key components:

- **RoPE (Rotary Position Embedding)**: Custom implementation of rotary positional embeddings for better handling of positional information
- **Multi-Head Self-Attention with KV Caching**: Efficient attention mechanism with key-value caching for faster inference
- **RMSNorm**: Root Mean Square Layer Normalization as used in LLaMA (instead of standard LayerNorm)
- **SiLU/Swish Activation**: Using SiLU activation function in the feed-forward networks
- **Flash Attention**: Leveraging PyTorch's `F.scaled_dot_product_attention` for memory-efficient attention computation

You can view the implementation of each component in `layers.py` ([here](layers.py))

### Key Architecture Features

**Attention Mechanism:**
- Custom `SelfAttn` module implementing multi-head self-attention with RoPE
- KV caching support for efficient autoregressive generation
- Causal masking for proper language modeling

**Feed-Forward Network:**
- `FeedForward` module with SiLU activation
- Standard up-projection and down-projection pattern

**Layer Structure:**
- `MultiHeadAttn` combines attention and feed-forward with RMSNorm
- Residual connections following the transformer architecture
- Pre-normalization pattern as used in modern transformers

### Training Features
- **Gradient Clipping**: Implemented to stabilize training
- **AdamW Optimizer**: With LLaMA-specific hyperparameters (betas=(0.9, 0.95))
- **Learning Rate**: 3e-4 as commonly used for transformer training
- **Safetensors**: Model checkpointing using safetensors format for security
- **TensorBoard**: Integrated logging for training metrics
- **Mixed Precision**: Float32 matmul precision optimization

### Data Processing
- **Tokenization**: Using tiktoken with GPT-2 encoding
- **Context Length**: Configurable context window (default 256 tokens)
- **Batched Training**: Efficient batched processing with DataLoader

The model supports both training and inference modes, with efficient KV caching during generation for faster autoregressive sampling.