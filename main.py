import torch
import tiktoken
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from model import LlamaTransformer
from safetensors.torch import save_file, load_file
import os
import time

torch.set_float32_matmul_precision("high")  # for gpu perf
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
writer = SummaryWriter()

# Hyperparameters
BATCH_SIZE = 32
CONTEXT_LEN = 256
LR = 3e-4
EPOCHS = 10
EVAL_INTERVAL = 500
VOCAB_SIZE = 50304


class TextDataset(Dataset):
    def __init__(self, data, context_len):
        self.data = data
        self.context_len = context_len

    def __len__(self):
        return len(self.data) - self.context_len

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.context_len]
        y = self.data[idx + 1 : idx + self.context_len + 1]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)


def evaluate(model, dataloader):
    model.eval()
    losses = []
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            logits = model(x)
            loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        losses.append(loss.item())
    return sum(losses) / len(losses)


def generate(model, tokenizer, max_tokens=100, temperature=1.0):
    model.eval()
    prompt = "ROMEO:"
    tokens = tokenizer.encode(prompt)
    x = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)

    generated = []
    past_key_values = None
    with torch.no_grad():
        for _ in range(max_tokens):
            logits, past_key_values = model(x, past_key_values=past_key_values, use_cache=True)
            logits = logits[:, -1, :] / temperature
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated.append(next_token.item())
            x = next_token

    return prompt + tokenizer.decode(generated)


def main():
    enc = tiktoken.get_encoding("gpt2")
    with open("input.txt", "r", encoding="utf-8") as f:
        text = f.read()
    tokens = enc.encode(text)

    dataset = TextDataset(tokens, CONTEXT_LEN)
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    # model
    model = LlamaTransformer(
        vocab_size=VOCAB_SIZE,
        embd_size=512,
        num_layers=6,
        n_heads=8,
        ff_dim=2048,
        max_len=CONTEXT_LEN,
        batch_size=BATCH_SIZE,
    ).to(device)
    # model = torch.compile(model)

    if len(os.sys.argv) > 1 and os.sys.argv[1] == "inference":  # type: ignore
        model_file = "model.safetensors"
        if os.path.exists(model_file):
            state_dict = load_file(model_file)
            model.load_state_dict(state_dict, strict=False)
            print(generate(model, enc))
            return
        else:
            print("No model file found. Training first...")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.95), eps=1e-5)
    step = 0
    best_val_loss = float("inf")

    for epoch in range(EPOCHS):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            t0 = time.time()

            # Forward and backward
            logits = model(x)
            loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # clip gradients
            optimizer.step()

            t1 = time.time()
            dt = (t1 - t0) * 1000
            writer.add_scalar("train_loss", loss.item(), step)
            writer.add_scalar("time taken", dt, step)
            if step % EVAL_INTERVAL == 0:
                val_loss = evaluate(model, val_loader)
                writer.add_scalar("val_loss", val_loss, step)
                print(f"Epoch {epoch}, Step {step}: train_loss={loss.item():.4f}, val_loss={val_loss:.4f}")

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    model_state = {k: v.cpu() for k, v in model.state_dict().items()}
                    save_file(model_state, "model.safetensors")

            step += 1

    test_loss = evaluate(model, test_loader)
    print(f"Test loss: {test_loss:.4f}")
    writer.close()


if __name__ == "__main__":
    print(f"using device {device}")
    main()
