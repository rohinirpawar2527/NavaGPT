
import torch
from pathlib import Path

from src.tokenizer import Tokenizer
from src.model import NavaGPT

torch.manual_seed(42)

device = "cuda" if torch.cuda.is_available() else "cpu"

text = Path("data/input.txt").read_text(encoding="utf-8")

tokenizer = Tokenizer()
tokenizer.fit(text)

data = torch.tensor(tokenizer.encode(text), dtype=torch.long)

block_size = 8
batch_size = 4
max_iters = 300
learning_rate = 0.001

split = int(0.8 * len(data))
train_data = data[:split]
val_data = data[split:]

if len(train_data) <= block_size or len(val_data) <= block_size:
    raise ValueError("Dataset is too small. Add more text to data/input.txt.")


def get_batch(source):
    starts = torch.randint(
        len(source) - block_size,
        (batch_size,)
    )

    x = torch.stack([
        source[i:i + block_size] for i in starts
    ])

    y = torch.stack([
        source[i + 1:i + block_size + 1] for i in starts
    ])

    return x.to(device), y.to(device)


model = NavaGPT(
    vocab_size=len(tokenizer.word_to_id),
    embed_dim=64,
    num_heads=4,
    num_layers=2,
    block_size=block_size
).to(device)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=learning_rate
)

print("Starting NavaGPT training...")
print("Vocabulary size:", len(tokenizer.word_to_id))
print("Training device:", device)

for step in range(max_iters):
    model.train()

    x, y = get_batch(train_data)

    logits, loss = model(x, y)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    if step % 50 == 0:
        print(f"Step {step}: Loss = {loss.item():.4f}")

Path("checkpoints").mkdir(exist_ok=True)

torch.save({
    "model_state_dict": model.state_dict(),
    "vocab_size": len(tokenizer.word_to_id),
    "word_to_id": tokenizer.word_to_id,
    "id_to_word": tokenizer.id_to_word,
    "block_size": block_size
}, "checkpoints/navagpt.pth")

print("\nTraining completed!")
print("Model saved: checkpoints/navagpt.pth")
