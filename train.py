
import torch
import csv
from pathlib import Path

from src.bpe_tokenizer import BPETokenizer
from src.model import NavaGPT


torch.manual_seed(42)

device = "cuda" if torch.cuda.is_available() else "cpu"

text = Path("data/input.txt").read_text(encoding="utf-8")

tokenizer = BPETokenizer()
tokenizer.fit(text, num_merges=100)

data = torch.tensor(tokenizer.encode(text), dtype=torch.long)

block_size = 8
batch_size = 4
max_iters = 300
learning_rate = 0.001
eval_interval = 50
eval_iters = 20

split = int(0.8 * len(data))

train_data = data[:split]
val_data = data[split:]

if len(train_data) <= block_size or len(val_data) <= block_size:
    raise ValueError("Dataset is too small.")

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


model_config = {
    "embed_dim": 64,
    "num_heads": 4,
    "num_layers": 2,
    "block_size": block_size,
    "dropout": 0.1
}

model = NavaGPT(
    vocab_size=len(tokenizer.vocab),
    **model_config
).to(device)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=learning_rate
)

history = []


@torch.no_grad()
def estimate_loss():
    model.eval()

    results = {}

    for name, source in [
        ("train", train_data),
        ("validation", val_data)
    ]:
        losses = []

        for _ in range(eval_iters):
            x, y = get_batch(source)
            _, loss = model(x, y)
            losses.append(loss.item())

        results[name] = sum(losses) / len(losses)

    model.train()
    return results


print("Starting NavaGPT training...")
print("Vocabulary size:", len(tokenizer.vocab))
print("Training device:", device)


for step in range(max_iters):

    if step % eval_interval == 0:
        losses = estimate_loss()

        history.append({
            "step": step,
            "training_loss": losses["train"],
            "validation_loss": losses["validation"]
        })

        print(
            f"Step {step}: "
            f"Training Loss = {losses['train']:.4f}, "
            f"Validation Loss = {losses['validation']:.4f}"
        )

    x, y = get_batch(train_data)

    _, loss = model(x, y)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()


Path("checkpoints").mkdir(exist_ok=True)
Path("experiments").mkdir(exist_ok=True)

torch.save({
    "model_state_dict": model.state_dict(),
    "vocab_size": len(tokenizer.vocab),
    "vocab": tokenizer.vocab,
    "inverse_vocab": tokenizer.inverse_vocab,
    "merges": tokenizer.merges,
    "model_config": model_config
}, "checkpoints/navagpt.pth")


with open("experiments/training_history.csv", "w", newline="") as file:
    writer = csv.DictWriter(
        file,
        fieldnames=["step", "training_loss", "validation_loss"]
    )
    writer.writeheader()
    writer.writerows(history)


print("\nTraining completed!")
print("Model saved successfully.")
print("Training history saved: experiments/training_history.csv")
