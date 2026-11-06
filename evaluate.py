
import torch
from pathlib import Path

from src.model import NavaGPT
from src.bpe_tokenizer import BPETokenizer


device = "cuda" if torch.cuda.is_available() else "cpu"

checkpoint = torch.load(
    "checkpoints/navagpt.pth",
    map_location=device,
    weights_only=True
)

tokenizer = BPETokenizer()
tokenizer.merges = checkpoint["merges"]
tokenizer.vocab = checkpoint["vocab"]
tokenizer.inverse_vocab = checkpoint["inverse_vocab"]

text = Path("data/input.txt").read_text(encoding="utf-8")
data = torch.tensor(tokenizer.encode(text), dtype=torch.long)

block_size = checkpoint["model_config"]["block_size"]

split = int(0.8 * len(data))
val_data = data[split:]

model = NavaGPT(
    vocab_size=checkpoint["vocab_size"],
    **checkpoint["model_config"]
).to(device)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

total_loss = 0.0
total_tokens = 0

with torch.no_grad():
    for start in range(0, len(val_data) - block_size, block_size):
        x = val_data[start:start + block_size].unsqueeze(0).to(device)
        y = val_data[start + 1:start + block_size + 1].unsqueeze(0).to(device)

        _, loss = model(x, y)

        total_loss += loss.item() * y.numel()
        total_tokens += y.numel()

if total_tokens == 0:
    raise ValueError("Not enough validation data.")

average_loss = total_loss / total_tokens
perplexity = torch.exp(torch.tensor(average_loss)).item()

print("\nNavaGPT BPE Evaluation Results")
print("--------------------------------")
print(f"Validation Loss: {average_loss:.4f}")
print(f"Perplexity: {perplexity:.4f}")
print("Evaluation completed!")
