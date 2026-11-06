
import torch

from src.model import NavaGPT
from src.generation import generate
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

model = NavaGPT(
    vocab_size=checkpoint["vocab_size"],
    **checkpoint["model_config"]
).to(device)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

prompt = "Artificial intelligence"

input_ids = torch.tensor(
    [tokenizer.encode(prompt)],
    dtype=torch.long,
    device=device
)

output = generate(
    model,
    input_ids,
    max_new_tokens=30,
    temperature=0.8,
    top_k=5
)

print("\nNavaGPT:")
print(tokenizer.decode(output[0].tolist()))
