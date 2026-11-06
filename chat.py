
import torch

from src.model import NavaGPT
from src.tokenizer import Tokenizer
from src.generation import generate


device = "cuda" if torch.cuda.is_available() else "cpu"

checkpoint = torch.load(
    "checkpoints/navagpt.pth",
    map_location=device,
    weights_only=True
)

tokenizer = Tokenizer()
tokenizer.word_to_id = checkpoint["word_to_id"]
tokenizer.id_to_word = checkpoint["id_to_word"]

model = NavaGPT(
    vocab_size=checkpoint["vocab_size"],
    embed_dim=64,
    num_heads=4,
    num_layers=2,
    block_size=checkpoint["block_size"]
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
    max_new_tokens=15,
    temperature=0.8,
    top_k=5
)

print("\nNavaGPT:")
print(tokenizer.decode(output[0].tolist()))
