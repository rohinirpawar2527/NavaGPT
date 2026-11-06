
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.transformer import TransformerBlock


class NavaGPT(nn.Module):
    def __init__(
        self,
        vocab_size,
        embed_dim=128,
        num_heads=4,
        num_layers=4,
        block_size=128,
        dropout=0.1
    ):
        super().__init__()

        self.block_size = block_size

        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(block_size, embed_dim)

        self.dropout = nn.Dropout(dropout)

        self.blocks = nn.Sequential(
            *[
                TransformerBlock(
                    embed_dim, num_heads, block_size, dropout
                )
                for _ in range(num_layers)
            ]
        )

        self.final_norm = nn.LayerNorm(embed_dim)
        self.language_head = nn.Linear(embed_dim, vocab_size)

    def forward(self, idx, targets=None):
        batch_size, sequence_length = idx.shape

        if sequence_length > self.block_size:
            raise ValueError("Sequence length exceeds block size.")

        positions = torch.arange(
            sequence_length, device=idx.device
        )

        token_embeddings = self.token_embedding(idx)
        position_embeddings = self.position_embedding(positions)

        x = token_embeddings + position_embeddings
        x = self.dropout(x)

        x = self.blocks(x)
        x = self.final_norm(x)

        logits = self.language_head(x)

        loss = None

        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1)
            )

        return logits, loss
