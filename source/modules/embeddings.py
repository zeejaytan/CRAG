import numpy as np
import torch
import torch.nn as nn


class AbsolutePositionalEmbedding(nn.Module):
    def __init__(self, max_length: int, embedding_dim: int):
        super().__init__()
        pe = torch.zeros(max_length, embedding_dim)
        position = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * (-np.log(10000.0) / embedding_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_length, embedding_dim)
        self.register_buffer("pe", pe)
        self.embedding = nn.Embedding.from_pretrained(pe.squeeze(0), freeze=True)

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        return self.embedding(positions)
