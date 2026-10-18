"""An original, small decoder-only transformer for the byte baseline.

PyTorch is deliberately imported only by this optional training module. Core
leaderboard and tournament commands do not require a machine-learning runtime.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class CausalAttention(nn.Module):
    def __init__(self, width: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(width, 3 * width, bias=False)
        self.projection = nn.Linear(width, width, bias=False)
        self.dropout = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, width = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q, k, v = (
            part.view(batch, length, self.heads, width // self.heads).transpose(1, 2)
            for part in (q, k, v)
        )
        attended = F.scaled_dot_product_attention(
            q, k, v, is_causal=True,
            dropout_p=self.dropout if self.training else 0.0,
        )
        return self.projection(attended.transpose(1, 2).contiguous().view(batch, length, width))


class DecoderBlock(nn.Module):
    def __init__(self, width: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(width)
        self.attention = CausalAttention(width, heads, dropout)
        self.mlp_norm = nn.LayerNorm(width)
        self.mlp = nn.Sequential(
            nn.Linear(width, 4 * width), nn.GELU(),
            nn.Linear(4 * width, width), nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.attention_norm(x))
        return x + self.mlp(self.mlp_norm(x))


class ByteGPT(nn.Module):
    """Learned positions, pre-norm blocks, and a tied 256-byte output head."""

    def __init__(
        self, n_layer: int, n_head: int, n_embd: int,
        context_length: int, dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.context_length = context_length
        self.token_embedding = nn.Embedding(256, n_embd)
        self.position_embedding = nn.Embedding(context_length, n_embd)
        self.blocks = nn.ModuleList(
            DecoderBlock(n_embd, n_head, dropout) for _ in range(n_layer)
        )
        self.final_norm = nn.LayerNorm(n_embd)
        self.output = nn.Linear(n_embd, 256, bias=False)
        self.output.weight = self.token_embedding.weight
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        length = tokens.shape[1]
        if length > self.context_length:
            raise ValueError("input exceeds the model context length")
        positions = torch.arange(length, device=tokens.device)
        x = self.token_embedding(tokens) + self.position_embedding(positions)
        for block in self.blocks:
            x = block(x)
        return self.output(self.final_norm(x))
