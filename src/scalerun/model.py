"""Readable GPT building blocks with the familiar nanoGPT model interface.

This is ScaleRun's implementation; naming follows karpathy/nanoGPT so recipes,
checkpoints and small experiments feel familiar. The leaderboard's ByteGPT
adapter returns full logits and keeps its existing constructor.
"""

from __future__ import annotations

from dataclasses import dataclass
import inspect
import math

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True

    def __post_init__(self) -> None:
        for name in ("block_size", "vocab_size", "n_layer", "n_head", "n_embd"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.n_embd % self.n_head:
            raise ValueError("n_embd must be divisible by n_head")
        if not isinstance(self.dropout, (int, float)) or not 0 <= self.dropout < 1:
            raise ValueError("dropout must be between 0 (inclusive) and 1 (exclusive)")
        if type(self.bias) is not bool:
            raise ValueError("bias must be a boolean")


class LayerNorm(nn.Module):
    """Layer normalization with an optional learned bias."""

    def __init__(self, ndim: int, bias: bool) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, width = x.shape
        q, k, v = self.c_attn(x).chunk(3, dim=-1)
        q, k, v = (
            part.reshape(batch, length, self.n_head, width // self.n_head).transpose(1, 2)
            for part in (q, k, v)
        )
        attended = F.scaled_dot_product_attention(
            q, k, v, is_causal=True,
            dropout_p=self.dropout if self.training else 0.0,
        )
        attended = attended.transpose(1, 2).contiguous().reshape(batch, length, width)
        return self.resid_dropout(self.c_proj(attended))


class MLP(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.c_proj(self.gelu(self.c_fc(x))))


class Block(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, config.bias)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        return x + self.mlp(self.ln_2(x))


class GPT(nn.Module):
    """Decoder-only GPT with tied embeddings and learned absolute positions.

    ``model(idx, targets)`` returns full logits and cross entropy in nats.
    ``model(idx)`` returns just the final position's logits and ``None``.
    """

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(config.vocab_size, config.n_embd),
            "wpe": nn.Embedding(config.block_size, config.n_embd),
            "drop": nn.Dropout(config.dropout),
            "h": nn.ModuleList(Block(config) for _ in range(config.n_layer)),
            "ln_f": LayerNorm(config.n_embd, config.bias),
        })
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight = self.transformer.wte.weight
        self.apply(self._init_weights)
        for name, parameter in self.named_parameters():
            if name.endswith("c_proj.weight"):
                nn.init.normal_(parameter, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def _hidden(self, idx: torch.Tensor) -> torch.Tensor:
        if idx.ndim != 2 or idx.shape[1] == 0:
            raise ValueError("input must have shape (batch, nonempty sequence)")
        length = idx.shape[1]
        if length > self.config.block_size:
            raise ValueError("input exceeds the model context length (block_size)")
        positions = torch.arange(length, device=idx.device)
        x = self.transformer.drop(self.transformer.wte(idx) + self.transformer.wpe(positions))
        for block in self.transformer.h:
            x = block(x)
        return self.transformer.ln_f(x)

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        hidden = self._hidden(idx)
        logits = self.lm_head(hidden if targets is not None else hidden[:, [-1], :])
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, self.config.vocab_size), targets.reshape(-1), ignore_index=-1,
            )
        return logits, loss

    def get_num_params(self, non_embedding: bool = True) -> int:
        count = sum(parameter.numel() for parameter in self.parameters())
        # Token embeddings also form the output matrix, so keep them in this count.
        return count - self.transformer.wpe.weight.numel() if non_embedding else count

    def crop_block_size(self, block_size: int) -> None:
        """Shorten learned positions before configuring an optimizer."""
        if type(block_size) is not int or not 0 < block_size <= self.config.block_size:
            raise ValueError("block_size must be positive and cannot increase the context")
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size].detach())
        self.transformer.wpe.num_embeddings = block_size

    def configure_optimizers(
        self, weight_decay: float, learning_rate: float,
        betas: tuple[float, float], device_type: str,
    ) -> torch.optim.AdamW:
        """Decay matrices; leave biases and normalization vectors undecayed."""
        trainable = [parameter for parameter in self.parameters() if parameter.requires_grad]
        groups = [
            {"params": [p for p in trainable if p.ndim >= 2], "weight_decay": weight_decay},
            {"params": [p for p in trainable if p.ndim < 2], "weight_decay": 0.0},
        ]
        options = {}
        if "fused" in inspect.signature(torch.optim.AdamW).parameters:
            options["fused"] = device_type == "cuda"
        return torch.optim.AdamW(groups, lr=learning_rate, betas=betas, **options)

    @torch.no_grad()
    def generate(
        self, idx: torch.Tensor, max_new_tokens: int,
        temperature: float = 1.0, top_k: int | None = None,
    ) -> torch.Tensor:
        if type(max_new_tokens) is not int or max_new_tokens < 0:
            raise ValueError("max_new_tokens must be a nonnegative integer")
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be positive and finite")
        if top_k is not None and (type(top_k) is not int or top_k < 1):
            raise ValueError("top_k must be a positive integer or None")
        if idx.ndim != 2 or idx.shape[1] == 0:
            raise ValueError("generation needs a nonempty token prompt")
        for _ in range(max_new_tokens):
            # Call GPT.forward directly so the ByteGPT full-logits adapter can
            # share this generation path without a second model implementation.
            logits, _ = GPT.forward(self, idx[:, -self.config.block_size:])
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                threshold = torch.topk(logits, min(top_k, logits.shape[-1])).values[:, [-1]]
                logits = logits.masked_fill(logits < threshold, float("-inf"))
            next_idx = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
            idx = torch.cat((idx, next_idx), dim=1)
        return idx


class ByteGPT(GPT):
    """Leaderboard adapter: the original constructor and full-token logits."""

    def __init__(
        self, n_layer: int, n_head: int, n_embd: int,
        context_length: int, dropout: float = 0.0,
        vocab_size: int = 256, bias: bool = True,
    ) -> None:
        super().__init__(GPTConfig(
            n_layer=n_layer, n_head=n_head, n_embd=n_embd,
            block_size=context_length, dropout=dropout, vocab_size=vocab_size, bias=bias,
        ))

    @property
    def context_length(self) -> int:
        return self.config.block_size

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.lm_head(self._hidden(tokens))
