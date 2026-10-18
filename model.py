"""The familiar ``from model import GPTConfig, GPT`` entry point.

The implementation lives in src/scalerun/model.py and is shared with contest runs.
Install once with ``pip install -e '.[train]'`` before running root scripts.
"""

from scalerun.model import Block, ByteGPT, CausalSelfAttention, GPT, GPTConfig, LayerNorm, MLP

__all__ = ["GPTConfig", "GPT", "ByteGPT", "LayerNorm", "CausalSelfAttention", "MLP", "Block"]
