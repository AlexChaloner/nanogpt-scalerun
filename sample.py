"""Sample an ScaleRun run with nanoGPT-style command-line settings.

Example: python sample.py --out_dir=runs/shakespeare --start='ROMEO:'
The manifest selects the tokenizer and verifies the saved model before loading.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable

import torch

from scalerun.model import GPT, GPTConfig
from scalerun.training import sha256_file


class TextTokenizer:
    """Small inference adapter for the tokenizer recorded in a run manifest."""

    def __init__(self, metadata: dict[str, Any]) -> None:
        self.vocab_size = metadata.get("vocab_size")
        name = metadata.get("name")
        if name == "utf8-bytes" and self.vocab_size == 256:
            self._encode: Callable[[str], list[int]] = lambda text: list(text.encode("utf-8"))
            self._decode: Callable[[list[int]], str] = lambda ids: bytes(ids).decode("utf-8", errors="replace")
        elif name == "char":
            chars = metadata.get("chars")
            if (
                not isinstance(chars, list) or len(chars) != self.vocab_size
                or any(not isinstance(c, str) or len(c) != 1 for c in chars)
                or len(set(chars)) != len(chars)
            ):
                raise ValueError("character tokenizer requires its complete unique chars list")
            indices = {char: index for index, char in enumerate(chars)}

            def encode(text: str) -> list[int]:
                unknown = sorted(set(text) - indices.keys())
                if unknown:
                    raise ValueError(f"prompt contains characters absent from the dataset: {unknown!r}")
                return [indices[char] for char in text]

            self._encode = encode
            self._decode = lambda ids: "".join(chars[index] for index in ids)
        elif name == "gpt2" and self.vocab_size == 50257 and metadata.get("encoding", "gpt2") == "gpt2":
            try:
                import tiktoken
            except ImportError as error:
                raise RuntimeError("GPT-2 tokenization requires tiktoken; install nanogpt-scalerun[data]") from error
            encoding = tiktoken.get_encoding("gpt2")
            self._encode = lambda text: encoding.encode(text, allowed_special={"<|endoftext|>"})
            self._decode = encoding.decode
        else:
            raise ValueError("unsupported tokenizer in run manifest")

    def encode(self, text: str) -> list[int]:
        return self._encode(text)

    def decode(self, ids: list[int]) -> str:
        return self._decode(ids)


def _model_config(model_args: dict[str, Any]) -> GPTConfig:
    arguments = dict(model_args)
    if "context_length" in arguments:
        arguments["block_size"] = arguments.pop("context_length")
    arguments.setdefault("vocab_size", 256)
    return GPTConfig(**arguments)


def _upgrade_byte_state(state: dict[str, torch.Tensor], model: GPT) -> dict[str, torch.Tensor]:
    """Read byte checkpoints produced before the familiar GPT naming existed."""
    if "token_embedding.weight" not in state:
        return state
    renamed = {}
    for name, value in state.items():
        name = name.replace("token_embedding.", "transformer.wte.")
        name = name.replace("position_embedding.", "transformer.wpe.")
        name = name.replace("blocks.", "transformer.h.")
        name = name.replace("attention_norm.", "ln_1.").replace("mlp_norm.", "ln_2.")
        name = name.replace("attention.qkv.", "attn.c_attn.")
        name = name.replace("attention.projection.", "attn.c_proj.")
        name = name.replace("mlp.0.", "mlp.c_fc.").replace("mlp.2.", "mlp.c_proj.")
        name = name.replace("final_norm.", "transformer.ln_f.").replace("output.", "lm_head.")
        renamed[name] = value
    # The old byte attention projections omitted bias. Zero bias preserves them.
    for name, value in model.state_dict().items():
        if name not in renamed and (name.endswith("attn.c_attn.bias") or name.endswith("attn.c_proj.bias")):
            renamed[name] = torch.zeros_like(value)
    return renamed


def load_run(out_dir: Path | str, device: str = "cpu") -> tuple[GPT, TextTokenizer]:
    """Load only tensors/basic metadata after checking the manifest's model hash."""
    out_dir = Path(out_dir)
    manifest = json.loads((out_dir / "run_manifest.json").read_text(encoding="utf-8"))
    artifact = out_dir / "model.pt"
    if artifact.exists():
        expected_hash = manifest.get("artifact_sha256")
        config = _model_config(manifest["recipe"]["model"])
    else:
        artifact = out_dir / "ckpt.pt"
        expected_hash = manifest.get("ckpt_sha256")
        config = None
    if not isinstance(expected_hash, str) or sha256_file(artifact) != expected_hash:
        raise ValueError("checkpoint hash does not match run_manifest.json")
    checkpoint = torch.load(artifact, map_location="cpu", weights_only=True)
    if config is None:
        if not isinstance(checkpoint, dict) or "model_args" not in checkpoint or "model" not in checkpoint:
            raise ValueError("ckpt.pt must contain model_args and model")
        config = _model_config(checkpoint["model_args"])
        state = checkpoint["model"]
    else:
        state = checkpoint
    tokenizer = TextTokenizer(manifest["dataset"]["tokenizer"])
    if config.vocab_size != tokenizer.vocab_size:
        raise ValueError("checkpoint vocabulary does not match its tokenizer")
    model = GPT(config)
    # Compiled upstream modules may prefix state names; loading still stays strict.
    state = {name.removeprefix("_orig_mod."): value for name, value in state.items()}
    model.load_state_dict(_upgrade_byte_state(state, model), strict=True)
    return model.to(device).eval(), tokenizer


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out_dir", type=Path, default=Path("out"))
    parser.add_argument("--start", default="\n", help="prompt text, or FILE:path to a UTF-8 prompt")
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=200, help="0 disables top-k truncation")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:N, or mps")
    args = parser.parse_args(argv)
    if args.num_samples < 1:
        parser.error("num_samples must be positive")
    if args.max_new_tokens < 0 or args.top_k < 0:
        parser.error("max_new_tokens and top_k must be nonnegative")
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        torch.manual_seed(args.seed)
        model, tokenizer = load_run(args.out_dir, device)
        prompt = Path(args.start[5:]).read_text(encoding="utf-8") if args.start.startswith("FILE:") else args.start
        tokens = tokenizer.encode(prompt)
        if not tokens:
            raise ValueError("start must encode at least one token")
        context = torch.tensor([tokens], dtype=torch.long, device=device)
        for sample_number in range(args.num_samples):
            if sample_number:
                print("\n---------------\n")
            generated = model.generate(
                context, args.max_new_tokens, temperature=args.temperature,
                top_k=args.top_k or None,
            )
            print(tokenizer.decode(generated[0].tolist()))
    except (OSError, KeyError, ValueError, RuntimeError) as error:
        parser.exit(1, f"sample: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
