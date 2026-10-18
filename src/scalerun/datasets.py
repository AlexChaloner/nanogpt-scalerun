"""Explicit token formats for nanoGPT-style data directories.

The original byte track remains schema 1. Schema 2 stores little-endian uint16
token IDs with JSON tokenizer metadata; no pickle metadata is executed.
"""

from __future__ import annotations

from array import array
from contextlib import contextmanager
import hashlib
import json
import math
import mmap
from pathlib import Path
import shutil
import sys


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path):
    from .cli import read_json
    return read_json(path)


def _identity(tokenizer, train_hash, val_hash):
    return hashlib.sha256(_canonical({"tokenizer": tokenizer, "dtype": "uint16",
                                     "train_sha256": train_hash, "val_sha256": val_hash})).hexdigest()


def _validate_tokenizer(tokenizer):
    if not isinstance(tokenizer, dict):
        raise ValueError("tokenizer metadata must be an object")
    if type(tokenizer.get("version")) is not int or tokenizer["version"] != 1:
        raise ValueError("unsupported tokenizer version")
    if tokenizer.get("name") == "char":
        if set(tokenizer) != {"name", "version", "vocab_size", "chars"}:
            raise ValueError("character metadata needs name, version, vocab_size, chars")
        chars = tokenizer["chars"]
        if (not isinstance(chars, list) or not chars or len(chars) > 65536
                or any(not isinstance(char, str) or len(char) != 1 for char in chars)
                or len(set(chars)) != len(chars)):
            raise ValueError("chars must contain unique Unicode characters")
        if type(tokenizer["vocab_size"]) is not int or tokenizer["vocab_size"] != len(chars):
            raise ValueError("character vocabulary size does not match chars")
    elif tokenizer.get("name") == "gpt2":
        if tokenizer != {"name": "gpt2", "version": 1, "vocab_size": 50257, "encoding": "gpt2"}:
            raise ValueError("GPT-2 metadata must identify the fixed 50257-token gpt2 encoding")
    else:
        raise ValueError("supported uint16 tokenizers are char and gpt2")
    return tokenizer


class TokenBuffer:
    """Read-only uint16 token stream, with token rather than byte offsets."""

    def __init__(self, path):
        self.path = Path(path)
        self.stream = self.mapping = self.view = None

    def __enter__(self):
        self.stream = self.path.open("rb")
        self.mapping = mmap.mmap(self.stream.fileno(), 0, access=mmap.ACCESS_READ)
        if sys.byteorder == "little":
            self.view = memoryview(self.mapping).cast("H")
        return self

    def __len__(self):
        return len(self.mapping) // 2

    def __getitem__(self, key):
        if self.view is not None:
            value = self.view[key]
            if isinstance(key, slice):
                try:
                    return value.tolist()
                finally:
                    value.release()
            return value
        if isinstance(key, slice):
            return [self[index] for index in range(*key.indices(len(self)))]
        if key < 0:
            key += len(self)
        if not 0 <= key < len(self):
            raise IndexError(key)
        return int.from_bytes(self.mapping[key * 2:key * 2 + 2], "little")

    def __exit__(self, *exc):
        if self.view is not None:
            self.view.release()
        self.mapping.close()
        self.stream.close()


@contextmanager
def open_tokens(data_dir, split, manifest):
    if split not in ("train", "val"):
        raise ValueError("split must be train or val")
    path = Path(data_dir) / manifest[split]["path"]
    if manifest["schema_version"] == 1:
        with path.open("rb") as stream:
            with mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as data:
                yield data
    else:
        with TokenBuffer(path) as data:
            yield data


def load_manifest(data_dir):
    """Verify hashes, token counts, vocabulary bounds, and tokenizer identity."""
    root = Path(data_dir)
    manifest = _read_json(root / "manifest.json")
    if not isinstance(manifest, dict):
        raise ValueError("dataset manifest must be an object")
    if manifest.get("schema_version") == 1:
        from .training import load_data_manifest
        return load_data_manifest(root)
    if set(manifest) != {"schema_version", "dataset_id", "dtype", "tokenizer", "train", "val"}:
        raise ValueError("invalid uint16 dataset manifest fields")
    if type(manifest["schema_version"]) is not int or manifest["schema_version"] != 2:
        raise ValueError("unsupported dataset manifest version")
    if manifest["dtype"] != "uint16":
        raise ValueError("schema 2 token streams must be little-endian uint16")
    tokenizer = _validate_tokenizer(manifest["tokenizer"])
    for split in ("train", "val"):
        info = manifest[split]
        if not isinstance(info, dict) or set(info) != {"path", "tokens", "sha256"}:
            raise ValueError(f"invalid {split} metadata")
        if info["path"] != f"{split}.bin":
            raise ValueError("token streams must be train.bin and val.bin")
        if type(info["tokens"]) is not int or info["tokens"] < 2:
            raise ValueError(f"{split} must contain at least two tokens")
        path = root / info["path"]
        if path.stat().st_size != info["tokens"] * 2:
            raise ValueError(f"{split} token count does not match the uint16 file size")
        if _sha(path) != info["sha256"]:
            raise ValueError(f"{split} data hash does not match the manifest")
        with TokenBuffer(path) as data:
            for first in range(0, len(data), 65536):
                if max(data[first:first + 65536]) >= tokenizer["vocab_size"]:
                    raise ValueError(f"{split} contains an ID outside its tokenizer vocabulary")
    expected = _identity(tokenizer, manifest["train"]["sha256"], manifest["val"]["sha256"])
    if manifest["dataset_id"] != expected:
        raise ValueError("dataset identity does not match its metadata")
    return manifest


def _destination(root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    for name in ("train.bin", "val.bin", "manifest.json", "meta.json"):
        if (root / name).exists():
            raise ValueError(f"prepared artifact already exists: {root / name}")
    return root


def _finish(root, tokenizer):
    manifest = {"schema_version": 2, "dtype": "uint16", "tokenizer": tokenizer}
    for split in ("train", "val"):
        path = root / f"{split}.bin"
        manifest[split] = {"path": path.name, "tokens": path.stat().st_size // 2, "sha256": _sha(path)}
    manifest["dataset_id"] = _identity(tokenizer, manifest["train"]["sha256"], manifest["val"]["sha256"])
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    (root / "meta.json").write_text(json.dumps(tokenizer, indent=2, ensure_ascii=False) + "\n")
    return load_manifest(root)


def _write_tokens(path, tokens):
    packed = array("H", tokens)
    if sys.byteorder != "little":
        packed.byteswap()
    with path.open("wb") as stream:
        packed.tofile(stream)


def prepare_text(input_path, output_dir, tokenizer="char", val_fraction=0.1):
    """Prepare a local UTF-8 corpus with a deterministic held-out suffix."""
    if type(val_fraction) not in (int, float) or not math.isfinite(val_fraction) or not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be between 0 and 1")
    text = Path(input_path).read_text(encoding="utf-8")
    boundary = int(len(text) * (1 - val_fraction))
    train, val = text[:boundary], text[boundary:]
    if tokenizer == "char":
        chars = sorted(set(text))
        metadata = {"name": "char", "version": 1, "vocab_size": len(chars), "chars": chars}
        _validate_tokenizer(metadata)
        mapping = {char: index for index, char in enumerate(chars)}
        encode = lambda source: [mapping[char] for char in source]
    elif tokenizer == "gpt2":
        try:
            import tiktoken
        except ImportError as exc:
            raise RuntimeError("GPT-2 preparation needs tiktoken: pip install -e '.[data]'") from exc
        metadata = {"name": "gpt2", "version": 1, "vocab_size": 50257, "encoding": "gpt2"}
        encode = tiktoken.get_encoding("gpt2").encode_ordinary
    else:
        raise ValueError("tokenizer must be char or gpt2")
    train_tokens, val_tokens = encode(train), encode(val)
    if min(len(train_tokens), len(val_tokens)) < 2:
        raise ValueError("both data splits need at least two tokens")
    root = _destination(output_dir)
    _write_tokens(root / "train.bin", train_tokens)
    _write_tokens(root / "val.bin", val_tokens)
    return _finish(root, metadata)


def import_tokens(train_path, val_path, output_dir, tokenizer="gpt2", meta_path=None):
    """Import existing nanoGPT uint16 binaries without guessing their encoding."""
    train_path, val_path = Path(train_path), Path(val_path)
    if train_path.resolve() == val_path.resolve() or _sha(train_path) == _sha(val_path):
        raise ValueError("training and validation token streams must be different")
    if tokenizer == "gpt2":
        metadata = {"name": "gpt2", "version": 1, "vocab_size": 50257, "encoding": "gpt2"}
    elif tokenizer == "char" and meta_path:
        metadata = _read_json(meta_path)
        _validate_tokenizer(metadata)
        if metadata["name"] != "char":
            raise ValueError("--tokenizer=char requires character metadata")
    else:
        raise ValueError("character token import requires safe --meta JSON with the original token-ID order")
    for source in (train_path, val_path):
        if source.stat().st_size < 4 or source.stat().st_size % 2:
            raise ValueError("token streams must contain whole little-endian uint16 IDs")
    root = _destination(output_dir)
    shutil.copyfile(train_path, root / "train.bin")
    shutil.copyfile(val_path, root / "val.bin")
    return _finish(root, metadata)
