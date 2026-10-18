"""Byte data preparation and a budgeted, optional PyTorch training baseline.

The FLOPs counter is an analytic *proxy*, not a hardware profiler. Its definition
is versioned here so that runs use the same budget accounting at every scale.
"""

from __future__ import annotations

import codecs
import hashlib
import json
import math
import mmap
import os
from pathlib import Path
import re
import shutil
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4


TOKENIZER = {"name": "utf8-bytes", "version": 1, "vocab_size": 256}
FLOPS_PROXY = {
    "name": "dense-parameters-plus-attention-v1",
    "formula": "6 * P * B * T + 12 * L * B * T * T * D per optimizer update",
    "definitions": "P=unique trainable parameters, B=batch size, T=context length, L=layers, D=embedding width",
    "includes": "Approximate dense parameter forward/backward work and attention QK/AV forward/backward matmuls; multiply-add counts as two FLOPs.",
    "excludes": "Optimizer, normalization, activations, data loading, evaluation, and actual kernel/hardware behavior. Embeddings are approximated as dense parameters.",
}


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _text_info(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    decoder = codecs.getincrementaldecoder("utf-8")("strict")
    size = 0
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            decoder.decode(block)
            digest.update(block)
            size += len(block)
    decoder.decode(b"", final=True)
    if size == 0:
        raise ValueError(f"input is empty: {path}")
    return {"tokens": size, "sha256": digest.hexdigest()}


def _dataset_id(train_hash: str, val_hash: str) -> str:
    return hashlib.sha256(_canonical({
        "tokenizer": TOKENIZER, "train_sha256": train_hash, "val_sha256": val_hash,
    })).hexdigest()


def prepare_data(train_path: Path | str, val_path: Path | str, output_dir: Path | str) -> dict[str, Any]:
    """Copy two explicit UTF-8 splits to a fixed, 256-byte vocabulary dataset.

    Identical files and complete-file containment are rejected. Detecting all
    partial or semantic overlap is the dataset curator's responsibility.
    Existing prepared datasets are never silently replaced.
    """
    train_path, val_path, output_dir = Path(train_path), Path(val_path), Path(output_dir)
    if train_path.resolve() == val_path.resolve() or os.path.samefile(train_path, val_path):
        raise ValueError("training and validation must be separate files")
    train_info, val_info = _text_info(train_path), _text_info(val_path)
    if train_info["sha256"] == val_info["sha256"]:
        raise ValueError("training and validation contents are identical")
    with train_path.open("rb") as train_stream, val_path.open("rb") as val_stream:
        with mmap.mmap(train_stream.fileno(), 0, access=mmap.ACCESS_READ) as train_map:
            with mmap.mmap(val_stream.fileno(), 0, access=mmap.ACCESS_READ) as val_map:
                larger, smaller = (train_map, val_map) if len(train_map) >= len(val_map) else (val_map, train_map)
                if larger.find(smaller) != -1:
                    raise ValueError("one entire data split occurs inside the other; provide disjoint splits")
    manifest = {
        "schema_version": 1,
        "dataset_id": _dataset_id(train_info["sha256"], val_info["sha256"]),
        "tokenizer": dict(TOKENIZER),
        "train": {"path": "train.bin", **train_info},
        "val": {"path": "val.bin", **val_info},
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename in ("train.bin", "val.bin", "manifest.json"):
        if (output_dir / filename).exists():
            raise ValueError(f"prepared data already exists: {output_dir / filename}")
    shutil.copyfile(train_path, output_dir / "train.bin")
    shutil.copyfile(val_path, output_dir / "val.bin")
    # Recheck destination hashes so a source changed during preparation cannot
    # leave a manifest that falsely describes the bytes actually used.
    for split in ("train", "val"):
        if sha256_file(output_dir / manifest[split]["path"]) != manifest[split]["sha256"]:
            raise ValueError("input changed during data preparation; prepare again in a new directory")
    _write_json(output_dir / "manifest.json", manifest)
    return manifest


def load_data_manifest(data_dir: Path | str) -> dict[str, Any]:
    """Verify a prepared dataset's tokenizer, byte counts, hashes and identity."""
    data_dir = Path(data_dir)
    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or type(manifest.get("schema_version")) is not int or manifest["schema_version"] != 1 or manifest.get("tokenizer") != TOKENIZER:
        raise ValueError("unsupported data manifest or tokenizer")
    for split in ("train", "val"):
        info = manifest.get(split, {})
        if info.get("path") != f"{split}.bin":
            raise ValueError(f"manifest must reference {split}.bin")
        path = data_dir / info["path"]
        if type(info.get("tokens")) is not int or info["tokens"] < 1 or path.stat().st_size != info["tokens"]:
            raise ValueError(f"{split} token count does not match the manifest")
        if sha256_file(path) != info.get("sha256"):
            raise ValueError(f"{split} data hash does not match the manifest")
    expected_id = _dataset_id(manifest["train"]["sha256"], manifest["val"]["sha256"])
    if manifest.get("dataset_id") != expected_id:
        raise ValueError("dataset identity does not match its manifest")
    return manifest


def _positive_int(value: Any, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _finite_number(value: Any, name: str, *, minimum: float = 0.0, exclusive: bool = False) -> float:
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    if value < minimum or (exclusive and value == minimum):
        raise ValueError(f"{name} must be {'greater than' if exclusive else 'at least'} {minimum}")
    return float(value)


def validate_recipe(recipe: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate the fixed baseline recipe; reject misspelled/ignored settings."""
    if not isinstance(recipe, dict) or set(recipe) != {"schema_version", "model", "training"} or type(recipe["schema_version"]) is not int or recipe["schema_version"] != 1:
        raise ValueError("recipe requires schema_version=1, model, and training")
    model, training = recipe["model"], recipe["training"]
    model_fields = {"n_layer", "n_head", "n_embd", "context_length", "dropout"}
    training_fields = {
        "batch_size", "learning_rate", "weight_decay", "beta1", "beta2",
        "grad_clip", "device", "log_interval", "eval_batch_size",
    }
    if not isinstance(model, dict) or set(model) != model_fields:
        raise ValueError(f"model fields must be {sorted(model_fields)}")
    if not isinstance(training, dict) or set(training) != training_fields:
        raise ValueError(f"training fields must be {sorted(training_fields)}")
    for field in ("n_layer", "n_head", "n_embd", "context_length"):
        _positive_int(model[field], field)
    if model["n_embd"] % model["n_head"]:
        raise ValueError("n_embd must be divisible by n_head")
    if _finite_number(model["dropout"], "dropout") >= 1:
        raise ValueError("dropout must be smaller than 1")
    for field in ("batch_size", "log_interval", "eval_batch_size"):
        _positive_int(training[field], field)
    _finite_number(training["learning_rate"], "learning_rate", exclusive=True)
    _finite_number(training["weight_decay"], "weight_decay")
    _finite_number(training["grad_clip"], "grad_clip", exclusive=True)
    for field in ("beta1", "beta2"):
        if _finite_number(training[field], field) >= 1:
            raise ValueError(f"{field} must be smaller than 1")
    if training["device"] not in ("auto", "cpu", "cuda"):
        raise ValueError("device must be auto, cpu, or cuda")
    return dict(model), dict(training)


def _torch() -> Any:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("training requires PyTorch: install evogpt[train]") from error
    return torch


def estimate_step_flops(parameter_count: int, model: dict[str, Any], batch_size: int) -> int:
    """Return the versioned integer dense-matmul proxy for one complete step."""
    context = model["context_length"]
    return (
        6 * parameter_count * batch_size * context
        + 12 * model["n_layer"] * batch_size * context * context * model["n_embd"]
    )


def _batch(torch: Any, data: mmap.mmap, starts: list[int], length: int, device: str) -> tuple[Any, Any]:
    # bytearray owns writable storage and avoids torch.frombuffer's read-only
    # warning; convert to int64 indices after making one compact byte buffer.
    raw = bytearray().join(data[start:start + length + 1] for start in starts)
    tokens = torch.frombuffer(raw, dtype=torch.uint8).reshape(len(starts), length + 1).long()
    return tokens[:, :-1].to(device), tokens[:, 1:].to(device)


def _evaluate(torch: Any, model: Any, data: mmap.mmap, context: int, batch_size: int, device: str) -> tuple[float, int]:
    """Evaluate every next-byte target exactly once, including the final tail.

    Each context-length window starts with a fresh context. Windows are batched
    only when equally long, so there is neither padding nor a skipped tail.
    """
    model.eval()
    target_count = len(data) - 1
    full_windows = target_count // context
    total_loss = 0.0
    with torch.no_grad():
        for first in range(0, full_windows, batch_size):
            starts = [window * context for window in range(first, min(first + batch_size, full_windows))]
            x, y = _batch(torch, data, starts, context, device)
            loss = torch.nn.functional.cross_entropy(model(x).reshape(-1, 256), y.reshape(-1), reduction="sum")
            total_loss += float(loss.item())
        tail = target_count % context
        if tail:
            x, y = _batch(torch, data, [full_windows * context], tail, device)
            total_loss += float(torch.nn.functional.cross_entropy(model(x).reshape(-1, 256), y.reshape(-1), reduction="sum").item())
    return total_loss / target_count, target_count


def train_run(
    config: dict[str, Any], recipe: dict[str, Any], data_dir: Path | str,
    scale_id: str, seed: int, output_dir: Path | str, recipe_id: str, revision: str,
) -> dict[str, Any]:
    """Train from scratch to the largest complete update within a scale budget.

    Every scale and seed starts from its own initialization. The immutable
    recipe and source revision identify the entry; no checkpoint is resumed.
    """
    from .core import config_digest, parse_config

    contest = parse_config(config)
    model_config, training = validate_recipe(recipe)
    scale = next((item for item in contest.scales if item.id == scale_id), None)
    if scale is None:
        raise ValueError(f"unknown scale: {scale_id}")
    if type(seed) is not int or seed not in config["seeds"]:
        raise ValueError("seed must be one of the contest's required seeds")
    if not isinstance(recipe_id, str) or not re.fullmatch(r"[a-z0-9][a-z0-9_.-]{0,63}", recipe_id):
        raise ValueError("recipe_id must be a lowercase slug of at most 64 characters")
    if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("revision must be a full, lowercase 40-character Git commit")
    data_dir, output_dir = Path(data_dir), Path(output_dir)
    data_manifest = load_data_manifest(data_dir)
    if data_manifest["dataset_id"] != config["dataset_id"]:
        raise ValueError("prepared dataset_id does not match the contest")
    context = model_config["context_length"]
    if data_manifest["train"]["tokens"] <= context or data_manifest["val"]["tokens"] < 2:
        raise ValueError("training needs context_length + 1 bytes; validation needs at least 2 bytes")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("run output directory must be empty; existing artifacts are never replaced")

    torch = _torch()
    from .model import ByteGPT

    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = training["device"]
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested, but CUDA is not available")
    model = ByteGPT(**model_config).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    step_flops = estimate_step_flops(parameter_count, model_config, training["batch_size"])
    if step_flops > scale.max_flops:
        raise ValueError(f"scale budget {scale.max_flops} cannot fit one update ({step_flops} proxy FLOPs)")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=training["learning_rate"],
        betas=(training["beta1"], training["beta2"]), weight_decay=training["weight_decay"],
    )
    generator = torch.Generator(device="cpu").manual_seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "train.jsonl"
    train_flops = train_tokens = step = 0
    with (data_dir / "train.bin").open("rb") as train_stream, (data_dir / "val.bin").open("rb") as val_stream:
        with mmap.mmap(train_stream.fileno(), 0, access=mmap.ACCESS_READ) as train_data:
            with mmap.mmap(val_stream.fileno(), 0, access=mmap.ACCESS_READ) as val_data:
                with log_path.open("w", encoding="utf-8") as log:
                    model.train()
                    # Check the next complete update before executing it. The
                    # same integer formula defines both accounting and budget.
                    while train_flops + step_flops <= scale.max_flops:
                        starts = torch.randint(
                            len(train_data) - context,
                            (training["batch_size"],), generator=generator,
                        ).tolist()
                        x, y = _batch(torch, train_data, starts, context, device)
                        optimizer.zero_grad(set_to_none=True)
                        loss = torch.nn.functional.cross_entropy(model(x).reshape(-1, 256), y.reshape(-1))
                        if not math.isfinite(float(loss.item())):
                            raise ValueError("training loss is not finite")
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), training["grad_clip"], error_if_nonfinite=True)
                        optimizer.step()
                        step += 1
                        train_flops += step_flops
                        train_tokens += training["batch_size"] * context
                        if step % training["log_interval"] == 0 or train_flops + step_flops > scale.max_flops:
                            log.write(json.dumps({
                                "step": step, "train_flops": train_flops,
                                "train_tokens": train_tokens, "train_loss": float(loss.item()),
                            }, sort_keys=True, allow_nan=False) + "\n")
                            log.flush()
                val_loss, eval_tokens = _evaluate(torch, model, val_data, context, training["eval_batch_size"], device)
    if not math.isfinite(val_loss):
        raise ValueError("validation loss is not finite")
    artifact_path = output_dir / "model.pt"
    # A state_dict only. Consumers must use torch.load(..., weights_only=True).
    torch.save({name: value.detach().cpu() for name, value in model.state_dict().items()}, artifact_path)
    run_id = uuid4().hex
    created_at = datetime.now(timezone.utc).isoformat()
    recipe_sha256 = hashlib.sha256(_canonical(recipe)).hexdigest()
    result = {
        "schema_version": 1, "run_id": run_id, "recipe_id": recipe_id,
        "recipe_sha256": recipe_sha256,
        "revision": revision, "scale_id": scale_id, "seed": seed,
        "val_loss": val_loss, "train_flops": train_flops,
        "train_tokens": train_tokens, "dataset_id": data_manifest["dataset_id"],
        "config_digest": config_digest(config), "created_at": created_at,
        "artifact_sha256": sha256_file(artifact_path), "log_sha256": sha256_file(log_path),
    }
    _write_json(output_dir / "run_manifest.json", {
        "schema_version": 1, "run_id": run_id, "created_at": created_at,
        "recipe": recipe, "recipe_sha256": recipe_sha256,
        "contest": config, "config_digest": result["config_digest"],
        "dataset": data_manifest, "revision": revision, "seed": seed,
        "framework": {"name": "torch", "version": str(torch.__version__), "device": device},
        "parameter_count": parameter_count, "step_flops": step_flops,
        "flops_proxy": FLOPS_PROXY, "steps": step,
        "eval_tokens": eval_tokens, "validation": "all next-byte pairs once, fixed windows, token-weighted cross entropy in nats",
        "artifact_sha256": result["artifact_sha256"], "log_sha256": result["log_sha256"],
    })
    _write_json(output_dir / "result.json", result)
    return result
