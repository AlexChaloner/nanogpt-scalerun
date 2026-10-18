"""Byte data preparation and a budgeted, optional PyTorch training baseline.

The FLOPs counter is an analytic *proxy*, not a hardware profiler. Its definition
is versioned here so that runs use the same budget accounting at every scale.
"""

from __future__ import annotations

import codecs
from contextlib import contextmanager, nullcontext
import hashlib
import json
import math
import mmap
import os
from pathlib import Path
import re
import shutil
import signal
import sys
import time
import threading
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4


TOKENIZER = {"name": "utf8-bytes", "version": 1, "vocab_size": 256}
FLOPS_PROXY = {
    "name": "dense-parameters-plus-attention-v1",
    "formula": "(6 * P * B * T + 12 * L * B * T * T * D) * A * W per optimizer update",
    "definitions": "P=unique trainable parameters, B=microbatch size, T=context length, L=layers, D=embedding width, A=gradient accumulation steps, W=world size",
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
    if isinstance(manifest, dict) and type(manifest.get("schema_version")) is int and manifest["schema_version"] == 2:
        from .datasets import load_manifest
        return load_manifest(data_dir)
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


def resolve_recipe(recipe: dict[str, Any], scale_id: str, scale_ids: list[str] | None = None) -> dict[str, Any]:
    """Return the schema-1 recipe a scaling policy uses at one scale.

    A schema-2 policy has shared model/training settings plus per-scale
    overrides. The whole policy is one entry digest; overrides cannot be
    chosen later without changing it.
    """
    if not isinstance(recipe, dict) or type(recipe.get("schema_version")) is not int or recipe["schema_version"] != 2:
        return recipe
    if set(recipe) != {"schema_version", "model", "training", "scales"}:
        raise ValueError("scaling policy requires schema_version=2, model, training, and scales")
    base_model, base_training, scales = recipe["model"], recipe["training"], recipe["scales"]
    if not isinstance(base_model, dict) or not isinstance(base_training, dict) or not isinstance(scales, dict):
        raise ValueError("scaling policy model, training, and scales must be objects")
    if scale_ids is not None and set(scales) - set(scale_ids):
        raise ValueError(f"scaling policy names unknown scales: {sorted(set(scales) - set(scale_ids))}")
    override = scales.get(scale_id, {})
    if not isinstance(override, dict) or set(override) - {"model", "training"} or not all(isinstance(value, dict) for value in override.values()):
        raise ValueError(f"scaling policy override for {scale_id} may contain only model and training objects")
    return {
        "schema_version": 1,
        "model": {**base_model, **override.get("model", {})},
        "training": {**base_training, **override.get("training", {})},
    }


def validate_recipe(recipe: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Reject ignored/misspelled settings, applying documented optional defaults."""
    if not isinstance(recipe, dict) or set(recipe) != {"schema_version", "model", "training"} or type(recipe["schema_version"]) is not int or recipe["schema_version"] != 1:
        raise ValueError("recipe requires schema_version=1, model, and training (or a schema_version=2 scaling policy)")
    model, training = recipe["model"], recipe["training"]
    model_fields = {"n_layer", "n_head", "n_embd", "context_length", "dropout"}
    model_defaults = {"vocab_size": 256, "bias": True}
    training_fields = {"batch_size", "learning_rate", "weight_decay", "beta1", "beta2", "grad_clip", "device", "log_interval", "eval_batch_size"}
    training_defaults = {
        "gradient_accumulation_steps": 1, "dtype": "float32", "compile": False,
        "decay_lr": False, "warmup_iters": 0, "lr_decay_iters": None,
        "min_lr": 0.0, "checkpoint_interval": 0, "eval_interval": 0,
    }
    if not isinstance(model, dict) or not model_fields <= set(model) or set(model) - model_fields - set(model_defaults):
        raise ValueError(f"model requires {sorted(model_fields)}; optional fields: {sorted(model_defaults)}")
    if not isinstance(training, dict) or not training_fields <= set(training) or set(training) - training_fields - set(training_defaults):
        raise ValueError(f"training requires {sorted(training_fields)}; optional fields: {sorted(training_defaults)}")
    model, training = {**model_defaults, **model}, {**training_defaults, **training}
    for field in ("n_layer", "n_head", "n_embd", "context_length", "vocab_size"):
        _positive_int(model[field], field)
    if type(model["bias"]) is not bool:
        raise ValueError("bias must be boolean")
    if model["n_embd"] % model["n_head"]:
        raise ValueError("n_embd must be divisible by n_head")
    if _finite_number(model["dropout"], "dropout") >= 1:
        raise ValueError("dropout must be smaller than 1")
    for field in ("batch_size", "log_interval", "eval_batch_size", "gradient_accumulation_steps"):
        _positive_int(training[field], field)
    _finite_number(training["learning_rate"], "learning_rate", exclusive=True)
    _finite_number(training["weight_decay"], "weight_decay")
    _finite_number(training["grad_clip"], "grad_clip")
    _finite_number(training["min_lr"], "min_lr")
    if training["min_lr"] > training["learning_rate"]:
        raise ValueError("min_lr cannot exceed learning_rate")
    for field in ("beta1", "beta2"):
        if _finite_number(training[field], field) >= 1:
            raise ValueError(f"{field} must be smaller than 1")
    for field in ("warmup_iters", "checkpoint_interval", "eval_interval"):
        if type(training[field]) is not int or training[field] < 0:
            raise ValueError(f"{field} must be a nonnegative integer")
    if training["lr_decay_iters"] is not None:
        _positive_int(training["lr_decay_iters"], "lr_decay_iters")
        if training["lr_decay_iters"] <= training["warmup_iters"]:
            raise ValueError("lr_decay_iters must exceed warmup_iters")
    for field in ("compile", "decay_lr"):
        if type(training[field]) is not bool:
            raise ValueError(f"{field} must be boolean")
    if training["dtype"] not in ("float32", "bfloat16", "float16"):
        raise ValueError("dtype must be float32, bfloat16, or float16")
    if training["device"] not in ("auto", "cpu", "cuda"):
        raise ValueError("device must be auto, cpu, or cuda")
    if not training["decay_lr"] and (training["warmup_iters"] or training["lr_decay_iters"] is not None or training["min_lr"]):
        raise ValueError("warmup_iters, lr_decay_iters and min_lr require decay_lr=True")
    return model, training


def _torch() -> Any:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("training requires PyTorch: install evogpt[train]") from error
    return torch


def estimate_step_flops(
    parameter_count: int, model: dict[str, Any], batch_size: int,
    gradient_accumulation_steps: int = 1, world_size: int = 1,
) -> int:
    """Integer dense-matmul proxy for all microbatches in one update."""
    _positive_int(gradient_accumulation_steps, "gradient_accumulation_steps")
    _positive_int(world_size, "world_size")
    context = model["context_length"]
    return (
        6 * parameter_count * batch_size * context
        + 12 * model["n_layer"] * batch_size * context * context * model["n_embd"]
    ) * gradient_accumulation_steps * world_size


def _batch(torch: Any, data: Any, starts: list[int], length: int, device: str) -> tuple[Any, Any]:
    if isinstance(data, (mmap.mmap, bytes, bytearray)):
        raw = bytearray().join(data[start:start + length + 1] for start in starts)
        tokens = torch.frombuffer(raw, dtype=torch.uint8).reshape(len(starts), length + 1).long()
    else:
        tokens = torch.tensor([list(data[start:start + length + 1]) for start in starts], dtype=torch.long)
    return tokens[:, :-1].to(device), tokens[:, 1:].to(device)


def _autocast(torch: Any, device: str, dtype: str) -> Any:
    if dtype == "float32":
        return nullcontext()
    return torch.autocast(device_type=device, dtype=getattr(torch, dtype))


def _evaluate(torch: Any, model: Any, data: Any, context: int, batch_size: int, device: str, dtype: str = "float32") -> tuple[float, int]:
    """Evaluate every next-token target exactly once, including the tail."""
    model.eval()
    target_count = len(data) - 1
    full_windows = target_count // context
    total_loss = 0.0
    with torch.no_grad():
        for first in range(0, full_windows, batch_size):
            starts = [window * context for window in range(first, min(first + batch_size, full_windows))]
            x, y = _batch(torch, data, starts, context, device)
            with _autocast(torch, device, dtype):
                logits = model(x)
                loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1), reduction="sum")
            total_loss += float(loss.item())
        tail = target_count % context
        if tail:
            x, y = _batch(torch, data, [full_windows * context], tail, device)
            with _autocast(torch, device, dtype):
                logits = model(x)
                loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1), reduction="sum")
            total_loss += float(loss.item())
    return total_loss / target_count, target_count


def get_lr(iter_num: int, training: dict[str, Any], total_updates: int) -> float:
    """Warmup then cosine decay, with nanoGPT's zero-based iteration counter."""
    if not training["decay_lr"]:
        return training["learning_rate"]
    warmup = training["warmup_iters"]
    decay_end = training["lr_decay_iters"] or total_updates
    if decay_end <= warmup:
        raise ValueError("lr_decay_iters (or budget-derived updates) must exceed warmup_iters")
    if iter_num < warmup:
        return training["learning_rate"] * (iter_num + 1) / (warmup + 1)
    if iter_num >= decay_end:
        return training["min_lr"]
    ratio = (iter_num - warmup) / (decay_end - warmup)
    return training["min_lr"] + 0.5 * (1 + math.cos(math.pi * ratio)) * (training["learning_rate"] - training["min_lr"])


class TrainingInterrupted(RuntimeError):
    """An explicitly stopped run has a resumable checkpoint, but no result."""


@contextmanager
def _deferred_sigint():
    """Finish the active update before honoring Ctrl+C and saving its budget."""
    interrupted = [False]
    if threading.current_thread() is not threading.main_thread():
        yield interrupted
        return
    previous = signal.getsignal(signal.SIGINT)

    def request_stop(signum, frame):
        if not interrupted[0]:
            print("Interrupt received; saving after the current complete update.", file=sys.stderr)
        interrupted[0] = True

    signal.signal(signal.SIGINT, request_stop)
    try:
        yield interrupted
    finally:
        signal.signal(signal.SIGINT, previous)


def _checkpoint_model_args(model: dict[str, Any]) -> dict[str, Any]:
    args = dict(model)
    args["block_size"] = args.pop("context_length")
    return args


def train_run(
    config: dict[str, Any], recipe: dict[str, Any], data_dir: Path | str,
    scale_id: str, seed: int, output_dir: Path | str, recipe_id: str, revision: str,
    resume: bool = False, stop_after_steps: int | None = None,
) -> dict[str, Any]:
    """Train within one budget; same-run resume preserves spent compute.

    stop_after_steps interrupts after an absolute update count, saving a
    checkpoint without creating a result. It does not change the final budget.
    """
    from .core import config_digest, parse_config, parse_result
    from .model import ByteGPT
    from .datasets import open_tokens

    contest = parse_config(config)
    scale = next((item for item in contest.scales if item.id == scale_id), None)
    if scale is None:
        raise ValueError(f"unknown scale: {scale_id}")
    # Check every scale now, so a broken large-scale setting fails before small runs.
    scale_ids = [item.id for item in contest.scales]
    for other in scale_ids:
        validate_recipe(resolve_recipe(recipe, other, scale_ids))
    model_config, training = validate_recipe(resolve_recipe(recipe, scale_id, scale_ids))
    if type(seed) is not int or seed not in config["seeds"]:
        raise ValueError("seed must be one of the contest's required seeds")
    if not isinstance(recipe_id, str) or not re.fullmatch(r"[a-z0-9][a-z0-9_.-]{0,63}", recipe_id):
        raise ValueError("recipe_id must be a lowercase slug of at most 64 characters")
    if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("revision must be a full, lowercase 40-character Git commit")
    if type(resume) is not bool:
        raise ValueError("resume must be boolean")
    if stop_after_steps is not None:
        _positive_int(stop_after_steps, "stop_after_steps")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("distributed training is not supported by the budgeted runner; launch one process")
    data_dir, output_dir = Path(data_dir), Path(output_dir)
    data_manifest = load_data_manifest(data_dir)
    if data_manifest["dataset_id"] != config["dataset_id"]:
        raise ValueError("prepared dataset_id does not match the contest")
    if model_config["vocab_size"] != data_manifest["tokenizer"]["vocab_size"]:
        raise ValueError("model vocab_size must match the dataset tokenizer")
    context = model_config["context_length"]
    if data_manifest["train"]["tokens"] <= context or data_manifest["val"]["tokens"] < 2:
        raise ValueError("training needs context_length + 1 tokens; validation needs at least 2 tokens")
    if resume:
        if not (output_dir / "ckpt.pt").is_file():
            raise ValueError("resume requires ckpt.pt in the existing run output directory")
    elif output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("run output directory must be empty; existing artifacts are never replaced")
    torch = _torch()
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
    if training["dtype"] == "float16" and device != "cuda":
        raise ValueError("float16 training requires CUDA; use float32 or bfloat16 on CPU")
    if training["dtype"] == "bfloat16" and device == "cuda" and not torch.cuda.is_bf16_supported():
        raise ValueError("this CUDA device does not support bfloat16")
    raw_model = ByteGPT(**model_config).to(device)
    parameter_count = sum(parameter.numel() for parameter in raw_model.parameters())
    accumulation = training["gradient_accumulation_steps"]
    step_flops = estimate_step_flops(parameter_count, model_config, training["batch_size"], accumulation)
    total_updates = scale.max_flops // step_flops
    if not total_updates:
        raise ValueError(f"scale budget {scale.max_flops} cannot fit one update ({step_flops} proxy FLOPs)")
    get_lr(0, training, total_updates)
    optimizer = raw_model.configure_optimizers(training["weight_decay"], training["learning_rate"], (training["beta1"], training["beta2"]), device)
    scaler = torch.amp.GradScaler("cuda", enabled=training["dtype"] == "float16")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    recipe_sha256 = hashlib.sha256(_canonical(recipe)).hexdigest()
    identity = {
        "recipe_sha256": recipe_sha256, "config_digest": config_digest(config),
        "dataset_id": data_manifest["dataset_id"], "revision": revision,
        "scale_id": scale_id, "seed": seed, "recipe_id": recipe_id,
        "device": device, "step_flops": step_flops, "max_flops": scale.max_flops,
        "flops_proxy": FLOPS_PROXY, "model_args": _checkpoint_model_args(model_config),
    }
    train_flops = train_tokens = step = 0
    log_path = output_dir / "train.jsonl"
    checkpoint_path = output_dir / "ckpt.pt"
    if resume:
        progress_manifest = json.loads((output_dir / "run_manifest.json").read_text(encoding="utf-8"))
        if progress_manifest.get("ckpt_sha256") != sha256_file(checkpoint_path):
            raise ValueError("checkpoint hash differs from run_manifest.json")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        if checkpoint.get("schema_version") != 1 or checkpoint.get("identity") != identity:
            raise ValueError("checkpoint provenance differs: resume requires unchanged recipe, dataset, source, scale, seed, device and contest")
        step = checkpoint["iter_num"]
        train_flops, train_tokens = checkpoint["train_flops"], checkpoint["train_tokens"]
        expected_tokens = step * training["batch_size"] * context * accumulation
        if type(step) is not int or type(train_flops) is not int or type(train_tokens) is not int or not 0 <= step <= total_updates or train_flops != step * step_flops or train_tokens != expected_tokens:
            raise ValueError("checkpoint compute counters are inconsistent with the budget")
        if not log_path.is_file() or sha256_file(log_path) != checkpoint.get("log_sha256"):
            raise ValueError("training log differs from the checkpoint; refusing to replace evidence")
        raw_model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        generator.set_state(checkpoint["data_rng_state"].cpu())
        torch.set_rng_state(checkpoint["rng_state"].cpu())
        if device == "cuda":
            torch.cuda.set_rng_state_all([state.cpu() for state in checkpoint["cuda_rng_state_all"]])
        if (output_dir / "result.json").is_file():
            result = json.loads((output_dir / "result.json").read_text())
            parse_result(result, contest)
            identity_fields = ("recipe_sha256", "config_digest", "dataset_id", "revision", "scale_id", "seed", "recipe_id")
            if step != total_updates or any(result.get(key) != identity[key] for key in identity_fields) or result.get("train_flops") != train_flops or result.get("train_tokens") != train_tokens:
                raise ValueError("completed result differs from its checkpoint")
            if result.get("artifact_sha256") != sha256_file(output_dir / "model.pt") or result.get("log_sha256") != sha256_file(log_path):
                raise ValueError("completed run artifact or log hash differs")
            return result
    output_dir.mkdir(parents=True, exist_ok=True)
    model = torch.compile(raw_model) if training["compile"] else raw_model

    def save_checkpoint() -> None:
        checkpoint = {
            "schema_version": 1, "model": raw_model.state_dict(),
            "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
            "model_args": identity["model_args"], "iter_num": step,
            "config": config, "recipe": recipe, "identity": identity,
            "train_flops": train_flops, "train_tokens": train_tokens,
            "rng_state": torch.get_rng_state(), "data_rng_state": generator.get_state(),
            "cuda_rng_state_all": torch.cuda.get_rng_state_all() if device == "cuda" else [],
            "log_sha256": sha256_file(log_path),
        }
        temporary = output_dir / "ckpt.pt.tmp"
        torch.save(checkpoint, temporary)
        temporary.replace(checkpoint_path)
        # Sampling an interrupted run uses the same verified tokenizer/model
        # metadata as a completed run. No result exists until the budget is met.
        _write_json(output_dir / "run_manifest.json", {
            "schema_version": 1, "complete": False,
            "recipe": recipe, "recipe_sha256": recipe_sha256,
            "contest": config, "config_digest": identity["config_digest"],
            "dataset": data_manifest, "revision": revision, "seed": seed,
            "scale_id": scale_id, "recipe_id": recipe_id,
            "framework": {"name": "torch", "version": str(torch.__version__), "device": device, "dtype": training["dtype"], "compile": training["compile"]},
            "parameter_count": parameter_count, "step_flops": step_flops,
            "flops_proxy": FLOPS_PROXY, "steps": step,
            "train_flops": train_flops, "train_tokens": train_tokens,
            "gradient_accumulation_steps": accumulation, "world_size": 1,
            "log_sha256": sha256_file(log_path), "ckpt_sha256": sha256_file(checkpoint_path),
        })

    with _deferred_sigint() as interrupted, open_tokens(data_dir, "train", data_manifest) as train_data, open_tokens(data_dir, "val", data_manifest) as val_data:
        with log_path.open("a" if resume else "w", encoding="utf-8") as log:
            model.train()
            while step < total_updates:
                started = time.perf_counter()
                lr = get_lr(step, training, total_updates)
                for group in optimizer.param_groups:
                    group["lr"] = lr
                optimizer.zero_grad(set_to_none=True)
                loss_sum = 0.0
                for _ in range(accumulation):
                    starts = torch.randint(len(train_data) - context, (training["batch_size"],), generator=generator).tolist()
                    x, y = _batch(torch, train_data, starts, context, device)
                    with _autocast(torch, device, training["dtype"]):
                        logits = model(x)
                        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
                    loss_value = float(loss.item())
                    if not math.isfinite(loss_value):
                        raise ValueError("training loss is not finite")
                    loss_sum += loss_value
                    scaler.scale(loss / accumulation).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(raw_model.parameters(), training["grad_clip"] or math.inf, error_if_nonfinite=True)
                scaler.step(optimizer)
                scaler.update()
                step += 1
                train_flops += step_flops
                update_tokens = training["batch_size"] * context * accumulation
                train_tokens += update_tokens
                if step % training["log_interval"] == 0 or step == total_updates:
                    log.write(json.dumps({"step": step, "train_flops": train_flops, "train_tokens": train_tokens, "train_loss": loss_sum / accumulation, "learning_rate": lr}, sort_keys=True, allow_nan=False) + "\n")
                    log.flush()
                    elapsed = time.perf_counter() - started
                    print(f"iter {step}: loss {loss_sum / accumulation:.4f}, lr {lr:.6g}, time {elapsed * 1000:.2f}ms, tokens/s {update_tokens / elapsed:.0f}", file=sys.stderr)
                if interrupted[0] and step < total_updates:
                    log.flush()
                    save_checkpoint()
                    raise TrainingInterrupted(f"interrupted after iter {step}; resume from {checkpoint_path}")
                if training["eval_interval"] and step % training["eval_interval"] == 0:
                    train_loss, _ = _evaluate(torch, model, train_data, context, training["eval_batch_size"], device, training["dtype"])
                    validation_loss, _ = _evaluate(torch, model, val_data, context, training["eval_batch_size"], device, training["dtype"])
                    print(f"step {step}: train loss {train_loss:.4f}, val loss {validation_loss:.4f}", file=sys.stderr)
                    model.train()
                if training["checkpoint_interval"] and step % training["checkpoint_interval"] == 0:
                    log.flush()
                    save_checkpoint()
                if stop_after_steps is not None and step >= stop_after_steps and step < total_updates:
                    log.flush()
                    save_checkpoint()
                    raise TrainingInterrupted(f"stopped after iter {step}; resume from {checkpoint_path}")
            log.flush()
            save_checkpoint()
        val_loss, eval_tokens = _evaluate(torch, model, val_data, context, training["eval_batch_size"], device, training["dtype"])
    if not math.isfinite(val_loss):
        raise ValueError("validation loss is not finite")
    artifact_path = output_dir / "model.pt"
    torch.save({name: value.detach().cpu() for name, value in raw_model.state_dict().items()}, artifact_path)
    run_id = uuid4().hex
    created_at = datetime.now(timezone.utc).isoformat()
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
        "schema_version": 1, "complete": True, "run_id": run_id, "created_at": created_at,
        "train_flops": train_flops, "train_tokens": train_tokens,
        "recipe": recipe, "recipe_sha256": recipe_sha256,
        "contest": config, "config_digest": result["config_digest"],
        "dataset": data_manifest, "revision": revision, "seed": seed,
        "framework": {"name": "torch", "version": str(torch.__version__), "device": device, "dtype": training["dtype"], "compile": training["compile"]},
        "parameter_count": parameter_count, "step_flops": step_flops,
        "flops_proxy": FLOPS_PROXY, "steps": step,
        "gradient_accumulation_steps": accumulation, "world_size": 1,
        "eval_tokens": eval_tokens, "validation": "all next-token pairs once, fixed windows, token-weighted cross entropy in nats",
        "artifact_sha256": result["artifact_sha256"], "log_sha256": result["log_sha256"],
        "ckpt_sha256": sha256_file(checkpoint_path),
    })
    _write_json(output_dir / "result.json", result)
    return result
