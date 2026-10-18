"""Train with the familiar nanoGPT command shape and EvoGPT compute accounting.

    python train.py config/train_shakespeare_char.py --device=cpu --compile=False

Without --contest, runs are local experiments. Official results require the
frozen contest config, one active scale, and a clean tracked source checkout.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

from configurator import parse_config
from evogpt import core
from evogpt.cli import read_json, source_revision, write_json
from evogpt.datasets import load_manifest, open_tokens
from evogpt.training import (
    TrainingInterrupted, _evaluate, estimate_step_flops, sha256_file,
    train_run, validate_recipe,
)


def build_recipe(settings, manifest):
    vocabulary = settings["vocab_size"] or manifest["tokenizer"]["vocab_size"]
    if vocabulary != manifest["tokenizer"]["vocab_size"]:
        raise ValueError("vocab_size must match the prepared tokenizer; 0 selects it automatically")
    recipe = {
        "schema_version": 1,
        "model": {"n_layer": settings["n_layer"], "n_head": settings["n_head"],
                  "n_embd": settings["n_embd"], "context_length": settings["block_size"],
                  "dropout": settings["dropout"], "vocab_size": vocabulary, "bias": settings["bias"]},
        "training": {key: settings[key] for key in (
            "batch_size", "gradient_accumulation_steps", "learning_rate", "weight_decay",
            "beta1", "beta2", "grad_clip", "device", "dtype", "compile", "decay_lr",
            "warmup_iters", "min_lr", "checkpoint_interval", "eval_interval",
            "log_interval", "eval_batch_size")},
    }
    recipe["training"]["lr_decay_iters"] = settings["lr_decay_iters"] or None
    if not settings["decay_lr"]:
        # Inactive scheduler values have no effect and do not enter the policy.
        recipe["training"].update(warmup_iters=0, lr_decay_iters=None, min_lr=0.0)
    validate_recipe(recipe)
    return recipe


def local_config(settings, manifest, recipe):
    import torch
    from evogpt.model import ByteGPT

    if settings["max_iters"] < 1:
        raise ValueError("max_iters must be positive")
    # Construction is on CPU and train_run resets the seed before initialization.
    model = ByteGPT(**recipe["model"])
    parameters = sum(parameter.numel() for parameter in model.parameters())
    step_flops = estimate_step_flops(parameters, recipe["model"], settings["batch_size"],
                                     settings["gradient_accumulation_steps"])
    del model
    budget = step_flops * settings["max_iters"]
    scale_id = settings["scale"]
    second_id = "next" if scale_id != "next" else "next-scale"
    return {
        "schema_version": 1, "contest_id": "evogpt-local-experiment",
        "dataset_id": manifest["dataset_id"],
        "scales": [{"id": scale_id, "max_flops": budget, "baseline_loss": None,
                    "baseline_is_fixture": False},
                   {"id": second_id, "max_flops": budget * 10, "baseline_loss": None,
                    "baseline_is_fixture": False}],
        "initial_scale_count": 1, "top_n": 1, "seeds": [settings["seed"]],
        "epoch_start": "2000-01-01T00:00:00Z", "interval_days": 7,
    }


def _local_source():
    repository = Path(__file__).resolve().parent
    process = subprocess.run(["git", "-C", str(repository), "rev-parse", "HEAD"],
                             text=True, capture_output=True)
    revision = process.stdout.strip() if process.returncode == 0 else "0" * 40
    status = subprocess.run(["git", "-C", str(repository), "status", "--porcelain"],
                            text=True, capture_output=True)
    return revision, bool(status.returncode or status.stdout)


def evaluate_saved_run(output, data_dir, manifest, device):
    import torch
    from evogpt.model import ByteGPT

    stored = read_json(output / "run_manifest.json")
    if stored["dataset"]["dataset_id"] != manifest["dataset_id"]:
        raise ValueError("evaluation dataset does not match the saved run")
    artifact = output / "model.pt"
    if sha256_file(artifact) != stored["artifact_sha256"]:
        raise ValueError("model.pt hash does not match the saved run")
    model_config, training = validate_recipe(stored["recipe"])
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ByteGPT(**model_config).to(device)
    model.load_state_dict(torch.load(artifact, map_location=device, weights_only=True))
    result = {}
    for split in ("train", "val"):
        with open_tokens(data_dir, split, manifest) as tokens:
            loss, count = _evaluate(torch, model, tokens, model_config["context_length"],
                                    training["eval_batch_size"], device, "float32")
        result[split] = {"loss": loss, "tokens": count}
    print(json.dumps(result, indent=2))


def main(argv=None):
    try:
        settings, config_path = parse_config(argv)
        if int(os.environ.get("WORLD_SIZE", "1")) != 1:
            raise ValueError("DDP/torchrun training is not implemented yet; use one training process")
        if settings["init_from"] not in ("scratch", "resume"):
            raise ValueError("init_from supports scratch or resume; pretrained GPT-2 import is not implemented")
        if settings["num_threads"] < 0 or settings["stop_after_steps"] < 0:
            raise ValueError("num_threads and stop_after_steps must be nonnegative")
        import torch
        if settings["num_threads"]:
            torch.set_num_threads(settings["num_threads"])
        data_dir = Path(settings["data_dir"]) if settings["data_dir"] else Path("data") / settings["dataset"]
        manifest = load_manifest(data_dir)
        output = Path(settings["out_dir"])
        if settings["eval_only"]:
            evaluate_saved_run(output, data_dir, manifest, settings["device"])
            return 0
        recipe = build_recipe(settings, manifest)
        official = bool(settings["contest"])
        if official:
            config = core.load_config(settings["contest"]).to_dict()
            revision = source_revision(config_path or Path(__file__), None)
            dirty = False
        else:
            config = local_config(settings, manifest, recipe)
            revision, dirty = _local_source()
            print("Local experiment: results do not belong to a calibrated competition.", file=sys.stderr)
        core.parse_config(config)
        try:
            result = train_run(config, recipe, data_dir, settings["scale"], settings["seed"],
                               output, settings["recipe_id"], revision,
                               resume=settings["init_from"] == "resume",
                               stop_after_steps=settings["stop_after_steps"] or None)
        except TrainingInterrupted as error:
            if not official:
                write_json(output / "experiment.json", {"ranked": False, "source_dirty": dirty,
                           "source_revision": revision, "settings": settings})
            print(str(error), file=sys.stderr)
            return 0
        if not official:
            write_json(output / "experiment.json", {"ranked": False, "source_dirty": dirty,
                       "source_revision": revision, "settings": settings})
        print(f"saved {output / 'ckpt.pt'}; validation loss {result['val_loss']:.6f}", file=sys.stderr)
        print(json.dumps(result, indent=2))
        return 0
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, ImportError, SyntaxError, ZeroDivisionError) as error:
        print(f"train: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
