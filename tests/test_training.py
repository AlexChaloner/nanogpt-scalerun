"""Real CPU training smoke tests, skipped when the optional torch extra is absent."""

from __future__ import annotations

import importlib.util
import json
import math
import mmap
from pathlib import Path
import tempfile
import unittest

from evogpt.training import (
    _evaluate, estimate_step_flops, prepare_data, sha256_file,
    train_run, validate_recipe,
)


def tiny_recipe() -> dict:
    return {
        "schema_version": 1,
        "model": {"n_layer": 1, "n_head": 1, "n_embd": 8, "context_length": 4, "dropout": 0.0},
        "training": {
            "batch_size": 2, "learning_rate": 0.001, "weight_decay": 0.1,
            "beta1": 0.9, "beta2": 0.95, "grad_clip": 1.0,
            "device": "cpu", "log_interval": 1, "eval_batch_size": 2,
        },
    }


def tiny_contest(dataset_id: str, max_flops: int) -> dict:
    return {
        "schema_version": 1, "contest_id": "training-test", "dataset_id": dataset_id,
        "scales": [
            {"id": "tiny", "max_flops": max_flops, "baseline_loss": 6.0, "baseline_is_fixture": True},
            {"id": "next", "max_flops": max_flops * 10, "baseline_loss": 5.0, "baseline_is_fixture": True},
        ],
        "initial_scale_count": 1, "top_n": 1, "seeds": [42],
        "epoch_start": "2026-01-01T00:00:00+00:00", "interval_days": 7,
    }


class PrepareDataTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.train = self.root / "train.txt"
        self.val = self.root / "val.txt"
        self.train.write_text("Training has Unicode: café, λ, 🐱.\n", encoding="utf-8")
        self.val.write_text("Validation uses its own text and symbols: 🚀.\n", encoding="utf-8")

    def test_dataset_identity_uses_bytes_and_tokenizer_not_names(self) -> None:
        first = prepare_data(self.train, self.val, self.root / "prepared")
        second = prepare_data(self.train, self.val, self.root / "again")
        self.assertEqual(first, second)
        self.assertEqual(first["train"]["tokens"], len(self.train.read_bytes()))
        self.assertEqual((self.root / "prepared/train.bin").read_bytes(), self.train.read_bytes())
        self.assertEqual(len(first["dataset_id"]), 64)

    def test_rejects_same_duplicate_and_contained_data(self) -> None:
        with self.assertRaisesRegex(ValueError, "separate"):
            prepare_data(self.train, self.train, self.root / "out")
        self.val.write_bytes(self.train.read_bytes())
        with self.assertRaisesRegex(ValueError, "identical"):
            prepare_data(self.train, self.val, self.root / "out")
        self.val.write_text("Training has Unicode", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "inside"):
            prepare_data(self.train, self.val, self.root / "out")

    def test_rejects_invalid_utf8_and_existing_artifacts(self) -> None:
        prepare_data(self.train, self.val, self.root / "out")
        with self.assertRaisesRegex(ValueError, "already exists"):
            prepare_data(self.train, self.val, self.root / "out")
        self.val.write_bytes(b"\xff\xfe")
        with self.assertRaises(UnicodeDecodeError):
            prepare_data(self.train, self.val, self.root / "invalid")

    def test_rejects_silently_ignored_recipe_settings(self) -> None:
        recipe = tiny_recipe()
        validate_recipe(recipe)
        recipe["training"]["batch_szie"] = 2
        with self.assertRaises(ValueError):
            validate_recipe(recipe)

    def test_recipe_schema_version_cannot_be_boolean(self) -> None:
        recipe = tiny_recipe()
        recipe["schema_version"] = True
        with self.assertRaises(ValueError):
            validate_recipe(recipe)


@unittest.skipUnless(importlib.util.find_spec("torch") is not None, "install evogpt[train] for CPU smoke tests")
class TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        import torch
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls) -> None:
        import torch
        torch.set_num_threads(cls.previous_threads)

    def setUp(self) -> None:
        from evogpt.model import ByteGPT
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        train, val = self.root / "train.txt", self.root / "val.txt"
        train.write_text("A small training corpus for the transformer, repeated for enough bytes.\n", encoding="utf-8")
        val.write_text("Distinct validation text!", encoding="utf-8")
        self.data = self.root / "data"
        self.manifest = prepare_data(train, val, self.data)
        self.recipe = tiny_recipe()
        model = ByteGPT(**self.recipe["model"])
        self.step_flops = estimate_step_flops(sum(p.numel() for p in model.parameters()), self.recipe["model"], 2)
        self.config = tiny_contest(self.manifest["dataset_id"], self.step_flops * 2 + self.step_flops // 2)

    def run_training(self, destination: str) -> dict:
        return train_run(
            self.config, self.recipe, self.data, "tiny", 42,
            self.root / destination, "tiny-byte", "a" * 40,
        )

    def test_real_training_respects_budget_and_repeats(self) -> None:
        import torch
        first, second = self.run_training("run-a"), self.run_training("run-b")
        self.assertTrue(math.isfinite(first["val_loss"]))
        self.assertEqual(first["val_loss"], second["val_loss"])
        self.assertEqual(first["train_flops"], 2 * self.step_flops)
        self.assertEqual(first["train_tokens"], 2 * 2 * 4)
        self.assertLessEqual(first["train_flops"], self.config["scales"][0]["max_flops"])
        self.assertGreater(first["train_flops"] + self.step_flops, self.config["scales"][0]["max_flops"])
        self.assertEqual((self.root / "run-a/train.jsonl").read_bytes(), (self.root / "run-b/train.jsonl").read_bytes())
        manifest = json.loads((self.root / "run-a/run_manifest.json").read_text())
        self.assertEqual(first["recipe_sha256"], manifest["recipe_sha256"])
        self.assertEqual(first["recipe_sha256"], second["recipe_sha256"])
        self.assertEqual(manifest["eval_tokens"], self.manifest["val"]["tokens"] - 1)
        self.assertEqual(first["artifact_sha256"], sha256_file(self.root / "run-a/model.pt"))
        self.assertEqual(first["log_sha256"], sha256_file(self.root / "run-a/train.jsonl"))
        first_state = torch.load(self.root / "run-a/model.pt", weights_only=True)
        second_state = torch.load(self.root / "run-b/model.pt", weights_only=True)
        self.assertTrue(all(torch.equal(first_state[name], second_state[name]) for name in first_state))

    def test_refuses_budget_that_cannot_fit_one_update(self) -> None:
        self.config["scales"][0]["max_flops"] = self.step_flops - 1
        self.config["scales"][1]["max_flops"] = (self.step_flops - 1) * 10
        with self.assertRaisesRegex(ValueError, "cannot fit one update"):
            self.run_training("too-small")
        self.assertFalse((self.root / "too-small").exists())

    def test_rechecks_dataset_hash_and_contest_identity(self) -> None:
        self.config["dataset_id"] = "different-dataset"
        with self.assertRaisesRegex(ValueError, "dataset_id"):
            self.run_training("mismatch")
        self.config["dataset_id"] = self.manifest["dataset_id"]
        with (self.data / "val.bin").open("r+b") as stream:
            stream.write(b"X")
        with self.assertRaisesRegex(ValueError, "hash"):
            self.run_training("tampered")

    def test_evaluation_covers_tail_once_and_reports_nats(self) -> None:
        import torch

        class UniformModel(torch.nn.Module):
            def forward(self, tokens):
                return torch.zeros((*tokens.shape, 256), device=tokens.device)

        with (self.data / "val.bin").open("rb") as stream:
            with mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as data:
                loss, count = _evaluate(torch, UniformModel(), data, 4, 2, "cpu")
                self.assertEqual(count, len(data) - 1)
                self.assertAlmostEqual(loss, math.log(256), places=5)

    def test_decoder_cannot_see_future_bytes(self) -> None:
        import torch
        from evogpt.model import ByteGPT
        model = ByteGPT(**self.recipe["model"]).eval()
        with torch.no_grad():
            first = model(torch.tensor([[1, 2, 3, 4]]))
            second = model(torch.tensor([[1, 2, 99, 100]]))
        self.assertTrue(torch.equal(first[:, :2], second[:, :2]))


if __name__ == "__main__":
    unittest.main()
