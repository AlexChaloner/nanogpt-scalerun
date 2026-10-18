"""Real CPU training smoke tests, skipped when the optional torch extra is absent."""

from __future__ import annotations

import copy
import importlib.util
import json
import math
import mmap
import signal
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from evogpt.training import (
    _evaluate, count_flops, measure_update_flops, prepare_data, sha256_file,
    train_run, validate_recipe, resolve_recipe, get_lr, TrainingInterrupted, load_data_manifest,
)

ROOT = Path(__file__).resolve().parents[1]


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


def tiny_policy() -> dict:
    recipe = tiny_recipe()
    return {
        "schema_version": 2, "model": recipe["model"], "training": recipe["training"],
        "scales": {"next": {"model": {"n_embd": 16, "n_head": 2}, "training": {"learning_rate": 0.0005}}},
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

    def test_optional_controls_reject_invalid_and_ignored_values(self) -> None:
        for options in ({"gradient_accumulation_steps": True}, {"compile": 1},
                        {"dtype": "fp16"}, {"checkpoint_interval": -1},
                        {"eval_interval": False}, {"warmup_iters": 2},
                        {"decay_lr": True, "min_lr": 2.0},
                        {"decay_lr": True, "warmup_iters": 2, "lr_decay_iters": 2}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                recipe = tiny_recipe()
                recipe["training"].update(options)
                validate_recipe(recipe)

    def test_scaling_policy_overrides_shared_settings_per_scale(self) -> None:
        policy = tiny_policy()
        small, small_training = validate_recipe(resolve_recipe(policy, "tiny", ["tiny", "next"]))
        large, large_training = validate_recipe(resolve_recipe(policy, "next", ["tiny", "next"]))
        self.assertEqual((small["n_embd"], small_training["learning_rate"]), (8, 0.001))
        self.assertEqual((large["n_embd"], large["n_head"], large_training["learning_rate"]), (16, 2, 0.0005))
        self.assertEqual(large["context_length"], small["context_length"])
        fixed = tiny_recipe()
        self.assertIs(resolve_recipe(fixed, "next"), fixed)

    def test_scaling_policy_rejects_unknown_scales_and_fields(self) -> None:
        policy = tiny_policy()
        with self.assertRaisesRegex(ValueError, "unknown scales"):
            resolve_recipe(policy, "tiny", ["tiny"])
        policy["scales"]["next"]["optimizer"] = {}
        with self.assertRaisesRegex(ValueError, "only model and training"):
            resolve_recipe(policy, "next", ["tiny", "next"])
        del policy["scales"]
        with self.assertRaisesRegex(ValueError, "schema_version=2"):
            resolve_recipe(policy, "tiny")

    def test_baseline_policy_grows_with_every_production_budget(self) -> None:
        policy = json.loads((ROOT / "examples/baseline.json").read_text())
        scale_ids = [scale["id"] for scale in json.loads((ROOT / "configs/contest.json").read_text())["scales"]]
        widths = [validate_recipe(resolve_recipe(policy, scale_id, scale_ids))[0]["n_embd"] for scale_id in scale_ids]
        self.assertEqual(widths, sorted(set(widths)))
        self.assertEqual(set(policy["scales"]), set(scale_ids))

    def test_warmup_and_cosine_reach_expected_endpoints(self) -> None:
        recipe = tiny_recipe()
        recipe["training"].update(decay_lr=True, warmup_iters=2, lr_decay_iters=10, min_lr=0.0001)
        _, options = validate_recipe(recipe)
        self.assertAlmostEqual(get_lr(0, options, 12), 0.001 / 3)
        self.assertAlmostEqual(get_lr(1, options, 12), 0.002 / 3)
        self.assertEqual(get_lr(2, options, 12), 0.001)
        self.assertAlmostEqual(get_lr(6, options, 12), 0.00055)
        self.assertEqual(get_lr(10, options, 12), 0.0001)
        self.assertEqual(get_lr(11, options, 12), 0.0001)

    def test_invalid_legacy_tokenizer_is_rejected_without_recursion(self) -> None:
        prepare_data(self.train, self.val, self.root / "out")
        path = self.root / "out/manifest.json"
        manifest = json.loads(path.read_text())
        manifest["tokenizer"]["name"] = "unknown"
        path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "unsupported"):
            load_data_manifest(path.parent)


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
        import torch
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        train, val = self.root / "train.txt", self.root / "val.txt"
        train.write_text("A small training corpus for the transformer, repeated for enough bytes.\n", encoding="utf-8")
        val.write_text("Distinct validation text!", encoding="utf-8")
        self.data = self.root / "data"
        self.manifest = prepare_data(train, val, self.data)
        self.recipe = tiny_recipe()
        self.step_flops = measure_update_flops(torch, *validate_recipe(self.recipe), "cpu")
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

    def test_scaling_policy_trains_its_model_for_each_scale(self) -> None:
        policy = tiny_policy()
        small = train_run(self.config, policy, self.data, "tiny", 42, self.root / "policy-tiny", "tiny-byte", "a" * 40)
        large = train_run(self.config, policy, self.data, "next", 42, self.root / "policy-next", "tiny-byte", "a" * 40)
        self.assertEqual(small["recipe_sha256"], large["recipe_sha256"])
        manifests = [json.loads((self.root / name / "run_manifest.json").read_text()) for name in ("policy-tiny", "policy-next")]
        self.assertLess(manifests[0]["parameter_count"], manifests[1]["parameter_count"])
        self.assertLessEqual(large["train_flops"], self.config["scales"][1]["max_flops"])

    def test_scaling_policy_checks_every_scale_before_training(self) -> None:
        policy = tiny_policy()
        policy["scales"]["next"]["model"]["n_head"] = 3
        with self.assertRaisesRegex(ValueError, "divisible"):
            train_run(self.config, policy, self.data, "tiny", 42, self.root / "broken", "tiny-byte", "a" * 40)
        self.assertFalse((self.root / "broken").exists())

    def test_counted_update_charges_every_accumulated_microbatch(self) -> None:
        import torch
        recipe = tiny_recipe()
        recipe["training"]["gradient_accumulation_steps"] = 3
        # AdamW is elementwise, so the counted optimizer step adds nothing.
        self.assertEqual(measure_update_flops(torch, *validate_recipe(recipe), "cpu"), 3 * self.step_flops)

    def test_cpu_attention_is_counted(self) -> None:
        import torch
        recipe = tiny_recipe()
        recipe["model"]["context_length"] = 8
        # Dense layers double with context; attention quadruples.
        self.assertGreater(measure_update_flops(torch, *validate_recipe(recipe), "cpu"), 2 * self.step_flops)

    def test_operations_without_a_formula_are_refused(self) -> None:
        import torch
        a = torch.ones(4)
        self.assertEqual(count_flops(torch, lambda: torch.ones(2, 3) @ torch.ones(3, 4)), 2 * 2 * 3 * 4)
        with self.assertRaisesRegex(ValueError, "aten.dot"):
            count_flops(torch, lambda: torch.dot(a, a))

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

    def test_accumulation_charges_all_microbatches_and_complete_updates(self) -> None:
        self.recipe["training"]["gradient_accumulation_steps"] = 2
        self.recipe["training"]["eval_interval"] = 1
        result = self.run_training("accumulated")
        self.assertEqual(result["train_flops"], 2 * self.step_flops)
        self.assertEqual(result["train_tokens"], 16)
        checkpoint = __import__("torch").load(self.root / "accumulated/ckpt.pt", weights_only=True)
        self.assertEqual(checkpoint["iter_num"], 1)
        self.assertEqual(checkpoint["train_flops"], result["train_flops"])

    def test_resume_preserves_rng_optimizer_counters_and_log(self) -> None:
        import torch
        self.recipe["model"]["dropout"] = 0.2
        self.recipe["training"].update(gradient_accumulation_steps=2, decay_lr=True,
                                       warmup_iters=1, min_lr=0.0001, checkpoint_interval=1)
        self.config["scales"][0]["max_flops"] = 8 * self.step_flops
        self.config["scales"][1]["max_flops"] = 80 * self.step_flops
        uninterrupted = self.run_training("full")
        out = self.root / "resumed"
        with self.assertRaisesRegex(TrainingInterrupted, "iter 1"):
            train_run(self.config, self.recipe, self.data, "tiny", 42, out,
                      "tiny-byte", "a" * 40, stop_after_steps=1)
        self.assertFalse((out / "result.json").exists())
        prefix = (out / "train.jsonl").read_bytes()
        interrupted = torch.load(out / "ckpt.pt", weights_only=True)
        self.assertEqual(interrupted["train_flops"], measure_update_flops(torch, *validate_recipe(self.recipe), "cpu"))
        resumed = train_run(self.config, self.recipe, self.data, "tiny", 42, out,
                            "tiny-byte", "a" * 40, resume=True)
        self.assertEqual(resumed["train_flops"], uninterrupted["train_flops"])
        self.assertEqual(resumed["train_tokens"], uninterrupted["train_tokens"])
        self.assertEqual(resumed["val_loss"], uninterrupted["val_loss"])
        self.assertTrue((out / "train.jsonl").read_bytes().startswith(prefix))
        self.assertEqual((out / "train.jsonl").read_bytes(), (self.root / "full/train.jsonl").read_bytes())
        full_state = torch.load(self.root / "full/model.pt", weights_only=True)
        resumed_state = torch.load(out / "model.pt", weights_only=True)
        self.assertTrue(all(torch.equal(full_state[key], resumed_state[key]) for key in full_state))
        before = {file.name: file.read_bytes() for file in out.iterdir()}
        completed = train_run(self.config, self.recipe, self.data, "tiny", 42, out,
                              "tiny-byte", "a" * 40, resume=True)
        self.assertEqual(completed, resumed)
        self.assertEqual(before, {file.name: file.read_bytes() for file in out.iterdir()})

    def test_resume_refuses_changed_budget_source_or_log(self) -> None:
        out = self.root / "partial"
        with self.assertRaises(TrainingInterrupted):
            train_run(self.config, self.recipe, self.data, "tiny", 42, out,
                      "tiny-byte", "a" * 40, stop_after_steps=1)
        changed_budget = copy.deepcopy(self.config)
        for scale in changed_budget["scales"]:
            scale["max_flops"] *= 2
        with self.assertRaisesRegex(ValueError, "provenance"):
            train_run(changed_budget, self.recipe, self.data, "tiny", 42, out,
                      "tiny-byte", "a" * 40, resume=True)
        with self.assertRaisesRegex(ValueError, "provenance"):
            train_run(self.config, self.recipe, self.data, "tiny", 42, out,
                      "tiny-byte", "b" * 40, resume=True)
        with (out / "train.jsonl").open("a") as stream:
            stream.write("tampered\n")
        with self.assertRaisesRegex(ValueError, "log differs"):
            train_run(self.config, self.recipe, self.data, "tiny", 42, out,
                      "tiny-byte", "a" * 40, resume=True)

    def test_uint16_character_training_uses_token_ids_and_vocabulary(self) -> None:
        from evogpt.datasets import prepare_text
        source = self.root / "characters.txt"
        source.write_text("abc λ🌱abcdef λ🌱abcdee" * 8, encoding="utf-8")
        self.data = self.root / "characters"
        self.manifest = prepare_text(source, self.data, tokenizer="char")
        self.recipe["model"]["vocab_size"] = self.manifest["tokenizer"]["vocab_size"]
        self.config["dataset_id"] = self.manifest["dataset_id"]
        result = self.run_training("chars-run")
        self.assertTrue(math.isfinite(result["val_loss"]))
        manifest = json.loads((self.root / "chars-run/run_manifest.json").read_text())
        self.assertEqual(manifest["eval_tokens"], self.manifest["val"]["tokens"] - 1)
        self.assertLess(result["val_loss"], math.log(256))

    def test_ctrl_c_saves_only_after_complete_update_and_restores_handler(self) -> None:
        import evogpt.training as trainer
        import torch
        out = self.root / "interrupted"
        previous_handler = signal.getsignal(signal.SIGINT)
        real_batch = trainer._batch
        requested = False

        def interrupt_once(*args, **kwargs):
            nonlocal requested
            batch = real_batch(*args, **kwargs)
            if not requested:
                requested = True
                signal.raise_signal(signal.SIGINT)
            return batch

        with patch("evogpt.training._batch", side_effect=interrupt_once):
            with self.assertRaisesRegex(TrainingInterrupted, "interrupted after iter 1"):
                self.run_training("interrupted")
        self.assertIs(signal.getsignal(signal.SIGINT), previous_handler)
        self.assertFalse((out / "result.json").exists())
        checkpoint = torch.load(out / "ckpt.pt", weights_only=True)
        self.assertEqual(checkpoint["train_flops"], self.step_flops)
        self.assertEqual(checkpoint["train_tokens"], 8)
        resumed = train_run(self.config, self.recipe, self.data, "tiny", 42, out,
                            "tiny-byte", "a" * 40, resume=True)
        complete = self.run_training("uninterrupted")
        self.assertEqual(resumed["val_loss"], complete["val_loss"])
        self.assertEqual(resumed["train_flops"], complete["train_flops"])

    def test_zero_grad_clip_disables_clipping(self) -> None:
        self.recipe["training"]["grad_clip"] = 0
        result = self.run_training("unclipped")
        self.assertTrue(math.isfinite(result["val_loss"]))

    def test_bfloat16_cpu_autocast_smoke(self) -> None:
        self.recipe["training"]["dtype"] = "bfloat16"
        result = self.run_training("bf16")
        self.assertTrue(math.isfinite(result["val_loss"]))
        self.assertEqual(result["train_flops"], 2 * self.step_flops)


if __name__ == "__main__":
    unittest.main()
