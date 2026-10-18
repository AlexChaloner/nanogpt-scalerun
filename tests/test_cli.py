"""Integration checks for review artifacts, calibration, and command failures."""

import contextlib
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

from scalerun.cli import main, read_json, source_revision, write_json
from scalerun import core
from scalerun.page import render_html


class CliTests(unittest.TestCase):
    def invoke(self, *arguments):
        output, errors = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(errors):
            code = main(list(arguments))
        return code, output.getvalue(), errors.getvalue()

    def test_demo_produces_minimax_ranking_and_frozen_promotion(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "demo"
            code, _, errors = self.invoke("demo", "--output", str(root))
            self.assertEqual(code, 0, errors)
            board = read_json(root / "leaderboard.json")
            self.assertEqual(board["overall"][0]["recipe_id"], "robust")
            self.assertEqual(board["scales"]["s0"]["rows"][0]["recipe_id"], "specialist")
            page = (root / "leaderboard.html").read_text(encoding="utf-8")
            self.assertIn("Synthetic demo", page)
            self.assertLess(page.index("robust"), page.index("fragile"))
            self.assertIn("SYNTHETIC DEMO", (root / "leaderboard.md").read_text())
            state = read_json(root / "state.week1.json")
            core.validate_state(core.load_config(root / "contest.json"), state)
            self.assertIn("robust@", json.dumps(state))
            code, _, errors = self.invoke(
                "leaderboard", "--config", str(root / "contest.json"),
                "--results", str(root / "results.json"), "--state", str(root / "state.week1.json"),
                "--output", str(root / "next.json"), "--markdown", str(root / "next.md"))
            self.assertEqual(code, 0, errors)
            self.assertIn("incomplete", (root / "next.md").read_text())
            code, _, errors = self.invoke("demo", "--output", str(root))
            self.assertEqual(code, 2)
            self.assertIn("empty", errors)

    def test_page_renders_an_empty_uncalibrated_season(self):
        contest = core.load_config(Path(__file__).parent.parent / "configs" / "contest.json")
        page = render_html(core.leaderboard(contest, []))
        self.assertIn("No entries yet.", page)
        self.assertIn("Uncalibrated", page)
        self.assertNotIn("Synthetic demo", page)

    def test_calibration_preserves_reference_provenance_and_changes_digest(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = {
                "schema_version": 1, "contest_id": "calibration-test", "dataset_id": "test-data",
                "scales": [{"id": f"s{i}", "max_flops": 10 ** (6 + i),
                            "baseline_loss": None, "baseline_is_fixture": False} for i in range(2)],
                "initial_scale_count": 1, "top_n": 1, "seeds": [0, 1],
                "epoch_start": "2026-01-05T00:00:00Z", "interval_days": 7,
            }
            write_json(root / "source.json", config)
            digest = core.config_digest(config)
            reference = []
            for i in range(2):
                for seed in (0, 1):
                    run_id = f"ref-{i}-{seed}"
                    reference.append({
                        "schema_version": 1, "run_id": run_id, "recipe_id": "baseline",
                        "recipe_sha256": "c" * 64,
                        "revision": "a" * 40, "scale_id": f"s{i}", "seed": seed,
                        "val_loss": 5.0 - i + seed * 0.2, "train_flops": 10 ** (6 + i),
                        "train_tokens": 64, "dataset_id": "test-data", "config_digest": digest,
                        "created_at": "2026-01-06T00:00:00Z",
                        "artifact_sha256": hashlib.sha256(run_id.encode()).hexdigest(),
                        "log_sha256": "b" * 64,
                    })
            write_json(root / "reference.json", reference)
            code, _, errors = self.invoke(
                "calibrate", "--config", str(root / "source.json"),
                "--results", str(root / "reference.json"), "--recipe-id", "baseline",
                "--revision", "a" * 40, "--output", str(root / "frozen.json"))
            self.assertEqual(code, 0, errors)
            frozen = read_json(root / "frozen.json")
            self.assertAlmostEqual(frozen["scales"][0]["baseline_loss"], 5.1)
            self.assertNotEqual(core.config_digest(frozen), digest)
            evidence = read_json(root / "frozen.calibration.json")
            self.assertEqual(evidence["source_config_digest"], digest)
            self.assertEqual(evidence["reference_results"], reference)
            self.assertIsNone(read_json(root / "source.json")["scales"][0]["baseline_loss"])

    def test_invalid_result_file_does_not_write_a_leaderboard(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            code, _, errors = self.invoke("demo", "--output", str(root / "demo"))
            self.assertEqual(code, 0, errors)
            (root / "bad.json").write_text('{"val_loss": NaN}', encoding="utf-8")
            code, _, errors = self.invoke(
                "leaderboard", "--config", str(root / "demo" / "contest.json"),
                "--results", str(root / "bad.json"), "--output", str(root / "board.json"))
            self.assertEqual(code, 2)
            self.assertTrue(errors)
            self.assertFalse((root / "board.json").exists())

    def test_a_recipe_in_another_repository_cannot_claim_the_runner_revision(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            recipe = root / "recipe.json"
            recipe.write_text("{}\n", encoding="utf-8")
            for arguments in [("init", "-q"), ("add", "recipe.json"),
                              ("-c", "user.name=Test", "-c", "user.email=test@example.com",
                               "commit", "-qm", "Track recipe")]:
                subprocess.run(["git", "-C", str(root), *arguments], check=True,
                               capture_output=True, text=True)
            with self.assertRaisesRegex(ValueError, "same Git checkout"):
                source_revision(recipe, None)

    def test_mixed_result_filenames_do_not_silently_lose_contestants(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            code, _, errors = self.invoke("demo", "--output", str(root / "demo"))
            self.assertEqual(code, 0, errors)
            records = read_json(root / "demo" / "results.json")
            write_json(root / "records" / "alice.json", records[:6])
            write_json(root / "records" / "bob" / "result.json", records[6:])
            write_json(root / "records" / "bob" / "run_manifest.json", {})
            (root / "records" / "bob" / "train.jsonl").write_text('{"step": 1}\n')
            code, output, errors = self.invoke(
                "validate", "--config", str(root / "demo" / "contest.json"),
                "--results", str(root / "records"))
            self.assertEqual(code, 0, errors)
            self.assertEqual(json.loads(output)["results"], len(records))


if __name__ == "__main__":
    unittest.main()
