"""Check token-format compatibility and the prepare/train/sample/resume workflow."""

from array import array
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from configurator import load_config
from scalerun.datasets import import_tokens, load_manifest, open_tokens, prepare_text


REPOSITORY = Path(__file__).resolve().parents[1]


class FamiliarConfigTests(unittest.TestCase):
    def test_assignments_arithmetic_and_typed_overrides(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "config.py"
            path.write_text("max_iters = 12\nlr_decay_iters = max_iters\ngradient_accumulation_steps = 2 * 3\n")
            result = load_config(path, {"device": "cpu", "compile": False})
            self.assertEqual(result["lr_decay_iters"], 12)
            self.assertEqual(result["gradient_accumulation_steps"], 6)
            with self.assertRaisesRegex(ValueError, "bool"):
                load_config(path, {"compile": "False"})

    def test_configs_do_not_execute_calls_or_imports(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "unsafe.py"
            marker = root / "marker"
            path.write_text(f"out_dir = __import__('pathlib').Path({str(marker)!r}).touch()\n")
            with self.assertRaises(ValueError):
                load_config(path)
            self.assertFalse(marker.exists())
            path.write_text("import os\n")
            with self.assertRaises(ValueError):
                load_config(path)


class TokenFormatTests(unittest.TestCase):
    def test_character_data_roundtrips_unicode_in_uint16(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "input.txt"
            text = "abcd café λ 🐱\n" * 7
            source.write_text(text, encoding="utf-8")
            manifest = prepare_text(source, root / "data", "char", 0.2)
            self.assertEqual(manifest["schema_version"], 2)
            decoded = ""
            for split in ("train", "val"):
                with open_tokens(root / "data", split, manifest) as tokens:
                    decoded += "".join(manifest["tokenizer"]["chars"][index] for index in tokens[:])
                    self.assertEqual(len(tokens) * 2, (root / "data" / f"{split}.bin").stat().st_size)
            self.assertEqual(decoded, text)
            self.assertFalse((root / "data/meta.pkl").exists())
            self.assertEqual(load_manifest(root / "data"), manifest)

    def test_existing_gpt2_ids_are_tokens_rather_than_individual_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name, tokens in (("train", [65, 50000, 50256, 256]), ("val", [300, 42, 12345])):
                data = array("H", tokens)
                if sys.byteorder != "little":
                    data.byteswap()
                (root / f"{name}.bin").write_bytes(data.tobytes())
            manifest = import_tokens(root / "train.bin", root / "val.bin", root / "prepared")
            with open_tokens(root / "prepared", "train", manifest) as tokens:
                self.assertEqual(tokens[:], [65, 50000, 50256, 256])
                self.assertEqual(tokens[-1], 256)
            (root / "prepared/train.bin").write_bytes(b"\x00\x00")
            with self.assertRaisesRegex(ValueError, "token count"):
                load_manifest(root / "prepared")

    def test_character_import_cannot_silently_select_gpt2_metadata(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "train.bin").write_bytes(b"\x01\x00\x02\x00")
            (root / "val.bin").write_bytes(b"\x03\x00\x04\x00")
            (root / "meta.json").write_text(json.dumps({"name": "gpt2", "version": 1,
                "vocab_size": 50257, "encoding": "gpt2"}))
            with self.assertRaisesRegex(ValueError, "character metadata"):
                import_tokens(root / "train.bin", root / "val.bin", root / "data", "char", root / "meta.json")


@unittest.skipUnless(importlib.util.find_spec("torch"), "requires nanogpt-scalerun[train]")
class FamiliarWorkflowTests(unittest.TestCase):
    def test_local_prepare_interrupted_train_sample_resume_and_evaluate(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "corpus.txt").write_text("abc def\n" * 20)
            environment = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", CUDA_VISIBLE_DEVICES="")

            def command(script, *arguments):
                result = subprocess.run([sys.executable, str(REPOSITORY / script), *map(str, arguments)],
                                        cwd=REPOSITORY, env=environment, text=True, capture_output=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                return result.stdout

            command("data/shakespeare_char/prepare.py", "--input", root / "corpus.txt", "--output", root / "data")
            args = ["--data_dir", root / "data", "--out_dir", root / "run", "--device=cpu",
                    "--compile=False", "--n_layer=1", "--n_head=1", "--n_embd=8", "--block_size=4",
                    "--batch_size=2", "--max_iters=3", "--warmup_iters=0", "--lr_decay_iters=3",
                    "--checkpoint_interval=1", "--log_interval=1", "--num_threads=1"]
            command("train.py", *args, "--stop_after_steps=1")
            self.assertTrue((root / "run/ckpt.pt").exists())
            self.assertFalse((root / "run/result.json").exists())
            partial = command("sample.py", "--out_dir", root / "run", "--start=a", "--max_new_tokens=2", "--device=cpu")
            self.assertTrue(partial.startswith("a"))
            command("train.py", *args, "--init_from=resume")
            record = json.loads((root / "run/result.json").read_text())
            manifest = json.loads((root / "run/run_manifest.json").read_text())
            self.assertEqual(manifest["steps"], 3)
            self.assertEqual(record["train_flops"], manifest["step_flops"] * 3)
            self.assertFalse(json.loads((root / "run/experiment.json").read_text())["ranked"])
            sampled = command("sample.py", "--out_dir", root / "run", "--start=a", "--max_new_tokens=2", "--device=cpu")
            self.assertTrue(sampled.startswith("a"))
            losses = json.loads(command("train.py", *args, "--eval_only=True"))
            self.assertGreater(losses["val"]["tokens"], 0)
            self.assertEqual(json.loads((root / "run/result.json").read_text()), record)


if __name__ == "__main__":
    unittest.main()
