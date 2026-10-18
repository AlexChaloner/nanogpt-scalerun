"""Model behavior and a real local checkpoint-to-text sampling round trip."""

from __future__ import annotations

from contextlib import redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


@unittest.skipUnless(importlib.util.find_spec("torch") is not None, "install evogpt[train]")
class ModelTests(unittest.TestCase):
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
        from evogpt.model import GPT, GPTConfig
        torch.manual_seed(42)
        self.model = GPT(GPTConfig(
            block_size=4, vocab_size=7, n_layer=2, n_head=2, n_embd=8,
        )).eval()

    def test_causality_and_target_loss_with_gradients(self) -> None:
        import torch
        from torch.nn import functional as F
        idx = torch.tensor([[1, 2, 3, 4]])
        targets = torch.tensor([[2, 3, 4, -1]])
        logits, loss = self.model(idx, targets)
        changed, _ = self.model(torch.tensor([[1, 2, 5, 6]]), targets)
        self.assertTrue(torch.equal(logits[:, :2], changed[:, :2]))
        self.assertEqual(tuple(logits.shape), (1, 4, 7))
        self.assertTrue(torch.equal(loss, F.cross_entropy(logits[0, :3], targets[0, :3])))
        self.assertGreater(loss.item(), 0)
        loss.backward()
        self.assertTrue(torch.isfinite(self.model.transformer.wte.weight.grad).all())
        self.assertGreater(self.model.transformer.wte.weight.grad.abs().sum().item(), 0)
        last, no_loss = self.model(idx)
        self.assertIsNone(no_loss)
        self.assertTrue(torch.equal(last, logits[:, [-1]]))

    def test_familiar_names_weight_tying_and_parameter_counts(self) -> None:
        self.assertIs(self.model.lm_head.weight, self.model.transformer.wte.weight)
        state = self.model.state_dict()
        for name in (
            "transformer.wte.weight", "transformer.wpe.weight",
            "transformer.h.0.attn.c_attn.weight", "transformer.h.0.mlp.c_proj.weight",
            "transformer.ln_f.weight", "lm_head.weight",
        ):
            self.assertIn(name, state)
        total = sum(p.numel() for p in self.model.parameters())
        self.assertEqual(self.model.get_num_params(False), total)
        self.assertEqual(self.model.get_num_params(), total - 4 * 8)

    def test_optimizer_decays_matrices_once_and_exempts_vectors(self) -> None:
        optimizer = self.model.configure_optimizers(0.1, 1e-3, (0.9, 0.95), "cpu")
        seen = []
        for group in optimizer.param_groups:
            for parameter in group["params"]:
                seen.append(id(parameter))
                self.assertEqual(group["weight_decay"], 0.1 if parameter.ndim >= 2 else 0.0)
        self.assertEqual(len(seen), len(set(seen)))
        self.assertEqual(set(seen), {id(p) for p in self.model.parameters()})

    def test_top_k_one_is_greedy_and_generation_crops_context(self) -> None:
        import torch
        prompt = torch.tensor([[1, 2, 3]])
        expected = prompt
        for _ in range(6):
            logits, _ = self.model(expected[:, -4:])
            expected = torch.cat((expected, logits[:, -1].argmax(-1, keepdim=True)), dim=1)
        generated = self.model.generate(prompt, 6, temperature=0.8, top_k=1)
        self.assertTrue(torch.equal(generated, expected))
        self.assertEqual(generated.shape, (1, 9))
        self.assertTrue(torch.equal(self.model.generate(prompt, 0), prompt))
        for kwargs in ({"temperature": 0}, {"temperature": float("nan")}, {"top_k": 0}):
            with self.assertRaises(ValueError):
                self.model.generate(prompt, 1, **kwargs)

    def test_byte_adapter_arbitrary_vocabulary_bias_and_crop(self) -> None:
        import torch
        from evogpt.model import ByteGPT
        model = ByteGPT(1, 1, 8, context_length=4, vocab_size=11, bias=False).eval()
        self.assertFalse(any(name.endswith("bias") for name, _ in model.named_parameters()))
        self.assertEqual(tuple(model(torch.tensor([[1, 2, 3]])).shape), (1, 3, 11))
        self.assertEqual(tuple(model.generate(torch.tensor([[1]]), 5, top_k=1).shape), (1, 6))
        model.crop_block_size(2)
        self.assertEqual(model.context_length, 2)
        self.assertEqual(model.transformer.wpe.weight.shape[0], 2)
        with self.assertRaisesRegex(ValueError, "context length"):
            model(torch.tensor([[1, 2, 3]]))
        with self.assertRaises(ValueError):
            model.crop_block_size(3)

    def _sample_module(self):
        path = Path(__file__).resolve().parents[1] / "sample.py"
        spec = importlib.util.spec_from_file_location("evogpt_sample_test", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def _save_sample_run(self, root: Path) -> dict:
        import torch
        from evogpt.training import sha256_file
        torch.save(self.model.state_dict(), root / "model.pt")
        manifest = {
            "recipe": {"model": {
                "n_layer": 2, "n_head": 2, "n_embd": 8,
                "context_length": 4, "dropout": 0.0, "vocab_size": 7, "bias": True,
            }},
            "dataset": {"tokenizer": {
                "name": "char", "version": 1, "vocab_size": 7,
                "chars": ["\n", "a", "b", "c", "d", "e", "f"],
            }},
            "artifact_sha256": sha256_file(root / "model.pt"),
        }
        (root / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        return manifest

    def test_sample_round_trip_uses_manifest_vocab_and_safe_loading(self) -> None:
        import torch
        sample = self._sample_module()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self._save_sample_run(root)
            with patch.object(torch, "load", wraps=torch.load) as loader:
                loaded, tokenizer = sample.load_run(root)
            self.assertIs(loader.call_args.kwargs["weights_only"], True)
            prompt = torch.tensor([[1, 2]])
            expected = self.model.generate(prompt, 5, top_k=1)
            actual = loaded.generate(prompt, 5, top_k=1)
            self.assertTrue(torch.equal(actual, expected))
            self.assertEqual(tokenizer.encode("ab"), [1, 2])
            self.assertEqual(tokenizer.decode([1, 2]), "ab")
            (root / "prompt.txt").write_text("ab", encoding="utf-8")
            output = io.StringIO()
            with redirect_stdout(output):
                result = sample.main([
                    f"--out_dir={root}", f"--start=FILE:{root / 'prompt.txt'}",
                    "--num_samples=2", "--max_new_tokens=5", "--top_k=1", "--device=cpu",
                ])
            self.assertEqual(result, 0)
            self.assertEqual(output.getvalue().count(tokenizer.decode(expected[0].tolist())), 2)
            with self.assertRaisesRegex(ValueError, "absent"):
                tokenizer.encode("z")
            with (root / "model.pt").open("ab") as stream:
                stream.write(b"tampered")
            with self.assertRaisesRegex(ValueError, "hash"):
                sample.load_run(root)

    def test_legacy_byte_checkpoint_names_load_without_changing_logits(self) -> None:
        import torch
        from evogpt.training import sha256_file
        sample = self._sample_module()
        old_state = {}
        for name, value in self.model.state_dict().items():
            if name.endswith(("attn.c_attn.bias", "attn.c_proj.bias")):
                continue
            name = name.replace("transformer.wte.", "token_embedding.")
            name = name.replace("transformer.wpe.", "position_embedding.")
            name = name.replace("transformer.h.", "blocks.")
            name = name.replace("ln_1.", "attention_norm.").replace("ln_2.", "mlp_norm.")
            name = name.replace("attn.c_attn.", "attention.qkv.")
            name = name.replace("attn.c_proj.", "attention.projection.")
            name = name.replace("mlp.c_fc.", "mlp.0.").replace("mlp.c_proj.", "mlp.2.")
            name = name.replace("transformer.ln_f.", "final_norm.").replace("lm_head.", "output.")
            old_state[name] = value
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest = self._save_sample_run(root)
            torch.save(old_state, root / "model.pt")
            manifest["artifact_sha256"] = sha256_file(root / "model.pt")
            (root / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            loaded, _ = sample.load_run(root)
            prompt = torch.tensor([[1, 2, 3]])
            self.assertTrue(torch.equal(loaded(prompt)[0], self.model(prompt)[0]))

    def test_sampling_full_checkpoint_fallback_requires_its_hash(self) -> None:
        import torch
        from dataclasses import asdict
        from evogpt.training import sha256_file
        sample = self._sample_module()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest = self._save_sample_run(root)
            (root / "model.pt").unlink()
            torch.save({"model": self.model.state_dict(), "model_args": asdict(self.model.config)}, root / "ckpt.pt")
            with self.assertRaisesRegex(ValueError, "hash"):
                sample.load_run(root)
            manifest["ckpt_sha256"] = sha256_file(root / "ckpt.pt")
            (root / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            loaded, _ = sample.load_run(root)
            self.assertTrue(torch.equal(loaded.lm_head.weight, self.model.lm_head.weight))

    def test_utf8_byte_tokenizer_preserves_prompt_and_replaces_incomplete_sequences(self) -> None:
        sample = self._sample_module()
        tokenizer = sample.TextTokenizer({"name": "utf8-bytes", "version": 1, "vocab_size": 256})
        self.assertEqual(tokenizer.decode(tokenizer.encode("café λ")), "café λ")
        self.assertEqual(tokenizer.decode([255]), "\ufffd")


if __name__ == "__main__":
    unittest.main()
