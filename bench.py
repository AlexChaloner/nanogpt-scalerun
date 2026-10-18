"""Measure synthetic GPT training throughput with familiar nanoGPT settings.

    python bench.py --device=cpu --batch_size=2 --block_size=32 --num_threads=1
    python bench.py --device=cuda --dtype=bfloat16 --compile=True

This measures model updates on one fixed random batch. It does not write a
checkpoint or submit a competition result. Reported TFLOPs/s uses the same
counted FLOPs as competition runs, not a hardware profiler or utilization metric.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import asdict
import json
import math
import time

import torch

from model import GPT, GPTConfig
from scalerun.training import FLOP_COUNTER, count_flops


def _boolean(value: str) -> bool:
    if value.lower() in ("true", "1"):
        return True
    if value.lower() in ("false", "0"):
        return False
    raise argparse.ArgumentTypeError("expected True or False")


def benchmark(args: argparse.Namespace) -> dict:
    """Time complete forward/backward/AdamW iterations after optional warmup."""
    if args.batch_size < 1 or args.max_iters < 1:
        raise ValueError("batch_size and max_iters must be positive")
    if args.warmup_iters < 0 or args.num_threads < 0:
        raise ValueError("warmup_iters and num_threads must be nonnegative")
    if args.num_threads:
        torch.set_num_threads(args.num_threads)
    requested_device = args.device
    if requested_device == "auto":
        requested_device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested_device)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("benchmark device must be cpu or cuda")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA requested, but CUDA is not available")
        with torch.cuda.device(device):
            if args.dtype == "bfloat16" and not torch.cuda.is_bf16_supported():
                raise ValueError("this CUDA device does not support bfloat16")
    elif args.dtype == "float16":
        raise ValueError("float16 training requires CUDA; use float32 or bfloat16 on CPU")
    torch.manual_seed(1337)
    config = GPTConfig(
        block_size=args.block_size, vocab_size=args.vocab_size,
        n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd,
    )
    raw_model = GPT(config).to(device).train()
    parameter_count = raw_model.get_num_params(non_embedding=False)
    optimizer = raw_model.configure_optimizers(0.1, 1e-3, (0.9, 0.95), device.type)
    model = torch.compile(raw_model) if args.compile else raw_model
    scaler = torch.amp.GradScaler("cuda", enabled=args.dtype == "float16")
    x = torch.randint(config.vocab_size, (args.batch_size, config.block_size), device=device)
    y = torch.randint(config.vocab_size, x.shape, device=device)

    def update(network=model) -> torch.Tensor:
        optimizer.zero_grad(set_to_none=True)
        precision = (
            nullcontext() if args.dtype == "float32"
            else torch.autocast(device_type=device.type, dtype=getattr(torch, args.dtype))
        )
        with precision:
            _, loss = network(x, y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        return loss.detach()

    # Count one eager update; compiled kernels do the same arithmetic.
    step_flops = count_flops(torch, lambda: update(raw_model))
    for _ in range(args.warmup_iters):
        update()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    for _ in range(args.max_iters):
        loss = update()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    last_loss = float(loss.item())
    if not math.isfinite(last_loss):
        raise RuntimeError("benchmark training produced a nonfinite loss")
    return {
        "device": str(device), "dtype": args.dtype, "compile": args.compile,
        "model": asdict(config), "batch_size": args.batch_size,
        "parameter_count": parameter_count, "num_threads": torch.get_num_threads(),
        "warmup_iters": args.warmup_iters, "measured_iters": args.max_iters,
        "elapsed_seconds": elapsed, "ms_per_update": 1000 * elapsed / args.max_iters,
        "tokens_per_second": args.batch_size * config.block_size * args.max_iters / elapsed,
        "counted_training_tflops": step_flops * args.max_iters / elapsed / 1e12,
        "counted_flops_per_update": step_flops,
        "flop_counter": FLOP_COUNTER["name"],
        "timing_scope": "forward, loss, backward and AdamW on a fixed synthetic batch; warmup and data loading excluded",
        "last_loss": last_loss,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--block_size", type=int, default=128)
    parser.add_argument("--n_layer", type=int, default=4)
    parser.add_argument("--n_head", type=int, default=4)
    parser.add_argument("--n_embd", type=int, default=256)
    parser.add_argument("--vocab_size", type=int, default=256)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--dtype", choices=("float32", "bfloat16", "float16"), default="float32")
    parser.add_argument("--compile", type=_boolean, default=False,
                        help="True/False; use warmup_iters >= 1 to exclude compilation startup")
    parser.add_argument("--warmup_iters", type=int, default=5)
    parser.add_argument("--max_iters", type=int, default=20)
    parser.add_argument("--num_threads", type=int, default=0, help="0 keeps PyTorch's thread setting")
    args = parser.parse_args(argv)
    try:
        print(json.dumps(benchmark(args), indent=2, allow_nan=False))
    except (ValueError, RuntimeError) as error:
        parser.exit(1, f"bench: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
