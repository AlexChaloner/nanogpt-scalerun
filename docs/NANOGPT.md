# Coming from nanoGPT

EvoGPT keeps the small prepare → train → sample workflow familiar to users of
[Karpathy's nanoGPT](https://github.com/karpathy/nanoGPT#quick-start), then adds
multiple compute budgets, worst-scale scoring, and weekly promotion. It is an
independent implementation. Familiar filenames and settings do not imply that
upstream configs, checkpoints, or benchmark results are interchangeable.

## Start here

```bash
python -m pip install -e '.[train]'
python data/shakespeare_char/prepare.py
python train.py config/train_shakespeare_char.py --device=cpu --compile=False
python sample.py --out_dir=runs/out-shakespeare-char --device=cpu
```

Use `python data/shakespeare_char/prepare.py --input /path/to/input.txt` to
prepare your own text. The preparation writes a character vocabulary and two
token streams. This is distinct from the byte-tokenized dataset prepared by
`evogpt prepare`.

The experiment files are deliberately easy to find:

```text
model.py                         GPTConfig and GPT imports
train.py                         config file + --key=value overrides
sample.py                        generate text from an EvoGPT checkpoint
bench.py                         synthetic training throughput benchmark
config/train_shakespeare_char.py small character-level preset
config/train_gpt2.py              GPT-2-sized architecture preset
data/shakespeare_char/prepare.py  character token preparation
data/shakespeare/prepare.py       GPT-2 BPE token preparation
data/openwebtext/prepare.py       import existing uint16 token streams
src/evogpt/                      implementation and tournament engine
```

`python train.py config/train_gpt2.py` uses familiar GPT-2-sized architecture
settings with random initialization. Reproducing GPT-2 quality requires
measured training on an appropriate corpus; this preset supplies a starting
configuration.

To inspect a trained model with a prompt and sampling controls:

```bash
python sample.py --out_dir=runs/out-shakespeare-char --device=cpu \
  --start='ROMEO:' --temperature=0.8 --top_k=40 --max_new_tokens=200
```

`--start=FILE:prompt.txt` reads a UTF-8 prompt. The sampler verifies the saved
model hash against `run_manifest.json` and selects the tokenizer recorded
there. Keep the checkpoint and its manifest together.

## What carries over

Upstream nanoGPT uses [Python config files followed by `--key=value`
overrides](https://github.com/karpathy/nanoGPT/blob/master/configurator.py).
EvoGPT accepts the same command style for its supported settings. Config files
contain scalar assignments, references to earlier settings, and simple
arithmetic such as `gradient_accumulation_steps = 5 * 8`. An unknown or
unsupported option is an error;
it is never silently ignored. An arbitrary upstream Python config may need
editing before it can be used here.

The model interface follows the recognizable [nanoGPT
API](https://github.com/karpathy/nanoGPT/blob/master/model.py): `GPTConfig`,
`GPT`, model-size settings such as `n_layer`, `n_head`, `n_embd`, and
`block_size`, and a forward call returning logits and an optional loss.
The bundled model supports autoregressive sampling. Its weights and module
layout belong to EvoGPT.

| Capability | EvoGPT |
| --- | --- |
| Prepare → train → sample scripts | Included |
| Python config + `--key=value` overrides | Included for supported scalar settings |
| `GPTConfig` / `GPT` and optional training loss | Included |
| Character `uint16` `train.bin` / `val.bin` | Included with `meta.json` metadata |
| GPT-2 BPE token preparation | Optional `tiktoken` dependency |
| Existing OpenWebText token binaries | Import supported; download/tokenization pipeline not bundled |
| Checkpoint resume | EvoGPT checkpoints; source and dataset consistency required |
| Gradient accumulation and LR warmup/cosine decay | Included |
| CUDA mixed precision and `torch.compile` | Included; accelerator behavior needs hardware validation |
| CPU and single-device training | Included |
| Standalone synthetic throughput benchmark | Included in `bench.py` |
| Multi-GPU `torchrun` / DDP | Not implemented |
| W&B logging | Not implemented; JSONL training logs are written locally |
| Upstream nanoGPT checkpoint loading | Not implemented |
| OpenAI GPT-2 pretrained initialization / finetuning | Not implemented |
| FineWeb speedrun shards and optimized kernels | Not implemented |
| Public measured benchmark and calibrated baseline | Must be established for each competition |

The upstream [training script](https://github.com/karpathy/nanoGPT/blob/master/train.py)
is the reference for its DDP, pretrained initialization, checkpoint, logging,
and optimization behavior. EvoGPT's supported subset is explicit rather than
a promise of full compatibility.

Common settings include:

- Model: `n_layer`, `n_head`, `n_embd`, `block_size`, `dropout`, `bias`.
- Training: `batch_size`, `gradient_accumulation_steps`, `learning_rate`,
  `weight_decay`, `beta1`, `beta2`, `grad_clip`, `max_iters`.
- Schedule: `decay_lr`, `warmup_iters`, `lr_decay_iters`, `min_lr`.
- Execution and output: `device`, `dtype`, `compile`, `out_dir`,
  `log_interval`, `eval_interval`, `checkpoint_interval`.

Progress logs report loss, learning rate, iteration time, and tokens per
second. `eval_interval` requests full, deterministic train/validation passes;
it is not upstream's sampled `eval_iters` setting. Full evaluation can be
expensive on a large corpus. For synthetic throughput measurements, run:

```bash
python bench.py --device=cpu --max_iters=5 --warmup_iters=1 --num_threads=4
```

`bench.py` reports JSON with milliseconds per update, tokens per second,
and estimated TFLOPs/s. It times forward, loss, backward, and AdamW on a fixed
synthetic batch; warmup and data loading are excluded. Estimated TFLOPs/s
uses the same analytic proxy as training. MFU reporting and distributed
profiling remain future work.

`--init_from=resume` continues an interrupted run from its checkpoint. Keep
the original config and `max_iters`: the contest budget, recipe, revision,
scale, seed, data, and device must still agree. This continuation includes
previously consumed compute. Increasing `max_iters` to extend a finished run
is not supported; resuming a finished run returns its existing result.

During training, Ctrl+C finishes the current complete update and saves a
resumable checkpoint before stopping. This preserves the compute counter at
an optimizer boundary.

To exercise interruption and resume without waiting for a failure:

```bash
python train.py config/train_shakespeare_char.py --device=cpu --compile=False \
  --out_dir=runs/resume-demo --stop_after_steps=50
python train.py config/train_shakespeare_char.py --device=cpu --compile=False \
  --out_dir=runs/resume-demo --init_from=resume
```

`stop_after_steps` is an absolute update count and does not change the planned
budget. To reevaluate a completed checkpoint on its original dataset, use
the same `out_dir` as the run:

```bash
python train.py config/train_shakespeare_char.py --eval_only=True --device=cpu
```

## Data and scores

Upstream nanoGPT's [character preparation](https://github.com/karpathy/nanoGPT/blob/master/data/shakespeare_char/prepare.py)
writes raw `uint16` token IDs with `meta.pkl` vocabulary metadata. EvoGPT's
character preparation also writes `uint16` streams, but uses explicit JSON
metadata and a manifest containing data hashes. It does not load pickled
upstream metadata. Reprepare character data from source text or import token
streams with their declared tokenizer information.

For GPT-2 BPE experiments, install `tiktoken`, then run
`python data/shakespeare/prepare.py`. To use prepared OpenWebText binaries,
run:

```bash
python data/openwebtext/prepare.py \
  --train /path/to/train.bin --val /path/to/val.bin
```

This imports existing GPT-2 token streams without downloading a large corpus.

Byte, character, and BPE losses measure different prediction targets. Dataset
identity includes tokenizer and split hashes, so these runs cannot be pooled
into one leaderboard. Fixed full-validation windows also differ from
upstream nanoGPT's sampled validation estimates. Choose and freeze the
dataset and evaluation protocol before measuring competition baselines.

## Experiments and official entries

Plain `train.py` runs are local experiments. They permit a dirty checkout and
record that source status in their experiment metadata. They do not become
official competition results merely because they produce a checkpoint.

An official run uses the frozen contest and one of its configured scale
budgets. It requires a clean, tracked source revision and a frozen recipe.
The `evogpt train` workflow in the [README](../README.md#start-a-real-competition)
shows preparation, reference calibration, entry identity, and result export.
Each scale starts from its own initialization; resume only continues the
same run and never supplies an uncharged head start at a larger scale.

The familiar frontend also accepts `--contest=configs/season.json`,
`--scale=s0`, `--seed=0`, and `--recipe_id=my-recipe` for official runs.
The contest must identify the same prepared dataset and tokenizer as the
training config. In this mode the frozen scale's FLOPs cap determines the
number of updates; `max_iters` controls local experiment budgets only.

## The speedrun comparison

The familiar leaderboard competition is
[KellerJordan's modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt),
which descends from nanoGPT through llm.c. Its main speedrun measures wall
time to a target FineWeb validation loss on specified hardware. Its
[rules](https://github.com/KellerJordan/modded-nanogpt#rules) preserve token
streams, require evidence across repeated runs, and compare records on the
same hardware.

EvoGPT instead measures validation loss at several estimated training FLOPs
budgets and ranks the weakest normalized score. It does not inherit FineWeb
scores, timing records, Muon/FP8 optimizations, or speedrun submission
compatibility. Useful future work includes a pinned curated corpus,
calibrated reference runs, independently replicated results, scale-dependent
recipe policies, and a trusted accelerator runner.

## Attribution

The familiar script names, config style, and model interface are inspired by
[nanoGPT](https://github.com/karpathy/nanoGPT). The competitive experiment
format is inspired by [modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt).
Both upstream repositories are MIT licensed. EvoGPT's multi-scale tournament
and implementation are maintained independently.
