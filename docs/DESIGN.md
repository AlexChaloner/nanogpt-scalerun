# Design

EvoGPT has two jobs: measure a recipe at multiple compute budgets, and keep a
reproducible tournament state over those measurements. Its orchestration uses
the Python standard library. The optional training baseline uses PyTorch.

## Why the weakest scale wins

A leaderboard at one training budget can reward ideas that work only at that
budget. EvoGPT makes the intended behavior explicit: a recipe should improve
the baseline across a range of budgets.

The minimum only makes sense if scores mean the same thing at every scale.
A relative loss reduction does not: the baseline curve flattens as compute
grows, so the same gain shows up as a smaller loss change at larger budgets,
and the minimum would almost always land on the largest scale. EvoGPT instead
scores the compute multiplier, the factor by which the baseline's budget must
grow to match the entry's loss. The calibrated baseline losses define that
curve. The geometric mean is a deterministic tiebreaker; it does not replace
the minimum as the objective.

Multipliers magnify noise where the baseline curve is flat: a small loss
difference at the largest scale is a large compute difference.

The minimum is sensitive to noisy measurements. Taking the median of five
seeds means one diverged or lucky run cannot move a score, but the leaderboard is not an uncertainty estimate.
Confidence intervals, paired statistical comparisons, and larger seed sets are
reasonable extensions for a mature competition.

## Separate configuration, evidence, and state

The competition configuration describes the frozen rules. Result records
describe measured runs. The state records the current round, active scales,
and advancing cohort. Keeping them separate makes a proposed promotion
inspectable without rerunning the experiments.

The configuration digest prevents an old result or state from quietly being
reinterpreted after a budget, baseline, seed, or other frozen setting changes.
A recipe ID and immutable revision identify the entry. Every result also
requires `recipe_sha256`, which binds all runs of the entry to the same frozen
recipe or scaling policy. This prevents results from successive code versions
or different recipe files at one revision being pooled into an imaginary
best-of-both recipe.

Leaderboard computation is read-only. Promotion writes a new state document;
the caller chooses whether and when to accept it. The GitHub workflow uses
this separation to upload a proposal artifact for maintainer review.

## Reference trainer

The bundled baseline is an original decoder-only GPT implementation inspired
by the simplicity of nanoGPT. Root `model.py`, `train.py`, `sample.py`, and
`config/*.py` provide its familiar experiment interface. The implementation
lives under `src/evogpt/`. `GPTConfig` and `GPT` expose common model-size names
and return logits with an optional training loss. This interface does not
claim compatibility with upstream checkpoints or benchmark results. See
[Coming from nanoGPT](NANOGPT.md) for the supported subset.

Byte-level tokenization avoids requiring a tokenizer download: UTF-8 input is
represented as integer bytes in a fixed vocabulary. The training and validation
splits are supplied as separate files. The preparation step records their
identities so that results can be tied to the data used. Character-level and
GPT-2 BPE preparation scripts additionally support `uint16` token streams,
explicit tokenizer metadata, and dataset manifests. Character vocabularies
use JSON metadata, not pickle. Tokenizer identity is part of dataset identity;
byte, character, and BPE scores require separate calibrated competitions.

Each seed controls model initialization and training sampling. Evaluation uses
fixed validation windows and inference mode, so a run's reported loss does not
depend on random validation sampling. Hardware and library differences can
still affect numerical reproducibility; exact seeds are not a promise of
bitwise-identical outputs on every device. The trainer supports gradient
accumulation, optional learning-rate warmup and cosine decay, single-device
mixed precision, compilation, and checkpoints that can continue the same
run. Optimizer, dataset, recipe, and compute progress must agree when resuming;
continuing a run does not reset its compute counter. Each competition scale
still starts from scratch.

The root training script also supports local experiments without requiring a
clean checkout. Their metadata records source status, and their configuration
identity differs from an official contest. Official runs retain clean-source
and frozen-config requirements. A familiar development loop must not turn
uncalibrated experiments into ranked competition evidence.

A recipe is a scaling policy. Schema 2 has shared `model` and `training`
settings plus per-scale overrides; schema 1 uses one fixed configuration at
every scale. The canonical SHA-256 of the whole policy is recorded in each
result and checked across the entry. Every scale's settings are validated
before any run starts, so a broken large-scale setting fails at the first
small run.

## Counted FLOPs

A formula written for one architecture misprices the next one, and charging
nothing for the optimizer makes optimizer-side matmuls free. So the trainer
counts what executes. Before a run, it builds a throwaway copy of the model
and runs one eager microbatch (forward and backward) and one optimizer step
under PyTorch's `FlopCounterMode`, on the run's device and dtype:

```text
update_flops = accumulation_steps × microbatch_flops + optimizer_step_flops
```

Every update has the same shapes, so one counted update prices them all. The
probe uses zero tokens because counts depend only on shapes. It runs before
the seed is reset, so it does not change the run's initialization.

The counter prices matrix multiplies, attention and convolutions; a
multiply-add is two FLOPs. PyTorch 2.6 has no formula for CPU attention, so
EvoGPT registers the CUDA formula for it and both devices charge the same
attention work. Counts still follow the kernel that runs: attention dropout
selects a different kernel and changes the count slightly.

Two kinds of work would otherwise be free, so training refuses them:
matmul-like operations without a formula, and operators outside ATen, such as
custom kernels registered with PyTorch. Code that bypasses PyTorch dispatch is
invisible to any dispatch-level counter; review must reject it.

Elementwise, normalization and reduction work is not charged, and neither is
memory: lookup tables add capacity at almost no counted cost. The count does
not reflect memory bandwidth, utilization, elapsed time or energy. Validation
is outside the training budget. Changing the counting policy starts a new
competition configuration.

## Weekly automation

Weekly progression chooses the top eligible immutable entries on the current
active prefix, then opens the next budget for the selected cohort. It does not
select a different recipe at each scale. Every subsequent promotion requires
complete evidence at all active scales.

The scheduled GitHub workflow reads committed configuration, state, and result
records. It uploads an artifact containing a proposed advancement when the
round is due and enough evidence is available. Maintainers inspect both the
state change and the underlying results before committing the proposed state.
The same workflow can be triggered manually.

Contributor training code is not executed by this workflow. Adding hosted
training requires a trusted runner, resource quotas, artifact provenance, and
an explicit review process; those are not supplied by a JSON result parser.

## Scope

This repository supplies a runnable baseline, result validation, compute-multiplier
leaderboards, deterministic advancement, tests, and reviewable automation.
It does not provide a distributed training service, a web leaderboard, a
submission sandbox, or verified public benchmark results.

Useful next steps are a real curated dataset and baseline calibration,
independent result replication, stronger provenance, confidence estimates,
and a trusted runner integration. These can be added without changing the
central objective: improve the weakest scale.
