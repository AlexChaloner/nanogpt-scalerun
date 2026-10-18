# Design

EvoGPT has two jobs: measure a recipe at multiple compute budgets, and keep a
reproducible tournament state over those measurements. Its orchestration uses
the Python standard library. The optional training baseline uses PyTorch.

## Why the weakest scale wins

A leaderboard at one training budget can reward ideas that work only at that
budget. EvoGPT makes the intended behavior explicit: a recipe should improve
the baseline across a range of budgets.

Loss normalization makes improvements comparable across scales whose reference
losses differ. Taking the minimum normalized score then focuses ranking on the
weakest scale. The mean is a deterministic tiebreaker; it does not replace the
minimum as the objective.

The minimum is sensitive to noisy measurements. Required seeds reduce the
effect of a single run, but the leaderboard is not an uncertainty estimate.
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

The bundled baseline is an original, small decoder-only GPT implementation
inspired by the simplicity of nanoGPT. It is not a copy of nanoGPT and does not
claim compatibility with nanoGPT checkpoints or benchmark results.

Byte-level tokenization avoids requiring a tokenizer download: UTF-8 input is
represented as integer bytes in a fixed vocabulary. The training and validation
splits are supplied as separate files. The preparation step records their
identities so that results can be tied to the data used.

Each seed controls model initialization and training sampling. Evaluation uses
fixed validation windows and inference mode, so a run's reported loss does not
depend on random validation sampling. Hardware and library differences can
still affect numerical reproducibility; exact seeds are not a promise of
bitwise-identical outputs on every device.

Recipe settings make small experiments practical. The bundled trainer uses
the same fixed recipe at every scale, with its canonical SHA-256 recorded in
each result and checked across the entry. A future trainer can support a
scale-dependent policy if the full policy is committed before the experiments
and every run identifies the same frozen policy digest.

## Estimated FLOPs are a proxy

For a batch size `B`, context length `T`, parameter count `P`, layer count `L`,
and embedding width `D`, the estimator charges each optimizer update:

```text
estimated_training_flops = 6 × P × B × T + 12 × L × B × T² × D
```

The first term approximates dense forward/backward work and the second charges
the sequence-length-dependent attention products. `P` includes all parameters
in the bundled model. Every executed update uses this same charge. Optimizer
operations and evaluation are excluded.

The trainer stops at an explicit estimate of training arithmetic. This is
useful for a reproducible starter competition, but an estimated FLOP count is
not an accelerator profiler.

The estimate does not account for memory bandwidth, communication, compiler
fusion, utilization, elapsed time, energy use, or optimizer operations.
Validation is outside the training budget. Two models with the same estimated
FLOPs can have different wall-clock costs, and a new
architecture can invalidate assumptions in the estimator.

Treat the default estimator as the policy for the provided baseline family.
Before accepting arbitrary model families, define the operations counted,
precision policy, forward/backward accounting, optimizer and auxiliary work,
and an independently reviewable measurement method. Changing that policy
starts a new competition configuration.

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

This repository supplies a runnable baseline, result validation, normalized
leaderboards, deterministic advancement, tests, and reviewable automation.
It does not provide a distributed training service, a web leaderboard, a
submission sandbox, or verified public benchmark results.

Useful next steps are a real curated dataset and baseline calibration,
independent result replication, stronger provenance, confidence estimates,
and a trusted runner integration. These can be added without changing the
central objective: improve the weakest scale.
