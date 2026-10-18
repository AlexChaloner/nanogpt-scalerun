# EvoGPT

**A language-model leaderboard where your weakest compute scale determines
your rank.**

EvoGPT takes the small, reproducible experiment style of nanoGPT and adds a
multi-scale tournament. Recipes compete at five 10× training-compute scales,
spanning four orders of magnitude. Every seven days, the top five eligible recipes
advance to the next budget. The working name is EvoGPT.

If you know [nanoGPT](https://github.com/karpathy/nanoGPT), start with the same
prepare → train → sample loop. `model.py`, `train.py`, `sample.py`, and Python
files under `config/` are the experiment entrypoints. The `evogpt` CLI adds
the competition rules and result accounting.

## Train a small GPT

Python 3.10 or newer is required. From a repository checkout:

```bash
cd evogpt
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[train]'

python data/shakespeare_char/prepare.py
python train.py config/train_shakespeare_char.py --device=cpu --compile=False
python sample.py --out_dir=runs/out-shakespeare-char --device=cpu
```

The preparation script downloads Tiny Shakespeare from its original source;
use `--input /path/to/input.txt` to prepare local text instead. The small
character-level preset is suitable for a setup check on CPU. Override familiar
settings directly, for example `--max_iters=200 --batch_size=8 --block_size=64`.
Set `--device=cuda` for GPU training.

To measure synthetic training throughput, run:

```bash
python bench.py --device=cpu --max_iters=5 --warmup_iters=1 --num_threads=4
```

This reports time per update, tokens per second, and counted TFLOPs/s for
the model and optimizer. It excludes data loading and writes no competition
result.

These are local experiments. They record their configuration and source
status without requiring a clean checkout. Official competition results need
a frozen contest, a scale, calibrated baselines, and a clean source revision;
the workflow below provides those steps. Local experiments cannot be ranked
against an official contest.

EvoGPT preserves familiar commands while using its own trainer, dataset
metadata, and checkpoints. See [Coming from nanoGPT](docs/NANOGPT.md) for
supported settings, compatibility limits, and what still needs building.

## The competition

| Scale | Training FLOPs | Opens |
| --- | ---: | --- |
| `s0` | 10¹⁵ | Initially |
| `s1` | 10¹⁶ | Initially |
| `s2` | 10¹⁷ | After the first promotion |
| `s3` | 10¹⁸ | After the second promotion |
| `s4` | 10¹⁹ | After the third promotion |

At each scale, average validation loss over seeds `0`, `1`, and `2`, then score
the relative reduction from a frozen baseline:

```text
scale_score = 1 - mean_validation_loss / baseline_validation_loss
overall_score = min(scale_score across all active scales)
```

Higher scores win. A recipe that improves every active scale by 4% ranks above
one that improves most scales by 20% but regresses at one. Each scale also has
its own leaderboard. Ties in the overall ranking use the mean scale score,
then the immutable entry ID.

This repository includes a Python CLI, a small original PyTorch GPT baseline,
result validation, reproducible tournament state, synthetic examples, and
GitHub automation for proposing weekly advancement. **It does not ship real
benchmark results.** Production baseline losses are deliberately unset until
you measure them.

## Try the tournament without a GPU

The scoring and tournament tools have no
runtime dependencies beyond the standard library. From a repository checkout:

```bash
cd evogpt
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .

evogpt demo --output demo-output
evogpt validate --config demo-output/contest.json --results demo-output/results.json
evogpt leaderboard --config demo-output/contest.json \
  --results demo-output/results.json --state demo-output/state.json
```

The demo writes a synthetic competition, clearly marked fake results, JSON and
Markdown leaderboards, and `state.week1.json` showing the first promotion.
Its specialist wins `s0`, while its more consistent recipe wins overall.
Demo records and their fixture baselines stay separate from real competitions.

## Start a real competition

Choose a dataset you can redistribute or identify reproducibly, and provide
separate UTF-8 training and validation files. The byte-level competition
baseline needs no external tokenizer; `evogpt prepare` uses your local files.

```bash
python -m pip install -e '.[train]'
evogpt prepare --train /path/to/train.txt --val /path/to/val.txt --output data
evogpt configure --config configs/contest.json --manifest data/manifest.json \
  --epoch-start 2026-11-02T00:00:00Z --output configs/season-uncalibrated.json
```

Replace the example launch time with your chosen future UTC start, leaving
time for reference calibration. The preparation step records data hashes and
token counts. The configure command binds the competition to that dataset.
Review the budgets, seeds, and promotion count before freezing them.

Training requires a clean Git checkout and a tracked recipe. Commit your
configured competition and any source or recipe changes before running:

```bash
git add configs/season-uncalibrated.json
git commit -m "Configure dataset and launch time for season one"
REFERENCE_REVISION=$(git rev-parse HEAD)

evogpt train --config configs/season-uncalibrated.json \
  --recipe examples/baseline.json --data data --scale s0 --seed 0 \
  --recipe-id baseline --revision "$REFERENCE_REVISION" \
  --output runs/reference/s0/0
```

Each run writes a checkpoint, a training log, a provenance manifest, and a
`result.json`. The baseline recipe selects CUDA when available and otherwise
uses the CPU. The larger budgets can be expensive; one small run is enough to
check your setup, but is not enough to calibrate or rank a competition.

A recipe is a scaling policy: shared `model` and `training` settings plus
per-scale overrides under `scales`. The baseline in `examples/baseline.json`
grows from about 2M parameters at `s0` to about 290M at `s4`, at roughly 20
training tokens per parameter. Its top scale trains on about 5B byte tokens,
so the training split should be at least that large to avoid repeating data.

Collect the reference recipe at **all five scales and all three seeds**, using
a separate output directory for every run. Keep its source revision fixed.
Then freeze the measured baseline losses:

```bash
evogpt calibrate --config configs/season-uncalibrated.json \
  --results runs/reference --recipe-id baseline --revision "$REFERENCE_REVISION" \
  --output configs/season.json
evogpt init --config configs/season.json --output competition/state.json
```

Calibration writes `configs/season.calibration.json` with the source
configuration and reference evidence. Freezing the baseline changes the
configuration digest. Reference runs belong to the uncalibrated configuration;
keep them outside contestant submissions. All ranked contestant runs must use
the new frozen configuration. Commit the frozen configuration and initial
state before beginning contestant training.

An entry is `recipe-id@40-character-Git-revision`. The training CLI defaults to
the current clean `HEAD`; an explicit `--revision` must match it. Results from
different revisions never pool, even when their recipe IDs match. An entry
also has one frozen `recipe_sha256`; changing settings requires a distinct
entry. It needs every configured seed at every active scale for an overall
score.

## Validate, rank, and advance

Store accepted contestant result records under `submissions/`. A result path
can be a JSON or JSONL file, or a directory containing run `result.json` files.
Multiple paths are accepted after `--results`.

```bash
evogpt validate --config configs/season.json --results submissions \
  --state competition/state.json
evogpt leaderboard --config configs/season.json --results submissions \
  --state competition/state.json --output leaderboard.json --markdown leaderboard.md
evogpt advance --config configs/season.json --results submissions \
  --state competition/state.json --as-of 2026-11-09T00:00:00Z \
  --output competition/state.proposed.json
```

The example advance is due one week after the example launch. Review the
proposed state and its supporting runs before making it the accepted state.
Only the selected immutable entries may compete at the next scale. Earlier
round evidence remains frozen, and new-scale results must belong to that
round's time window. After all five scales, advancement closes the final
ranking. A missed window is an error rather than permission to skip rounds.

An uncalibrated configuration can validate reference records, but cannot
produce a ranked competition or advance. Result validation checks schema,
finite losses, configured seeds, dataset and configuration identity, unique
entry/scale/seed coverage, and compute caps. It does not independently prove
that submitted training happened; review and replication remain necessary.

## Compute accounting

FLOPs are **counted from the operations training executes**. Before training,
the trainer runs one microbatch and one optimizer step on a throwaway copy of
the model under PyTorch's FLOP counter, then charges every update:

```text
FLOPs per update = gradient_accumulation_steps × counted microbatch FLOPs
                 + counted optimizer-step FLOPs
```

Matrix multiplies, attention and convolutions count wherever they run,
including inside the optimizer, so a new architecture or optimizer is priced
without a hand-written formula. Elementwise and normalization work is free.
Training refuses matmul-like operations the counter cannot price, and custom
operators outside PyTorch's core library. Work that bypasses PyTorch entirely
is invisible to the counter and is not allowed in submissions.

Every full training update must fit within the scale's budget. The trainer
stops before the next update would exceed it. Validation is excluded. Fixed
validation windows cover each held-out next-token target once; their losses
are weighted by token count. The byte track evaluates bytes; character and
GPT-2 token tracks have separate dataset identities and losses.

FLOPs do not price memory. Large lookup tables add capacity at almost no
counted cost, and equal FLOPs do not imply equal GPU time or energy use.

## Automation and development

GitHub Actions runs the tests and includes a promotion workflow scheduled for
Monday at 00:17 UTC, with a manual trigger as well. GitHub can delay scheduled
runs. This schedule assumes a Monday 00:00 UTC launch. For another launch time,
adjust the cron or trigger the workflow manually at each boundary so the next
scale keeps its full measurement window. The workflow reads this committed
directory layout:

```text
season/
  contest.json       # frozen, calibrated configuration
  state.json         # last accepted tournament state
  results/           # reviewed contestant result records
```

For the example setup above, copy `configs/season.json` to
`season/contest.json` and `competition/state.json` to `season/state.json`.
Place accepted result records under `season/results/` and commit the season
directory. Until the configuration and state exist, the schedule exits with
setup instructions.

When a round is due, download the `evogpt-weekly-promotion` artifact, review its
ranking and proposed state, then commit the accepted state as
`season/state.json`. The workflow has read-only repository permission. It does
not launch GPU jobs, execute contributor training code, or commit changes.

```bash
python -m unittest discover -s tests -v
evogpt --help
```

Read [the competition rules](docs/RULES.md), [the design](docs/DESIGN.md), and
[the contribution guide](CONTRIBUTING.md) for details. Read
[Coming from nanoGPT](docs/NANOGPT.md) for the training interface and remaining
upstream differences. The code is released
under the [MIT license](LICENSE).
