# EvoGPT

**A language-model leaderboard where your weakest compute scale determines
your rank.**

EvoGPT takes the small, reproducible experiment style of nanoGPT and adds a
multi-scale tournament. Recipes compete at five training-compute budgets,
each 10× larger than the last. Every seven days, the top five eligible recipes
advance to the next budget. The working name is EvoGPT.

| Scale | Estimated training FLOPs | Opens |
| --- | ---: | --- |
| `s0` | 10¹⁰ | Initially |
| `s1` | 10¹¹ | Initially |
| `s2` | 10¹² | After the first promotion |
| `s3` | 10¹³ | After the second promotion |
| `s4` | 10¹⁴ | After the third promotion |

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

Python 3.10 or newer is required. The scoring and tournament tools have no
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
separate UTF-8 training and validation files. No dataset is downloaded
automatically. The byte-level baseline needs no external tokenizer.

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
  --recipe examples/recipe.json --data data --scale s0 --seed 0 \
  --recipe-id baseline --revision "$REFERENCE_REVISION" \
  --output runs/reference/s0/0
```

Each run writes a checkpoint, a training log, a provenance manifest, and a
`result.json`. The default recipe selects CUDA when available and otherwise
uses the CPU. The larger budgets can be expensive; one small run is enough to
check your setup, but is not enough to calibrate or rank a competition.

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

The reference trainer uses an **analytic FLOPs proxy**, not hardware profiling:

```text
FLOPs per update = 6 × parameters × batch_size × context_length
                + 12 × layers × batch_size × context_length² × embedding_width
```

Every full training update must fit within the scale's budget. The trainer
stops before the next update would exceed it. Optimizer work and validation
are excluded. Fixed validation windows cover each held-out next-byte target
once; their losses are weighted by token count.

The proxy is suitable for the provided baseline family. New architectures
need an explicit reviewed accounting policy before their counts can be
compared. Equal estimated FLOPs do not imply equal GPU time or energy use.

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
[the contribution guide](CONTRIBUTING.md) for details. The code is released
under the [MIT license](LICENSE).
