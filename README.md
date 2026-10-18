# NanoGPT ScaleRun

A nanoGPT speedrun across five compute scales. The question: **which training
algorithm wins with scaling?**

The [nanoGPT speedrun](https://github.com/KellerJordan/modded-nanogpt) rewards
whatever is fastest at one size. A trick that only works at that size can still
win. ScaleRun trains every recipe at five budgets, 10× apart, and ranks it by
its **worst** one.

Leaderboard: <https://alexchaloner.github.io/nanogpt-scalerun/>

| Scale | Training FLOPs | Opens |
| --- | ---: | --- |
| `s0` | 10¹⁴ | Week 0 |
| `s1` | 10¹⁵ | Week 0 |
| `s2` | 10¹⁶ | Week 1 |
| `s3` | 10¹⁷ | Week 2 |
| `s4` | 10¹⁸ | Week 3 |

Each week the top five recipes are promoted to the next, more expensive scale.
The rest stop there. Only recipes that keep working as compute grows reach the
top.

## Scoring

Loss is validation cross-entropy per byte on FineWeb-Edu, taken as the median
over five seeds. At each scale, that loss is turned into a compute multiplier:
how many times the scale's budget the baseline would need to reach the same
loss.

```text
scale_score   = baseline FLOPs to reach your median loss / scale budget
overall_score = min(scale_score over open scales)
```

`1.4×` means the baseline needs 1.4 times the compute to match you. Below `1×`
means you're worse than the baseline. A multiplier means the same thing at
every scale, so taking the minimum is fair. A recipe at 1.4× everywhere beats
one at 3× on four scales and 0.9× on the fifth. Ties go to the geometric mean.

FLOPs are counted, not estimated. Before training, PyTorch's FLOP counter
measures one microbatch and one optimizer step, and every update is charged
that amount. New architectures and optimizers are priced automatically.
Training stops before an update would exceed the budget. Operations the
counter can't price are refused.

## Quick start

Python 3.10+.

```bash
git clone https://github.com/AlexChaloner/nanogpt-scalerun
cd nanogpt-scalerun
python3 -m venv .venv && source .venv/bin/activate
pip install -e '.[train,data]'

python data/fineweb_edu/prepare.py   # 6 GB of pinned FineWeb-Edu bytes
scalerun configure --config configs/contest.json \
  --manifest data/fineweb_edu/manifest.json \
  --epoch-start 2026-11-02T00:00:00Z --output configs/season-uncalibrated.json
git add configs/season-uncalibrated.json && git commit -m "Configure season"

scalerun train --config configs/season-uncalibrated.json \
  --recipe examples/baseline.json --recipe-id baseline --data data/fineweb_edu \
  --scale s0 --seed 0 --output runs/s0/0
```

Training only runs from a clean checkout, so commit your recipe first. `s0`
takes about a minute on a GPU. The familiar nanoGPT loop also
works:

```bash
python data/shakespeare_char/prepare.py
python train.py config/train_shakespeare_char.py --device=cpu --compile=False
python sample.py --out_dir=runs/out-shakespeare-char --device=cpu
```

See [Coming from nanoGPT](docs/NANOGPT.md) for what carries over.

To see a full tournament without a GPU, run `scalerun demo --output demo`.
It writes fake results, a leaderboard and a promotion.

## Recipes

A recipe is a JSON scaling policy: shared `model` and `training` settings plus
per-scale overrides. Choosing how to scale is part of the recipe. The whole
policy has one digest, so you can't swap in a different config at one scale
after seeing your results.

The baseline, `examples/baseline.json`, is a small GPT on raw bytes. It grows
from 0.3M parameters at `s0` to 32M at `s4`, at about 120 training bytes per
parameter. That's roughly where our sweeps found the compute optimum, the
byte-level equivalent of Chinchilla's 20 tokens per parameter.

An entry is `recipe-id@git-revision`. Every run must come from a clean
checkout, and runs from different revisions never pool. An entry needs all
five seeds at every open scale to be ranked.

## Running a season

Organizers calibrate the baseline at all five scales and five seeds, freeze
it, and start the clock:

```bash
scalerun calibrate --config configs/season-uncalibrated.json \
  --results runs/reference --recipe-id baseline --revision "$REV" \
  --output configs/season.json
scalerun init --config configs/season.json --output competition/state.json
```

Each week, rank the results and propose the promotion:

```bash
scalerun leaderboard --config configs/season.json --results submissions \
  --state competition/state.json --html leaderboard.html
scalerun advance --config configs/season.json --results submissions \
  --state competition/state.json --output competition/state.proposed.json
```

Commit the season as `season/contest.json`, `season/state.json` and
`season/results/`. GitHub Actions then proposes each Monday's promotion for
review and publishes the leaderboard to Pages. Nothing is promoted without a
human committing the new state.

Validation checks seeds, dataset and config identity, duplicates and FLOP
caps. It can't prove a run really happened, so results still need review or
replication.

## Limits

FLOPs don't price memory. Big lookup tables add capacity almost for free, and
equal FLOPs don't mean equal wall-clock time. Seed noise at small scales is
a few percent in compute terms, so smaller gaps are within noise.

## More

[Rules](docs/RULES.md) · [Design](docs/DESIGN.md) ·
[Coming from nanoGPT](docs/NANOGPT.md) · [Contributing](CONTRIBUTING.md) ·
MIT license.

```bash
python -m unittest discover -s tests
```
