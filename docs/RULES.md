# Competition rules

EvoGPT compares training recipes at five 10× training-compute scales spanning
four orders of magnitude. A recipe must work across scales: its overall score is its weakest
compute multiplier. Each week, a fixed number of eligible recipes advance
to the next compute budget.

This is an initial protocol and reference implementation, not a claim that any
particular recipe has achieved a measured improvement. Changes to the protocol
start a new competition configuration.

## 1. A frozen competition

A competition configuration fixes:

- The dataset and held-out validation split.
- The five compute budgets and their order.
- The baseline validation loss at each scale.
- The required seeds, promotion count, and week length.

The submitted source revision and recipe identify the evaluation settings and
FLOP counter. The bundled trainer's evaluation and FLOP counter are
described in [the design document](DESIGN.md); maintainers must review any
implementation that changes these policies.

The default budgets are:

| Scale | Maximum training FLOPs | Relative budget |
| --- | ---: | ---: |
| S0 | 1,000,000,000,000,000 | 1× |
| S1 | 10,000,000,000,000,000 | 10× |
| S2 | 100,000,000,000,000,000 | 100× |
| S3 | 1,000,000,000,000,000,000 | 1,000× |
| S4 | 10,000,000,000,000,000,000 | 10,000× |

The production configuration intentionally starts with uncalibrated baseline
losses. Collect real reference runs and calibrate before starting a ranked
competition. Synthetic demo scores are stored separately and must never be
used as production evidence.

Changing frozen settings changes the configuration digest. Results and state
from another digest are incompatible. Recalibrating a running competition is
a new competition, not a way to change its scores retrospectively.

## 2. What enters the leaderboard

An entry is an immutable pair of **recipe ID and revision**, bound to one
frozen recipe or scaling-policy digest. The revision must identify the exact
submitted implementation. Every result includes `recipe_sha256`, and that
digest must match across all seeds and scales for an entry. Changing code
means creating a new entry, including when the recipe ID stays the same.
Different recipe settings under one source revision cannot be pooled into a
single entry. Use a separate recipe ID for a distinct recipe at the same
revision.

A recipe is a scaling policy: one file with shared settings and per-scale
overrides, identified by one digest. The whole policy is committed before any
of its runs, so per-scale choices cannot be selected after seeing their
results; changing any scale's settings creates a new entry.

The default required seeds are `0`, `1`, and `2`. An entry needs one accepted
result for every required seed at every currently active scale to qualify for
the overall leaderboard. Missing runs cannot silently become good scores.
Duplicate or inconsistent records are rejected instead of being cherry-picked.

Submit the full run artifacts, recipe settings, code revision, data identity,
and compute accounting with each result. A result file is a machine-readable
claim. Maintainers still need to review reproducibility, dataset use, and
compute accounting before accepting it as competition evidence.

## 3. Scoring

For each recipe revision and scale, average the validation losses over the
required seeds, then find the compute the frozen baseline needs to reach that
mean loss:

```text
scale_score = baseline FLOPs needed to match mean_validation_loss / scale budget
overall_score = min(scale_score across all active scales)
```

The baseline curve passes through the calibrated loss at every budget and is
piecewise linear in log loss against log FLOPs. Beyond the smallest and
largest budgets it extends along the end segments. Calibrated baseline losses
must strictly decrease with compute.

Higher is better. A scale score of `1×` matches the reference; `2×` means the
baseline needs twice the budget to match; `0.5×` means the entry matches the
baseline at half its budget. A fixed loss reduction is worth more compute at
larger budgets, where the baseline curve is flatter, so multipliers compare
scales fairly where relative loss changes do not.

For example, consider two entries with complete seed coverage:

| Entry | S0 | S1 | S2 | Overall |
| --- | ---: | ---: | ---: | ---: |
| Steady | 1.6× | 1.5× | 1.4× | **1.4×** |
| Specialist | 3.0× | 2.5× | 0.9× | **0.9×** |

Steady ranks higher because its weakest scale is better. Strong results at one
scale cannot compensate for a regression at another.

Equal overall scores are ordered by the geometric mean of the active scale
scores, then
by stable entry identity. Comparisons use the stored numeric values; displayed
rounding does not create a tie. Rankings are deterministic. This starter
protocol does not claim statistical significance from a small score gap.

Per-scale leaderboards are also available. They are useful diagnostic views;
weekly promotion uses the complete overall ranking.

## 4. Weekly progression

Initially, S0 and S1 are active. Opening with two scales makes the first round
test transfer across compute budgets. By default, a round lasts seven days and
the top five eligible entries advance.

At a due round boundary:

1. Validate the submitted results against the frozen configuration.
2. Rank entries with complete coverage of the active scales and required seeds.
3. Select the top `N` entries, carrying their exact revisions forward.
4. Open the next scale for those entries and record the new round state.

The initial round accepts any valid entry. Subsequent rounds are restricted to
the previous round's selected entries. A round snapshots its accepted evidence
and ranking: earlier-scale measurements are frozen at their previous cutoff,
and the newly opened scale needs runs timestamped within its own weekly
window. Replacing an old measurement after seeing the next scale cannot change
the previous selection.

If fewer than `N` entries qualify, all qualifying entries advance. A round
with no complete entries cannot advance.

Advancement processes one due round at a time and must occur in the next
round's time window. The tool rejects an advance requested before the deadline
or after that window has already expired. Maintainers must resolve a missed
round explicitly rather than silently jumping several rounds.

Advancing requires new results at the newly opened budget; selection alone is
not evidence of performance there. Entries eliminated from the active cohort
do not return through a renamed revision. Maintainers can start a fresh
competition to admit a new cohort.

After S4, there is no higher scale to unlock. The final active cohort can be
ranked across all five scales. The scheduler does not invent extra budgets or
automatically launch training jobs.

The GitHub schedule proposes a state artifact for maintainer review. Commit
that state only after accepting the supporting results. A schedule is not
proof that training occurred, and elapsed weeks do not supply missing
measurements.

## 5. Compute accounting

Training FLOPs are **counted from executed PyTorch operations**: matrix
multiplies, attention and convolutions, including any in the optimizer. Each
update is charged the count of one probe update; see
[the design document](DESIGN.md) for how and for its limits. The count is not
wall-clock time or GPU utilization.

Every training step must fit within its scale's budget. Training stops before
the next whole step would exceed that cap. Evaluation is deterministic and
uses the same held-out bytes and evaluation policy for every compared run.
Preparation rejects identical files and complete-file containment. Checking
for partial or semantic overlap remains a dataset curation responsibility.

Submissions must run their training arithmetic through PyTorch operations the
counter can see. Training refuses matmul-like or custom operators without a
counting formula; code that bypasses PyTorch dispatch, auxiliary networks,
search, and preprocessing that does training work are not allowed. A
participant must not relabel expensive work as free preprocessing.

## 6. Reproducibility and review

- Keep validation data out of training, model selection loops, and manual tuning.
- Use the configured seeds and fixed evaluation procedure.
- Record exact source revisions, recipes, software versions, and data digests.
- Preserve original result artifacts. Correct a rejected run with an explicit
  replacement rather than silently editing accepted evidence.
- Do not execute unreviewed contributor code in a privileged workflow.

The default schedule evaluates checked-in result records and proposes a
reviewable promotion. GPU jobs, external submissions, and result verification
need a separate trusted execution setup.
