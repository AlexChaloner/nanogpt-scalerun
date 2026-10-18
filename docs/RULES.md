# Competition rules

EvoGPT compares training recipes across five orders of magnitude of training
compute. A recipe must work across scales: its overall score is its weakest
normalized scale score. Each week, a fixed number of eligible recipes advance
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
compute estimator. The bundled trainer's evaluation and estimator are
described in [the design document](DESIGN.md); maintainers must review any
implementation that changes these policies.

The default budgets are:

| Scale | Maximum estimated training FLOPs | Relative budget |
| --- | ---: | ---: |
| S0 | 10,000,000,000 | 1× |
| S1 | 100,000,000,000 | 10× |
| S2 | 1,000,000,000,000 | 100× |
| S3 | 10,000,000,000,000 | 1,000× |
| S4 | 100,000,000,000,000 | 10,000× |

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

The bundled trainer uses one fixed recipe at every scale. Any future
scale-dependent implementation must commit its full scaling policy in advance
and identify that policy with one digest; per-scale choices cannot be selected
after seeing their results.

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
required seeds, then compare that mean with the frozen baseline loss:

```text
scale_score = 1 - mean_validation_loss / baseline_validation_loss
overall_score = min(scale_score across all active scales)
```

Higher is better. A scale score of `0` matches the reference; `0.05` means a
5% reduction in validation cross-entropy; `-0.05` means a 5% increase. These
are relative changes in loss, not percentage changes in perplexity, accuracy,
or downstream task performance.

For example, consider two entries with complete seed coverage:

| Entry | S0 | S1 | S2 | Overall |
| --- | ---: | ---: | ---: | ---: |
| Steady | 0.06 | 0.05 | 0.04 | **0.04** |
| Specialist | 0.12 | 0.10 | -0.01 | **-0.01** |

Steady ranks higher because its weakest scale is better. Strong results at one
scale cannot compensate for a regression at another.

Equal overall scores are ordered by the mean of the active scale scores, then
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

The bundled trainer uses a documented **estimated FLOPs proxy**. It does not
profile executed accelerator instructions, measure GPU utilization, or enforce
equal wall-clock time. The proxy gives the starter implementation an explicit,
reproducible budget rule; see [the design document](DESIGN.md) for its limits.

Every training step must fit within its scale's budget. Training stops before
the next whole step would exceed that cap. Evaluation is deterministic and
uses the same held-out bytes and evaluation policy for every compared run.
Preparation rejects identical files and complete-file containment. Checking
for partial or semantic overlap remains a dataset curation responsibility.

For a serious open competition, freeze an accounting policy that covers model
changes, optimizer updates, auxiliary networks, search or preprocessing work,
and any other material training cost. A participant must not relabel expensive
work as free preprocessing. Extending the starter proxy to a substantially
different architecture requires review before results are comparable.

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
