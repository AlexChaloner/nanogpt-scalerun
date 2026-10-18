# Contributing

ScaleRun is a small reference implementation of a multi-scale language-model
competition. Contributions should keep the scoring rules reproducible and
the experiment artifacts inspectable.

## Development

Use Python 3.10 or newer in a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m unittest discover -s tests -v
```

Install the training extra when working on the optional PyTorch baseline:

```bash
python -m pip install -e '.[train]'
```

Keep the scoring and tournament code usable without PyTorch. Tests should
exercise a real rule or failure mode rather than duplicate implementation
details. A training change should include a small reproducible smoke run when
practical; a meaningful score claim needs the full configured seed coverage.

## Proposing a recipe

1. Read [the rules](docs/RULES.md) and use the exact competition configuration.
2. Commit the recipe implementation and record its immutable revision.
3. Run each required seed at every scale currently open to the entry.
4. Preserve the generated result records and supporting artifacts.
5. Validate the records locally and submit them in a pull request.

The pull request should explain the idea, identify the recipe ID and revision,
link the run artifacts, record the data identity and compute accounting, and
state how to reproduce the runs. Changes to the implementation create a new
entry; do not combine its results with an earlier revision.

The production reference losses start uncalibrated. Until maintainers have
collected and frozen real baseline runs, the repository is an experiment
harness rather than an established public benchmark. Demo records are
synthetic examples and belong only in the demo competition.

## Changing the protocol

Budget, seed, data, baseline, evaluation, or accounting changes affect the
meaning of a score. Discuss the intended behavior and migration in the pull
request, update the rules and tests, and create a fresh competition state where
the configuration digest changes. Do not reinterpret accepted records under
new rules.

Please report validation bypasses, revision pooling, budget overrun acceptance,
or nondeterministic advancement as correctness issues. Do not include secrets
or private datasets in issue reports or run artifacts.

## Reviewing promotions

Scheduled advancement produces a proposal for review. Check that the selected
entries have complete seed coverage, that submitted runs are accepted evidence,
and that the new state preserves the frozen competition configuration. The
schedule does not launch contributor code or certify the experiments.

Code and documentation contributions are licensed under the repository's
[MIT license](LICENSE). Contributed datasets and checkpoints must have their
own compatible terms and clear provenance.
