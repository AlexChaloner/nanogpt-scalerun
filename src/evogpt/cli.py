"""Command-line tools. Orchestration does not import PyTorch."""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from . import __version__
from . import core


def read_json(path):
    def pairs(items):
        output = {}
        for key, value in items:
            if key in output:
                raise ValueError(f"Duplicate JSON key: {key}")
            output[key] = value
        return output

    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle, object_pairs_hook=pairs,
                         parse_constant=lambda value: _invalid_number(value))


def _invalid_number(value):
    raise ValueError(f"Invalid JSON number: {value}")


def write_json(path, value):
    """Atomically replace a single local output; state transitions return new values."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def timestamp(value):
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
        raise ValueError("Timestamps must include the UTC timezone (Z or +00:00)")
    return parsed


def result_paths(paths):
    expanded = []
    for raw in paths:
        path = Path(raw)
        if path.is_dir():
            files = sorted(item for item in path.rglob("*")
                           if item.is_file() and item.suffix in (".json", ".jsonl")
                           and item.name not in ("run_manifest.json", "train.jsonl"))
            expanded.extend(files)
        elif path.is_file():
            expanded.append(path)
        else:
            raise ValueError(f"Result path does not exist: {path}")
    if len(set(path.resolve() for path in expanded)) != len(expanded):
        raise ValueError("The same result file was provided more than once")
    return expanded


def _results(args, contest):
    return core.load_results(result_paths(args.results), contest)


def _state(args, contest):
    return read_json(args.state) if args.state else core.initial_state(contest)


def _emit(value, path=None):
    if path:
        write_json(path, value)
    else:
        print(json.dumps(value, indent=2, sort_keys=True, allow_nan=False))


def _cell(value):
    return str(value).replace("|", "\\|").replace("\n", " ")


def render_markdown(board):
    """Readable board, using domain fields without generating a hosted service."""
    lines = ["# EvoGPT leaderboard", "", f"Contest: `{_cell(board['contest_id'])}` · "
             f"Status: **{_cell(board['status'])}** · Round: {board['round']}", "",
             "Higher score is better. Overall score is the weakest active scale.", ""]
    if any(scale["baseline_is_fixture"] for scale in board["scales"].values()):
        lines += ["> SYNTHETIC DEMO: these scores are fixtures, not measured training records.", ""]
    lines += ["## Overall", "", "| Rank | Entry | Worst scale | Mean score | Coverage |",
              "| ---: | --- | ---: | ---: | ---: |"]
    rank = 0
    for row in board["overall"]:
        scored = row["score"] is not None
        rank += int(scored)
        score = f"{row['score']:.2%}" if scored else row["status"]
        mean = f"{row['mean_score']:.2%}" if scored else "—"
        lines.append(f"| {rank if scored else '—'} | {_cell(row['entry_id'])} | {score} | "
                     f"{mean} | {row['completed_scales']}/{row['required_scales']} |")
    if not board["overall"]:
        lines += ["", "No entries yet."]
    for scale_id, scale in board["scales"].items():
        lines += ["", f"## {_cell(scale_id)} · {scale['max_flops']:,} counted FLOPs", "",
                  "| Entry | Mean validation loss | Score | Seeds complete |",
                  "| --- | ---: | ---: | --- |"]
        for row in scale["rows"]:
            score = f"{row['score']:.2%}" if row["score"] is not None else "—"
            loss = f"{row['mean_loss']:.6f}" if row["mean_loss"] is not None else "—"
            lines.append(f"| {_cell(row['entry_id'])} | {loss} | {score} | "
                         f"{'yes' if row['complete'] else 'no'} |")
    lines += ["", f"Configuration digest: `{board['config_digest']}`", ""]
    return "\n".join(lines)


def source_revision(recipe_path, requested):
    """Local runs identify the clean Git tree containing their recipe."""
    recipe = Path(recipe_path).resolve()
    def git(*arguments):
        run = subprocess.run(["git", "-C", str(recipe.parent), *arguments],
                             text=True, capture_output=True, check=False)
        if run.returncode:
            raise ValueError(run.stderr.strip() or "Recipe must be tracked in a Git repository")
        return run.stdout.strip()

    repository = Path(git("rev-parse", "--show-toplevel")).resolve()
    revision = git("rev-parse", "HEAD")
    if requested is not None and requested != revision:
        raise ValueError("--revision must match the current Git HEAD containing the recipe")
    sources = [recipe] + [Path(__file__).resolve().parent / name
                          for name in ("cli.py", "core.py", "training.py", "model.py")]
    for source in sources:
        source = source.resolve()
        if not source.is_relative_to(repository):
            raise ValueError("The recipe and installed EvoGPT source must belong to the same Git checkout; "
                             "install that checkout with pip install -e .")
        git("ls-files", "--error-unmatch", str(source))
    if git("status", "--porcelain", "--untracked-files=normal"):
        raise ValueError("Commit recipe/source changes before training; the source tree must be clean")
    return revision


def demo(output):
    root = Path(output)
    if root.exists() and any(root.iterdir()):
        raise ValueError(f"Demo destination must be empty: {root}")
    root.mkdir(parents=True, exist_ok=True)
    config = {
        "schema_version": 1, "contest_id": "evogpt-synthetic-demo",
        "dataset_id": "synthetic-demo-no-real-training",
        "scales": [{"id": f"s{i}", "max_flops": 10 ** (10 + i),
                    "baseline_loss": 5.0 - i * 0.5, "baseline_is_fixture": True}
                   for i in range(5)],
        "initial_scale_count": 2, "top_n": 2, "seeds": [0, 1, 2],
        "epoch_start": "2026-01-05T00:00:00Z", "interval_days": 7,
    }
    contest = core.parse_config(config)
    digest = core.config_digest(contest)
    fixtures = []
    # The specialist wins s0, but robust wins the minimax comparison.
    for name, scores in {"robust": (0.12, 0.10), "specialist": (0.24, 0.03),
                         "fragile": (0.07, -0.04)}.items():
        revision = hashlib.sha1(f"synthetic:{name}".encode()).hexdigest()
        for scale, score in zip(config["scales"], scores):
            for seed in config["seeds"]:
                run_id = f"demo-{name}-{scale['id']}-{seed}"
                fixtures.append({
                    "schema_version": 1, "run_id": run_id, "recipe_id": name,
                    "revision": revision, "scale_id": scale["id"], "seed": seed,
                    "recipe_sha256": hashlib.sha256(f"synthetic-policy:{name}".encode()).hexdigest(),
                    "val_loss": scale["baseline_loss"] * (1 - score) + (seed - 1) * 0.001,
                    "train_flops": scale["max_flops"], "train_tokens": 1024,
                    "dataset_id": config["dataset_id"], "config_digest": digest,
                    "created_at": "2026-01-06T00:00:00Z",
                    "artifact_sha256": hashlib.sha256((run_id + "fake-artifact").encode()).hexdigest(),
                    "log_sha256": hashlib.sha256((run_id + "fake-log").encode()).hexdigest(),
                })
    write_json(root / "contest.json", config)
    write_json(root / "results.json", fixtures)
    results = core.load_results([root / "results.json"], contest)
    state = core.initial_state(contest)
    write_json(root / "state.json", state)
    board = core.leaderboard(contest, results, state)
    write_json(root / "leaderboard.json", board)
    (root / "leaderboard.md").write_text(render_markdown(board), encoding="utf-8")
    next_state = core.advance(contest, results, state, timestamp("2026-01-12T00:00:00Z"))
    write_json(root / "state.week1.json", next_state)
    print(f"Synthetic demo written to {root.resolve()}")


def calibrate(args):
    contest = core.load_config(args.config)
    results = _results(args, contest)
    candidates = [result.to_dict() for result in results]
    candidates = [result for result in candidates
                  if result["recipe_id"] == args.recipe_id and result["revision"] == args.revision]
    config = contest.to_dict()
    if any(scale["baseline_loss"] is not None for scale in config["scales"]):
        raise ValueError("Calibration requires an uncalibrated config; create a new season to change baselines")
    for scale in config["scales"]:
        rows = [result for result in candidates if result["scale_id"] == scale["id"]]
        if sorted(result["seed"] for result in rows) != sorted(config["seeds"]):
            raise ValueError(f"Calibration needs exactly the required seeds for scale {scale['id']}")
        scale["baseline_loss"] = math.fsum(result["val_loss"] / len(rows) for result in rows)
        scale["baseline_is_fixture"] = False
    core.parse_config(config)
    output = Path(args.output)
    evidence = output.with_suffix(".calibration.json")
    if output.resolve() == Path(args.config).resolve():
        raise ValueError("Write calibration to a new config so original result provenance remains reproducible")
    write_json(evidence, {"schema_version": 1, "source_config": contest.to_dict(),
                         "source_config_digest": core.config_digest(contest),
                         "calibrated_config_digest": core.config_digest(config),
                         "recipe_id": args.recipe_id, "revision": args.revision,
                         "reference_results": candidates})
    write_json(output, config)
    print(f"Frozen baseline config: {output}; calibration evidence: {evidence}")


def parser():
    cli = argparse.ArgumentParser(description="EvoGPT multi-scale benchmark and weekly promotion tools")
    cli.add_argument("--version", action="version", version=__version__)
    commands = cli.add_subparsers(dest="command", required=True)

    def config_args(command, results=False, state=False):
        command.add_argument("--config", required=True)
        if results:
            command.add_argument("--results", nargs="+", default=[])
        if state:
            command.add_argument("--state")

    check = commands.add_parser("validate", help="Validate config, provenance and result uniqueness")
    config_args(check, results=True, state=True)
    board = commands.add_parser("leaderboard", help="Compute per-scale and minimax leaderboards")
    config_args(board, results=True, state=True)
    board.add_argument("--output")
    board.add_argument("--markdown")
    init = commands.add_parser("init", help="Create the initial weekly tournament state")
    config_args(init)
    init.add_argument("--output", required=True)
    advance = commands.add_parser("advance", help="Freeze one weekly boundary and promote the top N")
    config_args(advance, results=True, state=True)
    advance.add_argument("--as-of", help="UTC timestamp; defaults to now")
    advance.add_argument("--output", required=True)
    setup = commands.add_parser("configure", help="Bind a contest template to a prepared dataset")
    config_args(setup)
    setup.add_argument("--manifest", required=True)
    setup.add_argument("--epoch-start", required=True)
    setup.add_argument("--output", required=True)
    reference = commands.add_parser("calibrate", help="Freeze per-scale baselines from reference runs")
    config_args(reference, results=True)
    reference.add_argument("--recipe-id", required=True)
    reference.add_argument("--revision", required=True)
    reference.add_argument("--output", required=True)
    prepare = commands.add_parser("prepare", help="Prepare byte-tokenized train and validation files")
    prepare.add_argument("--train", required=True)
    prepare.add_argument("--val", required=True)
    prepare.add_argument("--output", required=True)
    train = commands.add_parser("train", help="Run the reference GPT within a counted FLOPs budget")
    config_args(train)
    train.add_argument("--recipe", required=True)
    train.add_argument("--data", required=True)
    train.add_argument("--scale", required=True)
    train.add_argument("--seed", required=True, type=int)
    train.add_argument("--recipe-id", required=True)
    train.add_argument("--revision", help="Source Git SHA; defaults to clean HEAD and must match HEAD")
    train.add_argument("--output", required=True)
    sample = commands.add_parser("demo", help="Generate clearly marked synthetic rankings and a promotion")
    sample.add_argument("--output", default="demo-output")
    return cli


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.command == "demo":
            demo(args.output)
        elif args.command == "prepare":
            from .training import prepare_data
            _emit(prepare_data(args.train, args.val, args.output))
        elif args.command == "configure":
            config = core.load_config(args.config).to_dict()
            if any(scale["baseline_loss"] is not None for scale in config["scales"]):
                raise ValueError("Configure the dataset before freezing baseline calibration")
            from .training import load_data_manifest
            manifest_path = Path(args.manifest)
            if manifest_path.name != "manifest.json":
                raise ValueError("--manifest must point to the prepared dataset's manifest.json")
            read_json(manifest_path)
            manifest = load_data_manifest(manifest_path.parent)
            config["dataset_id"] = manifest["dataset_id"]
            config["epoch_start"] = timestamp(args.epoch_start).isoformat().replace("+00:00", "Z")
            core.parse_config(config)
            _emit(config, args.output)
        elif args.command == "train":
            from .training import train_run
            config = core.load_config(args.config).to_dict()
            revision = source_revision(args.recipe, args.revision)
            _emit(train_run(config=config, recipe=read_json(args.recipe), data_dir=Path(args.data),
                            scale_id=args.scale, seed=args.seed, output_dir=Path(args.output),
                            recipe_id=args.recipe_id, revision=revision))
        elif args.command == "calibrate":
            calibrate(args)
        else:
            contest = core.load_config(args.config)
            if args.command == "init":
                _emit(core.initial_state(contest), args.output)
            else:
                results = _results(args, contest)
                state = _state(args, contest)
                core.validate_state(contest, state)
                if args.command == "validate":
                    _emit({"valid": True, "results": len(results),
                           "config_digest": core.config_digest(contest)})
                elif args.command == "leaderboard":
                    board = core.leaderboard(contest, results, state)
                    _emit(board, args.output)
                    if args.markdown:
                        path = Path(args.markdown)
                        path.parent.mkdir(parents=True, exist_ok=True)
                        path.write_text(render_markdown(board), encoding="utf-8")
                elif args.command == "advance":
                    as_of = timestamp(args.as_of) if args.as_of else datetime.now(timezone.utc)
                    _emit(core.advance(contest, results, state, as_of), args.output)
        return 0
    except (ValueError, OSError, KeyError, TypeError, RuntimeError, ImportError) as error:
        print(f"evogpt: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
