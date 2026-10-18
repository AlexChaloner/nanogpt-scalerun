"""Create a review artifact, keeping scheduled Actions read-only on the repository."""

import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path

from evogpt import core
from evogpt.cli import read_json, render_markdown, result_paths, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    contest = core.load_config(args.season / "contest.json")
    state = read_json(args.season / "state.json")
    records = args.season / "results"
    results = core.load_results(result_paths([records]) if records.exists() else [], contest)
    core.validate_state(contest, state)
    board = core.leaderboard(contest, results, state)
    args.output.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc)
    if board["status"] in ("complete", "uncalibrated"):
        summary = f"No promotion: contest is {board['status']}.\n"
    elif now < datetime.fromisoformat(board["deadline"].replace("Z", "+00:00")):
        summary = f"No promotion due before {board['deadline']}.\n"
    elif now >= (datetime.fromisoformat(board["deadline"].replace("Z", "+00:00"))
                 + timedelta(days=contest.interval_days)):
        summary = ("No promotion: a scheduled boundary was missed. Recover one boundary at a time "
                   "with an explicit historical --as-of timestamp, then commit the reviewed state.\n")
    elif not any(row["status"] == "eligible" for row in board["overall"]):
        summary = "No promotion: no entries have every required seed at every active scale.\n"
    else:
        proposal = core.advance(contest, results, state, now)
        write_json(args.output / "state.json", proposal)
        write_json(args.output / "leaderboard.json", board)
        summary = "Review and commit the proposed state.json before the next weekly boundary.\n\n"
        summary += render_markdown(board)
    (args.output / "summary.md").write_text(summary, encoding="utf-8")


if __name__ == "__main__":
    main()
