"""Deterministic scoring and auditable weekly promotions for EvoGPT.

This module uses only the standard library. It never imports or executes a
participant's code. A result is a claim backed by artifact hashes; independent
reproduction is a separate organizer responsibility.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable, Mapping


class ValidationError(ValueError):
    """A configuration, result, or persisted state violates the contest rules."""


_SLUG = re.compile(r"[a-z0-9][a-z0-9_.-]{0,63}\Z")
_RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")
_SHA1 = re.compile(r"[0-9a-f]{40}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


def _keys(raw: Any, required: set[str], label: str) -> None:
    if not isinstance(raw, dict):
        raise ValidationError(f"{label} must be a JSON object")
    missing, extra = required - raw.keys(), raw.keys() - required
    if missing or extra:
        raise ValidationError(
            f"{label} fields: missing={sorted(missing)}, unknown={sorted(extra)}"
        )


def _integer(value: Any, label: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValidationError(f"{label} must be an integer >= {minimum}")
    return value


def _number(value: Any, label: str, allow_zero: bool = False) -> float:
    description = "nonnegative" if allow_zero else "positive"
    if type(value) not in (int, float):
        raise ValidationError(f"{label} must be a finite {description} number")
    try:
        value = float(value)
    except (ValueError, OverflowError) as exc:
        raise ValidationError(f"{label} must be a finite {description} number") from exc
    if not math.isfinite(value) or value < 0 or (value == 0 and not allow_zero):
        raise ValidationError(f"{label} must be a finite {description} number")
    return value


def _text(value: Any, label: str, pattern: re.Pattern[str] | None = None) -> str:
    if not isinstance(value, str) or not value or len(value) > 512:
        raise ValidationError(f"{label} must be a nonempty string of at most 512 characters")
    if any(ord(c) < 32 or ord(c) == 127 for c in value):
        raise ValidationError(f"{label} may not contain control characters")
    if pattern is not None and pattern.fullmatch(value) is None:
        raise ValidationError(f"{label} has an invalid format")
    return value


def _timestamp(value: Any, label: str) -> datetime:
    if not isinstance(value, str):
        raise ValidationError(f"{label} must be an ISO 8601 UTC timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValidationError(f"{label} must be an ISO 8601 UTC timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
        raise ValidationError(f"{label} must explicitly use UTC (Z or +00:00)")
    return parsed.astimezone(timezone.utc)


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _canonical(raw: Any) -> bytes:
    try:
        return json.dumps(raw, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValidationError("Data must contain only finite JSON values") from exc


def _digest(raw: Any) -> str:
    return hashlib.sha256(_canonical(raw)).hexdigest()


def _json_loads(text: str, label: str) -> Any:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise ValidationError(f"{label}: duplicate JSON key {key!r}")
            result[key] = value
        return result

    def constant(value: str) -> None:
        raise ValidationError(f"{label}: nonfinite JSON number {value}")

    try:
        return json.loads(text, object_pairs_hook=pairs, parse_constant=constant)
    except (json.JSONDecodeError, UnicodeError) as exc:
        raise ValidationError(f"{label}: invalid JSON: {exc}") from exc


@dataclass(frozen=True)
class Scale:
    id: str
    max_flops: int
    baseline_loss: float | None
    baseline_is_fixture: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "max_flops": self.max_flops,
            "baseline_loss": self.baseline_loss,
            "baseline_is_fixture": self.baseline_is_fixture,
        }


@dataclass(frozen=True)
class Contest:
    contest_id: str
    dataset_id: str
    scales: tuple[Scale, ...]
    initial_scale_count: int
    top_n: int
    seeds: tuple[int, ...]
    epoch_start: datetime
    interval_days: int
    schema_version: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "contest_id": self.contest_id,
            "dataset_id": self.dataset_id,
            "scales": [scale.to_dict() for scale in self.scales],
            "initial_scale_count": self.initial_scale_count,
            "top_n": self.top_n,
            "seeds": list(self.seeds),
            "epoch_start": _iso(self.epoch_start),
            "interval_days": self.interval_days,
        }


def parse_config(raw: Mapping[str, Any]) -> Contest:
    _keys(raw, {"schema_version", "contest_id", "dataset_id", "scales", "initial_scale_count", "top_n", "seeds", "epoch_start", "interval_days"}, "config")
    if type(raw["schema_version"]) is not int or raw["schema_version"] != 1:
        raise ValidationError("config.schema_version must be 1")
    if not isinstance(raw["scales"], list) or len(raw["scales"]) < 2:
        raise ValidationError("config.scales must contain at least two scales")
    scales: list[Scale] = []
    for index, item in enumerate(raw["scales"]):
        _keys(item, {"id", "max_flops", "baseline_loss", "baseline_is_fixture"}, f"scales[{index}]")
        if type(item["baseline_is_fixture"]) is not bool:
            raise ValidationError("baseline_is_fixture must be a boolean")
        scales.append(Scale(
            _text(item["id"], "scale.id", _SLUG),
            _integer(item["max_flops"], "scale.max_flops", 1),
            None if item["baseline_loss"] is None else _number(item["baseline_loss"], "scale.baseline_loss"),
            item["baseline_is_fixture"],
        ))
    if len({scale.id for scale in scales}) != len(scales):
        raise ValidationError("Scale ids must be unique")
    if any(b.max_flops != a.max_flops * 10 for a, b in zip(scales, scales[1:])):
        raise ValidationError("Each successive scale must have exactly 10 times the FLOPs budget")
    calibrated = [scale.baseline_loss for scale in scales if scale.baseline_loss is not None]
    if any(later >= earlier for earlier, later in zip(calibrated, calibrated[1:])):
        raise ValidationError("Baseline losses must strictly decrease as compute grows")
    initial = _integer(raw["initial_scale_count"], "initial_scale_count", 1)
    if initial > len(scales):
        raise ValidationError("initial_scale_count cannot exceed the number of scales")
    if not isinstance(raw["seeds"], list) or not raw["seeds"]:
        raise ValidationError("seeds must be a nonempty list")
    seeds = tuple(_integer(seed, "seed") for seed in raw["seeds"])
    if len(set(seeds)) != len(seeds):
        raise ValidationError("seeds must be unique")
    epoch = _timestamp(raw["epoch_start"], "epoch_start")
    interval = _integer(raw["interval_days"], "interval_days", 1)
    try:
        epoch + timedelta(days=interval * len(scales))
    except OverflowError as exc:
        raise ValidationError("Contest schedule exceeds the supported datetime range") from exc
    return Contest(
        _text(raw["contest_id"], "contest_id", _SLUG),
        _text(raw["dataset_id"], "dataset_id"),
        tuple(scales), initial, _integer(raw["top_n"], "top_n", 1),
        seeds, epoch, interval,
    )


def load_config(path: str | Path) -> Contest:
    path = Path(path)
    return parse_config(_json_loads(path.read_text(encoding="utf-8"), str(path)))


def config_digest(config: Contest | Mapping[str, Any]) -> str:
    """Hash the validated, normalized config (timestamps UTC, losses floats)."""
    contest = config if isinstance(config, Contest) else parse_config(config)
    return _digest(contest.to_dict())


@dataclass(frozen=True)
class Result:
    run_id: str
    recipe_id: str
    revision: str
    recipe_sha256: str
    scale_id: str
    seed: int
    val_loss: float
    train_flops: int
    train_tokens: int
    dataset_id: str
    config_digest: str
    created_at: datetime
    artifact_sha256: str
    log_sha256: str
    schema_version: int = 1

    @property
    def entry_id(self) -> str:
        return f"{self.recipe_id}@{self.revision}"

    @property
    def key(self) -> tuple[str, str, int]:
        return (self.entry_id, self.scale_id, self.seed)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "recipe_id": self.recipe_id,
            "revision": self.revision,
            "recipe_sha256": self.recipe_sha256,
            "scale_id": self.scale_id,
            "seed": self.seed,
            "val_loss": self.val_loss,
            "train_flops": self.train_flops,
            "train_tokens": self.train_tokens,
            "dataset_id": self.dataset_id,
            "config_digest": self.config_digest,
            "created_at": _iso(self.created_at),
            "artifact_sha256": self.artifact_sha256,
            "log_sha256": self.log_sha256,
        }


def parse_result(raw: Mapping[str, Any], contest: Contest) -> Result:
    _keys(raw, {"schema_version", "run_id", "recipe_id", "revision", "recipe_sha256", "scale_id", "seed", "val_loss", "train_flops", "train_tokens", "dataset_id", "config_digest", "created_at", "artifact_sha256", "log_sha256"}, "result")
    if type(raw["schema_version"]) is not int or raw["schema_version"] != 1:
        raise ValidationError("result.schema_version must be 1")
    scale_id = _text(raw["scale_id"], "scale_id", _SLUG)
    scale = next((scale for scale in contest.scales if scale.id == scale_id), None)
    if scale is None:
        raise ValidationError(f"Unknown scale {scale_id!r}")
    seed = _integer(raw["seed"], "seed")
    if seed not in contest.seeds:
        raise ValidationError(f"Seed {seed} is not required by this contest")
    flops = _integer(raw["train_flops"], "train_flops", 1)
    if flops > scale.max_flops:
        raise ValidationError(f"Run exceeds scale {scale_id!r} FLOPs budget")
    if raw["dataset_id"] != contest.dataset_id:
        raise ValidationError("Result dataset_id does not match the contest")
    if raw["config_digest"] != config_digest(contest):
        raise ValidationError("Result config_digest does not match the contest")
    created_at = _timestamp(raw["created_at"], "created_at")
    return Result(
        _text(raw["run_id"], "run_id", _RUN_ID),
        _text(raw["recipe_id"], "recipe_id", _SLUG),
        _text(raw["revision"], "revision", _SHA1),
        _text(raw["recipe_sha256"], "recipe_sha256", _SHA256),
        scale_id, seed, _number(raw["val_loss"], "val_loss", allow_zero=True), flops,
        _integer(raw["train_tokens"], "train_tokens", 1),
        contest.dataset_id, config_digest(contest), created_at,
        _text(raw["artifact_sha256"], "artifact_sha256", _SHA256),
        _text(raw["log_sha256"], "log_sha256", _SHA256),
    )


def _results(results: Iterable[Result | Mapping[str, Any]], contest: Contest) -> list[Result]:
    parsed: list[Result] = []
    keys: set[tuple[str, str, int]] = set()
    run_ids: set[str] = set()
    recipe_digests: dict[str, str] = {}
    for raw in results:
        result = parse_result(raw.to_dict() if isinstance(raw, Result) else raw, contest)
        if recipe_digests.setdefault(result.entry_id, result.recipe_sha256) != result.recipe_sha256:
            raise ValidationError(f"Conflicting recipe_sha256 for entry {result.entry_id}; recipe policy must remain immutable")
        if result.key in keys:
            raise ValidationError(f"Duplicate entry/scale/seed: {result.key}; reruns cannot be cherry-picked")
        if result.run_id in run_ids:
            raise ValidationError(f"Duplicate run_id: {result.run_id}")
        keys.add(result.key)
        run_ids.add(result.run_id)
        parsed.append(result)
    return parsed


def load_results(paths: Iterable[str | Path] | str | Path, contest: Contest) -> list[Result]:
    """Read result objects/arrays in JSON files, or objects in .jsonl files."""
    if isinstance(paths, (str, Path)):
        paths = [paths]
    records: list[Any] = []
    for item in paths:
        path = Path(item)
        contents = path.read_text(encoding="utf-8")
        if path.suffix == ".jsonl":
            for line_no, line in enumerate(contents.splitlines(), 1):
                if line.strip():
                    records.append(_json_loads(line, f"{path}:{line_no}"))
        else:
            raw = _json_loads(contents, str(path))
            records.extend(raw if isinstance(raw, list) else [raw])
    return _results(records, contest)


def _mean(values: list[float]) -> float:
    return math.fsum(value / len(values) for value in values)


def median(values: Iterable[float]) -> float:
    """Median over seeds, so one diverged or lucky seed cannot move a score."""
    ordered = sorted(values)
    middle = len(ordered) // 2
    return ordered[middle] if len(ordered) % 2 else (ordered[middle - 1] + ordered[middle]) / 2


def _baseline_curve(contest: Contest) -> list[tuple[float, float]]:
    """(log FLOPs, log loss) at each calibrated scale."""
    points = [(math.log(s.max_flops), math.log(s.baseline_loss)) for s in contest.scales if s.baseline_loss is not None]
    if len(points) < 2:
        raise ValidationError("Compute multipliers need at least two calibrated baseline losses")
    return points


def baseline_loss_at(contest: Contest, flops: float) -> float:
    """The baseline curve's loss at any compute; the inverse of compute_multiplier."""
    points = _baseline_curve(contest)
    segments = list(zip(points, points[1:]))
    x = math.log(flops)
    (x0, y0), (x1, y1) = next(((a, b) for a, b in segments if x <= b[0]), segments[-1])
    return math.exp(y0 + (x - x0) * (y1 - y0) / (x1 - x0))


def compute_multiplier(contest: Contest, scale: Scale, loss: float) -> float:
    """Baseline compute needed to reach loss, as a multiple of scale's budget.

    The baseline curve is piecewise linear in log loss against log compute
    through the calibrated scales, extended past the ends along the end
    segments. 2.0 means the baseline needs twice this scale's FLOPs to match.
    """
    points = _baseline_curve(contest)
    if not loss > 0:
        raise ValidationError("Compute multiplier is not finite; validation loss must be positive")
    target = math.log(loss)
    segments = list(zip(points, points[1:]))
    (x0, y0), (x1, y1) = next(((a, b) for a, b in segments if target >= b[1]), segments[-1])
    log_multiplier = x0 + (target - y0) * (x1 - x0) / (y1 - y0) - math.log(scale.max_flops)
    try:
        return math.exp(log_multiplier)
    except OverflowError as exc:
        raise ValidationError("Compute multiplier is not finite; check the loss and baseline") from exc


def _geometric_mean(values: list[float]) -> float:
    return math.exp(_mean([math.log(value) for value in values]))


def _score(contest: Contest, evidence: Iterable[Result], active_count: int,
           allowed: list[str] | None) -> dict[str, Any]:
    evidence = list(evidence)
    active = contest.scales[:active_count]
    allowed_set = None if allowed is None else set(allowed)
    by_key = {result.key: result for result in evidence}
    all_entries = {r.entry_id for r in evidence}
    entries = sorted(entry for entry in all_entries if allowed_set is None or entry in allowed_set)
    if allowed is not None:
        entries = sorted(set(entries) | set(allowed))
    all_entries = sorted(all_entries | set(entries))
    uncalibrated = any(scale.baseline_loss is None for scale in active)
    scales: dict[str, Any] = {}
    entry_scores: dict[str, list[float]] = {entry: [] for entry in entries}
    completed: dict[str, int] = {entry: 0 for entry in entries}
    missing: dict[str, list[dict[str, Any]]] = {entry: [] for entry in entries}
    for index, scale in enumerate(contest.scales):
        rows: list[dict[str, Any]] = []
        if index < active_count:
            for entry in all_entries:
                present = [by_key[(entry, scale.id, seed)] for seed in contest.seeds if (entry, scale.id, seed) in by_key]
                if allowed_set is not None and entry not in allowed_set and not present:
                    continue
                absent = [seed for seed in contest.seeds if (entry, scale.id, seed) not in by_key]
                complete = not absent
                median_loss = median(r.val_loss for r in present) if present else None
                score = None
                if complete:
                    if entry in completed:
                        completed[entry] += 1
                    if scale.baseline_loss is not None:
                        score = compute_multiplier(contest, scale, median_loss)
                        if not 0 < score < math.inf:
                            raise ValidationError("Compute multiplier is not finite; check the loss and baseline")
                        if entry in entry_scores:
                            entry_scores[entry].append(score)
                else:
                    if entry in missing:
                        missing[entry].append({"scale_id": scale.id, "seeds": absent})
                recipe_id, revision = entry.split("@")
                rows.append({
                    "entry_id": entry, "recipe_id": recipe_id, "revision": revision,
                    "median_loss": median_loss, "score": score, "seeds": [r.seed for r in present],
                    "complete": complete,
                })
            rows.sort(key=lambda row: (row["score"] is None, -(row["score"] or 0.0), row["entry_id"]))
        scales[scale.id] = {
            "max_flops": scale.max_flops, "baseline_loss": scale.baseline_loss,
            "baseline_is_fixture": scale.baseline_is_fixture,
            "active": index < active_count, "rows": rows,
        }
    overall: list[dict[str, Any]] = []
    for entry in entries:
        complete = completed[entry] == active_count
        scores = entry_scores[entry]
        recipe_id, revision = entry.split("@")
        overall.append({
            "entry_id": entry, "recipe_id": recipe_id, "revision": revision,
            "status": "incomplete" if not complete else "uncalibrated" if uncalibrated else "eligible",
            "score": min(scores) if complete and not uncalibrated else None,
            "mean_score": _geometric_mean(scores) if complete and not uncalibrated else None,
            "completed_scales": completed[entry], "required_scales": active_count,
            "missing": missing[entry],
        })
    overall.sort(key=lambda row: (row["status"] != "eligible", -(row["score"] or 0.0), -(row["mean_score"] or 0.0), row["entry_id"]))
    return {"status": "uncalibrated" if uncalibrated else "active", "overall": overall, "scales": scales}


_STATE_KEYS = {"schema_version", "contest_id", "config_digest", "round", "active_scale_count", "eligible_entries", "round_started_at", "deadline", "complete", "history", "state_digest"}
_HISTORY_KEYS = {"round", "active_scale_count", "eligible_entries", "round_started_at", "deadline", "evidence", "evidence_sha256", "ranking", "promoted_entries"}


def _schedule(contest: Contest, round_no: int) -> tuple[datetime, datetime]:
    start = contest.epoch_start + timedelta(days=contest.interval_days * round_no)
    return start, start + timedelta(days=contest.interval_days)


def _state(contest: Contest, history: list[dict[str, Any]]) -> dict[str, Any]:
    total_rounds = len(contest.scales) - contest.initial_scale_count + 1
    complete = len(history) == total_rounds
    round_no = len(history) - 1 if complete else len(history)
    start, deadline = _schedule(contest, round_no)
    allowed = None if round_no == 0 else history[round_no - 1]["promoted_entries"]
    raw = {
        "schema_version": 1, "contest_id": contest.contest_id,
        "config_digest": config_digest(contest), "round": round_no,
        "active_scale_count": min(contest.initial_scale_count + round_no, len(contest.scales)),
        "eligible_entries": allowed, "round_started_at": _iso(start),
        "deadline": _iso(deadline), "complete": complete, "history": history,
    }
    raw["state_digest"] = _digest(raw)
    return raw


def initial_state(contest: Contest) -> dict[str, Any]:
    return _state(contest, [])


def _new_scale_ids(contest: Contest, round_no: int) -> set[str]:
    if round_no == 0:
        return {scale.id for scale in contest.scales[:contest.initial_scale_count]}
    return {contest.scales[contest.initial_scale_count + round_no - 1].id}


def _frozen(contest: Contest, history: list[dict[str, Any]]) -> list[Result]:
    return [parse_result(raw, contest) for snapshot in history for raw in snapshot["evidence"]]


def validate_state(contest: Contest, state: Mapping[str, Any]) -> None:
    """Recompute every snapshot's ranking and promotion before trusting a state.

    Digests detect accidental modification; they are not signatures. Organizers
    must retain state files in trusted, version-controlled storage.
    """
    _keys(state, _STATE_KEYS, "state")
    if not isinstance(state["history"], list):
        raise ValidationError("state.history must be a list")
    if type(state["schema_version"]) is not int or state["schema_version"] != 1:
        raise ValidationError("state.schema_version must be 1")
    if type(state["round"]) is not int or type(state["active_scale_count"]) is not int or type(state["complete"]) is not bool:
        raise ValidationError("State round/count/complete have invalid types")
    if state["contest_id"] != contest.contest_id or state["config_digest"] != config_digest(contest):
        raise ValidationError("State belongs to a different contest configuration")
    checksum = {key: value for key, value in state.items() if key != "state_digest"}
    if state["state_digest"] != _digest(checksum):
        raise ValidationError("State integrity checksum does not match")
    history = state["history"]
    total_rounds = len(contest.scales) - contest.initial_scale_count + 1
    if len(history) > total_rounds:
        raise ValidationError("State has more rounds than scales permit")
    frozen: list[Result] = []
    allowed: list[str] | None = None
    for index, snapshot in enumerate(history):
        _keys(snapshot, _HISTORY_KEYS, f"history[{index}]")
        start, deadline = _schedule(contest, index)
        metadata = {
            "round": index, "active_scale_count": contest.initial_scale_count + index,
            "eligible_entries": allowed, "round_started_at": _iso(start), "deadline": _iso(deadline),
        }
        if type(snapshot["round"]) is not int or type(snapshot["active_scale_count"]) is not int:
            raise ValidationError("History round/count must be integers")
        if any(snapshot[key] != value for key, value in metadata.items()):
            raise ValidationError(f"History {index} has altered schedule, roster, or scale metadata")
        if not isinstance(snapshot["evidence"], list):
            raise ValidationError("Snapshot evidence must be a list")
        evidence = _results(snapshot["evidence"], contest)
        if snapshot["evidence"] != [result.to_dict() for result in sorted(evidence, key=lambda result: result.key)]:
            raise ValidationError("Snapshot evidence must be normalized and sorted")
        if snapshot["evidence_sha256"] != _digest(snapshot["evidence"]):
            raise ValidationError("Snapshot evidence checksum does not match")
        open_ids = _new_scale_ids(contest, index)
        for result in evidence:
            if result.scale_id not in open_ids or not start <= result.created_at < deadline:
                raise ValidationError("Snapshot contains results from outside its unlocked scale/window")
            if allowed is not None and result.entry_id not in allowed:
                raise ValidationError("Snapshot contains an entry outside the frozen survivor roster")
        frozen = _results([*frozen, *evidence], contest)
        board = _score(contest, frozen, contest.initial_scale_count + index, allowed)
        ranking = [row["entry_id"] for row in board["overall"] if row["status"] == "eligible"]
        if not ranking:
            raise ValidationError("A frozen round must contain at least one complete, calibrated entry")
        promoted = ranking[:contest.top_n] if index < total_rounds - 1 else []
        if snapshot["ranking"] != ranking or snapshot["promoted_entries"] != promoted:
            raise ValidationError("Snapshot ranking or promotion does not match its evidence")
        allowed = promoted
    if dict(state) != _state(contest, history):
        raise ValidationError("State metadata does not match its verified history")


def _current_evidence(contest: Contest, results: Iterable[Result | Mapping[str, Any]],
                      state: Mapping[str, Any]) -> tuple[list[Result], list[Result]]:
    supplied = _results(results, contest)
    frozen = _frozen(contest, state["history"])
    by_key = {result.key: result for result in frozen}
    by_run = {result.run_id: result for result in frozen}
    recipe_digests = {result.entry_id: result.recipe_sha256 for result in frozen}
    for result in supplied:
        if recipe_digests.setdefault(result.entry_id, result.recipe_sha256) != result.recipe_sha256:
            raise ValidationError(f"recipe_sha256 conflicts with frozen recipe policy: {result.entry_id}")
        if result.key in by_key and result.to_dict() != by_key[result.key].to_dict():
            raise ValidationError(f"Result conflicts with frozen evidence: {result.key}")
        if result.run_id in by_run and result.to_dict() != by_run[result.run_id].to_dict():
            raise ValidationError(f"run_id conflicts with frozen evidence: {result.run_id}")
    if state["complete"]:
        return frozen, []
    start = _timestamp(state["round_started_at"], "round_started_at")
    deadline = _timestamp(state["deadline"], "deadline")
    open_ids = _new_scale_ids(contest, state["round"])
    allowed = state["eligible_entries"]
    current = [
        result for result in supplied
        if result.key not in by_key and result.scale_id in open_ids
        and start <= result.created_at < deadline
        and (allowed is None or result.entry_id in allowed)
    ]
    return [*frozen, *current], sorted(current, key=lambda result: result.key)


def leaderboard(contest: Contest, results: Iterable[Result | Mapping[str, Any]],
                state: Mapping[str, Any] | None = None) -> dict[str, Any]:
    state = initial_state(contest) if state is None else state
    validate_state(contest, state)
    evidence, _ = _current_evidence(contest, results, state)
    board = _score(contest, evidence, state["active_scale_count"], state["eligible_entries"])
    board.update({
        "contest_id": contest.contest_id, "config_digest": config_digest(contest),
        "top_n": contest.top_n, "round": state["round"], "deadline": state["deadline"],
        "active_scale_ids": [scale.id for scale in contest.scales[:state["active_scale_count"]]],
        "eligible_entries": state["eligible_entries"],
    })
    if state["complete"]:
        board["status"] = "complete"
    return board


def advance(contest: Contest, results: Iterable[Result | Mapping[str, Any]],
            state: Mapping[str, Any], as_of: datetime) -> dict[str, Any]:
    """Freeze one scheduled deadline and unlock one scale for its survivors.

    The clock must be inside the week immediately following this deadline.
    Use an explicit historical as_of to recover missed weeks deliberately.
    """
    validate_state(contest, state)
    if state["complete"]:
        raise ValidationError("Contest is already complete")
    if not isinstance(as_of, datetime) or as_of.tzinfo is None or as_of.utcoffset() != timedelta(0):
        raise ValidationError("as_of must be an aware UTC datetime")
    deadline = _timestamp(state["deadline"], "deadline")
    if as_of < deadline:
        raise ValidationError(f"Cannot advance before scheduled deadline {_iso(deadline)}")
    if as_of >= deadline + timedelta(days=contest.interval_days):
        raise ValidationError("as_of skips a scheduled week; advance each boundary explicitly")
    evidence, current = _current_evidence(contest, results, state)
    board = _score(contest, evidence, state["active_scale_count"], state["eligible_entries"])
    ranking = [row["entry_id"] for row in board["overall"] if row["status"] == "eligible"]
    if board["status"] == "uncalibrated":
        raise ValidationError("Cannot advance: active scales require calibrated baseline losses")
    if not ranking:
        raise ValidationError("Cannot advance: no entries have every required seed at every active scale")
    final = state["active_scale_count"] == len(contest.scales)
    raw_evidence = [result.to_dict() for result in current]
    snapshot = {
        "round": state["round"], "active_scale_count": state["active_scale_count"],
        "eligible_entries": state["eligible_entries"],
        "round_started_at": state["round_started_at"], "deadline": state["deadline"],
        "evidence": raw_evidence, "evidence_sha256": _digest(raw_evidence),
        "ranking": ranking, "promoted_entries": [] if final else ranking[:contest.top_n],
    }
    # JSON roundtrip detaches all nested containers from the caller's state.
    history = json.loads(_canonical([*state["history"], snapshot]))
    result = _state(contest, history)
    validate_state(contest, result)
    return result
