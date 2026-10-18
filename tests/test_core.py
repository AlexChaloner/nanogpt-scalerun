"""Rule-level tests: selection, evidence boundaries, and state reconstruction."""

from copy import deepcopy
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from evogpt.core import (
    ValidationError, advance, baseline_loss_at, compute_multiplier, config_digest, initial_state, leaderboard,
    load_config, load_results, parse_config, parse_result, validate_state,
)


UTC = timezone.utc
EPOCH = datetime(2026, 1, 5, tzinfo=UTC)
SHA_A = "a" * 40
SHA_B = "b" * 40


# Loss falls by sqrt(10) per 10x compute, so a multiplier is (baseline / loss) ** 2.
BASE = [10.0 * 10 ** (-i / 2) for i in range(4)]


def config(scales=4, seeds=(0, 1), top_n=2, initial=2):
    return {
        "schema_version": 1, "contest_id": "test-season", "dataset_id": "dataset-v1",
        "scales": [
            {"id": f"s{i}", "max_flops": 1000 * 10 ** i,
             "baseline_loss": BASE[i], "baseline_is_fixture": True}
            for i in range(scales)
        ],
        "initial_scale_count": initial, "top_n": top_n, "seeds": list(seeds),
        "epoch_start": "2026-01-05T00:00:00Z", "interval_days": 7,
    }


def record(contest, recipe="alpha", scale="s0", seed=0, loss=5.0, day=1, revision=SHA_A):
    raw = {
        "schema_version": 1,
        "run_id": f"{recipe}-{revision[0]}-{scale}-{seed}",
        "recipe_id": recipe, "revision": revision, "recipe_sha256": "e" * 64,
        "scale_id": scale, "seed": seed,
        "val_loss": loss, "train_flops": next(s.max_flops for s in contest.scales if s.id == scale),
        "train_tokens": 100, "dataset_id": contest.dataset_id,
        "config_digest": config_digest(contest),
        "created_at": (EPOCH + timedelta(days=day)).isoformat().replace("+00:00", "Z"),
        "artifact_sha256": "c" * 64, "log_sha256": "d" * 64,
    }
    return raw


def entry(contest, recipe, losses, day=1, revision=SHA_A):
    return [
        record(contest, recipe, f"s{i}", seed, loss, day, revision)
        for i, loss in enumerate(losses) for seed in contest.seeds
    ]


def resign(state):
    """An attacker may recompute public checksums: ranking must still verify."""
    for snapshot in state["history"]:
        snapshot["evidence_sha256"] = hashlib.sha256(json.dumps(
            snapshot["evidence"], sort_keys=True, separators=(",", ":"), allow_nan=False,
        ).encode()).hexdigest()
    state["state_digest"] = hashlib.sha256(json.dumps(
        {k: v for k, v in state.items() if k != "state_digest"},
        sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode()).hexdigest()
    return state


class ConfigTests(unittest.TestCase):
    def test_digest_normalizes_equivalent_json_and_utc(self):
        raw = config()
        other = deepcopy(raw)
        other["epoch_start"] = "2026-01-05T00:00:00+00:00"
        other["scales"][0]["baseline_loss"] = 10
        contest = parse_config(raw)
        self.assertEqual(config_digest(contest), config_digest(other))
        self.assertEqual(contest.to_dict()["epoch_start"], raw["epoch_start"])

    def test_strict_config_types_and_orders_of_magnitude(self):
        mutations = [
            lambda c: c.update(extra=True),
            lambda c: c.update(schema_version=True),
            lambda c: c.update(top_n=True),
            lambda c: c.update(initial_scale_count=5),
            lambda c: c.update(interval_days=0),
            lambda c: c.update(seeds=[True]),
            lambda c: c.update(seeds=[0, 0]),
            lambda c: c.update(seeds=[]),
            lambda c: c.update(epoch_start="2026-01-05T00:00:00"),
            lambda c: c.update(epoch_start="2026-01-05T01:00:00+01:00"),
            lambda c: c["scales"][0].update(max_flops=False),
            lambda c: c["scales"][1].update(max_flops=2000),
            lambda c: c["scales"][1].update(id="s0"),
            lambda c: c["scales"][0].update(baseline_loss=float("nan")),
            lambda c: c["scales"][0].update(baseline_loss=0),
            lambda c: c["scales"][0].update(baseline_loss=True),
            lambda c: c["scales"][0].update(baseline_is_fixture=1),
            lambda c: c["scales"][1].update(baseline_loss=c["scales"][0]["baseline_loss"]),
        ]
        for mutation in mutations:
            raw = config()
            mutation(raw)
            with self.subTest(raw=raw), self.assertRaises(ValidationError):
                parse_config(raw)

    def test_json_duplicate_keys_and_nonfinite_literals(self):
        contest = parse_config(config())
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "config.json"
            path.write_text('{"schema_version":1,"schema_version":1}')
            with self.assertRaisesRegex(ValidationError, "duplicate JSON key"):
                load_config(path)
            path.write_text('{"val_loss":NaN}')
            with self.assertRaisesRegex(ValidationError, "nonfinite JSON"):
                load_results(path, contest)
            path.write_text('{"val_loss":Infinity}')
            with self.assertRaisesRegex(ValidationError, "nonfinite JSON"):
                load_results(path, contest)


class ResultTests(unittest.TestCase):
    def setUp(self):
        self.contest = parse_config(config())

    def test_result_validation_rejects_bad_provenance_and_numbers(self):
        mutations = [
            {"schema_version": True}, {"seed": True}, {"seed": 99},
            {"train_flops": 1001}, {"train_flops": 1.0}, {"train_flops": 0},
            {"train_tokens": -1}, {"train_tokens": True}, {"val_loss": float("inf")},
            {"val_loss": float("nan")}, {"val_loss": False}, {"val_loss": -1},
            {"scale_id": "missing"}, {"dataset_id": "different"},
            {"config_digest": "0" * 64}, {"revision": "main"},
            {"recipe_sha256": "not-a-digest"},
            {"artifact_sha256": "a" * 63}, {"log_sha256": "A" * 64},
            {"created_at": "2026-01-06T00:00:00"},
            {"extra": 1},
        ]
        for mutation in mutations:
            raw = record(self.contest)
            raw.update(mutation)
            with self.subTest(mutation=mutation), self.assertRaises(ValidationError):
                parse_result(raw, self.contest)

    def test_json_array_jsonl_and_duplicate_seed_rejection(self):
        records = entry(self.contest, "alpha", [4.0, 5.0])
        with tempfile.TemporaryDirectory() as temp:
            a, b = Path(temp) / "runs.json", Path(temp) / "runs.jsonl"
            a.write_text(json.dumps(records[:2]))
            b.write_text("\n".join(json.dumps(r) for r in records[2:]) + "\n")
            self.assertEqual(len(load_results([a, b], self.contest)), 4)
            with self.assertRaisesRegex(ValidationError, "Duplicate entry/scale/seed"):
                load_results([a, a], self.contest)
            b.write_text(json.dumps(records[0] | {"run_id": "rerun", "val_loss": 0.1}))
            with self.assertRaisesRegex(ValidationError, "cherry-picked"):
                load_results([a, b], self.contest)

    def test_run_id_is_globally_unique(self):
        a, b = record(self.contest), record(self.contest, seed=1)
        b["run_id"] = a["run_id"]
        with self.assertRaisesRegex(ValidationError, "Duplicate run_id"):
            leaderboard(self.contest, [a, b])

    def test_pre_epoch_reference_validates_but_cannot_rank(self):
        raw = record(self.contest)
        raw["created_at"] = "2026-01-04T23:59:59Z"
        self.assertEqual(parse_result(raw, self.contest).created_at.day, 4)
        self.assertEqual(leaderboard(self.contest, [raw])["overall"], [])

    def test_same_entry_cannot_mix_recipe_policies_across_scales_or_seeds(self):
        for scale, seed in (("s0", 1), ("s1", 0)):
            a = record(self.contest)
            b = record(self.contest, scale=scale, seed=seed)
            b["recipe_sha256"] = "f" * 64
            with self.subTest(scale=scale, seed=seed), self.assertRaisesRegex(ValidationError, "recipe_sha256"):
                leaderboard(self.contest, [a, b])


class LeaderboardTests(unittest.TestCase):
    def setUp(self):
        self.contest = parse_config(config())

    def test_minimax_and_mean_tie_breaking(self):
        # Specialist has a better average, but a worse weakest scale.
        records = [
            *entry(self.contest, "specialist", [BASE[0] / 10, BASE[1]]),
            *entry(self.contest, "balanced", [BASE[0] / 2, BASE[1] / 2]),
            *entry(self.contest, "tie-better-mean", [BASE[0] / 4, BASE[1] / 2]),
            *entry(self.contest, "a-tie", [BASE[0] / 4, BASE[1] / 2]),
        ]
        rows = leaderboard(self.contest, records)["overall"]
        self.assertEqual([r["recipe_id"] for r in rows], ["a-tie", "tie-better-mean", "balanced", "specialist"])
        self.assertAlmostEqual(rows[0]["score"], 4.0)
        self.assertAlmostEqual(rows[0]["mean_score"], 8.0)
        self.assertAlmostEqual(rows[-1]["score"], 1.0)
        self.assertAlmostEqual(rows[-1]["mean_score"], 10.0)

    def test_multiplier_interpolates_and_extends_the_baseline_curve(self):
        contest = parse_config(config(initial=4))
        scale = {s.id: s for s in contest.scales}
        self.assertAlmostEqual(compute_multiplier(contest, scale["s0"], BASE[1]), 10.0)
        self.assertAlmostEqual(compute_multiplier(contest, scale["s1"], BASE[0]), 0.1)
        self.assertAlmostEqual(compute_multiplier(contest, scale["s1"], BASE[1]), 1.0)
        self.assertAlmostEqual(compute_multiplier(contest, scale["s3"], BASE[3] / 2), 4.0)
        self.assertAlmostEqual(compute_multiplier(contest, scale["s0"], BASE[0] * 2), 0.25)
        for flops in (100, 3000, 25_000, 10 ** 7):
            loss = baseline_loss_at(contest, flops)
            self.assertAlmostEqual(compute_multiplier(contest, scale["s1"], loss), flops / scale["s1"].max_flops)

    def test_take_median_seed_before_normalization(self):
        records = entry(self.contest, "alpha", [2.0, 3.0])
        records[1]["val_loss"] = 6.0
        board = leaderboard(self.contest, records)
        self.assertEqual(board["scales"]["s0"]["rows"][0]["median_loss"], 4.0)
        self.assertAlmostEqual(board["scales"]["s0"]["rows"][0]["score"], (10 / 4) ** 2)
        self.assertAlmostEqual(board["overall"][0]["score"], (BASE[1] / 3) ** 2)

    def test_one_diverged_seed_does_not_move_an_odd_median(self):
        contest = parse_config(config(seeds=(0, 1, 2, 3, 4)))
        records = entry(contest, "alpha", [2.0, 3.0])
        records[0]["val_loss"] = 1000.0
        board = leaderboard(contest, records)
        self.assertEqual(board["scales"]["s0"]["rows"][0]["median_loss"], 2.0)

    def test_never_combine_revisions_or_incomplete_seeds(self):
        records = [
            record(self.contest, seed=0, scale="s0", revision=SHA_A),
            record(self.contest, seed=1, scale="s0", revision=SHA_B),
            *entry(self.contest, "other", [4.0, 4.0]),
        ]
        rows = leaderboard(self.contest, records)["overall"]
        self.assertEqual(rows[0]["recipe_id"], "other")
        self.assertEqual(len(rows), 3)
        self.assertTrue(all(row["score"] is None for row in rows[1:]))
        self.assertTrue(all(row["status"] == "incomplete" for row in rows[1:]))
        self.assertEqual(rows[1]["missing"][0]["seeds"], [1])

    def test_uncalibrated_scale_has_no_overall_score(self):
        raw = config()
        raw["scales"][1]["baseline_loss"] = None
        contest = parse_config(raw)
        records = entry(contest, "alpha", [4.0, 5.0])
        board = leaderboard(contest, records)
        self.assertEqual(board["status"], "uncalibrated")
        self.assertEqual(board["overall"][0]["status"], "uncalibrated")
        self.assertIsNone(board["overall"][0]["score"])
        self.assertIsNone(board["scales"]["s1"]["rows"][0]["score"])
        with self.assertRaisesRegex(ValidationError, "calibrated"):
            advance(contest, records, initial_state(contest), EPOCH + timedelta(days=7))

    def test_locked_scales_and_deadline_results_do_not_count(self):
        records = entry(self.contest, "alpha", [4.0, 5.0, 1.0])
        records += entry(self.contest, "late", [0.1, 0.1], day=7)
        board = leaderboard(self.contest, records)
        self.assertEqual([r["recipe_id"] for r in board["overall"]], ["alpha"])
        self.assertFalse(board["scales"]["s2"]["active"])
        self.assertEqual(board["scales"]["s2"]["rows"], [])

    def test_loss_worse_than_baseline_has_multiplier_below_one(self):
        row = leaderboard(self.contest, entry(self.contest, "alpha", [BASE[0] * 2, BASE[1] * 10]))["overall"][0]
        self.assertAlmostEqual(row["score"], 0.01)
        self.assertAlmostEqual(row["mean_score"], 0.05)

    def test_zero_loss_cannot_be_scored(self):
        with self.assertRaisesRegex(ValidationError, "positive"):
            leaderboard(self.contest, entry(self.contest, "alpha", [0.0, 0.0]))


class PromotionTests(unittest.TestCase):
    def setUp(self):
        self.contest = parse_config(config(top_n=2))
        self.records = [
            *entry(self.contest, "alpha", [2.0, 3.0]),
            *entry(self.contest, "beta", [3.0, 4.0]),
            *entry(self.contest, "gamma", [5.0, 6.0]),
        ]
        self.initial = initial_state(self.contest)
        self.week1 = EPOCH + timedelta(days=7)

    def first_round(self):
        return advance(self.contest, self.records, self.initial, self.week1)

    def test_promotions_are_frozen_and_only_survivors_unlock(self):
        state = self.first_round()
        self.assertEqual(state["active_scale_count"], 3)
        self.assertEqual(state["eligible_entries"], [f"alpha@{SHA_A}", f"beta@{SHA_A}"])
        late = entry(self.contest, "late", [0.1, 0.1, 0.1], day=8)
        gamma = [record(self.contest, "gamma", "s2", seed, 0.1, day=8) for seed in self.contest.seeds]
        alpha = [record(self.contest, "alpha", "s2", seed, 9.0, day=8) for seed in self.contest.seeds]
        beta = [record(self.contest, "beta", "s2", seed, 4.0, day=8) for seed in self.contest.seeds]
        board = leaderboard(self.contest, [*self.records, *late, *gamma, *alpha, *beta], state)
        self.assertEqual([r["recipe_id"] for r in board["overall"]], ["beta", "alpha"])
        self.assertEqual(
            {r["recipe_id"] for r in board["scales"]["s0"]["rows"]},
            {"alpha", "beta", "gamma"},
        )
        self.assertEqual(
            {r["recipe_id"] for r in board["scales"]["s2"]["rows"]},
            {"alpha", "beta"},
        )
        self.assertEqual(state["history"][0]["ranking"][0], f"alpha@{SHA_A}")
        # Calling with the complete result directory is safe; frozen copies match.
        state2 = advance(self.contest, [*self.records, *alpha, *beta], state, self.week1 + timedelta(days=7))
        self.assertEqual(state2["eligible_entries"][0], f"beta@{SHA_A}")
        self.assertEqual(state["active_scale_count"], 3)  # Caller state was not mutated.
        state2["history"][0]["ranking"].reverse()
        self.assertEqual(state["history"][0]["ranking"][0], f"alpha@{SHA_A}")

    def test_late_lower_scale_rerun_cannot_replace_frozen_evidence(self):
        state = self.first_round()
        rerun = record(self.contest, "alpha", "s0", seed=0, loss=0.1, day=8)
        with self.assertRaisesRegex(ValidationError, "conflicts with frozen"):
            leaderboard(self.contest, [rerun], state)

    def test_future_scale_must_be_trained_in_its_unlocked_window(self):
        state = self.first_round()
        early = [record(self.contest, "alpha", "s2", seed, 1.0, day=6) for seed in self.contest.seeds]
        rows = leaderboard(self.contest, early, state)["overall"]
        self.assertTrue(all(r["status"] == "incomplete" for r in rows))
        at_unlock = [record(self.contest, "alpha", "s2", seed, 1.0, day=7) for seed in self.contest.seeds]
        self.assertEqual(leaderboard(self.contest, at_unlock, state)["overall"][0]["status"], "eligible")

    def test_source_revision_cannot_change_after_promotion(self):
        state = self.first_round()
        changed = [record(self.contest, "alpha", "s2", seed, 1.0, day=8, revision=SHA_B) for seed in self.contest.seeds]
        board = leaderboard(self.contest, changed, state)
        self.assertEqual(len(board["overall"]), 2)
        self.assertTrue(all(r["status"] == "incomplete" for r in board["overall"]))

    def test_recipe_policy_cannot_change_after_promotion(self):
        state = self.first_round()
        changed = record(self.contest, "alpha", "s2", seed=0, loss=1.0, day=8)
        changed["recipe_sha256"] = "f" * 64
        with self.assertRaisesRegex(ValidationError, "frozen recipe policy"):
            leaderboard(self.contest, [changed], state)
        with self.assertRaisesRegex(ValidationError, "frozen recipe policy"):
            advance(self.contest, [changed], state, EPOCH + timedelta(days=14))

    def test_clock_windows_are_exact_and_no_skipped_week(self):
        for as_of in (self.week1 - timedelta(microseconds=1), self.week1 + timedelta(days=7), self.week1.replace(tzinfo=None)):
            with self.subTest(as_of=as_of), self.assertRaises(ValidationError):
                advance(self.contest, self.records, self.initial, as_of)
        self.assertEqual(self.first_round()["round"], 1)

    def test_fewer_than_top_n_is_allowed_but_empty_round_is_not(self):
        one = entry(self.contest, "alpha", [5.0, 5.0])
        state = advance(self.contest, one, self.initial, self.week1)
        self.assertEqual(len(state["eligible_entries"]), 1)
        with self.assertRaisesRegex(ValidationError, "no entries"):
            advance(self.contest, one[:-1], self.initial, self.week1)

    def test_final_round_freezes_and_new_results_cannot_change_winner(self):
        state = self.first_round()
        s2 = [record(self.contest, "alpha", "s2", seed, 4.0, day=8) for seed in self.contest.seeds]
        state = advance(self.contest, s2, state, EPOCH + timedelta(days=14))
        s3 = [record(self.contest, "alpha", "s3", seed, 5.0, day=15) for seed in self.contest.seeds]
        final = advance(self.contest, s3, state, EPOCH + timedelta(days=21))
        self.assertTrue(final["complete"])
        self.assertEqual(final["active_scale_count"], 4)
        self.assertEqual(leaderboard(self.contest, [], final)["status"], "complete")
        extra = entry(self.contest, "new", [0.1] * 4, day=22)
        self.assertEqual(leaderboard(self.contest, [], final), leaderboard(self.contest, extra, final))
        with self.assertRaisesRegex(ValidationError, "already complete"):
            advance(self.contest, [], final, EPOCH + timedelta(days=22))

    def test_initial_all_scales_produces_one_final_freeze(self):
        contest = parse_config(config(scales=2, initial=2))
        records = entry(contest, "alpha", [4.0, 5.0])
        final = advance(contest, records, initial_state(contest), self.week1)
        self.assertTrue(final["complete"])
        self.assertIsNone(final["eligible_entries"])
        self.assertEqual(len(leaderboard(contest, [], final)["overall"]), 1)

    def test_tamper_detection_checks_rankings_even_after_resigning(self):
        original = self.first_round()
        for mutate in (
            lambda s: s.update(active_scale_count=4),
            lambda s: s.update(deadline="2026-02-02T00:00:00Z"),
            lambda s: s["history"][0]["ranking"].reverse(),
            lambda s: s["history"][0]["promoted_entries"].reverse(),
            lambda s: s["history"][0]["evidence"][0].update(scale_id="s3"),
            lambda s: s["history"][0].update(deadline="2026-01-20T00:00:00Z"),
        ):
            state = deepcopy(original)
            mutate(state)
            resign(state)
            with self.subTest(state=state), self.assertRaises(ValidationError):
                validate_state(self.contest, state)
        state = deepcopy(original)
        state["history"][0]["evidence"][0]["val_loss"] = 1.0
        with self.assertRaisesRegex(ValidationError, "checksum"):
            validate_state(self.contest, state)

    def test_config_change_invalidates_state_and_evidence(self):
        changed = config()
        changed["scales"][0]["baseline_loss"] = 9.0
        contest = parse_config(changed)
        with self.assertRaisesRegex(ValidationError, "different contest"):
            validate_state(contest, self.first_round())
        with self.assertRaisesRegex(ValidationError, "config_digest"):
            leaderboard(contest, self.records)


if __name__ == "__main__":
    unittest.main()
