"""Early-life absolute limit proposal. Synthetic frames only."""

from __future__ import annotations

import ast
import inspect
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pdm.zone_limit_proposal import (
    EARLY_FRACTION,
    EARLY_MIN_POINTS,
    MIN_EARLY_VALUES,
    _pair_from_pool,
    propose_absolute_limits,
)

_TOL = 1e-12
_WIDE_HEAD = [-100.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 100.0]
_REASON = (
    "Need at least 8 finite Training Data values in the early-life window. "
    "This snapshot has {n}."
)


def _approx(value: float):
    return pytest.approx(value, abs=_TOL, rel=0)


def _unit(unit_id: str, signals, *, gap_at: set[int] | None = None) -> pd.DataFrame:
    values = [float(v) if v is not None else np.nan for v in signals]
    gaps = gap_at or set()
    n = len(values)
    return pd.DataFrame(
        {
            "unit_id": unit_id,
            "timestamp_s": np.arange(n, dtype=np.float64),
            "signal": np.asarray(values, dtype=np.float64),
            "gap_before": [i in gaps for i in range(n)],
        }
    )


def _wide(n: int = 36, tail: float = 1000.0) -> list[float]:
    return _WIDE_HEAD + [tail] * (n - len(_WIDE_HEAD))


def _reason(n: int) -> str:
    return _REASON.format(n=n)


def _assert_refused(result: dict, direction: str, n: int) -> None:
    assert result == {
        "ok": False,
        "direction": direction,
        "n": n,
        "reason": _reason(n),
    }
    assert "yellow" not in result
    assert "red" not in result


def _assert_ok(result: dict, direction: str, pool) -> None:
    pool = np.asarray(pool, dtype=np.float64)
    assert result["ok"] is True
    assert result["direction"] == direction
    assert result["n"] == int(pool.size)
    assert type(result["n"]) is int
    assert result["early_fraction"] == 0.20
    assert result["early_min_points"] == 5
    yellow, red = result["yellow"], result["red"]
    assert type(yellow) is float
    assert type(red) is float
    assert math.isfinite(yellow) and math.isfinite(red)
    median = float(np.median(pool))
    if direction == "above":
        assert yellow < red
        assert yellow >= median
    else:
        assert yellow > red
        assert yellow <= median
    assert (yellow, red) != (300.0, 600.0)
    assert (yellow, red) != (1.0, 2.0)
    expect = _pair_from_pool(pool, direction)
    assert yellow == expect[0]
    assert red == expect[1]


def test_worked_example_above_and_below_ignore_failure_peak():
    # Lone 20-row unit: k_i = 5, pool [0, 1, 2, 3, 4], n = 5.
    # The count gate refuses before emitting the pairs. The pairs below are
    # that pool's quantile rule, which the public function applies once n >= 8.
    pool = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    above = _pair_from_pool(pool, "above")
    below = _pair_from_pool(pool, "below")
    assert above[0] == _approx(3.6)
    assert above[1] == _approx(4.3413)
    assert below[0] == _approx(0.4)
    assert below[1] == _approx(-0.3413)
    assert above[0] < 1000 and above[1] < 1000

    edited_pool = np.array([0.0, 1.0, 2.0, 3.0, -10.0])
    assert _pair_from_pool(edited_pool, "above") != above
    assert _pair_from_pool(edited_pool, "below") != below

    ramp = [float(i) if i < 5 else 1000.0 for i in range(20)]
    lone = _unit("u", ramp)
    _assert_refused(propose_absolute_limits(lone, ["u"], "above"), "above", 5)
    _assert_refused(propose_absolute_limits(lone, ["u"], "below"), "below", 5)
    peaked = lone.copy()
    peaked.loc[peaked["timestamp_s"] == 19, "signal"] = 1e6
    assert propose_absolute_limits(peaked, ["u"], "above") == propose_absolute_limits(
        lone, ["u"], "above"
    )

    # Same ramp plus eight early zeros so the public function returns pairs.
    pads = [_unit(f"p{i}", [0.0]) for i in range(8)]
    val = _unit("val", [1e9] * 36)
    test = _unit("test", [-1e9] * 36)
    frame = pd.concat([*pads, lone, val, test], ignore_index=True)
    train = [f"p{i}" for i in range(8)] + ["u"]
    public_pool = [0.0] * 8 + [0.0, 1.0, 2.0, 3.0, 4.0]
    base_above = propose_absolute_limits(frame, train, "above")
    base_below = propose_absolute_limits(frame, train, "below")
    _assert_ok(base_above, "above", public_pool)
    _assert_ok(base_below, "below", public_pool)
    assert base_above["yellow"] < 100 and base_above["red"] < 100

    tail = frame.copy()
    tail.loc[(tail["unit_id"] == "u") & (tail["timestamp_s"] == 19), "signal"] = 1e6
    tail_above = propose_absolute_limits(tail, train, "above")
    tail_below = propose_absolute_limits(tail, train, "below")
    assert tail_above["yellow"] == base_above["yellow"]
    assert tail_above["red"] == base_above["red"]
    assert tail_below["yellow"] == base_below["yellow"]
    assert tail_below["red"] == base_below["red"]

    prefix = frame.copy()
    prefix.loc[(prefix["unit_id"] == "u") & (prefix["timestamp_s"] == 4), "signal"] = -10.0
    prefix_above = propose_absolute_limits(prefix, train, "above")
    prefix_below = propose_absolute_limits(prefix, train, "below")
    assert (prefix_above["yellow"], prefix_above["red"]) != (
        base_above["yellow"],
        base_above["red"],
    )
    assert (prefix_below["yellow"], prefix_below["red"]) != (
        base_below["yellow"],
        base_below["red"],
    )

    leaked = frame.copy()
    mask = leaked["unit_id"].isin(["val", "test"])
    leaked.loc[mask, "signal"] = 1e9
    leaked.loc[mask, "timestamp_s"] = -1e12
    leaked.loc[mask, "rul"] = -1e9
    leaked.loc[mask, "event"] = 1
    leaked.loc[mask, "official_rul"] = 1e9
    leaked_above = propose_absolute_limits(leaked, train, "above")
    leaked_below = propose_absolute_limits(leaked, train, "below")
    assert leaked_above["yellow"] == base_above["yellow"]
    assert leaked_above["red"] == base_above["red"]
    assert leaked_below["yellow"] == base_below["yellow"]
    assert leaked_below["red"] == base_below["red"]


def test_constant_pool_minimum_separation():
    # 36 rows → k_i = 8. Tail is outside the prefix, so it must not move the pair.
    frame = _unit("u", [3.0] * 8 + [1000.0] * 28).drop(columns=["gap_before"])
    pool = [3.0] * 8
    above = propose_absolute_limits(frame, ["u"], "above")
    below = propose_absolute_limits(frame, ["u"], "below")
    _assert_ok(above, "above", pool)
    _assert_ok(below, "below", pool)
    assert above["yellow"] == _approx(3.0)
    assert above["red"] == _approx(3.0003)
    assert below["yellow"] == _approx(3.0)
    assert below["red"] == _approx(2.9997)
    q99 = float(np.quantile(np.array(pool), 0.99, method="linear"))
    assert above["red"] != q99
    assert above["red"] == _approx(above["yellow"] + 0.0003)
    twice = propose_absolute_limits(frame, ["u", "u"], "above")
    assert twice["n"] == above["n"] == 8
    assert twice["red"] == above["red"]


def test_wide_gap_uses_quantile_not_stacked_separation():
    frame = _unit("u", _wide(36, tail=1000.0))
    other_tail = _unit("u", _wide(36, tail=-1e6))
    still_eight = _unit("u", _wide(40, tail=1000.0))
    pool = list(_WIDE_HEAD)
    above = propose_absolute_limits(frame, ["u"], "above")
    below = propose_absolute_limits(frame, ["u"], "below")
    _assert_ok(above, "above", pool)
    _assert_ok(below, "below", pool)
    assert above["yellow"] == _approx(30)
    assert above["red"] == _approx(93)
    assert below["yellow"] == _approx(-30)
    assert below["red"] == _approx(-93)
    q99 = float(np.quantile(np.array(pool), 0.99, method="linear"))
    q01 = float(np.quantile(np.array(pool), 0.01, method="linear"))
    assert above["red"] == q99
    assert below["red"] == q01
    assert above["red"] > above["yellow"] + 1e-4
    assert below["red"] < below["yellow"] - 1e-4

    for other in (other_tail, still_eight):
        got_above = propose_absolute_limits(other, ["u"], "above")
        got_below = propose_absolute_limits(other, ["u"], "below")
        assert got_above["yellow"] == above["yellow"]
        assert got_above["red"] == above["red"]
        assert got_below["yellow"] == below["yellow"]
        assert got_below["red"] == below["red"]

    # Past n_i = 40, k_i becomes 9 and the former tail enters the pool.
    grown = _unit("u", _wide(41, tail=1000.0))
    grown_above = propose_absolute_limits(grown, ["u"], "above")
    assert grown_above["n"] == 9
    assert grown_above["red"] != above["red"]

    order = np.random.default_rng(0).permutation(len(frame))
    shuffled = frame.iloc[order].reset_index(drop=True)
    shuffled_above = propose_absolute_limits(shuffled, ["u"], "above")
    assert shuffled_above["yellow"] == above["yellow"]
    assert shuffled_above["red"] == above["red"]

    # An 8-row unit has k_i = 5, so this head must not be used as the lock.
    short = _unit("u", list(_WIDE_HEAD))
    _assert_refused(propose_absolute_limits(short, ["u"], "above"), "above", 5)


def test_val_and_test_mutations_do_not_change_proposal():
    train = _unit("train", _wide(36))
    validation = _unit("validation", [1e6] * 40)
    testing = _unit("testing", [-1e6] * 40)
    frame = pd.concat([validation, testing, train], ignore_index=True)
    frame["rul"] = 1.0
    frame["event"] = 0
    frame["official_rul"] = 2.0
    original = frame.copy(deep=True)
    above = propose_absolute_limits(frame, ["train"], "above")
    below = propose_absolute_limits(frame, ["train"], "below")
    pd.testing.assert_frame_equal(frame, original)
    _assert_ok(above, "above", _WIDE_HEAD)
    _assert_ok(below, "below", _WIDE_HEAD)

    mutated = frame.copy()
    mask = mutated["unit_id"].isin(["validation", "testing"])
    mutated.loc[mask, "signal"] = np.inf
    mutated.loc[mask, "timestamp_s"] = -np.inf
    mutated.loc[mask, "rul"] = np.nan
    mutated.loc[mask, "event"] = -1
    mutated.loc[mask, "official_rul"] = 1e308
    mutated.loc[mask, "gap_before"] = True
    got_above = propose_absolute_limits(mutated, ["train"], "above")
    got_below = propose_absolute_limits(mutated, ["train"], "below")
    assert got_above["yellow"] == above["yellow"]
    assert got_above["red"] == above["red"]
    assert got_below["yellow"] == below["yellow"]
    assert got_below["red"] == below["red"]


def test_train_prefix_edit_changes_proposal():
    # 37 rows keep k_i = 8. Dropping one prefix row leaves 36 rows, still k_i = 8,
    # so the proposal stays numeric and the slid-in tail value has to move it.
    frame = _unit("u", _wide(37, tail=50.0))
    above = propose_absolute_limits(frame, ["u"], "above")
    below = propose_absolute_limits(frame, ["u"], "below")
    _assert_ok(above, "above", _WIDE_HEAD)
    _assert_ok(below, "below", _WIDE_HEAD)

    tail = frame.copy()
    tail.loc[tail["timestamp_s"] == 36, "signal"] = -1e9
    assert propose_absolute_limits(tail, ["u"], "above")["yellow"] == above["yellow"]
    assert propose_absolute_limits(tail, ["u"], "above")["red"] == above["red"]
    assert propose_absolute_limits(tail, ["u"], "below")["yellow"] == below["yellow"]
    assert propose_absolute_limits(tail, ["u"], "below")["red"] == below["red"]

    # Index 7 is inside k_i = 8 (signal 100). Index 0 is the prefix low spike.
    high = frame.copy()
    high.loc[high["timestamp_s"] == 7, "signal"] = 0.0
    high_above = propose_absolute_limits(high, ["u"], "above")
    assert (high_above["yellow"], high_above["red"]) != (above["yellow"], above["red"])

    low = frame.copy()
    low.loc[low["timestamp_s"] == 0, "signal"] = 0.0
    low_below = propose_absolute_limits(low, ["u"], "below")
    assert (low_below["yellow"], low_below["red"]) != (below["yellow"], below["red"])

    removed = frame.loc[frame["timestamp_s"] != 7].copy()
    removed_above = propose_absolute_limits(removed, ["u"], "above")
    assert removed_above["ok"] is True
    assert (removed_above["yellow"], removed_above["red"]) != (above["yellow"], above["red"])


def test_gap_inside_prefix_counts_and_tail_gap_does_not():
    canonical = propose_absolute_limits(_unit("u", _wide(36, tail=999.0)), ["u"], "above")
    flagged = _unit("u", _wide(36, tail=999.0), gap_at={3, 20})
    flagged.loc[flagged.index >= 3, "timestamp_s"] += 1e12
    flagged["note"] = "gap row stays in the prefix"
    got = propose_absolute_limits(flagged, ["u"], "above")
    below = propose_absolute_limits(flagged, ["u"], "below")
    assert got["yellow"] == canonical["yellow"]
    assert got["red"] == canonical["red"]
    assert got["yellow"] == _approx(30)
    assert got["red"] == _approx(93)
    assert below["yellow"] == _approx(-30)
    assert below["red"] == _approx(-93)

    inside = flagged.copy()
    inside.loc[inside["timestamp_s"] == 3.0 + 1e12, "signal"] = 25.0
    inside_got = propose_absolute_limits(inside, ["u"], "above")
    assert (inside_got["yellow"], inside_got["red"]) != (got["yellow"], got["red"])

    outside = flagged.copy()
    outside.loc[outside["timestamp_s"] == 20.0 + 1e12, "signal"] = -1e12
    outside_got = propose_absolute_limits(outside, ["u"], "above")
    assert outside_got["yellow"] == got["yellow"]
    assert outside_got["red"] == got["red"]


def test_fewer_than_eight_early_values_is_disabled():
    assert EARLY_FRACTION == 0.20
    assert EARLY_MIN_POINTS == 5
    assert MIN_EARLY_VALUES == 8

    present = _unit("u", [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    _assert_refused(propose_absolute_limits(present, [], "above"), "above", 0)
    _assert_refused(propose_absolute_limits(present, ["missing"], "below"), "below", 0)

    non_finite = _unit("u", [np.nan, np.inf, -np.inf, np.nan])
    _assert_refused(propose_absolute_limits(non_finite, ["u"], "above"), "above", 0)

    _assert_refused(propose_absolute_limits(_unit("u", [1.0, 2.0, 3.0, 9.0]), ["u"], "below"), "below", 4)

    # 16-row units contribute the first 5 finite rows, which is still under 8.
    sixteen = _unit("u", [float(i) for i in range(16)])
    _assert_refused(propose_absolute_limits(sixteen, ["u"], "above"), "above", 5)

    seven = pd.concat([_unit(f"u{i}", [float(i)]) for i in range(7)], ignore_index=True)
    _assert_refused(propose_absolute_limits(seven, [f"u{i}" for i in range(7)], "below"), "below", 7)


def test_direction_is_respected_and_invalid_direction_raises():
    _assert_module_surface()
    frame = _unit("u", _wide(36))
    above = propose_absolute_limits(frame, ["u"], "above")
    below = propose_absolute_limits(frame, ["u"], "below")
    assert above["direction"] == "above"
    assert below["direction"] == "below"
    assert above["yellow"] > 0
    assert below["yellow"] < 0
    assert below["red"] < below["yellow"] <= float(np.median(_WIDE_HEAD))
    assert above["yellow"] != below["yellow"]
    assert above["red"] != below["red"]
    for bad in ("", "both", "up", "ABOVE", None):
        with pytest.raises(ValueError) as exc:
            propose_absolute_limits(frame, ["u"], bad)
        assert str(exc.value) == "Threshold direction must be above or below"
    with pytest.raises(ValueError) as exc:
        propose_absolute_limits(_unit("u", [1.0]), ["u"], "sideways")
    assert str(exc.value) == "Threshold direction must be above or below"


def test_different_train_ids_change_proposal():
    stay = _unit("stay", [3.0] * 8 + [1000.0] * 28)
    other = _unit("other", [10.0] * 8 + [1000.0] * 28)
    moved = _unit("moved", [50.0] * 8 + [0.0] * 28)
    ignored = _unit("ignored", [1e9] * 36)
    frame = pd.concat([moved, ignored, stay, other], ignore_index=True)
    stay_only = propose_absolute_limits(frame, ["stay"], "above")
    with_other = propose_absolute_limits(frame, ["other", "stay"], "above")
    with_moved = propose_absolute_limits(frame, ["stay", "moved"], "above")
    moved_only = propose_absolute_limits(frame, ["moved"], "below")
    _assert_ok(stay_only, "above", [3.0] * 8)
    assert (with_other["yellow"], with_other["red"]) != (stay_only["yellow"], stay_only["red"])
    assert (with_moved["yellow"], with_moved["red"]) != (stay_only["yellow"], stay_only["red"])
    assert with_moved["n"] == 16
    assert moved_only["direction"] == "below"
    assert moved_only["yellow"] != stay_only["yellow"]
    # Rows whose unit_id is not in the train list do not enter the pool.
    without_ignored = frame.loc[frame["unit_id"] != "ignored"]
    assert propose_absolute_limits(without_ignored, ["stay"], "above")["red"] == stay_only["red"]


def test_nan_train_signal_is_dropped_before_the_prefix():
    clean = _unit("u", _wide(36, tail=4.0))
    clean["timestamp_s"] = clean["timestamp_s"] + 10.0
    junk = pd.DataFrame(
        {
            "unit_id": ["u"] * 6,
            "timestamp_s": [-np.inf, np.nan, -1.0, 12.5, 13.5, 0.25],
            "signal": [1e18, -1e18, np.nan, np.nan, "bad", np.inf],
            "gap_before": [False] * 6,
        }
    )
    dirty = pd.concat([junk, clean], ignore_index=True)
    canonical = propose_absolute_limits(_unit("u", _wide(36, tail=1000.0)), ["u"], "above")
    got = propose_absolute_limits(dirty, ["u"], "above")
    got_below = propose_absolute_limits(dirty, ["u"], "below")
    assert got["ok"] is True
    assert got["n"] == 8
    assert got["yellow"] == canonical["yellow"]
    assert got["red"] == canonical["red"]
    assert got["yellow"] == _approx(30)
    assert got["red"] == _approx(93)
    assert got_below["yellow"] == _approx(-30)
    assert got_below["red"] == _approx(-93)


def _assert_module_surface() -> None:
    import pdm.zone_limit_proposal as mod

    source = Path(mod.__file__).read_text(encoding="utf-8")
    modules: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            modules.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module.split(".")[0])
    assert modules <= {"__future__", "math", "numpy", "pandas"}
    assert "math.ceil" in source
    assert "np.ceil" not in source and "numpy.ceil" not in source
    for banned in ("torch", "pdm.train", "pdm.worker", "validation", "DEFAULT_LIMITS"):
        assert banned not in source
    assert list(inspect.signature(propose_absolute_limits).parameters) == [
        "features",
        "train_unit_ids",
        "direction",
    ]
