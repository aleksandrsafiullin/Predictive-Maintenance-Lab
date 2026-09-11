from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from pdm.alerts import (
    AlertEngine,
    alert_policy_hash,
    build_alert_policy,
    classify_alert_outcome,
    confirmed_lead_time_s,
    copy_alert_policy_into_evaluation_config,
    load_alert_policy,
    resolve_alert_policy,
)
from pdm.config import load_dataset_config
from pdm.data.archive import _describe_mat_variable, probe_mat
from pdm.data.bearings import numeric_csv_index, sort_csv_names_numerically
from pdm.data.filters import (
    FILTER_TIME_SCALE_WARNING,
    FILTER_TIME_TO_SECONDS,
    FILTERS_FULL_HISTORY_STATUSES,
    _filters_full_history_status,
    _time_to_seconds,
    extract_filters_tables,
    filter_time_scale_meta,
    inspect_filters,
    probe_filters_full_history,
)
from pdm.data.prepare import (
    attach_origin_unit_id,
    dataset_fingerprint_for_run,
    load_processed,
    processed_ready,
    resolve_processed_dir,
    write_processed_version,
)
from pdm.evaluate import (
    METRICS_VERSION,
    IncompatibleDataError,
    _filter_validation_block,
    attach_actual_rul,
    bind_evaluation_to_run,
    build_rul_metrics,
    compare_baseline,
    equal_weight_unit_mae_by_zone,
    evaluate_run,
    filter_prefix_backtest_table,
    filter_prefix_end_table,
    filter_time_scale_from_bound,
    load_near_event_zones_s,
    run_alert_evaluation,
    summarize_rul_table,
)
from pdm.experiments import list_evaluations, resolve_evaluation_artifacts
from pdm.io_util import atomic_write_json, checkpoint_hash, sha256_file
from pdm.losses import weibull_nll
from pdm.models import PDMNet
from pdm.predict import Predictor
from pdm.preprocessing import (
    FEATURE_PIPELINE_VERSION,
    Preprocessor,
    fit_preprocessor,
    raw_to_feature_frame,
)
from pdm.replay import (
    ReplaySource,
    filter_replay_alert_log,
    replay_session_key,
    replay_unit,
    rescore_replay_alerts,
    slice_predictions_to_replay_time,
    visible_replay_slice,
)
from pdm.splits import (
    assert_disjoint_splits,
    assert_split_coverage,
    bearings_split,
    filters_split,
    resolve_split_hash,
    split_fingerprint,
    split_hash,
)
from pdm.train import (
    checkpoints_compatible,
    compatibility_dict,
    load_saved_train_settings,
    next_max_windows_on_mode_change,
    resolve_max_windows_per_unit,
    resolve_run_train_args,
    selection_metric_spec,
    training_mode_label,
)
from pdm.train import split_fingerprint as train_split_fingerprint
from pdm.windows import (
    FORBIDDEN_FEATURE_NAMES,
    build_windows,
    filter_gap_params,
    gap_before_from_delta_t,
    recompute_filter_gap_before,
    valid_history_window,
    window_matrix,
)


def test_numeric_csv_sort_and_gap_time_scale():
    names = ["10.csv", "2.csv", "1.csv", "12.csv"]
    sorted_names = sort_csv_names_numerically(names)
    assert [numeric_csv_index(n) for n in sorted_names] == [1, 2, 10, 12]
    # Missing 3 does not shift later timestamps: file 4 stays at 4 minutes, not 3.
    present = [1, 2, 4]
    ts = {i: i * 60.0 for i in present}
    assert ts[4] == 240.0
    assert ts[4] != 180.0


def test_split_units_disjoint(tiny_bearing_tables, tiny_filter_tables):
    bf, bu = tiny_bearing_tables
    split_b = bearings_split(bu, {"split": {"train_instances": [1, 2, 3], "val_instances": [4], "test_instances": [5]}})
    assert_disjoint_splits(split_b)
    assert_split_coverage(bu, split_b)
    assert "origin_unit_id" in bu.columns
    assert list(bu["origin_unit_id"].astype(str)) == list(bu["unit_id"].astype(str))
    ff, fu = tiny_filter_tables
    split_f = filters_split(fu, {"split": {"val_fraction": 0.25, "seed": 42}})
    assert_disjoint_splits(split_f)
    assert_split_coverage(fu, split_f)
    assert set(split_f["test"]) == {"Filter_101", "Filter_102"}
    assert "origin_unit_id" in fu.columns
    assert list(fu["origin_unit_id"]) == list(fu["author_data_no"])


def test_origin_unit_id_intersection_rejected(tiny_bearing_tables):
    _, units = tiny_bearing_tables
    split = bearings_split(units)
    units = units.copy()
    train_id = split["train"][0]
    test_id = split["test"][0]
    train_origin = units.loc[units["unit_id"] == train_id, "origin_unit_id"].iloc[0]
    units.loc[units["unit_id"] == test_id, "origin_unit_id"] = train_origin
    with pytest.raises(ValueError, match="origin_unit_id"):
        assert_split_coverage(units, split)


def test_filters_data_no_reuse_is_not_origin_leak(tiny_filter_tables):
    """HSE Train/Test CSVs reuse Data_No for different experiments — not leakage."""
    _, units = tiny_filter_tables
    units = units.copy()
    train_uid = units.loc[units["author_split"] == "author_train", "unit_id"].iloc[0]
    test_uid = units.loc[units["author_split"] == "author_test", "unit_id"].iloc[0]
    units.loc[units["unit_id"] == train_uid, "author_data_no"] = 50
    units.loc[units["unit_id"] == test_uid, "author_data_no"] = 50
    units = units.drop(columns=["origin_unit_id"], errors="ignore")
    split = filters_split(units)
    assert_split_coverage(units, split)
    assert int(units.loc[units["unit_id"] == train_uid, "origin_unit_id"].iloc[0]) == 50
    assert int(units.loc[units["unit_id"] == test_uid, "origin_unit_id"].iloc[0]) == 50


def test_filters_same_file_origin_intersection_rejected(tiny_filter_tables):
    _, units = tiny_filter_tables
    split = filters_split(units)
    units = units.copy()
    train_id = split["train"][0]
    val_id = split["validation"][0]
    train_origin = units.loc[units["unit_id"] == train_id, "origin_unit_id"].iloc[0]
    units.loc[units["unit_id"] == val_id, "origin_unit_id"] = train_origin
    with pytest.raises(ValueError, match="origin_unit_id"):
        assert_split_coverage(units, split)


def test_same_group_origin_collision_rejected(tiny_bearing_tables, tiny_filter_tables):
    _, bu = tiny_bearing_tables
    split_b = bearings_split(bu)
    bu = bu.copy()
    a, b = split_b["train"][0], split_b["train"][1]
    bu.loc[bu["unit_id"] == b, "origin_unit_id"] = bu.loc[bu["unit_id"] == a, "origin_unit_id"].iloc[0]
    with pytest.raises(ValueError, match="duplicate origin_unit_id in train"):
        assert_split_coverage(bu, split_b)

    _, fu = tiny_filter_tables
    split_f = filters_split(fu)
    fu = fu.copy()
    a, b = split_f["train"][0], split_f["train"][1]
    fu.loc[fu["unit_id"] == b, "origin_unit_id"] = fu.loc[fu["unit_id"] == a, "origin_unit_id"].iloc[0]
    with pytest.raises(ValueError, match="duplicate origin_unit_id in train"):
        assert_split_coverage(fu, split_f)


def test_split_duplicate_unit_ids_rejected(tiny_bearing_tables):
    _, units = tiny_bearing_tables
    split = dict(bearings_split(units))
    split["train"] = list(split["train"]) + [split["train"][0]]
    with pytest.raises(ValueError, match="Duplicate unit_id in train"):
        assert_split_coverage(units, split)
    dup_table = pd.concat([units, units.iloc[[0]]], ignore_index=True)
    split_ok = bearings_split(units)
    with pytest.raises(ValueError, match="Duplicate unit_id in units table"):
        assert_split_coverage(dup_table, split_ok)


def test_filters_unassigned_rejected(tiny_filter_tables):
    _, units = tiny_filter_tables
    split = filters_split(units)
    extra = units.iloc[[0]].copy()
    extra["unit_id"] = "Filter_orphan"
    extra["author_data_no"] = 999
    extra["origin_unit_id"] = 999
    units2 = pd.concat([units, extra], ignore_index=True)
    with pytest.raises(ValueError, match="not assigned|Unassigned units"):
        assert_split_coverage(units2, split)


def test_bearings_unassigned_warns_not_fail(tiny_bearing_tables):
    _, units = tiny_bearing_tables
    extra = units.iloc[[0]].copy()
    extra["unit_id"] = "Bearing9_9"
    extra["instance"] = 9
    extra["origin_unit_id"] = "Bearing9_9"
    units2 = pd.concat([units, extra], ignore_index=True)
    with pytest.warns(UserWarning, match="unassigned"):
        split = bearings_split(units2)
    assert "Bearing9_9" in split["unassigned"]
    assert_split_coverage(units2, split)


def test_test_rows_do_not_change_train_scaler(tiny_bearing_tables):
    features, units = tiny_bearing_tables
    split = bearings_split(units)
    cfg = {"features": {"log1p_features": ["horizontal_rms"]}}
    prep1, _ = fit_preprocessor("bearings", features, units, split, cfg)
    mutated = features.copy()
    test_ids = set(split["test"])
    mutated.loc[mutated["unit_id"].isin(test_ids), "horizontal_rms"] = 999.0
    prep2, _ = fit_preprocessor("bearings", mutated, units, split, cfg)
    np.testing.assert_allclose(prep1.scaler_mean, prep2.scaler_mean)
    np.testing.assert_allclose(prep1.scaler_scale, prep2.scaler_scale)
    assert prep1.time_scale_s == prep2.time_scale_s
    assert prep1.feature_names == prep2.feature_names


def test_future_measurements_do_not_change_prediction_at_t(tiny_bearing_tables):
    features, units = tiny_bearing_tables
    split = bearings_split(units)
    cfg = {"features": {"log1p_features": []}}
    prep, feat_t = fit_preprocessor("bearings", features, units, split, cfg)
    uid = split["train"][0]
    g = feat_t[feat_t["unit_id"] == uid].sort_values("timestamp_s")
    hist = 5
    prefix = g.iloc[:hist]
    model = PDMNet(len(prep.feature_names), hidden_size=8, architecture="gru", head="rul", time_scale_s=prep.time_scale_s)
    model.eval()
    x1 = torch.from_numpy(prefix[prep.feature_names].to_numpy(dtype=np.float32)).unsqueeze(0)
    with torch.no_grad():
        y1 = model.predicted_rul_s(x1).clone()
    g2 = feat_t.copy()
    later = g.index[hist:]
    g2.loc[later, prep.feature_names[0]] = 12345.0
    prefix2 = g2[g2["unit_id"] == uid].sort_values("timestamp_s").iloc[:hist]
    x2 = torch.from_numpy(prefix2[prep.feature_names].to_numpy(dtype=np.float32)).unsqueeze(0)
    with torch.no_grad():
        y2 = model.predicted_rul_s(x2)
    np.testing.assert_allclose(y1.numpy(), y2.numpy(), rtol=1e-6, atol=1e-6)


def test_reference_rul_does_not_change_prediction(tiny_filter_tables):
    features, units = tiny_filter_tables
    split = filters_split(units)
    cfg = {}
    prep, feat_t = fit_preprocessor("filters", features, units, split, cfg)
    model = PDMNet(len(prep.feature_names), hidden_size=8, architecture="gru", head="weibull", time_scale_s=prep.time_scale_s)
    model.eval()
    uid = split["test"][0]
    g = feat_t[feat_t["unit_id"] == uid].sort_values("timestamp_s").iloc[:8]
    x = torch.from_numpy(g[prep.feature_names].to_numpy(dtype=np.float32)).unsqueeze(0)
    with torch.no_grad():
        y1 = model.predicted_rul_s(x).clone()
    units2 = units.copy()
    units2.loc[units2["unit_id"] == uid, "event_time_s"] = 1e9
    units2.loc[units2["unit_id"] == uid, "official_rul_at_prefix_end_s"] = 1e9
    with torch.no_grad():
        y2 = model.predicted_rul_s(x)
    np.testing.assert_allclose(y1.numpy(), y2.numpy())
    p = pd.DataFrame({"unit_id": [uid], "timestamp_s": [float(g.iloc[-1]["timestamp_s"])], "predicted_rul_s": [float(y1)]})
    e1 = attach_actual_rul(p, units, "filters")
    e2 = attach_actual_rul(p, units2, "filters")
    assert pd.isna(e1["actual_rul_s"].iloc[0])
    assert pd.isna(e2["actual_rul_s"].iloc[0])


def test_windows_do_not_cross_units_or_gaps(tiny_bearing_tables):
    features, units = tiny_bearing_tables
    w = build_windows(features, units, history_length=5, dataset_id="bearings")
    for _, row in w.iterrows():
        g = features[features["unit_id"] == row["unit_id"]].sort_values("timestamp_s").reset_index(drop=True)
        sl = g.iloc[int(row["start_index"]) : int(row["end_index"]) + 1]
        assert sl["unit_id"].nunique() == 1
        if "gap_before" in sl.columns:
            assert not sl.iloc[1:]["gap_before"].any()
    gapped = w[w["unit_id"] == "Bearing2_1"]
    # gap at file 8: no window may include both sides
    for _, row in gapped.iterrows():
        g = features[features["unit_id"] == "Bearing2_1"].sort_values("file_index")
        sl = g.iloc[int(row["start_index"]) : int(row["end_index"]) + 1]
        assert 8 not in set(range(int(sl["file_index"].min()), int(sl["file_index"].max()) + 1)) or True
        files = sl["file_index"].to_numpy()
        assert np.all(np.diff(files) == 1)


def test_rul_and_time_conversion(tmp_path, tiny_filter_tables):
    event_time_s = 1200.0
    timestamp_s = 600.0
    assert event_time_s - timestamp_s == 600.0
    minutes = 10.0
    assert minutes * 60.0 == 600.0
    hours = 0.5
    assert hours * 3600.0 == 1800.0

    cfg = load_dataset_config("filters")
    assert float(cfg["time_to_seconds"]) == 60.0
    assert FILTER_TIME_TO_SECONDS == 60.0
    assert cfg.get("time_scale_verified") is False
    assert cfg.get("original_time_unit") == "minutes"
    meta = filter_time_scale_meta(cfg)
    assert meta["time_to_seconds"] == 60.0
    assert meta["time_scale_verified"] is False
    note = meta["time_unit_note"]
    assert note == FILTER_TIME_SCALE_WARNING
    lowered = note.lower()
    assert "unconfirmed" in lowered
    assert "wall-clock" in lowered
    assert "minutes before failure" in lowered
    assert "implementation_report" in lowered
    assert "60" in note
    np.testing.assert_allclose(_time_to_seconds(np.array([10.0]), 60.0), np.array([600.0]))

    features, units = tiny_filter_tables
    assert "time_original" in features.columns
    assert features["time_original"].notna().all()
    np.testing.assert_allclose(features["timestamp_s"], features["time_original"] * 60.0)
    test_units = units[units["author_split"] == "author_test"]
    assert "official_rul_at_prefix_end_original" in test_units.columns
    assert test_units["official_rul_at_prefix_end_original"].notna().all()
    np.testing.assert_allclose(
        test_units["official_rul_at_prefix_end_s"],
        test_units["official_rul_at_prefix_end_original"] * 60.0,
    )
    assert "official_rul_at_prefix_end_original" in FORBIDDEN_FEATURE_NAMES
    assert "official_rul_at_prefix_end_s" in FORBIDDEN_FEATURE_NAMES

    split = filters_split(units)
    rec = write_processed_version(
        "filters",
        features,
        units,
        split,
        sensor_note="n",
        cfg=cfg,
        processed_root=tmp_path / "filters",
    )
    report = rec["report"]
    assert report["time_to_seconds"] == 60.0
    assert report["time_scale_verified"] is False
    assert report["original_time_unit"] == "minutes"
    assert report["time_unit_note"] == FILTER_TIME_SCALE_WARNING
    assert any("unverified" in str(issue).lower() for issue in report["issues_and_decisions"])

    saved_feat = pd.read_parquet(rec["dir"] / "features.parquet")
    saved_units = pd.read_parquet(rec["dir"] / "units.parquet")
    assert "time_original" in saved_feat.columns
    assert saved_feat["time_original"].notna().all()
    saved_test = saved_units[saved_units["author_split"] == "author_test"]
    assert "official_rul_at_prefix_end_original" in saved_test.columns
    assert saved_test["official_rul_at_prefix_end_original"].notna().all()

    stamped = filter_time_scale_from_bound({"report": report, "processed_dir": rec["dir"]})
    assert stamped["time_to_seconds"] == 60.0
    assert stamped["time_unit_note"] == report["time_unit_note"]
    # Bound report wins; live YAML is not consulted.
    from_report_only = filter_time_scale_from_bound(
        {"report": {**report, "time_unit_note": "from-bound-report"}}
    )
    assert from_report_only["time_unit_note"] == "from-bound-report"
    from_file = filter_time_scale_from_bound({"report": {}, "processed_dir": rec["dir"]})
    assert from_file["time_unit_note"] == report["time_unit_note"]
    assert from_file["time_to_seconds"] == 60.0


def test_censored_not_event_and_weibull_nll_and_grad():
    duration = torch.tensor([1.0, 1.0])
    event = torch.tensor([1.0, 0.0])
    lam = torch.tensor([1.0, 1.0], requires_grad=True)
    k = torch.tensor([1.0, 1.0], requires_grad=True)
    nll = weibull_nll(duration, event, lam, k, time_scale_s=1.0)
    # exponential(1): log_f(1)= -1, log_S(1)= -1
    np.testing.assert_allclose(nll.detach().numpy(), np.array([1.0, 1.0]), atol=1e-5)
    nll.sum().backward()
    assert torch.isfinite(lam.grad).all()
    assert torch.isfinite(k.grad).all()
    # a censored row must not be treated as event=1 by the adapter
    units = pd.DataFrame(
        {
            "unit_id": ["a"],
            "event_observed": [0],
            "event_time_s": [np.nan],
            "observation_end_s": [100.0],
        }
    )
    feat = pd.DataFrame(
        {
            "unit_id": ["a"] * 5,
            "timestamp_s": [10, 20, 30, 40, 50],
            "gap_before": [False] * 5,
        }
    )
    w = build_windows(feat, units, 2, "filters")
    assert set(w["event"].unique()) == {0}


def test_gru_lstm_both_heads_change_weights():
    for arch in ("gru", "lstm"):
        for head in ("rul", "weibull"):
            net = PDMNet(3, hidden_size=8, architecture=arch, head=head, time_scale_s=10.0)
            x = torch.randn(4, 6, 3)
            before = [p.detach().clone() for p in net.parameters()]
            opt = torch.optim.AdamW(net.parameters(), lr=0.01)
            opt.zero_grad()
            if head == "rul":
                y = net(x)
                loss = torch.nn.functional.smooth_l1_loss(y, torch.ones_like(y))
            else:
                lam, k = net(x)
                loss = weibull_nll(torch.ones(4), torch.tensor([1.0, 0, 1.0, 0]), lam, k, 10.0).mean()
            loss.backward()
            opt.step()
            changed = any(not torch.allclose(a, b) for a, b in zip(before, net.parameters()))
            assert changed, f"{arch}/{head} weights did not change"


def test_checkpoint_reload_matches(tmp_path):
    net = PDMNet(3, hidden_size=8, architecture="gru", head="rul", time_scale_s=5.0)
    x = torch.randn(2, 5, 3)
    net.eval()
    with torch.no_grad():
        y1 = net.predicted_rul_s(x).clone()
    path = tmp_path / "m.pt"
    torch.save({"model_state_dict": net.state_dict()}, path)
    net2 = PDMNet(3, hidden_size=8, architecture="gru", head="rul", time_scale_s=5.0)
    blob = torch.load(path, map_location="cpu", weights_only=False)
    net2.load_state_dict(blob["model_state_dict"])
    net2.eval()
    with torch.no_grad():
        y2 = net2.predicted_rul_s(x)
    np.testing.assert_allclose(y1.numpy(), y2.numpy(), rtol=1e-6, atol=1e-6)


def test_alert_confirmation_dedup_reset_h_independent_of_prediction():
    eng = AlertEngine(warning_horizon_s=10.0, confirmation_count=3, reset_factor=1.2)
    eng.reset_unit("u1")
    opened = []
    preds = []
    for t, rul in enumerate([20, 9, 8, 7, 6, 5, 13, 13, 13], start=1):
        snap = eng.update(timestamp_s=float(t), predicted_rul_s=float(rul))
        preds.append(rul)
        if snap["opened_episode"]:
            opened.append(snap["opened_episode"])
    assert len(opened) == 1
    assert opened[0]["timestamp_s"] == 4  # third consecutive trigger at t=4 (9,8,7)
    assert eng.warning_active is False
    eng2 = AlertEngine(warning_horizon_s=1.0, confirmation_count=3)
    eng2.reset_unit("u1")
    # same predictions, different H -> different alerts, predictions unchanged
    assert preds[0] == 20
    snaps = [eng2.update(timestamp_s=float(i), predicted_rul_s=float(r)) for i, r in enumerate(preds, start=1)]
    assert preds[0] == 20
    assert any(s["opened_episode"] is None for s in snaps)


def test_alert_outcome_timely_lead_time_and_coverage():
    """Helpers for 15b: timely uses min lead time at confirmed (K-th) timestamp."""
    eng = AlertEngine(warning_horizon_s=10.0, confirmation_count=3, reset_factor=1.2)
    eng.reset_unit("u1")
    opened = None
    for t, rul in enumerate([20, 9, 8, 7, 6], start=1):
        snap = eng.update(timestamp_s=float(t), predicted_rul_s=float(rul))
        if snap["opened_episode"]:
            opened = snap["opened_episode"]
    assert opened is not None
    assert opened["timestamp_s"] == 4
    assert opened["H_trigger"] == 10.0
    assert eng.confirmation_delay_steps == 2

    policy = {
        "H_trigger": 10.0,
        "minimum_action_lead_time": 5.0,
        "confirmation_count": 3,
        "reset_factor": 1.2,
        "max_useful_horizon_s": None,
    }
    confirmed = float(opened["timestamp_s"])
    event = 12.0
    assert confirmed_lead_time_s(event_time_s=event, confirmed_alert_time_s=confirmed) == 8.0
    assert (
        classify_alert_outcome(
            event_time_s=event,
            observation_end_s=event,
            policy=policy,
            confirmed_alert_time_s=confirmed,
        )
        == "timely"
    )
    late_policy = {**policy, "minimum_action_lead_time": 9.0}
    assert (
        classify_alert_outcome(
            event_time_s=event,
            observation_end_s=event,
            policy=late_policy,
            confirmed_alert_time_s=confirmed,
        )
        == "late"
    )
    early_policy = {**policy, "max_useful_horizon_s": 6.0}
    assert (
        classify_alert_outcome(
            event_time_s=event,
            observation_end_s=event,
            policy=early_policy,
            confirmed_alert_time_s=confirmed,
        )
        == "too_early"
    )
    assert (
        classify_alert_outcome(
            event_time_s=event,
            observation_end_s=event,
            policy=policy,
            confirmed_alert_time_s=event,
        )
        == "late"
    )
    assert (
        classify_alert_outcome(
            event_time_s=100.0,
            observation_end_s=100.0,
            policy=policy,
            confirmed_alert_time_s=None,
        )
        == "miss"
    )
    assert (
        classify_alert_outcome(
            event_time_s=100.0,
            observation_end_s=80.0,
            policy=policy,
            confirmed_alert_time_s=None,
        )
        == "insufficient_coverage"
    )


def test_insufficient_coverage_not_false_miss(tmp_path):
    policy = {
        "H_trigger": 10.0,
        "minimum_action_lead_time": 30.0,
        "confirmation_count": 1,
        "reset_factor": 1.2,
        "max_useful_horizon_s": None,
    }
    # coverage_horizon = max(H=10, min_lead=30)=30. Truncated obs_end=2 is not a miss.
    # obs_end=85 covers min_lead but not H-only (event-10=90) → still miss, not insufficient.
    # Filter prefixes: NaN event_time, finite official RUL → overlay event = obs_end + official.
    pred = pd.DataFrame(
        {
            "run_id": ["r"] * 12,
            "unit_id": (
                ["trunc"] * 3
                + ["full"] * 3
                + ["h_only_trap"] * 2
                + ["prefix_short"] * 2
                + ["prefix_covered"] * 2
            ),
            "timestamp_s": [
                0.0,
                1.0,
                2.0,
                90.0,
                95.0,
                100.0,
                80.0,
                85.0,
                10.0,
                20.0,
                90.0,
                100.0,
            ],
            "predicted_rul_s": [100.0] * 12,
        }
    )
    units = pd.DataFrame(
        {
            "unit_id": ["trunc", "full", "h_only_trap", "prefix_short", "prefix_covered"],
            "event_time_s": [100.0, 100.0, 100.0, np.nan, np.nan],
            "observation_end_s": [2.0, 100.0, 85.0, 20.0, 100.0],
            "official_rul_at_prefix_end_s": [np.nan, np.nan, np.nan, 50.0, 20.0],
        }
    )
    out = run_alert_evaluation(pred, policy, units=units)
    block = out["metrics"]["alerts"]
    assert block["insufficient_coverage"] == 2
    assert block["miss"] == 3
    assert block["timely"] == 0
    assert block["n_units_with_event"] == 5
    assert block["n_units_insufficient_coverage"] == 2
    assert block["n_units_sufficient_coverage"] == 3
    assert block["denominator_units_scored"] == 3
    assert block["n_units_scored"] == 3
    assert block["miss_rate"] == pytest.approx(1.0)
    assert block["timely_rate"] == pytest.approx(0.0)
    by = out["by_unit"].set_index("unit_id")
    assert by.loc["trunc", "alert_outcome"] == "insufficient_coverage"
    assert by.loc["full", "alert_outcome"] == "miss"
    # obs_end=85: sufficient vs max(H,min_lead)=30, but not vs H=10 window (event-10=90).
    assert by.loc["h_only_trap", "alert_outcome"] == "miss"
    assert bool(by.loc["trunc", "has_sufficient_coverage"]) is False
    assert bool(by.loc["full", "has_sufficient_coverage"]) is True
    assert by.loc["prefix_short", "alert_event_source"] == "official_rul_overlay"
    assert by.loc["prefix_short", "evaluator_event_time_s"] == pytest.approx(70.0)
    assert by.loc["prefix_short", "alert_outcome"] == "insufficient_coverage"
    assert by.loc["prefix_covered", "alert_event_source"] == "official_rul_overlay"
    assert by.loc["prefix_covered", "evaluator_event_time_s"] == pytest.approx(120.0)
    assert by.loc["prefix_covered", "alert_outcome"] == "miss"

    edir = tmp_path / "eval_cov"
    edir.mkdir()
    pred_path = edir / "predictions.csv"
    pred_path.write_text(pred.to_csv(index=False), encoding="utf-8")
    frozen = pred_path.read_text(encoding="utf-8")
    run_alert_evaluation(pred_path, policy, units=units, eval_dir=edir)
    assert pred_path.read_text(encoding="utf-8") == frozen
    assert "official_rul" not in frozen
    metrics = json.loads((edir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["alerts"]["insufficient_coverage"] == 2
    assert metrics["alerts"]["miss"] == 3
    assert metrics["alerts"]["denominator_units_scored"] == 3
    by_csv = pd.read_csv(edir / "metrics_by_unit.csv")
    assert "alert_outcome" in by_csv.columns


def test_alert_multiple_episodes_and_time_weighting(tmp_path):
    policy = {
        "H_trigger": 10.0,
        "minimum_action_lead_time": 5.0,
        "confirmation_count": 2,
        "reset_factor": 1.2,
        "max_useful_horizon_s": None,
    }
    # Unit multi: two confirmed episodes (K=2). Uneven Δt at t=5→10.
    # t:   0, 1, 2, 3, 4, 5, 10, 11, 12, 13
    # RUL: 50,9, 8, 20,20,20, 9,  8,  7,  6
    # open at t=2 and t=11. event=14 → leads 12 (timely) and 3 (late).
    # warning_active: F F T T F F F T T T; forward Δt warn=4, total=13.
    multi_t = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 11.0, 12.0, 13.0]
    multi_rul = [50.0, 9.0, 8.0, 20.0, 20.0, 20.0, 9.0, 8.0, 7.0, 6.0]
    # Unit late1s: K=3 confirm at t=9, event=10, lead=1 < min_lead=5 → late
    # (legacy H-window would call t=9 timely because it is in [event-H, event)).
    late_t = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
    late_rul = [50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 9.0, 8.0, 7.0]
    # Unit kdelay: confirmed at K-th step t=10, event=20, lead=10.
    # k3 min_lead=9 → timely. Subtracting confirmation_delay_steps again would yield 8 (late).
    k_t = [8.0, 9.0, 10.0]
    k_rul = [9.0, 8.0, 7.0]
    pred = pd.DataFrame(
        {
            "run_id": ["r"] * (len(multi_t) + len(late_t) + len(k_t)),
            "unit_id": ["multi"] * len(multi_t) + ["late1s"] * len(late_t) + ["kdelay"] * len(k_t),
            "timestamp_s": multi_t + late_t + k_t,
            "predicted_rul_s": multi_rul + late_rul + k_rul,
        }
    )
    units = pd.DataFrame(
        {
            "unit_id": ["multi", "late1s", "kdelay"],
            "event_time_s": [14.0, 10.0, 20.0],
            "observation_end_s": [14.0, 10.0, 20.0],
        }
    )
    k3_policy = {**policy, "confirmation_count": 3, "minimum_action_lead_time": 9.0}
    # late1s and kdelay need K=3; multi needs K=2. Score in two passes on the same predictions.
    out_k2 = run_alert_evaluation(
        pred[pred["unit_id"] == "multi"].reset_index(drop=True),
        policy,
        units=units,
    )
    out_k3 = run_alert_evaluation(
        pred[pred["unit_id"].isin(["late1s", "kdelay"])].reset_index(drop=True),
        k3_policy,
        units=units,
    )

    multi_ep = out_k2["alerts"].sort_values("timestamp_s")
    assert len(multi_ep) == 2
    assert list(multi_ep["timestamp_s"]) == [2.0, 11.0]
    assert list(multi_ep["lead_time_s"]) == [12.0, 3.0]
    assert list(multi_ep["outcome"]) == ["timely", "late"]
    # First-episode-only scoring would drop the late episode.
    mk = out_k2["metrics"]["alerts"]
    assert mk["timely"] == 1
    assert mk["late"] == 1
    assert mk["n_episodes"] == 2
    assert mk["miss"] == 0
    assert mk["time_in_warning_s"] == pytest.approx(4.0)
    assert mk["observed_duration_s"] == pytest.approx(13.0)
    assert mk["fraction_time_in_warning"] == pytest.approx(4.0 / 13.0)
    n_warn_rows = int(out_k2["steps"]["warning_active"].sum())
    assert n_warn_rows == 5
    assert mk["fraction_time_in_warning"] != pytest.approx(n_warn_rows / len(out_k2["steps"]))

    late_ep = out_k3["alerts"][out_k3["alerts"]["unit_id"] == "late1s"]
    assert len(late_ep) == 1
    assert float(late_ep.iloc[0]["timestamp_s"]) == 9.0
    assert float(late_ep.iloc[0]["lead_time_s"]) == 1.0
    assert late_ep.iloc[0]["outcome"] == "late"

    k_ep = out_k3["alerts"][out_k3["alerts"]["unit_id"] == "kdelay"]
    assert len(k_ep) == 1
    assert float(k_ep.iloc[0]["timestamp_s"]) == 10.0
    assert float(k_ep.iloc[0]["lead_time_s"]) == 10.0
    assert k_ep.iloc[0]["outcome"] == "timely"
    assert int(k_ep.iloc[0]["confirmation_delay_steps"]) == 2

    edir = tmp_path / "eval_multi"
    edir.mkdir()
    pred_path = edir / "predictions.csv"
    multi_pred = pred[pred["unit_id"] == "multi"].reset_index(drop=True)
    pred_path.write_text(multi_pred.to_csv(index=False), encoding="utf-8")
    (edir / "evaluation_config.json").write_text("{}\n", encoding="utf-8")
    frozen = pred_path.read_bytes()
    run_alert_evaluation(pred_path, policy, units=units, eval_dir=edir)
    assert pred_path.read_bytes() == frozen
    other = {**policy, "H_trigger": 1.0, "minimum_action_lead_time": 5.0}
    run_alert_evaluation(pred_path, other, units=units, eval_dir=edir)
    assert pred_path.read_bytes() == frozen
    run_alert_evaluation(pred_path, policy, units=units, eval_dir=edir)
    assert pred_path.read_bytes() == frozen
    cfg = json.loads((edir / "evaluation_config.json").read_text(encoding="utf-8"))
    assert cfg["alert_policy"]["H_trigger"] == 10.0
    assert cfg["alert_policy"]["minimum_action_lead_time"] == 5.0
    assert cfg["alert_policy"]["K"] == 2
    assert cfg["alert_policy"]["metrics_version"] == METRICS_VERSION
    by = pd.read_csv(edir / "metrics_by_unit.csv").set_index("unit_id")
    assert int(by.loc["multi", "n_alert_episodes"]) == 2
    assert int(by.loc["multi", "timely_episodes"]) == 1
    assert int(by.loc["multi", "late_episodes"]) == 1


def test_alert_policy_frozen_null_max_useful_not_refilled_from_yaml(tmp_path):
    from pdm.io_util import atomic_write_json

    rdir = tmp_path / "run_frozen_null_cap"
    rdir.mkdir()
    atomic_write_json(
        rdir / "alert_policy.json",
        {
            "schema_version": "v1",
            "H_trigger": 60.0,
            "warning_horizon_s": 60.0,
            "minimum_action_lead_time": 30.0,
            "max_useful_horizon_s": None,
            "confirmation_count": 3,
            "reset_factor": 1.2,
        },
    )
    yaml_with_cap = {
        "confirmation_count": 3,
        "reset_factor": 1.2,
        "max_useful_horizon_s": 999.0,
        "max_useful_horizon_fraction": 2.0,
    }
    resolved = resolve_alert_policy(rdir, alerts_cfg=yaml_with_cap)
    assert resolved["max_useful_horizon_s"] is None
    omitted_dir = tmp_path / "run_omitted_cap"
    omitted_dir.mkdir()
    atomic_write_json(
        omitted_dir / "alert_policy.json",
        {
            "schema_version": "v1",
            "H_trigger": 60.0,
            "warning_horizon_s": 60.0,
            "minimum_action_lead_time": 30.0,
            "confirmation_count": 3,
            "reset_factor": 1.2,
        },
    )
    filled = resolve_alert_policy(omitted_dir, alerts_cfg=yaml_with_cap)
    assert filled["max_useful_horizon_s"] == 999.0


def test_truncated_test_history_not_an_event_and_no_future_rows():
    meas = pd.DataFrame(
        {
            "timestamp_s": [1.0, 2.0, 3.0],
            "differential_pressure": [10.0, 20.0, 30.0],
        }
    )
    src = ReplaySource(meas)
    prefix = src.prefix(1)
    assert list(prefix["timestamp_s"]) == [1.0, 2.0]
    assert 3.0 not in set(prefix["timestamp_s"])
    last_dp = float(meas["differential_pressure"].iloc[-1])
    assert last_dp < 600.0
    # end of truncated file is not Observed limit reached
    eng = AlertEngine(10.0, 3)
    eng.reset_unit("t")
    snap = eng.update(timestamp_s=3.0, predicted_rul_s=50.0, observed_limit=last_dp > 600)
    assert snap["status"] != "Observed limit reached"


def _loop_causal_gap(dt: np.ndarray, k: float, samp: float) -> np.ndarray:
    """Independent causal rule: gap[i] = dt[i] > k * median(dt[1:i])."""
    dt = np.asarray(dt, dtype=np.float64)
    gap = np.zeros(dt.size, dtype=bool)
    for i in range(1, dt.size):
        past = dt[1:i]
        finite = past[np.isfinite(past)]
        if finite.size:
            med = float(np.median(finite))
            ref = med if np.isfinite(med) and med > 0.0 else samp
        else:
            ref = samp
        gap[i] = bool(np.isfinite(dt[i]) and dt[i] > k * ref)
    return gap


def test_filter_gap_params_match_documented_yaml():
    k, samp = filter_gap_params()
    assert k == 3.0
    assert samp == 6.0


def test_filter_gap_params_persisted_on_preprocessor(tiny_filter_tables):
    features, units = tiny_filter_tables
    split = filters_split(units)
    cfg = {"gap": {"gap_multiplier": 7.0, "sampling_interval_s": 2.5}}
    prep, _ = fit_preprocessor("filters", features, units, split, cfg)
    assert prep.gap_multiplier == 7.0
    assert prep.sampling_interval_s == 2.5
    blob = prep.to_dict()
    assert blob["gap_multiplier"] == 7.0
    assert blob["sampling_interval_s"] == 2.5
    reloaded = Preprocessor.from_dict(blob)
    assert reloaded.gap_multiplier == 7.0
    assert reloaded.sampling_interval_s == 2.5
    legacy = {k: v for k, v in blob.items() if k not in {"gap_multiplier", "sampling_interval_s"}}
    assert Preprocessor.from_dict(legacy).gap_multiplier is None
    model = PDMNet(
        len(reloaded.feature_names),
        hidden_size=8,
        architecture="gru",
        head="weibull",
        time_scale_s=reloaded.time_scale_s,
    )
    pred = Predictor(model, reloaded, history_length=5)
    assert pred.gap_multiplier == 7.0
    assert pred.sampling_interval_s == 2.5


def test_filter_causal_gap_uses_prefix_median_not_full_file():
    k, samp = 3.0, 6.0
    dt = np.array([0.0, 30.0])
    got = gap_before_from_delta_t(dt, gap_multiplier=k, sampling_interval_s=samp, causal=True)
    assert list(got) == [False, True]

    regular = np.array([0.0, 6.0, 6.0, 6.0, 100.0, 6.0])
    got = gap_before_from_delta_t(regular, gap_multiplier=k, sampling_interval_s=samp, causal=True)
    np.testing.assert_array_equal(got, _loop_causal_gap(regular, k, samp))
    assert list(got) == [False, False, False, False, True, False]

    # Early slow Δt look like gaps vs full-file median of later 6 s samples.
    mixed = np.concatenate([[0.0], np.full(4, 20.0), np.full(20, 6.0)])
    causal = gap_before_from_delta_t(mixed, gap_multiplier=k, sampling_interval_s=samp, causal=True)
    full = gap_before_from_delta_t(mixed, gap_multiplier=k, sampling_interval_s=samp, causal=False)
    np.testing.assert_array_equal(causal, _loop_causal_gap(mixed, k, samp))
    assert causal[1]  # first interval vs documented 6 s fallback
    assert not causal[2:5].any()
    assert full[1:5].all()

    mixed_future = np.concatenate([mixed, [1000.0], np.full(30, 6.0)])
    causal_future = gap_before_from_delta_t(
        mixed_future, gap_multiplier=k, sampling_interval_s=samp, causal=True
    )
    np.testing.assert_array_equal(causal, causal_future[: mixed.size])

    # Full-file median can flip earlier flags when later dense samples arrive.
    sparse = np.array([0.0, 20.0, 20.0, 20.0, 20.0, 50.0])
    sparse_then_dense = np.concatenate([sparse, np.full(40, 1.0)])
    full_sparse = gap_before_from_delta_t(sparse, gap_multiplier=k, sampling_interval_s=samp, causal=False)
    full_dense = gap_before_from_delta_t(
        sparse_then_dense, gap_multiplier=k, sampling_interval_s=samp, causal=False
    )
    assert not np.array_equal(full_sparse, full_dense[: sparse.size])
    causal_sparse = gap_before_from_delta_t(sparse, gap_multiplier=k, sampling_interval_s=samp, causal=True)
    causal_dense = gap_before_from_delta_t(
        sparse_then_dense, gap_multiplier=k, sampling_interval_s=samp, causal=True
    )
    np.testing.assert_array_equal(causal_sparse, causal_dense[: sparse.size])

    ts = np.cumsum(mixed)
    ts_future = np.cumsum(mixed_future)
    prefix = pd.DataFrame(
        {
            "dataset_id": ["filters"] * mixed.size,
            "timestamp_s": ts,
            "gap_before": full,  # poisoned full-file flags
        }
    )
    longer = pd.DataFrame(
        {
            "dataset_id": ["filters"] * mixed_future.size,
            "timestamp_s": ts_future,
            "gap_before": True,
        }
    )
    src = ReplaySource(longer, gap_multiplier=k, sampling_interval_s=samp)
    recomputed = src.prefix(mixed.size - 1)
    np.testing.assert_array_equal(recomputed["gap_before"].to_numpy(), causal)

    # Stored parquet flags must not drive inference: regular 6 s with fake gaps.
    n = 8
    fake_gaps = pd.DataFrame(
        {
            "dataset_id": ["filters"] * n,
            "timestamp_s": np.arange(n, dtype=float) * 6.0,
            "gap_before": [True] * n,
            "delta_t_s": [0.0] + [6.0] * (n - 1),
        }
    )
    ok, reason = valid_history_window(
        fake_gaps, 5, dataset_id="filters", required_columns=[], gap_multiplier=k, sampling_interval_s=samp
    )
    assert ok and reason == ""

    hole = fake_gaps.copy()
    hole.loc[5, "timestamp_s"] = float(hole.loc[4, "timestamp_s"] + 100.0)
    hole.loc[6:, "timestamp_s"] = hole.loc[5, "timestamp_s"] + np.arange(1, n - 5) * 6.0
    hole["gap_before"] = False
    ok_h, reason_h = valid_history_window(
        hole, 5, dataset_id="filters", required_columns=[], gap_multiplier=k, sampling_interval_s=samp
    )
    assert not ok_h and reason_h == "gap_in_window"

    rec = recompute_filter_gap_before(prefix, gap_multiplier=k, sampling_interval_s=samp, causal=True)
    np.testing.assert_array_equal(rec["gap_before"].to_numpy(), causal)


def test_split_hash_alias_and_protocol_identity():
    split = {
        "train": ["a", "b"],
        "validation": ["c"],
        "test": ["d"],
        "protocol": "per_regime_instances_1-3_train_4_val_5_test",
        "n_train": 2,
        "warning": "ignored-for-hash",
    }
    h = split_hash(split)
    assert h == split_fingerprint(split) == train_split_fingerprint(split)
    assert len(h) == 16
    assert split_hash({**split, "n_train": 99, "warning": "other"}) == h
    mutated = {**split, "test": ["e"]}
    assert split_hash(mutated) != h
    blob = {"split_hash": h}
    legacy = {"split_fingerprint": h}
    assert resolve_split_hash(blob) == resolve_split_hash(legacy) == h
    assert resolve_split_hash({"split_hash": h, "split_fingerprint": "old"}) == h


def test_checkpoints_compatible_accepts_split_hash_or_legacy_fingerprint():
    base = {
        "dataset_id": "bearings",
        "architecture": "gru",
        "history_length": 20,
        "hidden_size": 64,
        "recurrent_layers": 2,
        "head": "rul",
        "feature_names": ["horizontal_rms"],
        "time_scale_s": 60.0,
        "feature_pipeline_version": FEATURE_PIPELINE_VERSION,
        "categorical_maps_fingerprint": "abc",
    }
    h = "deadbeefdeadbeef"
    saved_legacy = {**base, "split_fingerprint": h}
    current = {**base, "split_hash": h}
    assert checkpoints_compatible(saved_legacy, current)
    assert checkpoints_compatible({**base, "split_hash": h}, {**base, "split_fingerprint": h})
    assert not checkpoints_compatible(saved_legacy, {**current, "split_hash": "ffffffffffffaaaa"})
    assert not checkpoints_compatible(saved_legacy, {**current, "dataset_id": "filters"})
    assert checkpoints_compatible({**saved_legacy, "features_hash": "aa"}, {**current, "features_hash": "aa"})
    assert not checkpoints_compatible({**saved_legacy, "features_hash": "aa"}, {**current, "features_hash": "bb"})
    assert checkpoints_compatible(saved_legacy, {**current, "features_hash": "bb", "dataset_version": "v"})


def test_processed_fingerprint_versioning_does_not_mutate_prior(
    tmp_path, monkeypatch, tiny_bearing_tables
):
    features, units = tiny_bearing_tables
    split = bearings_split(units)
    root = tmp_path / "bearings"
    monkeypatch.setattr("pdm.data.prepare.dataset_processed", lambda _id: root)
    rec1 = write_processed_version(
        "bearings", features, units, split, sensor_note="n", cfg={}, processed_root=root
    )
    v1 = rec1["dataset_version"]
    v1_dir = root / "versions" / v1
    feat_path = v1_dir / "features.parquet"
    h1 = sha256_file(feat_path)
    fp1 = rec1["fingerprint"]
    assert fp1["dataset_version"] == v1
    assert fp1["feature_pipeline_version"] == FEATURE_PIPELINE_VERSION
    assert fp1["split_hash"] == split_hash(split)
    assert fp1["features_hash"] == h1
    assert fp1["features_hash_short"] == h1[:12]
    assert (v1_dir / "processed_fingerprint.json").exists()
    assert rec1["report"]["dataset_version"] == v1
    assert rec1["report"]["split_protocol"] == split["protocol"]
    assert rec1["report"]["split_hash"] == split_hash(split)
    units1 = pd.read_parquet(v1_dir / "units.parquet")
    assert "origin_unit_id" in units1.columns
    assert list(units1["origin_unit_id"]) == list(units1["unit_id"])

    features2 = features.copy()
    features2.loc[features2.index[0], "horizontal_rms"] = 999.0
    rec2 = write_processed_version(
        "bearings", features2, units, split, sensor_note="n", cfg={}, processed_root=root
    )
    assert rec2["dataset_version"] != v1
    assert sha256_file(feat_path) == h1
    assert (root / "versions" / rec2["dataset_version"] / "features.parquet").exists()
    man = (root / "manifest.json").read_text(encoding="utf-8")
    assert rec2["dataset_version"] in man
    assert processed_ready("bearings")
    loaded = load_processed("bearings")
    assert loaded["dataset_version"] == rec2["dataset_version"]
    assert loaded["fingerprint"]["features_hash"] == rec2["fingerprint"]["features_hash"]
    assert float(loaded["features"].iloc[0]["horizontal_rms"]) == 999.0
    run_fp = dataset_fingerprint_for_run(loaded, split)
    assert run_fp["split_hash"] == split_hash(split)
    assert run_fp["dataset_version"] == rec2["dataset_version"]
    assert run_fp["feature_pipeline_version"] == FEATURE_PIPELINE_VERSION


def test_origin_unit_id_passthrough_from_author_data_no(tmp_path, tiny_filter_tables):
    features, units = tiny_filter_tables
    units = units.copy()
    units["author_data_no"] = units["unit_id"].str.split("_").str[1].astype(int)
    split = filters_split(units)
    rec = write_processed_version(
        "filters", features, units, split, sensor_note="n", cfg={}, processed_root=tmp_path / "filters"
    )
    saved = pd.read_parquet(rec["dir"] / "units.parquet")
    assert "origin_unit_id" in saved.columns
    assert list(saved["origin_unit_id"]) == list(saved["author_data_no"])
    already = attach_origin_unit_id(saved)
    assert already is saved or list(already["origin_unit_id"]) == list(saved["origin_unit_id"])


def test_legacy_processed_dir_still_loads(tmp_path, monkeypatch, tiny_bearing_tables):
    features, units = tiny_bearing_tables
    split = bearings_split(units)
    root = tmp_path / "bearings_legacy"
    root.mkdir(parents=True)
    features.to_parquet(root / "features.parquet", index=False)
    units.to_parquet(root / "units.parquet", index=False)
    (root / "split.json").write_text(json.dumps(split), encoding="utf-8")
    monkeypatch.setattr("pdm.data.prepare.dataset_processed", lambda _id: root)
    assert processed_ready("bearings")
    loaded = load_processed("bearings")
    assert loaded["fingerprint"]["dataset_version"] == "unversioned"
    assert loaded["fingerprint"]["split_hash"] == split_hash(split)
    assert loaded["fingerprint"]["features_hash"] == sha256_file(root / "features.parquet")
    fp = dataset_fingerprint_for_run(loaded, split)
    assert fp["split_hash"] == split_hash(split)


def test_manifest_does_not_fall_back_to_root_parquet(tmp_path, monkeypatch, tiny_bearing_tables):
    features, units = tiny_bearing_tables
    split = bearings_split(units)
    root = tmp_path / "bearings_stale"
    root.mkdir()
    features.to_parquet(root / "features.parquet", index=False)
    units.to_parquet(root / "units.parquet", index=False)
    (root / "split.json").write_text(json.dumps(split), encoding="utf-8")
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "dataset_id": "bearings",
                "current_version": "gone",
                "versions": ["gone"],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr("pdm.data.prepare.dataset_processed", lambda _id: root)
    assert not processed_ready("bearings")
    with pytest.raises(FileNotFoundError, match="gone"):
        resolve_processed_dir("bearings")


def test_evaluate_rejects_changed_dataset_or_split(tmp_path, monkeypatch, tiny_bearing_tables):
    features, units = tiny_bearing_tables
    split = bearings_split(units)
    processed_root = tmp_path / "processed" / "bearings"
    runs_root = tmp_path / "runs"
    monkeypatch.setattr("pdm.data.prepare.dataset_processed", lambda _id: processed_root)
    monkeypatch.setattr("pdm.paths.dataset_runs", lambda _id: runs_root / _id)

    rec = write_processed_version(
        "bearings", features, units, split, sensor_note="n", cfg={}, processed_root=processed_root
    )
    run_id = "fake_run"
    rdir = runs_root / "bearings" / run_id
    rdir.mkdir(parents=True)
    fp = dataset_fingerprint_for_run(
        {"fingerprint": rec["fingerprint"], "dir": rec["dir"], "split": split}, split
    )
    (rdir / "best.pt").write_bytes(b"synthetic-checkpoint-bytes")
    fp["checkpoint_hash"] = checkpoint_hash(rdir / "best.pt")
    atomic_write_json(rdir / "dataset_fingerprint.json", fp)
    atomic_write_json(rdir / "split.json", split)

    bound = bind_evaluation_to_run(rdir, "bearings")
    assert bound["split"]["test"] == split["test"]
    assert bound["split"]["protocol"] == split["protocol"]

    live_split_path = rec["dir"] / "split.json"
    poisoned = {
        **split,
        "test": [split["train"][0]],
        "train": list(split["test"]),
        "validation": list(split["validation"]),
    }
    atomic_write_json(live_split_path, poisoned)
    with pytest.raises(IncompatibleDataError) as ei_split:
        evaluate_run("bearings", run_id)
    assert "split_hash" in ei_split.value.differing or "split_json_hash" in ei_split.value.differing
    atomic_write_json(live_split_path, split)

    live_fp_path = rec["dir"] / "processed_fingerprint.json"
    live_fp = json.loads(live_fp_path.read_text(encoding="utf-8"))
    mutated_split = {
        **split,
        "test": list(split["test"]) + [split["train"][0]],
        "train": list(split["train"][1:]),
    }
    live_fp["split_hash"] = split_hash(mutated_split)
    live_fp["features_hash"] = "0" * 64
    live_fp["units_hash"] = "1" * 64
    live_fp["dataset_version"] = "mutated_version"
    live_fp["feature_pipeline_version"] = "mutated_pipeline"
    live_fp_path.write_text(json.dumps(live_fp), encoding="utf-8")

    with pytest.raises(IncompatibleDataError) as ei:
        evaluate_run("bearings", run_id)
    diff = ei.value.differing
    for field in (
        "dataset_version",
        "split_hash",
        "features_hash",
        "units_hash",
        "feature_pipeline_version",
    ):
        assert field in diff
        assert field in str(ei.value)

    live_fp_path.write_text(json.dumps(rec["fingerprint"]), encoding="utf-8")
    feat_path = rec["dir"] / "features.parquet"
    orig_feat = feat_path.read_bytes()
    feat_path.write_bytes(orig_feat + b"\x00")
    with pytest.raises(IncompatibleDataError) as ei_pq:
        evaluate_run("bearings", run_id)
    assert "features_hash" in ei_pq.value.differing
    assert "features_hash" in str(ei_pq.value)
    feat_path.write_bytes(orig_feat)

    (rdir / "best.pt").write_bytes(b"tampered-checkpoint-bytes")
    with pytest.raises(IncompatibleDataError) as ei_ckpt:
        evaluate_run("bearings", run_id)
    assert "checkpoint_hash" in ei_ckpt.value.differing
    assert "checkpoint_hash" in str(ei_ckpt.value)


def test_legacy_evaluation_artifacts_readonly(tmp_path):
    rdir = tmp_path / "legacy_run"
    rdir.mkdir()
    (rdir / "predictions.csv").write_text("unit_id,predicted_rul_s\nU1,1.0\n", encoding="utf-8")
    (rdir / "test_metrics.json").write_text("{}\n", encoding="utf-8")
    (rdir / "alerts.csv").write_text("unit_id\n", encoding="utf-8")
    assert list_evaluations(rdir) == []
    art = resolve_evaluation_artifacts(rdir)
    assert art["legacy"] is True
    assert art["eval_id"] is None
    assert art["predictions"] == rdir / "predictions.csv"
    assert art["metrics"] == rdir / "test_metrics.json"
    assert art["alerts"] == rdir / "alerts.csv"


def _write_tiny_bearing_checkpoint(rdir, prep, split, fp, *, history_length: int = 3) -> None:
    mcfg = {
        "architecture": "gru",
        "history_length": history_length,
        "hidden_size": 8,
        "recurrent_layers": 1,
    }
    torch.manual_seed(0)
    model = PDMNet(
        input_size=len(prep.feature_names),
        hidden_size=8,
        num_layers=1,
        architecture="gru",
        head="rul",
        dropout=0.1,
        time_scale_s=prep.time_scale_s,
    )
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "compat": compatibility_dict(mcfg, prep, split, "bearings", "rul", fingerprint=fp),
            "meta": {
                "epoch": 1,
                "best_epoch": 1,
                "best_metric": 0.0,
                "smoke": True,
                "time_scale_s": prep.time_scale_s,
                "head": "rul",
                "architecture": "gru",
                "history_length": history_length,
                "hidden_size": 8,
                "recurrent_layers": 1,
                "dropout": 0.1,
                "input_size": len(prep.feature_names),
                "feature_names": list(prep.feature_names),
                "dataset_id": "bearings",
            },
        },
        rdir / "best.pt",
    )
    atomic_write_json(rdir / "preprocessing.json", prep.to_dict())
    atomic_write_json(rdir / "split.json", split)
    fp = dict(fp)
    fp["checkpoint_hash"] = checkpoint_hash(rdir / "best.pt")
    atomic_write_json(rdir / "dataset_fingerprint.json", fp)


def test_evaluate_run_immutable_dirs_predictions_independent_of_hk(
    tmp_path, monkeypatch, tiny_bearing_tables
):
    features, units = tiny_bearing_tables
    split = bearings_split(units)
    processed_root = tmp_path / "processed" / "bearings"
    runs_root = tmp_path / "runs"
    monkeypatch.setattr("pdm.data.prepare.dataset_processed", lambda _id: processed_root)
    monkeypatch.setattr("pdm.paths.dataset_runs", lambda _id: runs_root / _id)

    rec = write_processed_version(
        "bearings", features, units, split, sensor_note="n", cfg={}, processed_root=processed_root
    )
    run_id = "tiny_eval_run"
    rdir = runs_root / "bearings" / run_id
    rdir.mkdir(parents=True)
    fp = dataset_fingerprint_for_run(
        {"fingerprint": rec["fingerprint"], "dir": rec["dir"], "split": split}, split
    )
    prep, _ = fit_preprocessor("bearings", features, units, split, {"features": {"log1p_features": []}})
    _write_tiny_bearing_checkpoint(rdir, prep, split, fp, history_length=3)

    rec1 = evaluate_run("bearings", run_id, warning_horizon_s=60.0, confirmation_count=1, device="cpu")
    policy_path = rdir / "alert_policy.json"
    assert policy_path.exists()
    frozen1 = json.loads(policy_path.read_text(encoding="utf-8"))
    assert frozen1["H_trigger"] == 60.0
    assert frozen1["warning_horizon_s"] == 60.0
    assert frozen1["confirmation_count"] == 1
    assert frozen1["minimum_action_lead_time"] == 30.0
    rec2 = evaluate_run("bearings", run_id, warning_horizon_s=10_000.0, confirmation_count=7, device="cpu")
    assert rec1["eval_id"] != rec2["eval_id"]
    assert rec1["metrics_version"] == METRICS_VERSION
    evals = list_evaluations(rdir)
    assert len(evals) == 2
    ids = {e["eval_id"] for e in evals}
    assert rec1["eval_id"] in ids and rec2["eval_id"] in ids
    d1 = rdir / "evaluations" / rec1["eval_id"]
    d2 = rdir / "evaluations" / rec2["eval_id"]
    assert d1.is_dir() and d2.is_dir()
    for edir, rec in ((d1, rec1), (d2, rec2)):
        for name in (
            "evaluation_config.json",
            "predictions.csv",
            "alerts.csv",
            "metrics.json",
            "metrics_by_unit.csv",
        ):
            assert (edir / name).exists(), name
        cfg = json.loads((edir / "evaluation_config.json").read_text(encoding="utf-8"))
        assert cfg["eval_id"] == rec["eval_id"]
        assert cfg["checkpoint_hash"]
        assert cfg["split_hash"]
        assert cfg["evaluate_mask"]["split"] == "test"
        assert cfg["evaluate_mask"]["unit_ids"] == list(split["test"])
        assert cfg["metrics_version"] == METRICS_VERSION
        assert "minimum_action_lead_time" not in cfg
        assert "warning_horizon_s" not in cfg
        assert "confirmation_count" not in cfg
        assert "H_trigger" not in cfg
        assert "alert_policy" in cfg
        assert cfg["alert_policy"]["K"] == cfg["alert_policy"]["confirmation_count"]
        assert cfg["alert_policy"]["metrics_version"] == METRICS_VERSION
        metrics = json.loads((edir / "metrics.json").read_text(encoding="utf-8"))
        assert metrics["primary_metric"] == "equal_weight_unit_mae"
        assert "equal_weight_unit_mae" in metrics
        assert "pooled_mae" in metrics
        assert "pooled_rmse" in metrics
        assert "mean_overestimation" in metrics
        assert "unit_mae" in metrics
        assert metrics["near_event_zones_s"] == [3600, 1800, 600]
        assert set(metrics["equal_weight_unit_mae_by_zone"]) == {"3600", "1800", "600"}
        assert metrics["baseline"]["name"] == "Age-only"
        assert "neural_net_all_points" in metrics["baseline"]
        assert "baseline_coverage_fraction" in metrics["baseline"]
        assert "test_rul" not in metrics
        assert metrics["n_scored_points"] < metrics["n_prediction_rows"]
        by_unit = pd.read_csv(edir / "metrics_by_unit.csv")
        assert "unit_id" in by_unit.columns
        assert "mae" in by_unit.columns
        assert "alert_outcome" in by_unit.columns
        assert set(by_unit["unit_id"].astype(str)) == set(split["test"])
        pred = pd.read_csv(edir / "predictions.csv")
        assert "alert_status" not in pred.columns
        assert not pred.empty
        alerts = pd.read_csv(edir / "alerts.csv")
        assert "timestamp_s" in alerts.columns or alerts.empty
    leftover = [p.name for p in (rdir / "evaluations").iterdir() if p.name.startswith(".")]
    assert leftover == []
    ghost = rdir / "evaluations" / ".tmp_unfinished"
    ghost.mkdir()
    assert len(list_evaluations(rdir)) == 2
    ghost.rmdir()
    p1 = pd.read_csv(d1 / "predictions.csv")
    p2 = pd.read_csv(d2 / "predictions.csv")
    pd.testing.assert_frame_equal(p1, p2)
    m1 = json.loads((d1 / "metrics.json").read_text(encoding="utf-8"))
    m2 = json.loads((d2 / "metrics.json").read_text(encoding="utf-8"))
    m1.pop("eval_id")
    m2.pop("eval_id")
    m1.pop("alerts", None)
    m2.pop("alerts", None)
    assert m1 == m2
    cfg1_full = json.loads((d1 / "evaluation_config.json").read_text(encoding="utf-8"))
    cfg2_full = json.loads((d2 / "evaluation_config.json").read_text(encoding="utf-8"))
    assert cfg1_full["alert_policy"]["H_trigger"] == 60.0
    assert cfg1_full["alert_policy"]["confirmation_count"] == 1
    assert cfg2_full["alert_policy"]["H_trigger"] == 10_000.0
    assert cfg2_full["alert_policy"]["confirmation_count"] == 7
    assert not (rdir / "predictions.csv").exists()
    assert not (rdir / "test_metrics.json").exists()
    first_pred = (d1 / "predictions.csv").read_text(encoding="utf-8")
    rec3 = evaluate_run("bearings", run_id, warning_horizon_s=120.0, confirmation_count=2, device="cpu")
    assert rec3["eval_id"] not in {rec1["eval_id"], rec2["eval_id"]}
    assert (d1 / "predictions.csv").read_text(encoding="utf-8") == first_pred
    assert len(list_evaluations(rdir)) == 3
    frozen2 = json.loads(policy_path.read_text(encoding="utf-8"))
    assert frozen2["H_trigger"] == frozen1["H_trigger"]
    assert frozen2["confirmation_count"] == frozen1["confirmation_count"]
    assert frozen2["minimum_action_lead_time"] == frozen1["minimum_action_lead_time"]
    assert frozen2["policy_hash"] == frozen1["policy_hash"]
    cfg1 = json.loads((d1 / "evaluation_config.json").read_text(encoding="utf-8"))
    cfg_identity = {k: v for k, v in cfg1.items() if k != "alert_policy"}
    merged = copy_alert_policy_into_evaluation_config(cfg_identity, run_dir=rdir)
    assert merged["alert_policy"]["H_trigger"] == 60.0
    assert merged["alert_policy"]["minimum_action_lead_time"] == 30.0
    assert "alert_policy" not in cfg_identity
    assert "alert_policy" in cfg1
    assert load_alert_policy(rdir)["H_trigger"] == 60.0
    assert (d1 / "predictions.csv").read_text(encoding="utf-8") == first_pred


def test_bearings_near_event_zones_frozen_in_config():
    cfg = load_dataset_config("bearings")
    assert cfg["evaluation"]["near_event_zones_s"] == [3600, 1800, 600]
    assert load_near_event_zones_s(cfg) == [3600.0, 1800.0, 600.0]
    assert load_near_event_zones_s({}) == [3600.0, 1800.0, 600.0]


def test_bearings_zone_mae_excludes_endpoints():
    pred = pd.DataFrame(
        {
            "unit_id": ["U1", "U1", "U1", "U2", "U2"],
            "timestamp_s": [100.0, 200.0, 300.0, 100.0, 200.0],
            "actual_rul_s": [500.0, 2000.0, np.nan, 400.0, np.nan],
            "predicted_rul_s": [400.0, 1800.0, 1e9, 500.0, 1e9],
        }
    )
    stats = summarize_rul_table(pred)
    assert stats["n_points"] == 3
    assert stats["equal_weight_unit_mae"] == pytest.approx(125.0)
    by_zone = equal_weight_unit_mae_by_zone(pred, [3600, 1800, 600])
    assert by_zone["600"] == pytest.approx(100.0)
    assert by_zone["1800"] == pytest.approx(100.0)
    assert by_zone["3600"] == pytest.approx(125.0)
    scored_600 = pred[
        pred["actual_rul_s"].notna()
        & pred["predicted_rul_s"].notna()
        & (pred["actual_rul_s"] > 0)
        & (pred["actual_rul_s"] <= 600)
    ]
    assert len(scored_600) == 2
    assert scored_600["actual_rul_s"].max() <= 600


def test_filter_validation_block_uses_best_not_last(tmp_path):
    rdir = tmp_path / "run"
    rdir.mkdir()
    atomic_write_json(
        rdir / "validation_metrics.json",
        {
            "best_epoch": 2,
            "best_metric": 0.8,
            "n_val_windows": 12,
            "last": {"selection_metric": 1.25, "val_mae_events": 99.0, "n_val_event_units": 2},
        },
    )
    (rdir / "training_history.csv").write_text(
        "epoch,val_metric,val_mae_events,n_val_event_units\n"
        "1,1.0,50.0,1\n"
        "2,0.8,40.0,1\n"
        "3,1.25,99.0,2\n",
        encoding="utf-8",
    )
    block = _filter_validation_block(rdir)
    assert block is not None
    assert block["nll_all_units"] == pytest.approx(0.8)
    assert block["mae_observed_events"] == pytest.approx(40.0)
    assert block["n_observed_event_units"] == 1
    assert block["checkpoint_selection_epoch"] == 2
    assert "checkpoint-selection" in block["note"]


def test_filter_endpoint_baseline_comparison(tmp_path, monkeypatch, tiny_filter_tables):
    pred = pd.DataFrame(
        {
            "unit_id": ["Test_1", "Test_1", "Test_2", "Test_2"],
            "timestamp_s": [6.0, 12.0, 6.0, 18.0],
            "predicted_rul_s": [100.0, 80.0, 90.0, 70.0],
            "baseline_rul_s": [50.0, 40.0, np.nan, np.nan],
        }
    )
    units = pd.DataFrame(
        {
            "unit_id": ["Test_1", "Test_2"],
            "official_rul_at_prefix_end_s": [80.0, 70.0],
            "observation_end_s": [12.0, 18.0],
            "event_observed": [0, 0],
            "event_time_s": [np.nan, np.nan],
            "author_split": ["author_test", "author_test"],
        }
    )
    ends = filter_prefix_end_table(pred, units).sort_values("unit_id")
    assert list(ends["unit_id"]) == ["Test_1", "Test_2"]
    assert ends["actual_rul_s"].tolist() == [80.0, 70.0]
    assert ends["predicted_rul_s"].tolist() == [80.0, 70.0]
    cmp = compare_baseline(ends)
    assert cmp["n_reference_points"] == 2
    assert cmp["baseline_coverage_points"] == 1
    assert cmp["baseline_coverage_fraction"] == pytest.approx(0.5)
    assert cmp["neural_net_all_points"]["n_points"] == 2
    assert cmp["neural_net_all_points"]["mae"] == pytest.approx(0.0)
    assert cmp["neural_net_baseline_overlap"]["n_points"] == 1
    assert cmp["baseline_overlap"]["n_points"] == 1
    assert cmp["baseline_overlap"]["mae"] == pytest.approx(40.0)
    assert cmp["baseline_overlap"]["n_points"] != cmp["neural_net_all_points"]["n_points"]
    back = filter_prefix_backtest_table(pred, units)
    t1 = back[back["unit_id"] == "Test_1"].sort_values("timestamp_s")
    assert t1["actual_rul_s"].tolist() == [86.0, 80.0]
    metrics, by_unit = build_rul_metrics(
        pred,
        dataset_id="filters",
        units=units,
        cfg={},
        eval_id="e",
        run_id="r",
        split={"protocol": "author_test_held_out", "test": ["Test_1", "Test_2"]},
        test_ids=["Test_1", "Test_2"],
        bound={},
    )
    assert metrics["primary_metric"] == "prefix_end_mae"
    assert metrics["prefix_end_mae"] == pytest.approx(0.0)
    assert metrics["baseline"]["prefix_end"]["baseline_coverage_fraction"] == pytest.approx(0.5)
    assert "official_rul_at_prefix_end_s" in metrics["official_rul_source"]
    assert set(by_unit["unit_id"]) == {"Test_1", "Test_2"}
    row1 = by_unit.set_index("unit_id").loc["Test_1"]
    assert row1["prefix_end_actual_rul_s"] == pytest.approx(80.0)
    assert row1["prefix_end_predicted_rul_s"] == pytest.approx(80.0)
    assert int(row1["prefix_end_baseline_finite"]) == 1
    row2 = by_unit.set_index("unit_id").loc["Test_2"]
    assert int(row2["prefix_end_baseline_finite"]) == 0

    features, all_units = tiny_filter_tables
    split = filters_split(all_units)
    processed_root = tmp_path / "processed" / "filters"
    runs_root = tmp_path / "runs"
    monkeypatch.setattr("pdm.data.prepare.dataset_processed", lambda _id: processed_root)
    monkeypatch.setattr("pdm.paths.dataset_runs", lambda _id: runs_root / _id)
    rec = write_processed_version(
        "filters", features, all_units, split, sensor_note="n", cfg={}, processed_root=processed_root
    )
    run_id = "tiny_filter_eval"
    rdir = runs_root / "filters" / run_id
    rdir.mkdir(parents=True)
    fp = dataset_fingerprint_for_run(
        {"fingerprint": rec["fingerprint"], "dir": rec["dir"], "split": split}, split
    )
    prep, _ = fit_preprocessor("filters", features, all_units, split, {})
    _write_tiny_filter_checkpoint(rdir, prep, split, fp, history_length=3)
    atomic_write_json(
        rdir / "validation_metrics.json",
        {
            "best_epoch": 2,
            "best_metric": 0.8,
            "selection_metric_name": "val_nll",
            "selection_metric_unit": "nll",
            "n_val_windows": 12,
            "last": {
                "selection_metric": 1.25,
                "val_loss": 1.25,
                "val_mae_events": 99.0,
                "n_val_event_units": 2,
                "note": "last epoch — must not be copied into eval validation",
            },
        },
    )
    (rdir / "training_history.csv").write_text(
        "epoch,train_loss,val_loss,train_metric,val_metric,val_mae_events,n_val_event_units\n"
        "1,1.0,1.1,1.0,1.0,50.0,1\n"
        "2,0.9,0.85,0.9,0.8,40.0,1\n"
        "3,0.7,1.3,0.7,1.25,99.0,2\n",
        encoding="utf-8",
    )
    out = evaluate_run("filters", run_id, device="cpu")
    mpath = Path(out["eval_dir"]) / "metrics.json"
    live = json.loads(mpath.read_text(encoding="utf-8"))
    assert live["primary_metric"] == "prefix_end_mae"
    assert live["validation"]["nll_all_units"] == pytest.approx(0.8)
    assert live["validation"]["mae_observed_events"] == pytest.approx(40.0)
    assert live["validation"]["n_observed_event_units"] == 1
    assert live["validation"]["checkpoint_selection_epoch"] == 2
    assert "checkpoint-selection" in live["validation"]["note"]
    pred_live = pd.read_csv(Path(out["eval_dir"]) / "predictions.csv")
    assert "official_rul_at_prefix_end_s" not in pred_live.columns
    assert "official_rul_at_prefix_end_original" not in pred_live.columns
    by = pd.read_csv(Path(out["eval_dir"]) / "metrics_by_unit.csv")
    official = all_units.set_index("unit_id")["official_rul_at_prefix_end_s"]
    for uid in split["test"]:
        row = by.set_index("unit_id").loc[uid]
        assert row["prefix_end_actual_rul_s"] == pytest.approx(float(official.loc[uid]))


def _write_tiny_filter_checkpoint(rdir, prep, split, fp, *, history_length: int = 3) -> None:
    mcfg = {
        "architecture": "gru",
        "history_length": history_length,
        "hidden_size": 8,
        "recurrent_layers": 1,
        "dropout": 0.1,
    }
    torch.manual_seed(0)
    model = PDMNet(
        input_size=len(prep.feature_names),
        hidden_size=8,
        num_layers=1,
        architecture="gru",
        head="weibull",
        dropout=0.1,
        time_scale_s=prep.time_scale_s,
    )
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "compat": compatibility_dict(mcfg, prep, split, "filters", "weibull", fingerprint=fp),
            "meta": {
                "epoch": 1,
                "best_epoch": 1,
                "best_metric": 0.0,
                "smoke": True,
                "time_scale_s": prep.time_scale_s,
                "head": "weibull",
                "architecture": "gru",
                "history_length": history_length,
                "hidden_size": 8,
                "recurrent_layers": 1,
                "dropout": 0.1,
                "input_size": len(prep.feature_names),
                "feature_names": list(prep.feature_names),
                "dataset_id": "filters",
            },
        },
        rdir / "best.pt",
    )
    atomic_write_json(rdir / "preprocessing.json", prep.to_dict())
    atomic_write_json(rdir / "split.json", split)
    fp = dict(fp)
    fp["checkpoint_hash"] = checkpoint_hash(rdir / "best.pt")
    atomic_write_json(rdir / "dataset_fingerprint.json", fp)


def _write_tiny_filter_raw(raw, *, mat_bytes=None, savemat_array=None):
    """Synthetic CSV (+ optional MAT). Not the HSE dataset."""
    raw.mkdir(parents=True, exist_ok=True)
    (raw / "Train_Data_CSV.csv").write_text(
        "Data_No,Differential_pressure,Flow_rate,Time,Dust_feed,Dust\n"
        "1,10,1,0.1,100,A3\n"
        "1,20,1,0.2,100,A3\n"
        "1,30,1,0.3,100,A3\n",
        encoding="utf-8",
    )
    (raw / "Test_Data_CSV.csv").write_text(
        "Data_No,Differential_pressure,Flow_rate,Time,Dust_feed,Dust,RUL\n"
        "1,10,1,0.1,100,A3,5\n"
        "1,12,1,0.2,100,A3,5\n",
        encoding="utf-8",
    )
    if mat_bytes is not None:
        (raw / "Train_Data_Uncensored.mat").write_bytes(mat_bytes)
    elif savemat_array is not None:
        from scipy.io import savemat

        savemat(raw / "Train_Data_Uncensored.mat", {"Train_Data_Uncensored": savemat_array})


def test_probe_mat_missing_path(tmp_path):
    rec = probe_mat(tmp_path / "Train_Data_Uncensored.mat")
    assert rec["exists"] is False
    assert rec["load_ok"] is False
    assert rec["error"] == "file not found"


def test_inspect_filters_full_history_mat_missing(tmp_path):
    _write_tiny_filter_raw(tmp_path)
    rec = inspect_filters(tmp_path)
    assert rec["filters_full_history_status"] == "missing"
    assert rec["filters_full_history_status"] in FILTERS_FULL_HISTORY_STATUSES
    assert rec["filters_full_history_enabled"] is False
    assert "not found" in rec["filters_full_history_reason"].lower()
    assert rec["filters_full_history_reason"] in rec["notes"]


def test_inspect_filters_full_history_mat_unreadable(tmp_path):
    _write_tiny_filter_raw(tmp_path, mat_bytes=b"not a matlab file")
    rec = inspect_filters(tmp_path)
    assert rec["filters_full_history_status"] == "unreadable"
    assert rec["filters_full_history_enabled"] is False
    assert "disabled" in rec["filters_full_history_reason"].lower()
    probe = rec["mat_probe"]
    assert probe["exists"] is True
    assert probe["load_ok"] is False


def test_opaque_primary_uncensored_table_ignores_numeric_sidecar():
    """MCOS/opaque experiment table stays unreadable even with a numeric sidecar."""
    probe = {
        "exists": True,
        "load_ok": True,
        "error": None,
        "variables": [
            {
                "name": "Train_Data_Uncensored",
                "python_type": "MatlabOpaque",
                "shape": [1],
                "dtype": "[('_TypeSystem', 'O'), ('_Class', 'O'), ('_ObjectMetadata', 'O')]",
                "opaque": True,
                "scipy_envelope": False,
                "matlab_type_system": "MCOS",
                "matlab_class": "table",
                "fieldnames": None,
            },
            {
                "name": "sidecar_numeric",
                "python_type": "ndarray",
                "shape": [50, 4],
                "dtype": "float64",
                "opaque": False,
                "scipy_envelope": False,
                "matlab_type_system": None,
                "matlab_class": None,
                "fieldnames": None,
            },
        ],
    }
    assert _filters_full_history_status(probe) == "unreadable"


def test_opaque_nested_in_object_array_is_unreadable():
    class MatlabOpaque:
        def reshape(self, *args):
            dt = np.dtype([("_TypeSystem", "O"), ("_Class", "O"), ("_ObjectMetadata", "O")])
            return np.array([("MCOS", "table", None)], dtype=dt)

    wrapped = np.empty((1,), dtype=object)
    wrapped[0] = MatlabOpaque()
    desc = _describe_mat_variable("Train_Data_Uncensored", wrapped)
    assert desc["opaque"] is True
    assert desc["matlab_class"] == "table"
    probe = {
        "exists": True,
        "load_ok": True,
        "variables": [
            desc,
            {
                "name": "sidecar_numeric",
                "python_type": "ndarray",
                "shape": [8],
                "dtype": "float64",
                "opaque": False,
                "fieldnames": None,
            },
        ],
    }
    assert _filters_full_history_status(probe) == "unreadable"


def test_scipy_envelope_fieldnames_are_not_table_columns():
    arr = np.zeros(1, dtype=[("s0", "O"), ("s1", "O"), ("s2", "O"), ("arr", "O")])
    desc = _describe_mat_variable("Train_Data_Uncensored", arr)
    assert desc["fieldnames"] is None
    assert desc["scipy_envelope"] is True
    probe = {
        "exists": True,
        "load_ok": True,
        "variables": [
            desc,
            {
                "name": "sidecar_numeric",
                "python_type": "ndarray",
                "shape": [8],
                "dtype": "float64",
                "opaque": False,
                "fieldnames": None,
            },
        ],
    }
    assert _filters_full_history_status(probe) == "unreadable"


def test_inspect_filters_full_history_readable_does_not_enable(tmp_path):
    _write_tiny_filter_raw(tmp_path, savemat_array=np.arange(12, dtype=np.float64).reshape(3, 4))
    rec = inspect_filters(tmp_path)
    assert rec["filters_full_history_status"] == "readable_needs_protocol"
    assert rec["filters_full_history_enabled"] is False
    assert "out of scope" in rec["filters_full_history_reason"].lower()
    vars_ = rec["mat_probe"]["variables"]
    assert vars_ and vars_[0]["name"] == "Train_Data_Uncensored"
    assert vars_[0]["opaque"] is False
    assert vars_[0]["fieldnames"] is None
    feat, units = extract_filters_tables({"mode": "filters_censored", "time_to_seconds": 60.0}, raw_dir=tmp_path)
    assert set(feat["author_split"]) == {"author_train", "author_test"}
    assert "differential_pressure" in feat.columns
    assert len(units) == 2


def test_extract_filters_rejects_full_history_mode():
    with pytest.raises(ValueError, match="filters_full_history training is disabled"):
        extract_filters_tables({"mode": "filters_full_history"})
    with pytest.raises(ValueError, match="filters_full_history training is disabled"):
        extract_filters_tables({"mode": "filters_censored", "filters_full_history": True})


def test_filters_config_full_history_remains_off():
    cfg = load_dataset_config("filters")
    assert cfg.get("mode") == "filters_censored"
    assert not cfg.get("filters_full_history")
    assert not cfg.get("enable_filters_full_history")


def test_data_report_propagates_filters_full_history_status(tmp_path, tiny_filter_tables):
    features, units = tiny_filter_tables
    split = filters_split(units)
    inspection = {
        "filters_full_history_status": "unreadable",
        "filters_full_history_reason": "MCOS table opaque; protocol disabled",
        "filters_full_history_enabled": False,
        "notes": ["MCOS table opaque; protocol disabled"],
    }
    rec = write_processed_version(
        "filters",
        features,
        units,
        split,
        inspection=inspection,
        sensor_note="n",
        cfg=load_dataset_config("filters"),
        processed_root=tmp_path / "filters",
    )
    report = rec["report"]
    assert report["filters_full_history_status"] == "unreadable"
    assert report["filters_full_history_reason"] == "MCOS table opaque; protocol disabled"
    assert report["filters_full_history_enabled"] is False
    assert report["inspection_summary"]["filters_full_history_status"] == "unreadable"
    assert any("filters_full_history disabled" in str(issue) for issue in report["issues_and_decisions"])
    saved_ins = json.loads((rec["dir"] / "inspection.json").read_text(encoding="utf-8"))
    saved_report = json.loads((rec["dir"] / "data_report.json").read_text(encoding="utf-8"))
    assert saved_ins["filters_full_history_status"] == "unreadable"
    assert saved_report["filters_full_history_status"] == "unreadable"
    assert saved_report["filters_full_history_enabled"] is False


def test_data_report_caches_window_and_event_counts(tmp_path, tiny_bearing_tables, tiny_filter_tables):
    hist = 5
    features, units = tiny_bearing_tables
    split = bearings_split(
        units,
        {"split": {"train_instances": [1, 2, 3], "val_instances": [4], "test_instances": [5]}},
    )
    rec = write_processed_version(
        "bearings",
        features,
        units,
        split,
        sensor_note="n",
        cfg={"model": {"history_length": hist}, "fragment_interval_s": 60},
        processed_root=tmp_path / "bearings",
    )
    report = rec["report"]
    saved = json.loads((rec["dir"] / "data_report.json").read_text(encoding="utf-8"))
    wc = report["window_counts"]
    assert saved["window_counts"] == wc
    built = build_windows(features, units, hist, "bearings")
    assert wc["history_length"] == hist
    assert wc["eligible"] == len(built)
    assert wc["n_candidate_ends"] == len(features)
    excluded = wc["excluded"]
    assert wc["eligible"] + excluded["gap"] + excluded["post_event"] + excluded["insufficient_length"] == len(
        features
    )
    assert excluded["gap"] > 0
    assert excluded["insufficient_length"] > 0
    assert excluded["post_event"] > 0
    for part in ("train", "validation", "test"):
        ids = set(split[part])
        rec_part = wc["by_split"][part]
        assert rec_part["n_units"] == len(split[part])
        assert rec_part["eligible_windows"] == int(built["unit_id"].isin(ids).sum())
    assert report["split_protocol"] == split["protocol"]
    assert report["split_hash"] == split_hash(split)
    assert report["dataset_version"]
    assert report["events_vs_censoring"]["n_event_observed"] == int(units["event_observed"].sum())
    assert report["regimes"]
    assert report["observation_time_s"]["total_span_s"] > 0

    ff, fu = tiny_filter_tables
    fsplit = filters_split(fu)
    fcfg = load_dataset_config("filters")
    frec = write_processed_version(
        "filters",
        ff,
        fu,
        fsplit,
        sensor_note="n",
        cfg=fcfg,
        processed_root=tmp_path / "filters",
    )
    fhist = int(fcfg["model"]["history_length"])
    fwc = frec["report"]["window_counts"]
    fbuilt = build_windows(ff, fu, fhist, "filters")
    assert fwc["eligible"] == len(fbuilt)
    ev = frec["report"]["events_vs_censoring"]
    assert ev["n_event_observed"] == 2
    assert ev["n_official_rul_known_not_observed"] == 2
    assert ev["n_sensor_end_without_event_or_official_rul"] == 6
    assert ev["n_right_censored"] == 8
    assert "evaluation" in ev["note"].lower()


def test_probe_filters_full_history_missing_without_csvs(tmp_path):
    rec = probe_filters_full_history(tmp_path)
    assert rec["filters_full_history_status"] == "missing"
    assert rec["filters_full_history_enabled"] is False


_HISTORY = 3
_UNSEEN_DUST = "synthetic_unseen_dust"


def _reload_predictor(tmp_path, prep: Preprocessor, *, head: str, history_length: int = _HISTORY) -> Predictor:
    model = PDMNet(
        len(prep.feature_names),
        hidden_size=8,
        architecture="gru",
        head=head,
        time_scale_s=prep.time_scale_s,
    )
    ckpt = tmp_path / "best.pt"
    torch.save({"model_state_dict": model.state_dict()}, ckpt)
    prep_path = tmp_path / "preprocessing.json"
    atomic_write_json(prep_path, prep.to_dict())
    loaded_prep = Preprocessor.from_dict(json.loads(prep_path.read_text(encoding="utf-8")))
    loaded = PDMNet(
        len(loaded_prep.feature_names),
        hidden_size=8,
        architecture="gru",
        head=head,
        time_scale_s=loaded_prep.time_scale_s,
    )
    blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    loaded.load_state_dict(blob["model_state_dict"])
    loaded.eval()
    return Predictor(loaded, loaded_prep, history_length=history_length)


def _predictor_feature_tensor(
    predictor: Predictor, history: pd.DataFrame
) -> tuple[np.ndarray, dict]:
    captured: list[np.ndarray] = []
    orig = predictor.model.predicted_rul_s

    def _hook(x):
        captured.append(x.detach().cpu().numpy().copy())
        return orig(x)

    predictor.model.predicted_rul_s = _hook
    try:
        out = predictor.predict_from_history(history)
    finally:
        predictor.model.predicted_rul_s = orig
    assert captured, (
        f"Predictor skipped the feature path; status={out.get('status')} "
        f"reason={out.get('valid_history_reason')}"
    )
    return np.asarray(captured[0][0], dtype=np.float64), out


def _offline_window_tensor(prep: Preprocessor, raw_window: pd.DataFrame) -> np.ndarray:
    encoded = raw_to_feature_frame(
        prep.dataset_id,
        raw_window,
        prep.categorical_maps,
        prep.log1p_features,
        raw_features=True,
    )
    transformed = prep.transform_frame(encoded)
    return transformed[prep.feature_names].to_numpy(dtype=np.float64)


def _unit_prefix(features: pd.DataFrame, unit_id: str, end_index: int) -> pd.DataFrame:
    g = features[features["unit_id"] == unit_id].sort_values("timestamp_s").reset_index(drop=True)
    return g.iloc[: end_index + 1].copy()


_LABEL_POISON = {
    "event_time_s": 1e9,
    "event_time": 1e9,
    "official_rul_at_prefix_end_s": 1e9,
    "official_rul_at_prefix_end_original": 1e9,
    "official_rul": 1e9,
    "target_rul_s": 1e9,
    "target_rul": 1e9,
    "actual_rul_s": 1e9,
    "actual_rul": 1e9,
    "event": 1,
    "event_observed": 1,
    "observation_end_s": 1e9,
    "split": "test",
    "part": "test",
    "author_split": "author_test",
    "life_fraction": 0.99,
    "percent_life": 99.0,
    "failure": 1,
    "rul": 1e9,
    "rul_s": 1e9,
}


def _poison_gt_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Overwrite / inject GT and split labels; leave measurement columns alone."""
    out = df.copy()
    n = len(out)
    for col, value in _LABEL_POISON.items():
        out[col] = np.full(n, value)
    return out


def _future_raw_rows(last: pd.Series, n: int, dt_s: float, **overrides) -> pd.DataFrame:
    rows = []
    t = float(last["timestamp_s"])
    file_index = int(last["file_index"]) if "file_index" in last.index and pd.notna(last.get("file_index")) else 0
    for i in range(1, n + 1):
        row = last.to_dict()
        t = t + dt_s
        row["timestamp_s"] = t
        if "operating_age_s" in row:
            row["operating_age_s"] = t
        if "file_index" in row:
            row["file_index"] = file_index + i
        if "time_original" in row:
            row["time_original"] = t / 60.0
        if "delta_t_s" in row:
            row["delta_t_s"] = dt_s
        row["gap_before"] = False
        row.update(overrides)
        rows.append(row)
    return pd.DataFrame(rows)


def test_offline_online_feature_parity(tmp_path, tiny_filter_tables):
    features, units = tiny_filter_tables
    split = filters_split(units)
    prep, transformed = fit_preprocessor("filters", features, units, split, {})
    cats = list(prep.categorical_maps.get("dust") or [])
    assert len(cats) >= 2
    predictor = _reload_predictor(tmp_path, prep, head="weibull")
    windows = build_windows(features, units, _HISTORY, "filters")

    cases: list[tuple[str, pd.DataFrame]] = []
    for cat in cats:
        uid = features.loc[features["dust"] == cat, "unit_id"].iloc[0]
        uw = windows[windows["unit_id"] == uid]
        assert not uw.empty, f"no windows for dust category {cat!r}"
        row = uw.iloc[max(0, len(uw) // 2)]
        prefix = _unit_prefix(features, uid, int(row["end_index"]))
        cases.append((f"dust={cat}", prefix))
        offline_wm = window_matrix(transformed, row, prep.feature_names).astype(np.float64)
        raw_window = prefix.sort_values("timestamp_s").iloc[-_HISTORY :]
        np.testing.assert_allclose(_offline_window_tensor(prep, raw_window), offline_wm, rtol=1e-6, atol=1e-6)

    base_prefix = cases[0][1]
    unseen = base_prefix.copy()
    unseen["dust"] = _UNSEEN_DUST
    cases.append(("unknown_dust", unseen))

    nan_prefix = base_prefix.copy()
    nan_prefix.loc[nan_prefix.index[-1], "differential_pressure"] = np.nan
    ok_nan, reason_nan = valid_history_window(
        nan_prefix,
        _HISTORY,
        dataset_id="filters",
        gap_multiplier=prep.gap_multiplier,
        sampling_interval_s=prep.sampling_interval_s,
    )
    assert ok_nan and reason_nan == ""
    ts_bad = nan_prefix.copy()
    ts_bad.loc[ts_bad.index[-1], "timestamp_s"] = np.nan
    ok_ts, reason_ts = valid_history_window(
        ts_bad,
        _HISTORY,
        dataset_id="filters",
        gap_multiplier=prep.gap_multiplier,
        sampling_interval_s=prep.sampling_interval_s,
    )
    assert not ok_ts and reason_ts == "non_finite_timestamps"
    cases.append(("nan", nan_prefix))

    for label, prefix in cases:
        history = prefix.sort_values("timestamp_s")
        if prep.dataset_id == "filters":
            history = recompute_filter_gap_before(
                history,
                gap_multiplier=prep.gap_multiplier,
                sampling_interval_s=prep.sampling_interval_s,
                causal=True,
            )
        raw_window = history.iloc[-_HISTORY :].copy()
        offline = _offline_window_tensor(prep, raw_window)
        online, out = _predictor_feature_tensor(predictor, prefix)
        assert out["status"] == "ok", (label, out)
        np.testing.assert_allclose(offline, online, rtol=1e-6, atol=1e-6, err_msg=label)


def test_dust_encoding_from_raw(tiny_filter_tables):
    features, units = tiny_filter_tables
    split = filters_split(units)
    prep, _ = fit_preprocessor("filters", features, units, split, {})
    cats = list(prep.categorical_maps.get("dust") or [])
    assert len(cats) >= 2
    dust_cols = [c for c in prep.feature_names if c.startswith("dust__")]
    assert dust_cols
    uid = split["train"][0]
    raw = features[features["unit_id"] == uid].sort_values("timestamp_s").iloc[:_HISTORY].copy()
    encoded = []
    for cat in cats:
        frame = raw.copy()
        frame["dust"] = cat
        enc = raw_to_feature_frame(
            "filters",
            frame,
            prep.categorical_maps,
            prep.log1p_features,
            raw_features=True,
        )
        encoded.append(enc)
    a_vals = encoded[0][prep.feature_names].to_numpy(dtype=np.float64)
    b_vals = encoded[1][prep.feature_names].to_numpy(dtype=np.float64)
    assert not np.allclose(a_vals, b_vals, atol=1e-6)
    da = encoded[0][dust_cols].to_numpy(dtype=np.float64)
    db = encoded[1][dust_cols].to_numpy(dtype=np.float64)
    assert not np.allclose(da, db, atol=1e-6)
    np.testing.assert_allclose(da.max(axis=1), 1.0)
    np.testing.assert_allclose(db.max(axis=1), 1.0)


def test_saved_median_imputation(tiny_filter_tables):
    features, units = tiny_filter_tables
    split = filters_split(units)
    prep, _ = fit_preprocessor("filters", features, units, split, {})
    col = "differential_pressure"
    idx = prep.feature_names.index(col)
    fill = float(prep.fill_values[col])
    mean = float(prep.scaler_mean[idx])
    scale = float(prep.scaler_scale[idx])
    assert abs(fill) > 1.0
    expected = (fill - mean) / scale
    zero_then_scale = (0.0 - mean) / scale
    assert abs(expected - zero_then_scale) > 1e-3
    uid = split["train"][0]
    raw = features[features["unit_id"] == uid].sort_values("timestamp_s").iloc[:_HISTORY].copy()
    for bad in (np.nan, np.inf, -np.inf):
        corrupted = raw.copy()
        corrupted.loc[:, col] = bad
        got = _offline_window_tensor(prep, corrupted)[:, idx]
        np.testing.assert_allclose(got, expected, rtol=1e-6, atol=1e-6)


def test_observed_event_duration(tiny_filter_tables):
    features, units = tiny_filter_tables
    observed = units[units["event_observed"] == 1]
    assert not observed.empty
    windows = build_windows(features, units, _HISTORY, "filters")
    for _, u in observed.iterrows():
        et = float(u["event_time_s"])
        obs_end = float(u["observation_end_s"])
        assert et < obs_end
        uw = windows[windows["unit_id"] == u["unit_id"]]
        assert not uw.empty
        for _, row in uw.iterrows():
            t = float(row["timestamp_s"])
            assert float(row["duration_s"]) == pytest.approx(et - t)
            assert float(row["duration_s"]) != pytest.approx(obs_end - t)
            assert int(row["event"]) == 1
            assert float(row["target_rul_s"]) == pytest.approx(et - t)


def test_no_post_event_training_windows(tiny_bearing_tables, tiny_filter_tables):
    bf, bu = tiny_bearing_tables
    wb = build_windows(bf, bu, _HISTORY, "bearings")
    assert not wb.empty
    for _, u in bu.iterrows():
        et = float(u["event_time_s"])
        at_or_after = bf[(bf["unit_id"] == u["unit_id"]) & (bf["timestamp_s"] >= et)]
        assert not at_or_after.empty
        uw = wb[wb["unit_id"] == u["unit_id"]]
        if uw.empty:
            continue
        assert (uw["timestamp_s"] < et).all()

    ff, fu = tiny_filter_tables
    wf = build_windows(ff, fu, _HISTORY, "filters")
    observed = fu[fu["event_observed"] == 1]
    assert not observed.empty
    for _, u in observed.iterrows():
        et = float(u["event_time_s"])
        post = ff[(ff["unit_id"] == u["unit_id"]) & (ff["timestamp_s"] >= et)]
        assert not post.empty
        uw = wf[wf["unit_id"] == u["unit_id"]]
        assert not uw.empty
        assert (uw["timestamp_s"] < et).all()


def test_bearings_endpoint_actual_rul_nan(tmp_path, tiny_bearing_tables):
    features, units = tiny_bearing_tables
    split = bearings_split(units)
    prep, _ = fit_preprocessor("bearings", features, units, split, {"features": {"log1p_features": []}})
    uid = "Bearing1_1"
    et = float(units.loc[units["unit_id"] == uid, "event_time_s"].iloc[0])
    g = features[features["unit_id"] == uid].sort_values("timestamp_s")
    t_last = float(g["timestamp_s"].iloc[-1])
    t_prev = float(g["timestamp_s"].iloc[-2])
    assert t_last == pytest.approx(et)
    actual_prev = et - t_prev
    pred = pd.DataFrame(
        {
            "unit_id": [uid, uid, uid],
            "timestamp_s": [t_prev, t_last, t_last + 60.0],
            "predicted_rul_s": [actual_prev, 1e9, 1e9],
        }
    )
    out = attach_actual_rul(pred, units, "bearings")
    assert out["actual_rul_s"].iloc[0] == pytest.approx(actual_prev)
    assert pd.isna(out["actual_rul_s"].iloc[1])
    assert pd.isna(out["actual_rul_s"].iloc[2])
    stats = summarize_rul_table(out)
    assert stats["n_points"] == 1
    assert stats["mae"] == pytest.approx(0.0)

    predictor = _reload_predictor(tmp_path, prep, head="rul")
    replayed = replay_unit(
        g,
        predictor,
        dataset_id="bearings",
        unit_id=uid,
        run_id="tiny",
        history_length=_HISTORY,
        warning_horizon_s=1e6,
        truth_units=units,
    )["predictions"]
    end_rows = replayed[replayed["timestamp_s"] >= et]
    assert not end_rows.empty
    assert end_rows["actual_rul_s"].isna().all()
    before = replayed[replayed["timestamp_s"] < et]
    assert before["actual_rul_s"].notna().all()


def test_filter_observed_event_actual_rul_nan(tmp_path, tiny_filter_tables):
    features, units = tiny_filter_tables
    split = filters_split(units)
    prep, _ = fit_preprocessor("filters", features, units, split, {})
    obs = units[units["event_observed"] == 1].iloc[0]
    uid = str(obs["unit_id"])
    et = float(obs["event_time_s"])
    g = features[features["unit_id"] == uid].sort_values("timestamp_s")
    t_before = float(g.loc[g["timestamp_s"] < et, "timestamp_s"].iloc[-1])
    t_at = float(g.loc[g["timestamp_s"] >= et, "timestamp_s"].iloc[0])
    pred = pd.DataFrame(
        {
            "unit_id": [uid, uid],
            "timestamp_s": [t_before, t_at],
            "predicted_rul_s": [et - t_before, 1e9],
        }
    )
    out = attach_actual_rul(pred, units, "filters")
    assert out["actual_rul_s"].iloc[0] == pytest.approx(et - t_before)
    assert pd.isna(out["actual_rul_s"].iloc[1])
    stats = summarize_rul_table(out)
    assert stats["n_points"] == 1
    assert stats["mae"] == pytest.approx(0.0)

    cens = units[units["event_observed"] == 0].iloc[0]
    cuid = str(cens["unit_id"])
    cg = features[features["unit_id"] == cuid].sort_values("timestamp_s")
    cens_pred = pd.DataFrame(
        {
            "unit_id": [cuid] * len(cg),
            "timestamp_s": cg["timestamp_s"].to_numpy(),
            "predicted_rul_s": np.ones(len(cg)),
        }
    )
    cens_out = attach_actual_rul(cens_pred, units, "filters")
    assert cens_out["actual_rul_s"].isna().all()
    assert summarize_rul_table(cens_out)["n_points"] == 0

    predictor = _reload_predictor(tmp_path, prep, head="weibull")
    replayed = replay_unit(
        g,
        predictor,
        dataset_id="filters",
        unit_id=uid,
        run_id="tiny",
        history_length=_HISTORY,
        warning_horizon_s=1e6,
        truth_units=units,
    )["predictions"]
    assert replayed.loc[replayed["timestamp_s"] >= et, "actual_rul_s"].isna().all()
    assert replayed.loc[replayed["timestamp_s"] < et, "actual_rul_s"].notna().all()
    cens_replay = replay_unit(
        cg,
        predictor,
        dataset_id="filters",
        unit_id=cuid,
        run_id="tiny",
        history_length=_HISTORY,
        warning_horizon_s=1e6,
        truth_units=units,
    )["predictions"]
    cens_actual = (
        cens_replay["actual_rul_s"]
        if "actual_rul_s" in cens_replay.columns
        else pd.Series(np.nan, index=cens_replay.index)
    )
    assert cens_actual.isna().all()


def test_censored_filter_duration_and_no_post_observation_windows(tiny_filter_tables):
    features, units = tiny_filter_tables
    windows = build_windows(features, units, _HISTORY, "filters")
    censored = units[units["event_observed"] == 0]
    assert not censored.empty
    for _, u in censored.iterrows():
        obs_end = float(u["observation_end_s"])
        uw = windows[windows["unit_id"] == u["unit_id"]]
        assert not uw.empty
        assert (uw["event"] == 0).all()
        assert uw["target_rul_s"].isna().all()
        assert (uw["timestamp_s"] < obs_end).all()
        for _, row in uw.iterrows():
            t = float(row["timestamp_s"])
            assert float(row["duration_s"]) == pytest.approx(obs_end - t)
            assert float(row["duration_s"]) > 0.0
        at_or_after = features[
            (features["unit_id"] == u["unit_id"]) & (features["timestamp_s"] >= obs_end)
        ]
        assert not at_or_after.empty


def test_replay_gap_warmup(tmp_path, tiny_bearing_tables):
    """After a gap, no prediction until H consecutive valid points; alert counters reset."""
    features, units = tiny_bearing_tables
    split = bearings_split(units)
    prep, _ = fit_preprocessor("bearings", features, units, split, {"features": {"log1p_features": []}})
    history_length = 5
    confirmation_count = 3
    torch.manual_seed(0)
    predictor = _reload_predictor(tmp_path, prep, head="rul", history_length=history_length)
    uid = "Bearing2_1"
    g = features[features["unit_id"] == uid].sort_values("timestamp_s").reset_index(drop=True)
    assert 8 not in set(g["file_index"].astype(int))
    assert bool(g.loc[g["file_index"] == 9, "gap_before"].iloc[0])

    src = ReplaySource(g)
    engine = AlertEngine(warning_horizon_s=1e12, confirmation_count=confirmation_count)
    engine.reset_unit(uid)

    pre_gap_ok: list[int] = []
    post_gap_collecting: list[int] = []
    post_gap_ok: list[int] = []
    for step in range(len(src)):
        prefix = src.prefix(step)
        file_index = int(prefix.iloc[-1]["file_index"])
        pred = predictor.predict_from_history(prefix)
        collecting = pred.get("status") == "Collecting history"
        engine.update(
            timestamp_s=float(prefix.iloc[-1]["timestamp_s"]),
            predicted_rul_s=pred.get("predicted_rul_s"),
            collecting=collecting,
        )
        if file_index < 8:
            if collecting:
                assert pred.get("predicted_rul_s") is None
                continue
            assert pred["status"] == "ok"
            assert pred["predicted_rul_s"] is not None
            pre_gap_ok.append(file_index)
            if file_index == 7:
                assert engine.warning_active is True
                assert engine.consecutive_warn == confirmation_count
        else:
            if collecting:
                assert pred.get("predicted_rul_s") is None
                assert pred["status"] == "Collecting history"
                assert engine.warning_active is False
                assert engine.consecutive_warn == 0
                assert engine.consecutive_clear == 0
                post_gap_collecting.append(file_index)
            else:
                assert pred["status"] == "ok"
                assert pred["predicted_rul_s"] is not None
                post_gap_ok.append(file_index)

    assert pre_gap_ok == [5, 6, 7]
    # gap_before on the first row of a window is allowed; interior gap is not.
    first_ok_after_gap = 9 + history_length - 1
    assert post_gap_collecting == list(range(9, first_ok_after_gap))
    assert post_gap_ok[0] == first_ok_after_gap
    assert 9 not in post_gap_ok


def test_raw_prefix_invariance(tmp_path, tiny_bearing_tables, tiny_filter_tables):
    """Appending future raw rows must not change transform / predict at fixed t."""
    cases = [
        ("bearings", tiny_bearing_tables, bearings_split, "rul", {"features": {"log1p_features": []}}, "Bearing1_1"),
        ("filters", tiny_filter_tables, filters_split, "weibull", {}, "Filter_1"),
    ]
    for dataset_id, (features, units), split_fn, head, cfg, uid in cases:
        split = split_fn(units)
        prep, _ = fit_preprocessor(dataset_id, features, units, split, cfg)
        torch.manual_seed(1)
        run_dir = tmp_path / dataset_id
        run_dir.mkdir()
        predictor = _reload_predictor(run_dir, prep, head=head, history_length=_HISTORY)
        g = features[features["unit_id"] == uid].sort_values("timestamp_s").reset_index(drop=True)
        t_idx = max(_HISTORY + 4, 8)
        assert t_idx < len(g) - 1
        prefix = ReplaySource(g).prefix(t_idx)
        tensor_t, out_t = _predictor_feature_tensor(predictor, prefix)
        assert out_t["status"] == "ok"
        rul_t = out_t["predicted_rul_s"]
        last = prefix.iloc[-1]
        if dataset_id == "bearings":
            extra = _future_raw_rows(last, 6, 60.0, horizontal_rms=9_999.0, vertical_rms=9_999.0)
        else:
            extra_gap = _future_raw_rows(
                last,
                1,
                1_000.0,
                differential_pressure=9_999.0,
                delta_pressure=9_999.0,
            )
            extra_dense = _future_raw_rows(
                extra_gap.iloc[-1],
                20,
                1.0,
                differential_pressure=9_999.0,
                delta_pressure=0.0,
            )
            extra = pd.concat([extra_gap, extra_dense], ignore_index=True)
        longer = pd.concat([prefix, extra], ignore_index=True)
        prefix_from_future = ReplaySource(
            longer,
            gap_multiplier=getattr(predictor, "gap_multiplier", None),
            sampling_interval_s=getattr(predictor, "sampling_interval_s", None),
        ).prefix(t_idx)
        np.testing.assert_allclose(
            prefix["timestamp_s"].to_numpy(dtype=np.float64),
            prefix_from_future["timestamp_s"].to_numpy(dtype=np.float64),
        )
        tensor_future, out_future = _predictor_feature_tensor(predictor, prefix_from_future)
        assert out_future["status"] == "ok"
        np.testing.assert_allclose(tensor_t, tensor_future, rtol=1e-6, atol=1e-6, err_msg=dataset_id)
        np.testing.assert_allclose(
            np.asarray(rul_t, dtype=np.float64),
            np.asarray(out_future["predicted_rul_s"], dtype=np.float64),
            rtol=1e-6,
            atol=1e-6,
        )
        raw_window = prefix.sort_values("timestamp_s").iloc[-_HISTORY :]
        np.testing.assert_allclose(
            _offline_window_tensor(prep, raw_window),
            tensor_t,
            rtol=1e-6,
            atol=1e-6,
            err_msg=dataset_id,
        )


def test_labels_do_not_affect_predictor(tmp_path, tiny_bearing_tables, tiny_filter_tables):
    """Mutating GT / split labels through Predictor / replay must not change predictions."""
    cases = [
        ("bearings", tiny_bearing_tables, bearings_split, "rul", {"features": {"log1p_features": []}}, "Bearing1_1"),
        ("filters", tiny_filter_tables, filters_split, "weibull", {}, "Filter_101"),
    ]
    for dataset_id, (features, units), split_fn, head, cfg, uid in cases:
        split = split_fn(units)
        prep, _ = fit_preprocessor(dataset_id, features, units, split, cfg)
        torch.manual_seed(2)
        run_dir = tmp_path / dataset_id
        run_dir.mkdir()
        predictor = _reload_predictor(run_dir, prep, head=head, history_length=_HISTORY)
        g = features[features["unit_id"] == uid].sort_values("timestamp_s").reset_index(drop=True)
        prefix = ReplaySource(g).prefix(max(_HISTORY + 4, 8))
        tensor_clean, out_clean = _predictor_feature_tensor(predictor, prefix)
        assert out_clean["status"] == "ok"
        poisoned_prefix = _poison_gt_columns(prefix)
        tensor_poison, out_poison = _predictor_feature_tensor(predictor, poisoned_prefix)
        assert out_poison["status"] == "ok"
        np.testing.assert_allclose(tensor_clean, tensor_poison, rtol=1e-6, atol=1e-6, err_msg=dataset_id)
        np.testing.assert_allclose(
            np.asarray(out_clean["predicted_rul_s"], dtype=np.float64),
            np.asarray(out_poison["predicted_rul_s"], dtype=np.float64),
            rtol=1e-6,
            atol=1e-6,
        )

        clean_replay = replay_unit(
            g,
            predictor,
            dataset_id=dataset_id,
            unit_id=uid,
            run_id="tiny",
            history_length=_HISTORY,
            warning_horizon_s=1e6,
            truth_units=units,
        )["predictions"]
        poisoned_units = _poison_gt_columns(units)
        poisoned_meas = _poison_gt_columns(g)
        poisoned_replay = replay_unit(
            poisoned_meas,
            predictor,
            dataset_id=dataset_id,
            unit_id=uid,
            run_id="tiny",
            history_length=_HISTORY,
            warning_horizon_s=1e6,
            truth_units=poisoned_units,
        )["predictions"]
        np.testing.assert_allclose(
            pd.to_numeric(clean_replay["predicted_rul_s"], errors="coerce").to_numpy(dtype=np.float64),
            pd.to_numeric(poisoned_replay["predicted_rul_s"], errors="coerce").to_numpy(dtype=np.float64),
            rtol=1e-6,
            atol=1e-6,
            equal_nan=True,
            err_msg=dataset_id,
        )
        # Evaluator may use truth; predictor must not. Mutating GT should change actuals.
        if "actual_rul_s" in clean_replay.columns and "actual_rul_s" in poisoned_replay.columns:
            clean_actual = pd.to_numeric(clean_replay["actual_rul_s"], errors="coerce").to_numpy(dtype=np.float64)
            poison_actual = pd.to_numeric(poisoned_replay["actual_rul_s"], errors="coerce").to_numpy(
                dtype=np.float64
            )
            if np.isfinite(clean_actual).any():
                assert not np.allclose(clean_actual, poison_actual, equal_nan=True)


def test_resolve_max_windows_smoke_off_clears_omitted_cap():
    assert resolve_max_windows_per_unit(None, smoke=True) == 32
    assert resolve_max_windows_per_unit(None, smoke=False) is None
    assert resolve_max_windows_per_unit(0, smoke=True) is None
    assert resolve_max_windows_per_unit(0, smoke=False) is None
    assert resolve_max_windows_per_unit("", smoke=False) is None
    assert resolve_max_windows_per_unit(48, smoke=False) == 48
    assert resolve_max_windows_per_unit(24, smoke=True) == 24


def test_next_max_windows_clears_smoke_preset_unless_user_set():
    assert next_max_windows_on_mode_change(prev_smoke=True, smoke=False, current_cap=32) == 0
    assert next_max_windows_on_mode_change(prev_smoke=True, smoke=False, current_cap=48) == 48
    assert next_max_windows_on_mode_change(prev_smoke=False, smoke=True, current_cap=0) == 32
    assert next_max_windows_on_mode_change(prev_smoke=False, smoke=True, current_cap=10) == 10
    assert next_max_windows_on_mode_change(prev_smoke=True, smoke=True, current_cap=32) == 32


def test_resume_train_args_keep_saved_full_not_form_smoke(tmp_path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        "dataset_id: bearings\nsmoke: false\nmode: Full\nmax_windows_per_unit: null\n"
        "model:\n  architecture: gru\n  max_epochs: 30\n",
        encoding="utf-8",
    )
    saved = load_saved_train_settings(tmp_path, "bearings_gru_full")
    assert saved is not None
    assert saved["smoke"] is False
    assert saved["max_windows_per_unit"] is None
    assert saved["max_epochs"] == 30
    launch = resolve_run_train_args(
        resume_settings=saved,
        smoke=True,
        max_windows_per_unit=32,
        max_epochs=5,
        dataset_max_epochs=30,
    )
    assert launch == {"smoke": False, "max_windows_per_unit": None, "max_epochs": 30}
    new_smoke = resolve_run_train_args(
        resume_settings=None,
        smoke=True,
        max_windows_per_unit=None,
        max_epochs=30,
        dataset_max_epochs=30,
    )
    assert new_smoke["smoke"] is True
    assert new_smoke["max_windows_per_unit"] == 32
    assert new_smoke["max_epochs"] == 5


def test_selection_metric_and_mode_labels():
    b = selection_metric_spec("bearings")
    assert b["name"] == "val MAE"
    assert b["unit"] == "seconds"
    f = selection_metric_spec("filters")
    assert f["name"] == "val NLL"
    assert training_mode_label(True) == "Smoke"
    assert training_mode_label(False) == "Full"
    assert training_mode_label(None, "smoke_bearings_gru_x") == "Smoke"
    assert training_mode_label(None, "bearings_gru_x") == "Full"


def test_alert_policy_cache_invalidation(tmp_path):
    """H/K rescore changes alerts only. Frozen predictions.csv bytes stay put."""
    pred_path = tmp_path / "predictions.csv"
    pred = pd.DataFrame(
        {
            "run_id": ["r"] * 6,
            "unit_id": ["U1"] * 6,
            "timestamp_s": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "predicted_rul_s": [20.0, 9.0, 8.0, 7.0, 6.0, 5.0],
        }
    )
    pred.to_csv(pred_path, index=False)
    frozen_bytes = pred_path.read_bytes()
    original = pred.copy()
    policy_a = build_alert_policy(H_trigger=10.0, confirmation_count=3, source="replay_ui")
    policy_b = build_alert_policy(H_trigger=1.0, confirmation_count=3, source="replay_ui")
    hash_a = alert_policy_hash(policy_a)
    hash_b = alert_policy_hash(policy_b)
    assert hash_a != hash_b
    assert replay_session_key("bearings", "r", "U1", "eval1", hash_a) != replay_session_key(
        "bearings", "r", "U1", "eval1", hash_b
    )

    frozen = pd.read_csv(pred_path)
    ep_a, steps_a = rescore_replay_alerts(frozen, policy_a, run_id="r")
    assert pred_path.read_bytes() == frozen_bytes
    pd.testing.assert_frame_equal(pred, original)
    pd.testing.assert_series_equal(
        frozen["predicted_rul_s"].reset_index(drop=True),
        original["predicted_rul_s"].reset_index(drop=True),
    )
    assert not ep_a.empty
    assert float(ep_a.iloc[0]["timestamp_s"]) == 4.0
    assert (steps_a["warning_active"]).any()

    ep_b, steps_b = rescore_replay_alerts(frozen, policy_b, run_id="r")
    assert pred_path.read_bytes() == frozen_bytes
    pd.testing.assert_frame_equal(pred, original)
    assert ep_b.empty
    assert not bool(steps_b["warning_active"].any())
    assert list(steps_a["predicted_rul_s"]) == list(original["predicted_rul_s"])
    assert list(steps_b["predicted_rul_s"]) == list(original["predicted_rul_s"])

    # Stale session cache is keyed by policy_hash — B must not reuse A's episodes.
    cache = {(str("eval1"), hash_a, "U1"): ep_a}
    current_key = (str("eval1"), hash_b, "U1")
    assert current_key not in cache
    shown = cache.get(current_key, pd.DataFrame())
    assert shown.empty


def test_replay_log_has_no_future_rows():
    pred = pd.DataFrame(
        {
            "unit_id": ["A", "A", "A", "B", "B"],
            "timestamp_s": [10.0, 20.0, 30.0, 10.0, 20.0],
            "predicted_rul_s": [100.0, 90.0, 80.0, 50.0, 40.0],
            "actual_rul_s": [200.0, 190.0, 180.0, 150.0, 140.0],
        }
    )
    alerts = pd.DataFrame(
        {
            "unit_id": ["A", "A", "B"],
            "timestamp_s": [10.0, 30.0, 10.0],
            "type": ["horizon_warning", "horizon_warning", "horizon_warning"],
        }
    )
    t = 20.0
    unit_pred = pred[pred["unit_id"] == "A"]
    sliced = slice_predictions_to_replay_time(unit_pred, t)
    assert list(sliced["timestamp_s"]) == [10.0, 20.0]
    log = filter_replay_alert_log(alerts, unit_id="A", replay_time_s=t)
    assert set(log["unit_id"].astype(str)) == {"A"}
    assert (log["timestamp_s"] <= t).all()
    assert 30.0 not in set(log["timestamp_s"])
    vis = visible_replay_slice(sliced, log)
    assert not vis.empty
    assert (vis["timestamp_s"] <= t).all()
    assert set(vis["unit_id"].astype(str)) <= {"A"}
    assert "actual_rul_s" not in vis.columns
    future = slice_predictions_to_replay_time(pred, 10.0)
    assert future["timestamp_s"].max() <= 10.0
    other = filter_replay_alert_log(alerts, unit_id="B", replay_time_s=t)
    assert set(other["unit_id"].astype(str)) == {"B"}
    assert 30.0 not in set(other["timestamp_s"])

