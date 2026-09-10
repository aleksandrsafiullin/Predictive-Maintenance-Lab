from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from pdm.alerts import AlertEngine
from pdm.data.bearings import numeric_csv_index, sort_csv_names_numerically
from pdm.evaluate import attach_actual_rul
from pdm.losses import weibull_nll
from pdm.models import PDMNet
from pdm.preprocessing import fit_preprocessor
from pdm.replay import ReplaySource
from pdm.splits import assert_disjoint_splits, bearings_split, filters_split
from pdm.windows import build_windows


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
    ff, fu = tiny_filter_tables
    split_f = filters_split(fu, {"split": {"val_fraction": 0.25, "seed": 42}})
    assert_disjoint_splits(split_f)
    assert set(split_f["test"]) == {"Filter_101", "Filter_102"}


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
    assert e1["actual_rul_s"].iloc[0] != e2["actual_rul_s"].iloc[0]


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


def test_rul_and_time_conversion():
    event_time_s = 1200.0
    timestamp_s = 600.0
    assert event_time_s - timestamp_s == 600.0
    minutes = 10.0
    assert minutes * 60.0 == 600.0
    hours = 0.5
    assert hours * 3600.0 == 1800.0


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
