import json

import numpy as np
import pandas as pd
import pytest
import torch

import pdm.red_entry_training as training
from pdm.models.red_entry_recurrent import RedEntryRecurrent
from pdm.red_entry_inference import forecast_red_entry_prefix
from pdm.red_entry_training import (
    load_red_entry_run,
    masked_unit_hazard_loss,
    sample_origin_indices,
    train_red_entry_run,
)


def fixture_snapshot():
    frame = pd.DataFrame(
        [
            {
                "unit_id": u,
                "physical_unit_id": u,
                "timestamp_s": float(t),
                "signal": float(t + 1),
                "gap_before": False,
            }
            for u in ("a", "b", "c")
            for t in range(6)
        ]
    )
    return {
        "snapshot_id": "s1",
        "features": frame,
        "units": pd.DataFrame({"unit_id": ["a", "b", "c"], "physical_unit_id": ["a", "b", "c"]}),
        "split": {"train": ["a"], "validation": ["b"], "test": ["c"], "holdout": []},
        "schema": {
            "source_kind": "generic_sensor_csv",
            "signal_column": "signal",
            "signal_unit": "u",
            "thresholds": {"mode": "absolute", "red": 5.0, "direction": "above"},
            "columns": {
                "signal": {"role": "sensor"},
                "unit_id": {"role": "metadata"},
                "timestamp_s": {"role": "metadata"},
            },
        },
    }


class Store:
    def __init__(self, root):
        self.root = root

    def run_path(self, p, r):
        return self.root / p / "runs" / r

    def project_path(self, p):
        return self.root / p

    def get(self, p):
        return {"active_snapshot_id": "other"}


@pytest.fixture
def bound(monkeypatch, tmp_path):
    data = fixture_snapshot()
    store = Store(tmp_path)
    monkeypatch.setattr(training, "load_snapshot", lambda *a: data)
    monkeypatch.setattr(training, "load_zone_limits", lambda *a: None)
    monkeypatch.setattr(training, "project_store", lambda: store)
    return data, store


def test_loss_mask_and_equal_unit_weight():
    logits = torch.tensor([[0.0, 8.0], [0.0, -8.0], [2.0, 0.0]], requires_grad=True)
    y = torch.zeros_like(logits)
    mask = torch.tensor([[True, False], [True, False], [True, False]])
    loss = masked_unit_hazard_loss(logits, y, mask, ["a", "a", "b"])
    expected = (
        torch.nn.functional.softplus(logits[0, 0]) + torch.nn.functional.softplus(logits[2, 0])
    ) / 2
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert torch.all(logits.grad[:, 1] == 0)
    # Duplicating overlapping origins of one unit does not alter its weight.
    duplicated = masked_unit_hazard_loss(
        logits.detach()[[0, 0, 1, 2]], y[[0, 0, 1, 2]], mask[[0, 0, 1, 2]], ["a", "a", "a", "b"]
    )
    torch.testing.assert_close(duplicated, loss.detach())


def test_prefix_one_ignores_padding():
    torch.manual_seed(2)
    for architecture in ("gru", "lstm"):
        m = RedEntryRecurrent(architecture, 2, 4, 3).eval()
        x = torch.randn(1, 16, 2)
        changed = x.clone()
        changed[:, 1:] = 10000
        torch.testing.assert_close(m(x, torch.tensor([1])), m(changed, torch.tensor([1])))
        assert m(x, torch.tensor([1])).shape == (1, 3)


def test_sampling_physical_cap_early_and_outcome_blind():
    origins = pd.DataFrame(
        {
            "physical_unit_id": ["p"] * 100,
            "unit_id": ["a"] * 50 + ["b"] * 50,
            "timestamp_s": list(range(50)) * 2,
            "event_time_s": range(100),
        }
    )
    a = sample_origin_indices(origins, 32, 8)
    assert len(a) == 32 and set(range(8)).issubset(a) and set(range(50, 58)).issubset(a)
    origins["event_time_s"] = -999
    np.testing.assert_array_equal(a, sample_origin_indices(origins, 32, 8))


def test_save_reload_prefix_and_integrity(bound):
    data, store = bound
    m = train_red_entry_run(
        "p", None, "gru", {"epochs": 2, "hidden_size": 4, "horizons_s": [1.0, 2.0, 4.0]}
    )
    assert m["snapshot_id"] == "s1" and m["selection"]["test_used"] is False
    first = forecast_red_entry_prefix("p", m["run_id"], "c", 0.0)
    second = forecast_red_entry_prefix("p", m["run_id"], "c", 0.0)
    assert first == second and first["prediction_status"] == "available"
    assert first["input_quality"]["real_history_length"] == 1
    p = [x["probability"] for x in first["probability_by_horizon"] if x["probability"] is not None]
    assert np.all(np.diff(p) >= 0) and first["red_entry_corridor"]["earliest_s"] is None
    run = load_red_entry_run("p", m["run_id"])
    saved = np.load(run["dir"] / "predictions.npz")
    targets = __import__("joblib").load(run["dir"] / "targets.joblib")
    b = training._batch(
        data,
        targets,
        run["feature_state"],
        run["effective_schema"],
        saved["origin_indices"],
        run["params"],
    )
    np.testing.assert_allclose(training.predict_hazard(run["model"], b), saved["hazard"], atol=1e-7)
    assert forecast_red_entry_prefix("p", m["run_id"], "c", 4.0)["already_red"]
    with pytest.raises(ValueError, match="RED rule"):
        forecast_red_entry_prefix(
            "p", m["run_id"], "c", 0.0, thresholds={"mode": "absolute", "red": 50.0}
        )
    path = run["dir"] / "feature_state.json"
    state = json.loads(path.read_text())
    state["feature_names"].reverse()
    path.write_text(json.dumps(state))
    with pytest.raises(ValueError, match="integrity"):
        load_red_entry_run("p", m["run_id"])


def test_changed_future_rejected_but_causal_forecast_ignores_future(bound, monkeypatch):
    data, _ = bound
    m = train_red_entry_run(
        "p", "s1", "lstm", {"epochs": 1, "hidden_size": 4, "horizons_s": [1.0, 2.0]}
    )
    run = load_red_entry_run("p", m["run_id"])
    original = forecast_red_entry_prefix("p", m["run_id"], "c", 0.0)
    data["features"].loc[
        (data["features"].unit_id == "c") & (data["features"].timestamp_s > 0), "signal"
    ] = 900.0
    with pytest.raises(ValueError, match="content changed"):
        load_red_entry_run("p", m["run_id"])
    # Inject an already integrity-checked run to isolate causal inference itself.
    import pdm.red_entry_inference as inference

    monkeypatch.setattr(inference, "load_red_entry_run", lambda *a: run)
    assert original == forecast_red_entry_prefix("p", m["run_id"], "c", 0.0)


def test_baseline_no_entry_and_cancellation(bound):
    data, _ = bound
    m = train_red_entry_run("p", "s1", "always_no_entry", {"horizons_s": [1.0, 2.0]})
    result = forecast_red_entry_prefix("p", m["run_id"], "c", 0.0)
    assert [p["probability"] for p in result["probability_by_horizon"]] == [0.0, 0.0]
    with pytest.raises(InterruptedError):
        train_red_entry_run(
            "p", "s1", "gru", {"epochs": 1, "horizons_s": [1.0]}, should_stop=lambda: True
        )


def test_boosting_frozen_reload_and_prefix_one(bound):
    _, _ = bound
    m = train_red_entry_run("p", "s1", "hazard_boosting", {"horizons_s": [1.0, 2.0]})
    a = forecast_red_entry_prefix("p", m["run_id"], "c", 0.0)
    b = forecast_red_entry_prefix("p", m["run_id"], "c", 0.0)
    assert a == b and a["input_quality"]["real_history_length"] == 1
    assert m["selection"]["test_used"] is False
    run = load_red_entry_run("p", m["run_id"])
    assert run["contract"]["model_provenance"]["adapter_version"] == "red_entry_boosting_v1"


def test_masked_unknown_nan_is_not_negative_evidence():
    logits = torch.tensor([[0.0, float("nan")]], requires_grad=True)
    targets = torch.tensor([[1.0, float("nan")]])
    loss = masked_unit_hazard_loss(logits, targets, torch.tensor([[True, False]]), ["a"])
    torch.testing.assert_close(loss, torch.tensor(np.log(2), dtype=torch.float32))
    loss.backward()
    assert logits.grad[0, 1] == 0
