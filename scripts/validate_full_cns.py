#!/usr/bin/env python3
"""Validate the saved full-CNS model against real held-out equipment prefixes."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from pdm.data.prepare import load_processed
from pdm.forecasting import load_interval_profile, predict_failure_interval
from pdm.io_util import atomic_write_json
from pdm.train import load_trained_model
from pdm.visualization.simulation import neuron_details, simulate_step


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path, default=Path("output/full-cns-runtime-validation.json"))
    args = parser.parse_args()
    torch.set_num_threads(1)
    model, prep, meta = load_trained_model(args.run)
    profile = load_interval_profile(args.run)
    bundle = load_processed("bearings")
    recorded = pd.read_parquet(args.run / "test_forecast.parquet")
    report = {"run": args.run.name, "n_nodes": model.n_nodes, "recurrent_pairs": model.W_res._nnz(),
              "synapses": model.provenance["n_synapses"], "cases": []}
    for uid in bundle["split"]["test"]:
        frame = bundle["features"].loc[bundle["features"].unit_id == uid].sort_values("timestamp_s").iloc[:26]
        trace = simulate_step(frame, uid, float(frame.iloc[-2].timestamp_s), model, prep, meta["history_length"])
        start = time.perf_counter()
        appended = simulate_step(frame, uid, float(frame.iloc[-1].timestamp_s), model, prep, meta["history_length"], trace)
        elapsed = time.perf_counter() - start
        fresh = simulate_step(frame, uid, float(frame.iloc[-1].timestamp_s), model, prep, meta["history_length"])
        np.testing.assert_allclose(appended["states"], fresh["states"], atol=2e-7)
        rewound = simulate_step(frame, uid, float(frame.iloc[-2].timestamp_s), model, prep, meta["history_length"], appended)
        np.testing.assert_allclose(rewound["states"], trace["states"], atol=2e-7)
        assert fresh["states"].shape == (2, model.n_nodes)
        estimates = predict_failure_interval(fresh["timestamps_s"], fresh["raw_rul_s"], profile)
        expected = recorded.loc[recorded.unit_id == uid].set_index("timestamp_s")
        comparable = estimates.loc[estimates.timestamp_s.isin(expected.index) & estimates.predicted_rul_s.notna()].set_index("timestamp_s")
        columns = ["raw_rul_s", "predicted_rul_s", "lower_rul_s", "upper_rul_s"]
        difference = np.abs(comparable[columns].to_numpy() - expected.loc[comparable.index, columns].to_numpy())
        np.testing.assert_allclose(comparable[columns], expected.loc[comparable.index, columns], atol=.05, rtol=2e-5)
        index = int(np.flatnonzero(np.diff(model.W_res.crow_indices().numpy()))[0])
        _, neighbors = neuron_details(fresh, model, prep, index)
        np.testing.assert_allclose(neighbors["Recurrent contribution"].sum(),
                                   fresh["processing"]["recurrent_drive"][-1, index], atol=1e-6)
        report["cases"].append({"unit_id": uid, "measurements": fresh["n_history"],
                                "cached_step_seconds": elapsed, "max_saved_forecast_difference_s": float(difference.max()),
                                "cached_prefix_matches": True, "rewind_matches": True,
                                "retained_full_state_frames": len(fresh["states"]), "inspected_body_id": model.node_order[index]})
    report["passed"] = True
    atomic_write_json(args.output, report)
    print(report)


if __name__ == "__main__":
    main()
