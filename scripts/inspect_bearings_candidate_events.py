"""Diagnose saved recurrent event fitting on Train/reused Validation only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from pdm.data.project_prepare import load_snapshot
from pdm.io_util import atomic_write_json
from pdm.learned_trajectory import (
    _context_model_kwargs,
    _normalized,
    _objective_weights,
    load_learned_bundle,
)
from pdm.trajectory_data import build_trajectory_frame, slice_frame
from pdm.trajectory_evaluation import entry_corridor
from pdm.trajectory_objectives import first_entry_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    directory, output = Path(args.candidate), Path(args.output)
    if output.exists():
        raise FileExistsError("Keep existing diagnostic receipts unchanged")
    frozen = json.loads((directory / "frozen_candidate.json").read_text())
    contract = json.loads((directory / "candidate_artifact_contract.json").read_text())
    if contract["engine_id"] not in {"gru", "lstm"}:
        raise ValueError("This event-only diagnostic supports recurrent candidates")
    if contract["params"] != frozen["config"]:
        raise ValueError("Saved candidate configuration differs from its frozen protocol")
    contract["dir"] = str(directory)
    bundle = load_learned_bundle(contract)
    data = load_snapshot(frozen["project_id"], frozen["snapshot_id"],
                         feature_partitions=("train", "validation"))
    if data["fingerprint"] != frozen["fingerprint"]:
        raise ValueError("Candidate snapshot fingerprint differs")
    config = bundle["config"]
    grid = np.asarray(config["horizons_s"], float)
    cutoff = np.flatnonzero(grid == 1800.)
    if len(cutoff) != 1:
        raise ValueError("Fixed thirty-minute cutoff is unavailable")
    torch.set_num_threads(2)
    records, summaries = [], {}
    for partition in ("train", "validation"):
        frame = build_trajectory_frame(data, data["split"][partition], config)
        if frame["feature_names"] != bundle["feature_names"]:
            raise ValueError("Candidate input feature contract differs")
        x = _normalized(frame, bundle["scaler"])
        pieces = []
        with torch.no_grad():
            for start in range(0, len(x), 64):
                pieces.append(bundle["model"](
                    torch.as_tensor(x[start:start + 64]),
                    torch.as_tensor(frame["current"][start:start + 64]),
                    **_context_model_kwargs(config, slice_frame(frame, np.arange(start, min(start + 64, len(x))))),
                )["event_logits"].numpy())
        logits = torch.as_tensor(np.concatenate(pieces))
        probability = logits.softmax(-1).numpy()
        # All forecasts are fixed before consulting their future label evidence.
        groups = np.asarray(frame["physical_unit_id"], str)
        observed, warning = frame["event_observed"], frame["warning_eligible"]
        at_risk = frame["current"] < frame["red_threshold"]
        index = frame["event_allowed"][:, :-1].argmax(1)
        actual_left, actual_right = np.r_[0., grid][index], grid[index]
        weights = _objective_weights(frame)["event"]
        evidence = {
            "event_allowed_mask": torch.as_tensor(frame["event_allowed"]),
            "event_observed_mask": torch.as_tensor(observed),
            "no_entry_prefix": torch.as_tensor(frame["no_entry_prefix"]),
        }
        summary = {"rows": len(x), "observed_first_ever_event_units": int(len(np.unique(groups[warning & observed])))}
        for k in (int(cutoff[0]) + 1, len(grid)):
            mass = probability[:, :k].sum(1)
            collapsed = np.column_stack((probability[:, :k], probability[:, k:].sum(1)))
            corridors = [entry_corridor(row, grid[:k]) for row in collapsed]
            lower = np.asarray([row["earliest_s"] if row["earliest_s"] is not None else np.nan for row in corridors])
            upper = np.asarray([row["latest_s"] if row["latest_s"] is not None else np.nan for row in corridors])
            finite = np.isfinite(lower) & np.isfinite(upper)
            known = at_risk & (observed | (frame["no_entry_prefix"] >= k))
            likelihood_known = at_risk & (observed | (frame["no_entry_prefix"] > 0))
            positive = warning & observed & (index < k)
            lead_eligible = positive & (actual_left >= 600.)
            contained = finite & (lower <= actual_left + 1e-7) & (upper >= actual_right - 1e-7)
            alert = warning & finite & (mass >= .5) & (upper - lower <= 900.)
            useful = lead_eligible & contained & alert
            loss = first_entry_loss(logits, prefix_length=k, **evidence).numpy()
            summary[f"event_nll_{int(grid[k - 1])}s"] = float((loss * weights).sum())
            summary[f"useful_event_units_{int(grid[k - 1])}s"] = len(np.unique(groups[useful]))
            for group in np.unique(groups):
                member = groups == group
                supported = member & warning & known
                censored = member & warning & likelihood_known & ~known
                pos, lead = member & positive, member & lead_eligible
                containing = lead & contained
                rows = member & warning & likelihood_known
                records.append({
                    "partition": partition, "physical_unit_id": group,
                    "horizon_s": float(grid[k - 1]),
                    "first_ever_known_origins": int(supported.sum()),
                    "partial_censored_origins": int(censored.sum()),
                    "positive_origins": int(pos.sum()), "lead_eligible_positive_origins": int(lead.sum()),
                    "censored_nll": float(loss[rows].mean()) if rows.any() else None,
                    "brier": float(((mass[supported] - positive[supported]) ** 2).mean()) if supported.any() else None,
                    "positive_finite_fraction": float(finite[pos].mean()) if pos.any() else None,
                    "positive_bracket_containment": float(contained[pos].mean()) if pos.any() else None,
                    "maximum_probability_at_10min_actual_lead": float(mass[lead].max()) if lead.any() else None,
                    "minimum_containing_width_at_10min_actual_lead_s": float((upper - lower)[containing].min()) if containing.any() else None,
                    "useful_event": bool(useful[member].any()),
                    "false_alert_origins": int((supported & alert & ~positive).sum()),
                })
        summaries[partition] = summary
    atomic_write_json(output, {
        "candidate": directory.name, "saved_artifact_hashes": contract["artifacts"],
        "snapshot_id": frozen["snapshot_id"], "test_access": False,
        "feature_read_partitions": data["loaded_feature_partitions"],
        "scope": "Training-fit diagnosis and reused Validation, no refit or selection",
        "warning_policy": "finite unconditional90% corridor<=900s, contains actual bracket, achieved left-edge lead>=600s, probability>=.5",
        "summary": summaries, "by_physical_unit": records,
    })
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
