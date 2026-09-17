"""Read-only post-evaluation diagnostics. Never fits, selects or edits issued states."""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from pdm.data.prepare import load_processed
from pdm.io_util import atomic_write_json, read_json, sha256_file
from pdm.monitoring.bundle import bundle_root, load_bundle


def main(bundle_id):
    bundle, _, _ = load_bundle(bundle_id, load_models=False)
    data = load_processed(bundle["profile"]["dataset_id"], bundle["dataset_version"])
    meta = data["units"].set_index("unit_id")
    for ep in (bundle_root(bundle_id) / "evaluations").glob("*/evaluation.json"):
        root = ep.parent
        e = read_json(ep)
        event = pd.read_csv(root / "event_forecasts.csv")
        states = pd.read_parquet(root / "state_history.parquet")
        episodes = pd.read_parquet(root / "alert_episodes.parquet")
        per_unit = []
        lead = []
        for uid, g in event.groupby("unit_id"):
            valid = g.point.notna() & g.actual_remaining.notna()
            error = g.loc[valid, "point"] - g.loc[valid, "actual_remaining"]
            near = valid & g.actual_remaining.between(0, 1800, inclusive="right")
            s = states.loc[states.unit_id.eq(uid)]
            a = episodes.loc[episodes.unit_id.eq(uid)] if len(episodes) else episodes
            outcome = meta.loc[uid]
            endpoint_prediction = g.loc[g.as_of.eq(float(outcome.observation_end_s)), "point"]
            official_target = outcome.get("official_rul_at_prefix_end_s", np.nan)
            official_prediction = float(endpoint_prediction.iloc[-1]) if len(endpoint_prediction) and pd.notna(endpoint_prediction.iloc[-1]) else None
            duration = float(s.as_of.max() - s.as_of.min())
            gap = s.as_of.diff().fillna(0).to_numpy()
            previous = s.display_zone.shift().fillna("gray")
            usable = float(gap[previous.ne("gray").to_numpy()].sum())
            record = {
                "unit_id": uid,
                "time_basis": bundle["profile"]["time_basis"],
                "event_observed": int(outcome.event_observed),
                "observed_followup": duration,
                "usable_assessment_time": usable,
                "issued_assessments": len(s),
                "mae": float(error.abs().mean()) if len(error) else None,
                "median_absolute_error": float(error.abs().median()) if len(error) else None,
                "mean_signed_error": float(error.mean()) if len(error) else None,
                "mean_overestimation": float(error.clip(lower=0).mean()) if len(error) else None,
                "near_1800_saved_time_units_mae": float(
                    (g.loc[near, "point"] - g.loc[near, "actual_remaining"]).abs().mean()
                )
                if near.any()
                else None,
                "assessment_availability": float(s.health_state.ne("unknown").mean()),
                "critical_measurements": int(s.display_zone.eq("red").sum()),
                "diagnostic_episodes": int(a.kind.eq("diagnostic").sum()) if len(a) else 0,
                "prognostic_episodes": int(a.kind.eq("prognostic").sum()) if len(a) else 0,
                "episodes_per_1000_usable_time_units": len(a) * 1000 / usable if usable else None,
                "diagnostic_first_lead": None,
                "prognostic_first_lead": None,
                "critical_first_lead": None,
                "official_prefix_end_rul": float(official_target) if pd.notna(official_target) else None,
                "official_prefix_end_prediction": official_prediction,
                "official_prefix_end_absolute_error": abs(official_prediction - float(official_target)) if official_prediction is not None and pd.notna(official_target) else None,
            }
            if outcome.event_observed and np.isfinite(outcome.event_time_s):
                for kind in ("diagnostic", "prognostic"):
                    candidates = a.loc[a.kind.eq(kind)] if len(a) else a
                    if len(candidates):
                        record[kind + "_first_lead"] = float(
                            outcome.event_time_s - candidates.confirmed_at.min()
                        )
                critical = s.loc[s.display_zone.eq("red")]
                if len(critical):
                    record["critical_first_lead"] = float(
                        outcome.event_time_s - critical.as_of.min()
                    )
            per_unit.append(record)
            for _, row in a.iterrows():
                resolved = row.get("resolved_at")
                observed_until = (
                    float(resolved) if pd.notna(resolved) else float(outcome.observation_end_s)
                )
                lead.append(
                    {
                        "unit_id": uid,
                        "episode_id": row.episode_id,
                        "kind": row.kind,
                        "confirmed_at": float(row.confirmed_at),
                        "lead_to_observed_event": float(outcome.event_time_s - row.confirmed_at)
                        if outcome.event_observed
                        else None,
                        "observed_episode_duration": max(0.0, observed_until - row.confirmed_at),
                        "resolution_observed": bool(pd.notna(resolved)),
                        "time_basis": bundle["profile"]["time_basis"],
                        "not_a_false_alarm_label": row.kind == "diagnostic",
                    }
                )
        out = root / "diagnostics"
        out.mkdir(exist_ok=True)
        frame = pd.DataFrame(per_unit)
        frame.to_csv(out / "event_monitoring_per_unit.csv", index=False)
        pd.DataFrame(lead).to_csv(out / "episode_lead_and_duration.csv", index=False)
        scored = frame.dropna(subset=["mae"])
        atomic_write_json(
            out / "summary.json",
            {
                "bundle_id": bundle_id,
                "evaluation_id": e["eval_id"],
                "split": e["split"],
                "unit_mean_mae": float(scored.mae.mean()) if len(scored) else None,
                "unit_median_mae": float(scored.mae.median()) if len(scored) else None,
                "worst_observed_event_unit": str(scored.loc[scored.mae.idxmax(), "unit_id"])
                if len(scored)
                else None,
                "independent_units": len(frame),
                "observed_events": int(frame.event_observed.sum()),
                "official_prefix_end_mae": float(frame.official_prefix_end_absolute_error.mean()) if frame.official_prefix_end_absolute_error.notna().any() else None,
                "official_prefix_end_expected": int(frame.official_prefix_end_rul.notna().sum()),
                "official_prefix_end_available": int(frame.official_prefix_end_absolute_error.notna().sum()),
                "official_RUL_role": "evaluator-only prefix-end target; never an observed in-prefix failure or runtime input",
                "probability_reliability_status": "not_estimable_for_operational_calibration",
                "reason": "No independent admitted reliability/timing calibration; original censoring and endpoint conventions preserved. No ordinary Brier with censored rows treated as negatives.",
                "test_used_for_tuning": False,
                "input_evaluation_sha256": sha256_file(ep),
                "meaning": "Diagnostic lead times are descriptive, not timely-detection recall or false-positive labels.",
            },
        )
        print(str(out), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("bundle_id")
    main(p.parse_args().bundle_id)
