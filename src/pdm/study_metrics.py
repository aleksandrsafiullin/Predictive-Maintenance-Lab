"""Maintenance usefulness, distinct from point forecast ranking."""
from __future__ import annotations

import numpy as np
import pandas as pd

from pdm.alerts import alerts_from_predictions


def warning_settings(dataset_id):
    horizon = 1800.0 if dataset_id == "bearings" else 420.0
    return {"H_trigger": horizon, "minimum_action_lead_time": horizon / 2,
            "max_useful_horizon_s": horizon, "confirmation_count": 3, "reset_factor": 1.2}


def useful_warning_metrics(predictions, units, policy):
    episodes, _ = alerts_from_predictions(predictions, policy=policy)
    lookup = units.set_index("unit_id")
    horizon = float(policy.get("max_useful_horizon_s") or policy["H_trigger"])
    lead_min = float(policy["minimum_action_lead_time"])
    records, observed, scorable, timely, repeated = [], 0, 0, 0, 0
    for uid, frame in predictions.groupby("unit_id", sort=False):
        meta = lookup.loc[uid]
        event = float(meta.get("event_time_s", np.nan))
        end = float(meta.observation_end_s)
        observed_event = bool(meta.get("event_observed", 0)) and np.isfinite(event)
        observed += int(observed_event)
        ready = ~frame.get("prediction_status", pd.Series("ok", index=frame.index)).eq("Collecting history")
        confirmations = ready.astype(int).rolling(int(policy["confirmation_count"])).sum().eq(int(policy["confirmation_count"]))
        possible = bool((confirmations & frame.timestamp_s.between(event - horizon, event - lead_min)).any()) if observed_event else False
        scorable += int(possible)
        found = False
        unit_episodes = episodes[episodes.unit_id.astype(str).eq(str(uid))].sort_values("timestamp_s")
        repeated += max(0, len(unit_episodes) - 1)
        for _, row in unit_episodes.iterrows():
            stamp = float(row.timestamp_s)
            lead = event - stamp if observed_event else None
            if observed_event:
                if lead > horizon:
                    outcome = "too_early"
                elif lead < lead_min:
                    outcome = "late"
                elif found:
                    outcome = "duplicate"
                else:
                    outcome = "useful"
                    found = True
            else:
                outcome = "false_alarm" if end - stamp >= horizon else "unknown"
            records.append({"unit_id": uid, "timestamp_s": stamp, "lead_time_s": lead, "outcome": outcome})
        timely += int(found and possible)
    table = pd.DataFrame(records, columns=["unit_id", "timestamp_s", "lead_time_s", "outcome"])
    counts = table.outcome.value_counts().to_dict()
    known = len(table) - counts.get("unknown", 0)
    precision = counts.get("useful", 0) / known if known else None
    # Missing history cannot remove a failure from the laboratory target.
    recall = timely / observed if observed else None
    finite = pd.to_numeric(predictions.predicted_rul_s, errors="coerce")
    finite = finite[np.isfinite(finite)]
    zero_only = bool(len(finite) and finite.eq(0).all())
    result = {"version": "useful_warnings_v2", "observed_events": observed,
              "scorable_warning_units": scorable, "unscorable_warning_units": observed - scorable,
              "timely_warning_units": timely, "missed_warning_units": scorable - timely,
              "unwarned_observed_units": observed - timely, "repeated_episode_count": repeated,
              "scorable_timely_recall": timely / scorable if scorable else None,
              "episode_count": len(table), "known_episode_count": known,
              "episode_outcomes": counts, "useful_precision": precision, "timely_recall": recall,
              "degenerate_constant_zero": zero_only,
              "warning_goal_met": not zero_only and precision is not None and recall is not None and precision >= .8 and recall >= .9,
              "evidence_status": "laboratory_only_few_independent_failures"}
    return result, table
