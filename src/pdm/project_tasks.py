"""Shared dispatch contract for project training; historical jobs remain signal jobs."""
from __future__ import annotations

SIGNAL_ENGINES = ("gru", "lstm", "quantile_boosting", "full_cns")
RED_ENTRY_ENGINES = ("gru", "lstm", "hazard_boosting", "full_cns", "kaplan_meier",
                     "always_no_entry", "trend_to_red")
TASK_ENGINES = {"signal_forecast": SIGNAL_ENGINES, "red_entry": RED_ENTRY_ENGINES}


def validate_training_job(job: dict) -> str:
    task = job.get("task", "signal_forecast")
    if (task not in TASK_ENGINES or job.get("engine_id") not in TASK_ENGINES[task]
            or not isinstance(job.get("params"), dict)):
        raise ValueError("Project training requires one supported engine and parameter mapping for its task")
    return task


def train_project_job(job: dict, *, should_stop=None, status_cb=None) -> dict:
    task = validate_training_job(job)
    if task == "red_entry":
        from pdm.red_entry_training import train_red_entry_run
        trainer = train_red_entry_run
    else:
        from pdm.signal_training import train_signal_run
        trainer = train_signal_run
    return trainer(job["project_id"], job.get("snapshot_id"), job["engine_id"],
                   job["params"], should_stop=should_stop, status_cb=status_cb)
