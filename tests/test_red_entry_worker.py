"""Shared task routing prevents signal/event engine confusion before launch."""
import pytest

from pdm.project_tasks import RED_ENTRY_ENGINES, SIGNAL_ENGINES, validate_training_job


@pytest.mark.parametrize("engine", RED_ENTRY_ENGINES)
def test_event_engines_accepted_by_launch_contract(engine):
    assert validate_training_job({"task": "red_entry", "engine_id": engine, "params": {}}) == "red_entry"


@pytest.mark.parametrize("engine", SIGNAL_ENGINES)
def test_old_jobs_keep_signal_task(engine):
    assert validate_training_job({"engine_id": engine, "params": {}}) == "signal_forecast"


@pytest.mark.parametrize("task,engine", [("red_entry", "quantile_boosting"),
                                         ("signal_forecast", "hazard_boosting"),
                                         ("legacy_rul", "gru")])
def test_cross_task_engines_rejected(task, engine):
    with pytest.raises(ValueError, match="for its task"):
        validate_training_job({"task": task, "engine_id": engine, "params": {}})
