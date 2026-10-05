"""Isolated replay controls and dispatch; no project data or training required."""
from __future__ import annotations

import sys
from types import ModuleType

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from pdm import project_results_ui as ui, signal_inference as inference


def test_span_options_default_and_saved_support():
    spans, index = ui.forecast_span_options([60, 1800, 3600, 9000])
    assert spans == [600, 1200, 1800, 3600, 7200, 9000]
    assert spans[index] == 1800
    spans, index = ui.forecast_span_options([60, 420])
    assert spans == [420] and index == 0


def test_cache_identity_includes_span():
    args = ('project', 'run', 'snapshot', 'unit', 60, {'red': 2})
    assert ui.replay_forecast_key(*args, 600) != ui.replay_forecast_key(*args, 1800)
    assert ui.replay_forecast_key(*args, 600) == ui.replay_forecast_key(*args, 600)


def _mock_inference(monkeypatch, mode='learned_joint_trajectories'):
    run = {'project_id': 'p', 'run_id': 'r', 'snapshot_id': 's',
           'params': {'forecast_mode': mode, 'horizons_s': [60, 1800]},
           'schema': {'thresholds': {'red': 2}}}
    data = {'features': pd.DataFrame({'unit_id': ['u'], 'timestamp_s': [60], 'signal': [1]}),
            'split': {'test': ['u']}}
    monkeypatch.setattr(inference, 'load_signal_run', lambda *args: run)
    monkeypatch.setattr(inference, 'load_snapshot', lambda *args: data)
    return run, data


def test_public_span_dispatch_and_empty_rollout(monkeypatch):
    _mock_inference(monkeypatch)
    calls = []
    def predict(run, data, prefix, unit_id, stop, **kwargs):
        calls.append((prefix, kwargs))
        return {'as_of_s': None if prefix.empty else 60,
                'funnel': {'issued_horizon_s': kwargs['prediction_horizon_s']}}
    monkeypatch.setattr('pdm.learned_trajectory.forecast_learned_prefix', predict)
    result = inference.forecast_prefix('p', 'r', 'u', 0, prediction_horizon_s=600, rollout_steps=1)
    assert calls[-1][0].empty
    assert calls[-1][1] == {'prediction_horizon_s': 600}
    assert result['rollout']['direct_through_s'] is None
    result = inference.forecast_prefix('p', 'r', 'u', 60, prediction_horizon_s=600, rollout_steps=1)
    assert result['rollout']['direct_through_s'] == 660
    with pytest.raises(ValueError, match='saved RED rule'):
        inference.forecast_prefix('p', 'r', 'u', 60, thresholds={'red': 3}, prediction_horizon_s=600)


@pytest.mark.parametrize('span', [True, 0, -1, float('nan'), float('inf')])
def test_public_span_validation(monkeypatch, span):
    _mock_inference(monkeypatch)
    with pytest.raises(ValueError, match='finite positive'):
        inference.forecast_prefix('p', 'r', 'u', 60, prediction_horizon_s=span)


def test_legacy_engine_rejects_span_before_loading_snapshot(monkeypatch):
    _mock_inference(monkeypatch, mode='direct')
    monkeypatch.setattr(inference, 'load_snapshot', lambda *args: pytest.fail('must reject before data load'))
    with pytest.raises(ValueError, match='saved learned joint'):
        inference.forecast_prefix('p', 'r', 'u', 60, prediction_horizon_s=600)


@pytest.mark.parametrize('background', [False, True])
def test_replay_selector_request_cache_and_review_crop(monkeypatch, background):
    fixture = ModuleType('_forecast_span_fixture')
    fixture.snapshot = {'snapshot_id': 's', 'schema': {}, 'features': pd.DataFrame({
        'unit_id': ['u'] * 4, 'timestamp_s': [0, 60, 1200, 2400],
        'signal': [1, 2, 3, 4], 'gap_before': [False] * 4})}
    monkeypatch.setitem(sys.modules, fixture.__name__, fixture)
    calls, reviews = [], []
    def forecast(project, run, unit, as_of, **kwargs):
        calls.append(kwargs)
        span = kwargs['prediction_horizon_s']
        return {'project_id': project, 'run_id': run, 'snapshot_id': 's',
                'as_of_s': as_of, 'points': [{'target_time_s': as_of + span, 'value': 2}],
                'funnel': {'mode': 'learned_joint_trajectories', 'issued_horizon_s': span}}
    monkeypatch.setattr(ui, 'forecast_prefix', forecast)
    monkeypatch.setattr(ui, '_show_forecast_context', lambda *args: None)
    def figure(result, schema, theme, future_actual=None):
        import plotly.graph_objects as go
        reviews.append((result, future_actual))
        return go.Figure()
    monkeypatch.setattr(ui, 'replay_figure', figure)
    class Worker:
        def request(self, key, fn):
            return {'done': True, 'error': None, 'result': fn()}
        def cancel(self):
            pass
    monkeypatch.setattr(ui, '_background_forecasts', Worker)
    at = AppTest.from_string(f'''
from pdm.project_results_ui import _play_fragment
from _forecast_span_fixture import snapshot
_play_fragment('p', 'r', 'u', snapshot, zone_limits={{'red': 2}},
               background={background!r}, learned_horizons_s=[60, 1800, 9000])
''').run()
    assert not at.exception
    selector = next(w for w in at.selectbox if w.label == 'Forecast span')
    assert selector.value == 1800
    assert calls[-1]['prediction_horizon_s'] == 1800
    first_key = at.session_state['project_last_forecast']['key']
    at.run()
    assert len(calls) == 1  # unchanged request is cached
    selector = next(w for w in at.selectbox if w.label == 'Forecast span')
    selector.set_value(600)
    at.run()
    assert not at.exception
    assert calls[-1]['prediction_horizon_s'] == 600
    assert at.session_state['project_last_forecast']['key'] != first_key
    next(w for w in at.toggle if w.label == 'Show future actual measurements').set_value(True)
    at.run()
    assert not at.exception
    result, review = reviews[-1]
    assert result['funnel']['issued_horizon_s'] == 600
    assert result['points'][-1]['target_time_s'] == 600
    assert list(review.timestamp_s) == [60]
    assert all(row['timestamp_s'] <= result['as_of_s'] for row in result['observed_prefix'])
