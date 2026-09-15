import copy

import numpy as np
import pandas as pd
import pytest
import torch

from pdm.benchmark import add_survival_scores, compare_evaluations
from pdm.config import load_dataset_config
from pdm.models import PDMNet
from pdm.predict import Predictor, prepare_history_window
from pdm.preprocessing import fit_preprocessor
from pdm.splits import bearings_split


def evaluation(rid, values, *, dataset='bearings', split='validation'):
    frame = pd.DataFrame({'unit_id': ['u1', 'u1', 'u2', 'u2'], 'timestamp_s': [10., 20., 10., 20.],
                          'predicted_rul_s': values, 'actual_rul_s': [20., 10., 20., 10.],
                          'valid_history_reason': ['', '', '', '']})
    return {'run': {'architecture': 'gru', 'smoke': False, 'n_nodes': None},
            'config': {'run_id': rid, 'eval_id': 'evaluation-' + rid, 'dataset_id': dataset,
                       'dataset_version': 'v', 'features_hash': 'f', 'units_hash': 'u', 'split_hash': 's',
                       'quality_policy_hash': 'q', 'metrics_version': 'v2',
                       'evaluate_mask': {'split': split, 'unit_ids': ['u1', 'u2']}},
            'metrics': {}, 'predictions': frame}


def test_common_cohort_ranking_and_missing_prediction():
    a = evaluation('a', [21., 11., 21., 11.])
    b = evaluation('b', [25., 15., 25., 15.])
    result = compare_evaluations([a, b])
    table = result['table'].set_index('run_id')
    assert table.loc['a', 'rank'] == 1
    assert table.loc['a', 'near_30m_mae_s'] == 1
    b['predictions'].loc[0, 'predicted_rul_s'] = np.nan
    table = compare_evaluations([a, b])['table'].set_index('run_id')
    assert table.loc['b', 'prediction_coverage'] == .75
    assert not table.loc['b', 'rank_eligible']
    assert table.loc['a', 'common_points'] == 4


def test_missing_export_row_cannot_hide_a_difficult_point():
    a, b = evaluation('a', [21., 11., 21., 11.]), evaluation('b', [25., 15., 25., 15.])
    b['predictions'] = b['predictions'].iloc[1:]
    table = compare_evaluations([a, b])['table'].set_index('run_id')
    assert table.loc['a', 'common_points'] == 4
    assert table.loc['b', 'prediction_coverage'] == .75
    assert not table.loc['b', 'rank_eligible']


def test_official_test_labels_are_not_observed_prefix_failures():
    ev = evaluation('a', [21., 11., 21., 11.], dataset='filters', split='test')
    ev['predictions']['outcome_observed'] = 0
    ev['metrics']['alerts'] = {'n_units_with_event': 2}
    row = compare_evaluations([ev])['table'].iloc[0]
    assert row.observed_events == 0
    assert row.primary_score == 1


def test_recurrent_identity_ignores_unused_reservoir_defaults(tmp_path):
    import json

    from pdm.experiments import _enrich_run_identity
    (tmp_path / 'experiment_snapshot.json').write_text(json.dumps({'model': {
        'architecture': 'gru', 'hidden_size': 8, 'recurrent_layers': 2,
        'reservoir': {'n_nodes': 1000, 'graph_mode': 'synthetic_fixture'}}}))
    row = _enrich_run_identity({}, tmp_path)
    assert row['n_nodes'] == 16
    assert row['graph_mode'] is None
    assert row['is_synthetic'] is False


@pytest.mark.parametrize('field', ['dataset_version', 'quality_policy_hash', 'metrics_version', 'split_hash'])
def test_incompatible_identities_cannot_be_ranked(field):
    a, b = evaluation('a', [20]*4), evaluation('b', [20]*4)
    b['config'][field] = 'different'
    with pytest.raises(ValueError, match='Incompatible'):
        compare_evaluations([a, b])


def test_ground_truth_and_duplicate_checks():
    a, b = evaluation('a', [20]*4), evaluation('b', [20]*4)
    b['predictions'].loc[0, 'actual_rul_s'] = 999
    with pytest.raises(ValueError, match='Actual RUL differs'):
        compare_evaluations([a, b])
    b = copy.deepcopy(a)
    b['predictions'] = pd.concat([b['predictions'], b['predictions'].iloc[[0]]])
    with pytest.raises(ValueError, match='duplicate'):
        compare_evaluations([b])


def test_censored_likelihood_and_scale_are_comparable():
    pred = pd.DataFrame({'unit_id': ['censored', 'event'], 'timestamp_s': [10., 10.],
                         'weibull_scale_s': [10., 10.], 'weibull_shape': [1., 1.]})
    units = pd.DataFrame({'unit_id': ['censored', 'event'], 'event_observed': [0, 1],
                          'observation_end_s': [20., 20.], 'event_time_s': [np.nan, 20.]})
    scores = add_survival_scores(pred, units)
    assert scores.survival_nll.iloc[0] == pytest.approx(1.)
    assert scores.survival_nll.iloc[1] == pytest.approx(1. + np.log(10.))


def test_full_matrix_has_nine_main_models_and_paired_validation_study():
    from pdm.batch import evaluation_splits, matrix_tasks

    tasks = matrix_tasks(['bearings', 'filters'])
    assert len(tasks) == 14
    main = [t for t in tasks if t['role'] == 'main']
    assert len(main) == 9
    assert all(evaluation_splits(t) == ('validation', 'test') for t in main)
    paired = [t for t in tasks if t['dataset_id'] == 'filters' and t['architecture'] == 'gru']
    assert {(t['seed'], t['events_only']) for t in paired} == {(seed, events) for seed in (42,43,44) for events in (False,True)}
    assert all(evaluation_splits(t) == ('validation',) for t in paired if t['role'] == 'ablation')


@pytest.mark.parametrize('architecture', ['gru', 'lstm'])
@pytest.mark.parametrize('head', ['rul', 'weibull'])
def test_actual_recurrent_trace_matches_prediction_and_cell_memory(tiny_bearing_tables, architecture, head):
    f, u = tiny_bearing_tables
    cfg = load_dataset_config('bearings')
    prep, _ = fit_preprocessor('bearings', f, u, bearings_split(u), cfg)
    model = PDMNet(len(prep.feature_names), hidden_size=8, num_layers=2, architecture=architecture,
                   head=head, time_scale_s=prep.time_scale_s)
    predictor = Predictor(model, prep, 5)
    prefix = f[f.unit_id == 'Bearing1_1'].iloc[:12]
    plain = predictor.predict_from_history(prefix)
    trace = predictor.predict_from_history(prefix, with_trace=True)
    assert trace['predicted_rul_s'] == plain['predicted_rul_s']
    assert trace['states'].shape == (5, 16)
    window = prepare_history_window(prefix, prep, 5)
    with torch.no_grad():
        _, hidden = model.encoder.rnn(torch.from_numpy(np.array(window['inputs'], order='C', copy=True)).unsqueeze(0))
    if architecture == 'lstm':
        np.testing.assert_allclose(trace['cell_states'][-1], hidden[1][:, 0].reshape(-1).numpy(), atol=1e-6)
        hidden = hidden[0]
    np.testing.assert_allclose(trace['states'][-1], hidden[:, 0].reshape(-1).numpy(), atol=1e-6)
    if head == 'weibull':
        assert plain['lower_rul_s'] <= plain['predicted_rul_s'] <= plain['upper_rul_s']


@pytest.mark.parametrize('architecture', ['gru', 'lstm'])
def test_seek_and_replay_export_use_identical_forecasts(tiny_bearing_tables, architecture):
    from pdm.replay import replay_unit
    from pdm.visualization.simulation import window_forecast_history
    f, u = tiny_bearing_tables
    cfg = load_dataset_config('bearings')
    prep, _ = fit_preprocessor('bearings', f, u, bearings_split(u), cfg)
    model = PDMNet(len(prep.feature_names), hidden_size=8, architecture=architecture, time_scale_s=prep.time_scale_s)
    prefix = f[f.unit_id == 'Bearing1_1'].iloc[:16]
    points = window_forecast_history(prefix, model, prep, 5)
    cache = {r['timestamp_s']: r for r in points.to_dict('records')}
    rewind = window_forecast_history(prefix.iloc[:9], model, prep, 5, cached=cache)
    replay = window_forecast_history(prefix, model, prep, 5, cached={r['timestamp_s']: r for r in rewind.to_dict('records')})
    pd.testing.assert_frame_equal(points, replay)
    export = replay_unit(prefix, Predictor(model, prep, 5), dataset_id='bearings', unit_id='Bearing1_1', run_id='test',
                         history_length=5, warning_horizon_s=60, truth_units=u)['predictions']
    np.testing.assert_allclose(points.predicted_rul_s, export.predicted_rul_s, equal_nan=True, rtol=0, atol=0)
