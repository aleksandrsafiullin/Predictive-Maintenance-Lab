"""Independent synthetic optimizer wiring QA; never warning-quality evidence."""
import copy
import importlib.util
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from pdm import learned_trajectory as learned
from pdm.io_util import sha256_file
from pdm.models.signal_distribution import SignalDistribution
from tests.test_known_cdf_timing_integration import config as base_config, frame as base_frame

CONTRACT = 'optimizer_sampling_contract'
TERMS = ('energy', 'width', 'miss', 'event', 'phase_energy', 'red_corridor_width',
         'red_corridor_miss', 'red_finite_bound', 'event_cdf')
BASELINE = Path(__file__).resolve().parents[1] / 'output/bearings-learned-funnel-20261002/review/pre_event_stratified_learned_trajectory.py'


def config(**overrides):
    return base_config(**{'batch_sampling': 'physical_event_stratified', 'epochs': 2,
                          'batch_size': 3, 'min_delta': 1e10, **overrides})


def frame(offset=0.):
    source = base_frame()
    indices = np.array([0, 1, 2, 3, 4, 0, 1])
    data = {k: (v[indices].copy() if isinstance(v, np.ndarray) else copy.deepcopy(v))
            for k, v in source.items()}
    data['physical_unit_id'] = ['A', 'A', 'A', 'A', 'B', 'B', 'C']
    data['current'] = np.array([.1, .2, .3, 1.2, .5, .6, .7], np.float32) + offset
    data['event_allowed'][:] = [0, 0, 0, 1]
    data['event_observed'][:] = False
    data['no_entry_prefix'][:] = [0, 1, 0, 0, 0, 0, 3]
    data['event_allowed'][0] = [1, 0, 0, 0]
    data['event_observed'][0] = True
    data['mask'][[2, 4]] = False
    data['x'] = np.arange(28, dtype=np.float32).reshape(7, 2, 2) / 28 + offset
    data['raw_x'] = data['x'][:, :, :1].copy()
    return data


def populations(data):
    result = learned._objective_weights(data)
    result['red_corridor_width'] = learned._at_risk_weights(data)
    for key in ('red_corridor_miss', 'red_finite_bound', 'event_cdf'):
        result[key] = result['event'].copy()
    return result


def save_run(bundle, directory, full=False):
    artifacts = learned.save_learned_bundle(bundle, directory)
    fields = learned._event_distribution_metadata(bundle['model'], bundle['config'])
    for name in ('training_contract.json', 'model_input_contract.json'):
        (directory / name).write_text(json.dumps(fields))
        artifacts[name] = sha256_file(directory / name)
    run = {'dir': directory, 'engine_id': 'gru', 'params': copy.deepcopy(bundle['config']),
           'artifacts': artifacts, **copy.deepcopy(fields)}
    if full:
        run.update(schema_version=1, selection=copy.deepcopy(bundle['selection']))
    return run


def equal(a, b):
    assert a['scaler'] == b['scaler']
    for key, value in a['model'].state_dict().items():
        assert torch.equal(value, b['model'].state_dict()[key]), key
    pa = learned.predict_learned(a, frame(.01))
    pb = learned.predict_learned(b, frame(.01))
    for key in pa:
        np.testing.assert_array_equal(pa[key], pb[key])


@pytest.mark.parametrize('zero_events', [False, True])
def test_actual_loop_nine_terms_population_reduction_seed_budgets_and_causal_inputs(monkeypatch, zero_events):
    train, val = frame(), frame(.01)
    if zero_events:
        for data in (train, val):
            data['event_observed'][:] = False
            data['no_entry_prefix'][:] = 0
    before = copy.deepcopy((train, val))
    cfg = config()
    coeff = {k: cfg['phase_weight' if k == 'phase_energy' else k + '_weight'] for k in TERMS}
    assert all(value > 0 for value in coeff.values())
    weights = populations(train)
    assert not np.array_equal(weights['energy'], weights['event'])
    calls, plans, batches, gradients, models, checkpoints, built_plans = [], [], [], [], [], [], []
    factory, planner, sampler, reducer = learned._make_model, learned._event_stratified_plan, learned._optimizer_batches, learned._optimizer_term
    def make_model(*args, **kwargs):
        model = factory(*args, **kwargs)
        models.append(model)
        def inputs(module, args):
            data = train if module.training else val
            idx = np.array([np.flatnonzero(data['current'] == c)[0] for c in args[1].numpy()])
            w = learned._weights(train)
            ref = learned._normalization_reference(train, cfg)
            mean = (ref * w[:, None, None]).sum((0, 1)) / ref.shape[1]
            var = ((ref - mean) ** 2 * w[:, None, None]).sum((0, 1)) / ref.shape[1]
            expected = (data['x'][idx] - mean) / np.maximum(np.sqrt(var), 1e-5)
            np.testing.assert_array_equal(args[0].numpy(), expected)
            assert args[2] is None
        model.register_forward_pre_hook(inputs)
        return model
    def plan(data, prefix):
        assert data is train
        plans.append(prefix)
        result = planner(data, prefix)
        built_plans.append(result)
        return result
    def sampling(*args, **kwargs):
        for idx, pi in sampler(*args, **kwargs):
            batches.append((idx.copy(), pi.copy()))
            yield idx, pi
    def objective(self, moments, y, mask, red, **kwargs):
        training = self.training
        j = len([c for c in calls if c[0] == training])
        idx = batches[j][0] if training else np.arange((j % 3) * 3, min((j % 3) * 3 + 3, 7))
        data = train if training else val
        assert torch.equal(y, torch.as_tensor(data['y'][idx]))
        seed = cfg['seed'] + (j // 3) * 10000 + int(idx[0]) if training else cfg['seed'] + 900000 + int(idx[0])
        assert kwargs['generator'].initial_seed() == seed
        for key in TERMS:
            field = 'phase_weight' if key == 'phase_energy' else key + '_weight'
            assert kwargs[field] == coeff[key]
        assert tuple(kwargs['event_objective_prefix_lengths']) == (1, 3)
        calls.append((training, idx.copy(), copy.deepcopy(kwargs)))
        anchor = next(self.parameters()).reshape(-1)[0]
        # Finite deterministic row losses, with a known small gradient before clipping.
        return {key: torch.as_tensor(data['current'][idx]) * ((t + 1) / 100.) + anchor * ((t + 1) / 1000.)
                if key != 'width' else anchor * 0 + torch.zeros(len(idx))
                for t, key in enumerate(TERMS)}
    def optimizer(model, config):
        opt = torch.optim.SGD(model.parameters(), lr=0.)
        step = opt.step
        def step_spy(*args, **kwargs):
            gradients.append(float(next(model.parameters()).grad.reshape(-1)[0]))
            return step(*args, **kwargs)
        opt.step = step_spy
        return opt
    monkeypatch.setattr(learned, '_make_model', make_model)
    monkeypatch.setattr(learned, '_event_stratified_plan', plan)
    monkeypatch.setattr(learned, '_optimizer_batches', sampling)
    monkeypatch.setattr(learned, '_make_optimizer', optimizer)
    monkeypatch.setattr(SignalDistribution, 'objective', objective)
    def report(record):
        if record['stage'] == 'training':
            checkpoints.append(copy.deepcopy(models[0].state_dict()))
    bundle = learned.fit_learned_model('gru', train, val, cfg, report=report)
    assert plans == [1]
    plan_data = built_plans[0]
    for idx, pi in batches:
        np.testing.assert_array_equal(pi, plan_data['probabilities'][idx])
    metadata = bundle['selection']['optimizer_sampling_train']
    for key, value in plan_data['counts'].items():
        assert metadata[key] == value
    assert metadata['term_weight_sha256'] == {k: hashlib.sha256(v.tobytes()).hexdigest() for k, v in weights.items()}
    assert [len(i) for i, _ in batches] == [3, 2, 2] * 2
    assert len(gradients) == 6 and sum(len(i) for i, _ in batches) == 14
    anchor = float(next(bundle['model'].parameters()).reshape(-1)[0])
    for epoch, record in enumerate(bundle['trace']):
        expected = {key: 0. for key in TERMS}
        for step, (idx, pi) in enumerate(batches[epoch * 3:epoch * 3 + 3]):
            expected_gradient = 0.
            for t, key in enumerate(TERMS):
                values = np.zeros(len(idx)) if key == 'width' else train['current'][idx] * ((t + 1) / 100.) + anchor * ((t + 1) / 1000.)
                estimate = np.mean(values * weights[key][idx] / pi)
                expected[key] += estimate * len(idx) / 7
                if key != 'width':
                    expected_gradient += coeff[key] * (t + 1) / 1000. * np.mean(weights[key][idx] / pi)
            assert gradients[epoch * 3 + step] == pytest.approx(expected_gradient, rel=3e-6, abs=1e-7)
        for key in TERMS:
            assert record['train'][key] == pytest.approx(expected[key], rel=3e-6, abs=1e-7)
            t = TERMS.index(key)
            values = np.zeros(7) if key == 'width' else val['current'] * ((t + 1) / 100.) + anchor * ((t + 1) / 1000.)
            assert record['validation'][key] == pytest.approx(np.sum(values * populations(val)[key]), rel=3e-6, abs=1e-7)
        for phase in ('train', 'validation'):
            assert record[phase]['total'] == pytest.approx(sum(coeff[k] * record[phase][k] for k in TERMS), rel=3e-6)
        assert record['gradient_norm_mean'] >= 0 and record['clipped_batch_fraction'] == 0.
        diagnostics = record['optimizer_sampling']
        drawn = np.concatenate([i for i, _ in batches[epoch * 3:epoch * 3 + 3]])
        assert diagnostics['draw_count'] == 7
        assert diagnostics['distinct_origin_count'] == len(set(drawn))
        assert diagnostics['duplicate_origin_draw_count'] == 7 - len(set(drawn))
        assert sum(g['draw_count'] for g in diagnostics['group_stratum_draw_counts']) == 7
        for g, group in enumerate(diagnostics['group_stratum_draw_counts']):
            assert group['physical_unit_id'] == plan_data['groups'][g]
            assert group['draw_count'] == int((plan_data['row_group'][drawn] == g).sum())
            for st, name in enumerate(plan_data['strata']):
                expected_count = int(((plan_data['row_group'][drawn] == g) & (plan_data['row_stratum'][drawn] == st)).sum())
                assert group['stratum_counts'][name] == expected_count
        for key in TERMS:
            assert diagnostics['term_batch_estimates'][key]['count'] == 3
            assert diagnostics['term_batch_estimates'][key]['std'] >= 0
    assert bundle['selection']['best_epoch'] == 1
    assert bundle['selection']['best_score'] == bundle['trace'][0]['validation']['total']
    assert bundle['selection']['restored_best_checkpoint']
    for key, value in bundle['model'].state_dict().items():
        assert torch.equal(value, checkpoints[0][key])
    for old, actual in zip(before, (train, val)):
        for key, value in old.items():
            if isinstance(value, np.ndarray):
                np.testing.assert_array_equal(value, actual[key])
            else:
                assert value == actual[key]
    # Scaler uses original causal physical-group weights; optimizer evidence never enters it.
    ref = learned._normalization_reference(train, cfg)
    w = learned._weights(train)
    mean = (ref * w[:, None, None]).sum((0, 1)) / ref.shape[1]
    var = ((ref - mean) ** 2 * w[:, None, None]).sum((0, 1)) / ref.shape[1]
    np.testing.assert_array_equal(bundle['scaler']['mean'], mean)
    np.testing.assert_array_equal(bundle['scaler']['std'], np.maximum(np.sqrt(var), 1e-5))


@pytest.fixture(scope='module')
def positive_bundle():
    return learned.fit_learned_model('gru', frame(), frame(.01), config(epochs=1))


def test_minimal_and_full_reload_contract_all_surfaces(positive_bundle, tmp_path):
    expected = positive_bundle['selection'][CONTRACT]
    for full in (False, True):
        run = save_run(positive_bundle, tmp_path / str(full), full)
        equal(positive_bundle, learned.load_learned_bundle(run))
        assert run[CONTRACT] == expected
        for name in ('learned_model.json', 'training_contract.json', 'model_input_contract.json'):
            assert json.loads((run['dir'] / name).read_text())[CONTRACT] == expected


@pytest.mark.parametrize('surface', ['learned_model.json', 'selection', 'manifest', 'training_contract.json', 'model_input_contract.json'])
def test_all_contract_fields_rehashed_tamper_missing_rejected(positive_bundle, tmp_path, surface):
    contract = positive_bundle['selection'][CONTRACT]
    for field in [None, *contract]:
        for action in ('delete', 'tamper'):
            run = save_run(positive_bundle, tmp_path, full=True)
            target = 'learned_model.json' if surface == 'selection' else surface
            saved = run if target == 'manifest' else json.loads((tmp_path / target).read_text())
            parent = saved['selection'] if surface == 'selection' else saved
            if field is None:
                if action == 'delete':
                    del parent[CONTRACT]
                else:
                    parent[CONTRACT] = {'invalid': True}
            elif action == 'delete':
                del parent[CONTRACT][field]
            else:
                parent[CONTRACT][field] = {'invalid': True}
            if target != 'manifest':
                (tmp_path / target).write_text(json.dumps(saved))
                run['artifacts'][target] = sha256_file(tmp_path / target)
            with pytest.raises(ValueError):
                learned.load_learned_bundle(run)


@pytest.mark.parametrize('surface', ['manifest', 'learned_model.json', 'training_contract.json', 'model_input_contract.json'])
@pytest.mark.parametrize('field,wrong', [('batch_sampling', 'physical_group'), ('event_objective_horizons_s', [180.]), ('horizons_s', [60., 120., 240.])])
def test_rehashed_duplicate_params_rejected(positive_bundle, tmp_path, surface, field, wrong):
    run = save_run(positive_bundle, tmp_path, full=True)
    saved = run if surface == 'manifest' else json.loads((tmp_path / surface).read_text())
    key = 'config' if surface == 'learned_model.json' else 'params'
    saved[key] = {**positive_bundle['config'], field: wrong}
    if surface != 'manifest':
        (tmp_path / surface).write_text(json.dumps(saved))
        run['artifacts'][surface] = sha256_file(tmp_path / surface)
    with pytest.raises(ValueError):
        learned.load_learned_bundle(run)


@pytest.mark.parametrize('omission', ['training_contract.json', 'model_input_contract.json', 'selection'])
def test_full_manifest_requires_verified_contracts_and_selection(positive_bundle, tmp_path, omission):
    run = save_run(positive_bundle, tmp_path, full=True)
    if omission == 'selection':
        del run[omission]
    else:
        del run['artifacts'][omission]
    with pytest.raises(ValueError):
        learned.load_learned_bundle(run)


@pytest.mark.parametrize('mode', [None, 'row_permutation', 'physical_group'])
def test_old_modes_literal_pre_port_baseline_and_bundle_compatibility(mode, tmp_path):
    spec = importlib.util.spec_from_file_location('pdm._event_sampler_pre_port_qa', BASELINE)
    baseline = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(baseline)
    cfg = base_config(batch_sampling=mode or 'row_permutation', epochs=1)
    if mode is None:
        del cfg['batch_sampling']
    old = baseline.fit_learned_model('gru', frame(), frame(.01), cfg)
    old_rng = torch.random.get_rng_state().clone()
    new = learned.fit_learned_model('gru', frame(), frame(.01), cfg)
    assert torch.equal(old_rng, torch.random.get_rng_state())
    assert old['trace'] == new['trace'] and old['selection'] == new['selection']
    equal(old, new)
    assert learned._event_stratified_sampling_settings(cfg) is None
    assert CONTRACT not in new['selection']
    artifacts = baseline.save_learned_bundle(old, tmp_path)
    equal(old, learned.load_learned_bundle({'dir': tmp_path, 'engine_id': 'gru', 'params': cfg, 'artifacts': artifacts}))
    for surface in ('learned_model.json', 'selection', 'manifest', 'training_contract.json', 'model_input_contract.json'):
        run = save_run(new, tmp_path / ('surface-' + surface), full=True)
        target = 'learned_model.json' if surface == 'selection' else surface
        saved = run if target == 'manifest' else json.loads((run['dir'] / target).read_text())
        parent = saved['selection'] if surface == 'selection' else saved
        parent[CONTRACT] = {'mode': 'physical_event_stratified'}
        if target != 'manifest':
            (run['dir'] / target).write_text(json.dumps(saved))
            run['artifacts'][target] = sha256_file(run['dir'] / target)
        with pytest.raises(ValueError):
            learned.load_learned_bundle(run)


@pytest.mark.parametrize('mutation', ['missing_active_hash', 'group_name_outside_scaler'])
def test_consistently_rehashed_dynamic_metadata_semantic_tampering_rejected(positive_bundle, tmp_path, mutation):
    run = save_run(positive_bundle, tmp_path, full=True)
    saved = json.loads((tmp_path / 'learned_model.json').read_text())
    dynamic = saved['selection']['optimizer_sampling_train']
    if mutation == 'missing_active_hash':
        del dynamic['term_weight_sha256']['event_cdf']
    else:
        dynamic['groups'][0]['physical_unit_id'] = 'synthetic-forged-group'
    # Both copies agree and the artifact hash is valid. Rejection must therefore
    # rely on semantic binding, not a stale hash or metadata/manifest mismatch.
    run['selection'] = copy.deepcopy(saved['selection'])
    (tmp_path / 'learned_model.json').write_text(json.dumps(saved))
    run['artifacts']['learned_model.json'] = sha256_file(tmp_path / 'learned_model.json')
    with pytest.raises(ValueError):
        learned.load_learned_bundle(run)
