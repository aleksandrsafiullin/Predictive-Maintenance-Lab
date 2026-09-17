"""Read-only causal/runtime/trace/export verification on real saved measurements."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

from pdm.data.prepare import load_processed
from pdm.io_util import atomic_write_json, read_json, sha256_file
from pdm.monitoring.bundle import bundle_root, load_bundle, runtime_args
from pdm.monitoring.runtime import replay_monitoring


@threadpool_limits.wrap(limits=1)
def verify(bundle_id):
    torch.set_num_threads(1)
    bundle, predictor, sensor = load_bundle(bundle_id)
    data = load_processed(bundle['profile']['dataset_id'], bundle['dataset_version'])
    uid = max(data['split']['validation'], key=lambda u: int(data['features'].unit_id.eq(u).sum()))
    frame = data['features'].loc[lambda f: f.unit_id.eq(uid)].head(66).copy()
    args = runtime_args(bundle, predictor, sensor)
    rows, state = replay_monitoring(frame, **args)
    checks = []
    for n in (20, 40, 60, len(frame)):
        prefix = frame.iloc[:n]
        now = float(prefix.timestamp_s.iloc[-1])
        seek, seek_state = replay_monitoring(frame, as_of=now, **args)
        assert seek == rows[:n]
        changed = frame.copy()
        changed.loc[changed.timestamp_s > now, bundle['profile']['signal_name']] = 1e9
        changed['event_time_s'], changed['life_fraction'], changed['RUL'] = 1., .99, -123.
        altered, _ = replay_monitoring(changed, as_of=now, **args)
        assert altered == seek
        plain = predictor.predict_from_history(prefix)
        traced = predictor.predict_from_history(prefix, with_trace=True)
        a, b = plain.get('predicted_rul_s'), traced.get('predicted_rul_s')
        assert (a is None and b is None) or np.isclose(a, b, rtol=1e-5, atol=1e-4), (n,a,b)
        runtime = seek[-1]['event_forecast']['point']
        if runtime is not None:
            assert np.isclose(runtime, a, rtol=1e-6, atol=1e-4)
        json.dumps(seek, allow_nan=False)
        checks.append({'measurements': n, 'seek_equals_sequential': True,
            'future_and_ground_truth_independent': True, 'trace_equals_predict': True,
            'runtime_point': runtime, 'history': seek[-1]['history']})
    exports = []
    for path in sorted((bundle_root(bundle_id)/'evaluations').glob('*/evaluation.json')):
        evaluation = read_json(path)
        for name, digest in evaluation['artifact_hashes'].items():
            if name != 'feedback.jsonl':
                assert sha256_file(path.parent/name) == digest
        saved = [json.loads(line) for line in (path.parent/'observations.jsonl').read_text().splitlines()]
        states = pd.read_parquet(path.parent/'state_history.parquet')
        signals = pd.read_parquet(path.parent/'signal_forecasts.parquet')
        assert len(states) == len(saved) == evaluation['processed_measurements']
        assert states.display_zone.tolist() == [r['condition']['display_zone'] for r in saved]
        flat = [f for r in saved for f in r['signal_forecasts']]
        assert len(signals) == len(flat)
        np.testing.assert_allclose(pd.to_numeric(signals.point), pd.to_numeric(pd.Series([f['point'] for f in flat])), equal_nan=True)
        if evaluation['split'] == 'validation':
            saved_prefix = [r for r in saved if r['unit_id'] == uid][:len(rows)]
            assert rows == saved_prefix
        exports.append({'eval_id': evaluation['eval_id'], 'split': evaluation['split'],
            'measurements': len(saved), 'hashes_valid': True, 'json_parquet_runtime_equal': True})
    return {'bundle_id': bundle_id, 'real_unit': uid, 'checks': checks, 'exports': exports,
        'status': 'passed', 'test_used_for_tuning': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('bundle_id')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = verify(args.bundle_id)
    atomic_write_json(Path(args.output), result)
    print(json.dumps(result, indent=2))
