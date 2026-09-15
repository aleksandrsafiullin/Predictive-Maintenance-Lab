import copy

import numpy as np
import pytest

from pdm.config import load_dataset_config
from pdm.data.bearings import _read_vibration_csv
from pdm.data.quality import QUALITY_POLICY_HASH, admit_dataset, training_admission
from pdm.splits import bearings_split, filters_split
from pdm.windows import build_windows, recompute_filter_gap_before


def test_invalid_middle_fragment_creates_gap_without_moving_endpoint(tiny_bearing_tables):
    f, u = tiny_bearing_tables
    cfg = load_dataset_config('bearings')
    split = bearings_split(u, cfg)
    f.loc[(f.unit_id == 'Bearing1_1') & (f.file_index == 10), 'horizontal_rms'] = np.inf
    clean, units, active, audit, report = admit_dataset('bearings', f, u, split, cfg)
    assert report['excluded_measurements'] == 1
    assert units.set_index('unit_id').loc['Bearing1_1', 'event_time_s'] == 1800
    assert clean.loc[(clean.unit_id == 'Bearing1_1') & (clean.file_index == 11), 'gap_before'].item()
    assert active['train'] == split['train']
    windows = build_windows(clean, units, 5, 'bearings')
    assert not windows[(windows.unit_id == 'Bearing1_1') & windows.timestamp_s.between(660, 840)].shape[0]
    assert 'nonfinite_required_measurement' in ';'.join(audit.reasons)


def test_bad_endpoint_excludes_unit_instead_of_relabelling(tiny_bearing_tables):
    f, u = tiny_bearing_tables
    cfg = load_dataset_config('bearings')
    split = bearings_split(u, cfg)
    f.loc[(f.unit_id == 'Bearing1_1') & (f.file_index == 30), 'horizontal_rms'] = np.nan
    clean, units, active, audit, report = admit_dataset('bearings', f, u, split, cfg)
    assert 'Bearing1_1' not in set(units.unit_id)
    assert 'Bearing1_1' not in active['train']
    assert 'Bearing1_1' in report['original_split']['train']
    assert 'unreliable_recorded_endpoint' in report['excluded_units']['Bearing1_1']


def test_duplicates_and_degradation(tiny_bearing_tables):
    import pandas as pd
    f, u = tiny_bearing_tables
    cfg = load_dataset_config('bearings')
    split = bearings_split(u, cfg)
    extra = f.iloc[[5]].copy()
    f = pd.concat([f, extra], ignore_index=True)
    f.loc[(f.unit_id == 'Bearing1_1') & (f.file_index == 29), 'horizontal_rms'] = 1e7
    clean, _, _, audit, report = admit_dataset('bearings', f, u, split, cfg)
    assert report['excluded_measurements'] == 1
    assert clean.horizontal_rms.max() == 1e7
    assert (audit.reasons == 'duplicate_identical').sum() == 1
    f.iloc[-1, f.columns.get_loc('horizontal_rms')] = -10
    clean, _, _, audit, report = admit_dataset('bearings', f, u, split, cfg)
    assert (audit.reasons.str.contains('duplicate_conflict')).sum() == 2


def test_censored_rows_retained_and_ablation_keeps_validation(tiny_filter_tables):
    f, u = tiny_filter_tables
    cfg = load_dataset_config('filters')
    split = filters_split(u, cfg)
    clean, units, active, audit, report = admit_dataset('filters', f, u, split, cfg)
    assert len(clean) == len(f)
    assert len(units) == len(u)
    assert (units.event_observed == 0).sum() == 8
    bundle = dict(features=clean, units=units, split=active, fingerprint={'quality_policy_hash': QUALITY_POLICY_HASH})
    ablation, counts = training_admission(bundle, cfg, 5, events_only=True)
    assert len(ablation['train']) < len(active['train'])
    assert ablation['validation'] == active['validation']
    assert active['train'] == split['train']
    bad = copy.deepcopy(bundle)
    bad['fingerprint'] = {}
    with pytest.raises(ValueError, match='Prepare data'):
        training_admission(bad, cfg, 5)


def test_quality_gap_survives_filter_causal_recomputation(tiny_filter_tables):
    f, _ = tiny_filter_tables
    unit = f[f.unit_id == 'Filter_1'].copy()
    unit['quality_gap_before'] = False
    unit.loc[unit.index[10], 'quality_gap_before'] = True
    rebuilt = recompute_filter_gap_before(unit)
    assert rebuilt.iloc[10].gap_before


@pytest.mark.parametrize('content', [b'a,b\n1,2,3,4\n', b'a,b\n1,2\n3,broken\n', b'a,b\n1\n2\n'])
def test_raw_parser_does_not_silently_reshape_or_truncate(content):
    with pytest.raises(ValueError):
        _read_vibration_csv(content, 2)
