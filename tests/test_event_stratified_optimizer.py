"""Deterministic synthetic sampler evidence; no data, fits, or quality claims."""
import importlib.util
import itertools
from pathlib import Path

import numpy as np
import pytest
import torch

from pdm import learned_trajectory as learned


def frame(h=32):
    allowed = np.zeros((6, h + 1), bool)
    allowed[0, 2] = allowed[3, 5] = allowed[4, 9] = True
    return dict(x=np.zeros((6, 2, 1), np.float32), mask=np.ones((6, h), bool),
                current=np.array([0., 0., 0., 0., 0., 1.]), red_threshold=np.ones(6),
                event_allowed=allowed, event_observed=np.array([1, 0, 0, 1, 1, 0], bool),
                no_entry_prefix=np.array([0, 30, 0, 0, 0, 0]),
                physical_unit_id=['a'] * 3 + ['b'] * 3)


def test_settings_explicit_dense_first_prefix_and_legacy_none():
    config = dict(batch_sampling='physical_event_stratified', horizons_s=list(np.arange(1, 33)*60.),
                  event_objective_horizons_s=[1800., 1920.])
    settings = learned._event_stratified_sampling_settings(config)
    assert settings['prefix_length'] == 30 and settings['prefix_horizon_s'] == 1800.
    assert settings['gradient_scale'] == 1 and settings['replacement'] is True
    for mode in ('row_permutation', 'physical_group'):
        assert learned._event_stratified_sampling_settings({'batch_sampling': mode}) is None
    for bad in (None, [], [1799.], [1920., 1800.]):
        with pytest.raises(ValueError):
            learned._event_stratified_sampling_settings({**config, 'event_objective_horizons_s': bad})
    for grid in ([0., 60.], [60., 180.], [60., np.nan], [[60.]], []):
        with pytest.raises(ValueError):
            learned._event_stratified_sampling_settings({**config, 'horizons_s': grid})


@pytest.mark.parametrize('prefix', [29, 30, 31])
def test_canonical_boundary_observed_priority_disjoint_unknown_and_red(prefix):
    f = frame()
    # Exact class 29 enters by lead30; censor30 proves survival through30.
    f['event_allowed'][0] = False
    f['event_allowed'][0, 29] = True
    f['no_entry_prefix'][0] = 32  # Observed wins.
    f['event_allowed'][3] = False
    f['event_allowed'][3, [28, 30]] = True  # Disjoint bracket straddles30.
    f['event_allowed'][4] = False
    f['event_observed'][4] = False
    f['no_entry_prefix'][4] = 29
    p = learned._event_stratified_plan(f, prefix)
    expected = {29: [1, 1, 2, 2, 1, 3], 30: [0, 1, 2, 2, 2, 3], 31: [0, 2, 2, 0, 2, 3]}
    assert p['row_stratum'].tolist() == expected[prefix]
    assert sum(p['counts']['stratum_origin_counts'].values()) == 6
    assert p['probabilities'].sum() == pytest.approx(1)


@pytest.mark.parametrize('key,bad', [('current', np.full(6, np.nan)),
    ('red_threshold', np.zeros(6)), ('mask', np.ones((5,32),bool)),
    ('event_observed', np.zeros(5,bool)), ('physical_unit_id', ['a']*5),
    ('no_entry_prefix', np.full(6,33)), ('event_allowed',np.zeros((6,32),bool)),
    ('physical_unit_id', ['a']*5+[None])])
def test_invalid_frames_rejected(key,bad):
    with pytest.raises(ValueError):
        learned._event_stratified_plan({**frame(),key:bad},30)


@pytest.mark.parametrize('size', [1,2,3,4,6,10])
def test_budget_balance_replacement_and_identifier_renaming(size):
    f=frame();p=learned._event_stratified_plan(f,30)
    renamed={**f,'physical_unit_id':['z']*3+['q']*3}
    q=learned._event_stratified_plan(renamed,30)
    a=list(learned._optimizer_batches(f,size,np.random.default_rng(123),'physical_event_stratified',event_stratified_plan=p))
    b=list(learned._optimizer_batches(renamed,size,np.random.default_rng(123),'physical_event_stratified',event_stratified_plan=q))
    assert [len(i) for i,_ in a] == [len(i) for i in np.array_split(np.arange(6),int(np.ceil(6/size)))]
    for (idx,pi),(other,other_pi) in zip(a,b):
        assert np.array_equal(idx,other) and np.array_equal(pi,other_pi)
        assert np.array_equal(pi,p['probabilities'][idx])
        group_counts=np.bincount(p['row_group'][idx],minlength=2)
        assert np.ptp(group_counts)<=1
        for g, strata in enumerate(p['members']):
            counts=[np.isin(idx,rows).sum() for rows in strata]
            assert np.ptp(counts)<=1
    assert sum(len(i) for i,_ in a)==6
    if size>=6:
        assert any(len(np.unique(idx))<len(idx) for seed in range(10)
                   for idx,_ in learned._optimizer_batches(f,size,np.random.default_rng(seed),
                       'physical_event_stratified',event_stratified_plan=p))


class Branch(Exception):
    def __init__(self,options): self.options=options


class ExhaustRNG:
    """Explore every actual RNG call; equal outcomes have equal probability."""
    def __init__(self,answers): self.answers=iter(answers)
    def draw(self,options):
        try: return next(self.answers)
        except StopIteration: raise Branch(options)
    def choice(self,a,size,replace):
        pool=range(a) if np.isscalar(a) else a
        choices=list(itertools.product(pool,repeat=size) if replace else itertools.permutations(pool,size))
        return np.asarray(self.draw(choices),dtype=int)
    def permutation(self,a):
        return np.asarray(self.draw(sorted(set(itertools.permutations(a)))),dtype=int)


@pytest.mark.parametrize('size',[1,2,3])
def test_exhaust_actual_sampler_marginals_reductions_and_gradients(size):
    f=frame();p=learned._event_stratified_plan(f,30)
    pending=[([],1.)];draws=np.zeros(6);means=np.zeros(7);gradients=np.zeros(7);mass=0.
    weights=list(learned._objective_weights(f).values())+[learned._at_risk_weights(f),np.zeros(6,np.float32)]
    weight=np.asarray(weights)
    values=np.arange(1.,7.)**2
    while pending:
        answers,prob=pending.pop()
        try:
            idx,pi=next(learned._optimizer_batches(f,size,ExhaustRNG(answers),'physical_event_stratified',event_stratified_plan=p))
        except Branch as branch:
            pending.extend((answers+[option],prob/len(branch.options)) for option in branch.options)
            continue
        mass+=prob;draws+=prob*np.bincount(idx,minlength=6)/len(idx)
        theta=torch.ones(7,dtype=torch.float64,requires_grad=True)
        estimates=torch.stack([learned._optimizer_term(theta[k]*torch.as_tensor(values[idx]),w,idx,pi)
                               for k,w in enumerate(weight)])
        means+=prob*estimates.detach().numpy()
        gradients+=prob*torch.autograd.grad(estimates.sum(),theta)[0].numpy()
    assert mass==pytest.approx(1.)
    assert draws==pytest.approx(p['probabilities'],abs=1e-12)
    assert means==pytest.approx(weight@values,abs=1e-10)
    assert gradients==pytest.approx(weight@values,abs=1e-10)
    assert means[-1]==gradients[-1]==0


def test_old_sampler_source_and_rng_parity():
    baseline=Path('output/bearings-learned-funnel-20261002/review/pre_event_stratified_learned_trajectory.py')
    spec=importlib.util.spec_from_file_location('pre_event_port',baseline)
    old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    for mode in ('row_permutation','physical_group'):
        a=np.random.default_rng(555);b=np.random.default_rng(555)
        for size in (1,2,4,6,10):
            left=list(learned._optimizer_batches(frame(),size,a,mode));right=list(old._optimizer_batches(frame(),size,b,mode))
            for (i,p),(j,q) in zip(left,right):
                assert np.array_equal(i,j)
                assert p is q is None or np.array_equal(p,q)
            assert a.bit_generator.state==b.bit_generator.state
    import inspect
    for name in ('_weights','_objective_weights','_at_risk_weights','_optimizer_term','_make_optimizer'):
        assert inspect.getsource(getattr(learned,name))==inspect.getsource(getattr(old,name))


@pytest.mark.parametrize('mutation',["probabilities","counts","row_group","row_stratum"])
def test_plan_memberships_and_exact_probabilities_are_required(mutation):
    import copy
    f=frame();p=copy.deepcopy(learned._event_stratified_plan(f,30))
    if mutation=='probabilities': p[mutation][:]=1/6
    elif mutation=='counts': p[mutation]['total_origin_count']+=1
    else: p[mutation][0]=99
    with pytest.raises(ValueError):
        list(learned._optimizer_batches(f,3,np.random.default_rng(1),'physical_event_stratified',event_stratified_plan=p))
    with pytest.raises(ValueError):
        list(learned._optimizer_batches(f,3,np.random.default_rng(1),'physical_event_stratified'))
