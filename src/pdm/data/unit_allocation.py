"""Shared deterministic allocation of complete physical unit identities."""
from __future__ import annotations

import math

import numpy as np


def allocate_whole_units(primary_ids, automatic_roles, weights, seed):
    """Largest remainder, minimum one, then seeded permutation of sorted IDs."""
    primary = sorted(primary_ids)
    automatic = list(automatic_roles)
    if not automatic or len(primary) < len(automatic):
        raise ValueError(f"Primary pool needs at least {len(automatic)} physical units for automatic groups; found {len(primary)}")
    if len(primary) != len(set(primary)):
        raise ValueError("Duplicate physical unit IDs")
    total_weight = sum(float(weights[name]) for name in automatic)
    proportions = {name: float(weights[name]) / total_weight for name in automatic}
    quotas = {name: len(primary) * proportions[name] for name in automatic}
    counts = {name: int(math.floor(quotas[name])) for name in automatic}
    leftover = len(primary) - sum(counts.values())
    order = sorted(automatic, key=lambda name: (-(quotas[name] % 1), automatic.index(name)))
    for name in order[:leftover]:
        counts[name] += 1
    for name in automatic:
        if counts[name] == 0:
            donor = max((candidate for candidate in automatic if counts[candidate] > 1),
                        key=lambda candidate: (counts[candidate] - quotas[candidate], counts[candidate]),
                        default=None)
            if donor is None:
                raise ValueError("Too few primary units for the requested automatic groups")
            counts[donor] -= 1
            counts[name] = 1
    permutation = np.random.default_rng(seed).permutation(primary).tolist()
    allocated, cursor = {}, 0
    for name in automatic:
        allocated[name] = sorted(permutation[cursor:cursor+counts[name]])
        cursor += counts[name]
    return allocated
