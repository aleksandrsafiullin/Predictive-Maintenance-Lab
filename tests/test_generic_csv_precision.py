"""Canonical public measurements must survive a lossless CSV export/import."""
from pathlib import Path

import numpy as np
import pandas as pd

from pdm.data.generic_csv import read_generic_csv


def test_observed_measurements_and_age_keep_float64_precision(tmp_path: Path):
    # Actual HSE canonical values that the default CSV converter rounded by
    # one or more ULPs, including the clock and a pressure measurement.
    expected = pd.DataFrame({
        "unit_id": ["Train_1"] * 3,
        "timestamp_s": [0.1, 0.3, 0.6],
        "pressure": [0.271267, 98.19878, 100.0],
        "operating_age_s": [0.1, 0.3, 0.6],
        "physical_unit_id": ["Train_1"] * 3,
        "component_cycle_id": ["0"] * 3,
    })
    source = tmp_path / "history.csv"
    expected.to_csv(source, index=False, float_format="%.17g")
    features, units, _ = read_generic_csv(
        {"primary": [{"path": str(source), "relative_path": source.name}]},
        "pressure",
        context_mapping={name: name for name in (
            "operating_age_s", "physical_unit_id", "component_cycle_id")},
        age_source="laboratory_proxy",
    )
    for target, original in (("timestamp_s", "timestamp_s"),
                             ("signal", "pressure"),
                             ("operating_age_s", "operating_age_s")):
        np.testing.assert_array_equal(features[target], expected[original])
    assert features["physical_unit_id"].tolist() == ["Train_1"] * 3
    assert features["component_cycle_id"].astype(str).tolist() == ["0"] * 3
    assert units["unit_id"].tolist() == ["Train_1"]
