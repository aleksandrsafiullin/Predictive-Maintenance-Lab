"""Resume must detect damaged future-red model artifacts."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from pdm.io_util import sha256_file


def _runner():
    script = Path(__file__).resolve().parents[1] / "scripts" / "run_future_red_matrix.py"
    spec = importlib.util.spec_from_file_location("run_future_red_matrix", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_resume_rejects_modified_checkpoint_and_missing_outputs(tmp_path: Path) -> None:
    for name in ("predictions.parquet", "baseline_predictions.parquet",
                 "per_unit_metrics.csv", "baseline_metrics.json", "model.pt"):
        (tmp_path / name).write_bytes(b"saved artifact")
    names = ("predictions.parquet", "baseline_predictions.parquet",
             "per_unit_metrics.csv", "baseline_metrics.json", "model.pt")
    (tmp_path / "manifest.json").write_text(json.dumps({
        "artifact_hashes": {name: sha256_file(tmp_path / name) for name in names},
    }))
    valid = _runner()._completed_output_is_valid
    assert valid(tmp_path, "gru")
    (tmp_path / "model.pt").write_bytes(b"changed checkpoint")
    assert not valid(tmp_path, "gru")
    (tmp_path / "model.pt").write_bytes(b"saved artifact")
    (tmp_path / "predictions.parquet").write_bytes(b"altered predictions")
    assert not valid(tmp_path, "gru")
    (tmp_path / "predictions.parquet").write_bytes(b"saved artifact")
    (tmp_path / "predictions.parquet").unlink()
    assert not valid(tmp_path, "gru")
