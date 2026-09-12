from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np
import torch
from torch import nn

from pdm.preprocessing import FEATURE_PIPELINE_VERSION, Preprocessor, raw_to_feature_frame
from pdm.windows import recompute_filter_gap_before, resolve_filter_gap_params, valid_history_window


@runtime_checkable
class HasPredictedRUL(Protocol):
    """Duck-typed forecast module: GRU/LSTM ``PDMNet`` or a reservoir."""

    def eval(self) -> nn.Module: ...

    def to(self, device: Any) -> nn.Module: ...

    def predicted_rul_s(self, x: torch.Tensor) -> torch.Tensor: ...


class Predictor:
    """Uses only allowed features already observed up to t. No trial length / RUL / split."""

    def __init__(
        self,
        model: nn.Module | HasPredictedRUL,
        preprocessor: Preprocessor,
        history_length: int,
        device: str = "cpu",
    ) -> None:
        self.model = model.to(device)
        self.model.eval()
        self.prep = preprocessor
        self.history_length = int(history_length)
        self.device = device
        self.gap_multiplier: float | None = None
        self.sampling_interval_s: float | None = None
        if getattr(preprocessor, "dataset_id", None) == "filters":
            # Saved on preprocessing.json; live yaml only if the run snapshot omitted them.
            self.gap_multiplier, self.sampling_interval_s = resolve_filter_gap_params(
                gap_multiplier=getattr(preprocessor, "gap_multiplier", None),
                sampling_interval_s=getattr(preprocessor, "sampling_interval_s", None),
                allow_yaml_fallback=True,
            )
        if list(self.prep.feature_names) != list(getattr(model, "feature_names", self.prep.feature_names)):
            # feature order is owned by the preprocessor / checkpoint schema
            pass

    @torch.no_grad()
    def predict_from_history(self, history_rows) -> dict[str, Any]:
        """history_rows: raw measurement rows sorted by time, only observations <= t."""
        n = len(history_rows)
        history = history_rows
        ds = self.prep.dataset_id
        if ds == "filters":
            history = recompute_filter_gap_before(
                history,
                gap_multiplier=self.gap_multiplier,
                sampling_interval_s=self.sampling_interval_s,
                causal=True,
            )
        ok, reason = valid_history_window(
            history,
            self.history_length,
            dataset_id=ds,
            gap_multiplier=self.gap_multiplier,
            sampling_interval_s=self.sampling_interval_s,
        )
        if not ok:
            return {
                "predicted_rul_s": None,
                "status": "Collecting history",
                "n_history": n,
                "valid_history_reason": reason,
            }
        window = history.sort_values("timestamp_s").iloc[-self.history_length :].copy()
        if self.prep.feature_pipeline_version != FEATURE_PIPELINE_VERSION:
            raise ValueError(
                "Preprocessor feature_pipeline_version is missing or not v2_raw_first; "
                "retrain the model so preprocessing.json includes feature_pipeline_version "
                f"'{FEATURE_PIPELINE_VERSION}'."
            )
        if not self.prep.dataset_id:
            raise ValueError("Preprocessor missing dataset_id for raw feature pipeline")
        encoded = raw_to_feature_frame(
            self.prep.dataset_id,
            window,
            self.prep.categorical_maps,
            self.prep.log1p_features,
            raw_features=True,
        )
        transformed = self.prep.transform_frame(encoded)
        arr = transformed[self.prep.feature_names].to_numpy(dtype=np.float32)
        x = torch.from_numpy(arr).unsqueeze(0).to(self.device)
        rul = self.model.predicted_rul_s(x)
        value = float(rul.detach().cpu().reshape(-1)[0].item())
        if not np.isfinite(value) or value < 0:
            return {
                "predicted_rul_s": None,
                "status": "No valid prediction",
                "n_history": n,
                "valid_history_reason": "",
            }
        return {
            "predicted_rul_s": value,
            "status": "ok",
            "n_history": n,
            "valid_history_reason": "",
        }
