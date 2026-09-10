from __future__ import annotations

from typing import Any

import numpy as np
import torch

from pdm.models import PDMNet
from pdm.preprocessing import Preprocessor


class Predictor:
    """Uses only allowed features already observed up to t. No trial length / RUL / split."""

    def __init__(
        self,
        model: PDMNet,
        preprocessor: Preprocessor,
        history_length: int,
        device: str = "cpu",
    ) -> None:
        self.model = model.to(device)
        self.model.eval()
        self.prep = preprocessor
        self.history_length = int(history_length)
        self.device = device
        if list(self.prep.feature_names) != list(getattr(model, "feature_names", self.prep.feature_names)):
            # feature order is owned by the preprocessor / checkpoint schema
            pass

    @torch.no_grad()
    def predict_from_history(self, history_rows) -> dict[str, Any]:
        """history_rows: DataFrame of raw-or-transformed? Expect RAW feature table rows sorted by time, only <= t."""
        n = len(history_rows)
        if n < self.history_length:
            return {
                "predicted_rul_s": None,
                "status": "Collecting history",
                "n_history": n,
            }
        window = history_rows.iloc[-self.history_length :].copy()
        transformed = self.prep.transform_frame(window)
        arr = transformed[self.prep.feature_names].to_numpy(dtype=np.float32)
        x = torch.from_numpy(arr).unsqueeze(0).to(self.device)
        rul = self.model.predicted_rul_s(x)
        value = float(rul.detach().cpu().reshape(-1)[0].item())
        if not np.isfinite(value) or value < 0:
            return {"predicted_rul_s": None, "status": "No valid prediction", "n_history": n}
        return {"predicted_rul_s": value, "status": "ok", "n_history": n}
