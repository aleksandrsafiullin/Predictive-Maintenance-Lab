from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np
import pandas as pd
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


def prepare_history_window(
    history_rows: pd.DataFrame,
    prep: Preprocessor,
    history_length: int,
    *,
    gap_multiplier: float | None = None,
    sampling_interval_s: float | None = None,
) -> dict[str, Any]:
    """Causal last-``history_length`` window. Same prefix slice as ``Predictor``.

    Does not pad with future frames. Returns ``ok=False`` with
    ``status='Collecting history'`` when the prefix is not yet a valid window.
    """
    n = len(history_rows)
    history = history_rows
    ds = prep.dataset_id
    gap_m = gap_multiplier
    samp = sampling_interval_s
    if ds == "filters":
        if gap_m is None:
            gap_m = getattr(prep, "gap_multiplier", None)
        if samp is None:
            samp = getattr(prep, "sampling_interval_s", None)
        gap_m, samp = resolve_filter_gap_params(
            gap_multiplier=gap_m,
            sampling_interval_s=samp,
            allow_yaml_fallback=True,
        )
        history = recompute_filter_gap_before(
            history,
            gap_multiplier=gap_m,
            sampling_interval_s=samp,
            causal=True,
        )
    ok, reason = valid_history_window(
        history,
        int(history_length),
        dataset_id=ds,
        gap_multiplier=gap_m,
        sampling_interval_s=samp,
    )
    if not ok:
        return {
            "ok": False,
            "status": "Collecting history",
            "n_history": n,
            "valid_history_reason": reason,
            "window": None,
            "inputs": None,
        }
    window = history.sort_values("timestamp_s").iloc[-int(history_length) :].copy()
    if prep.feature_pipeline_version != FEATURE_PIPELINE_VERSION:
        raise ValueError(
            "Preprocessor feature_pipeline_version is missing or not v2_raw_first; "
            "retrain the model so preprocessing.json includes feature_pipeline_version "
            f"'{FEATURE_PIPELINE_VERSION}'."
        )
    if not prep.dataset_id:
        raise ValueError("Preprocessor missing dataset_id for raw feature pipeline")
    encoded = raw_to_feature_frame(
        prep.dataset_id,
        window,
        prep.categorical_maps,
        prep.log1p_features,
        raw_features=True,
    )
    transformed = prep.transform_frame(encoded)
    arr = transformed[prep.feature_names].to_numpy(dtype=np.float32)
    return {
        "ok": True,
        "status": "ok",
        "n_history": n,
        "valid_history_reason": "",
        "window": window,
        "inputs": arr,
    }


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
    def predict_from_history(
        self,
        history_rows,
        unit_id: str | None = None,
        with_trace: bool = False,
        *,
        lazy: bool = False,
    ) -> dict[str, Any]:
        """history_rows: raw measurement rows sorted by time, only observations <= t.

        Pass ``with_trace=True`` for the architecture's actual computed states.
        Trace collection preserves the point forecast and the saved memory mode.
        """
        if with_trace:
            from pdm.visualization.trace import is_reservoir_module, predict_with_trace

            if is_reservoir_module(self.model):
                uid = str(unit_id) if unit_id is not None else ""
                result = predict_with_trace(
                    history_rows,
                    uid,
                    self.model,
                    self.prep,
                    lazy=lazy,
                    history_length=self.history_length,
                    device=self.device,
                    gap_multiplier=self.gap_multiplier,
                    sampling_interval_s=self.sampling_interval_s,
                )
            else:
                from pdm.visualization.recurrent_trace import recurrent_trace

                result = recurrent_trace(history_rows, self.model, self.prep, self.history_length, self.device)
        else:
            result = self._predict_from_prepared(history_rows)
        timestamps = pd.to_numeric(history_rows.get("timestamp_s", pd.Series(dtype=float)), errors="coerce")
        result["timestamp_s"] = float(timestamps.max()) if timestamps.notna().any() else None
        result["ready"] = result.get("predicted_rul_s") is not None and result.get("status") in {"ok", "predicted"}
        result["state_mode"] = getattr(self.model, "state_mode", "window_reset")
        return result

    def _predict_from_prepared(self, history_rows) -> dict[str, Any]:
        if getattr(self.model, "state_mode", None) == "continuous":
            from pdm.visualization.simulation import continuous_trace

            trace = continuous_trace(history_rows, self.model, self.prep, self.history_length)
            result = {key: trace[key] for key in (
                "predicted_rul_s", "status", "n_history", "valid_history_reason"
            )}
            if result["status"] == "predicted":
                result["status"] = "ok"
            result["raw_rul_s"] = float(trace["raw_rul_s"][-1]) if len(trace.get("raw_rul_s", [])) else None
            result["forecast_method"] = "continuous_raw_readout"
            return result
        prepared = prepare_history_window(
            history_rows,
            self.prep,
            self.history_length,
            gap_multiplier=self.gap_multiplier,
            sampling_interval_s=self.sampling_interval_s,
        )
        n = prepared["n_history"]
        if not prepared["ok"]:
            return {
                "predicted_rul_s": None,
                "status": "Collecting history",
                "n_history": n,
                "valid_history_reason": prepared["valid_history_reason"],
            }
        arr = prepared["inputs"]
        x = torch.from_numpy(np.array(arr, order="C", copy=True)).unsqueeze(0).to(self.device)
        output = model_forecast(self.model, x)
        value = output["predicted_rul_s"]
        if not np.isfinite(value) or value < 0:
            return {
                "predicted_rul_s": None,
                "status": "No valid prediction",
                "n_history": n,
                "valid_history_reason": "",
            }
        return {
            **output,
            "status": "ok",
            "n_history": n,
            "valid_history_reason": "",
        }


@torch.no_grad()
def model_forecast(model, x):
    """Point forecast and optional distribution from the same saved model head."""
    import math

    from pdm.losses import weibull_median_rul

    result = {"forecast_method": "window_point", "interval_method": None}
    if getattr(model, "head_type", None) == "weibull":
        lam, k = model(x)
        value = float(weibull_median_rul(lam, k, model.time_scale_s).reshape(-1)[0].cpu())
        scale = float(lam.reshape(-1)[0].cpu()) * model.time_scale_s
        shape = float(k.reshape(-1)[0].cpu())
        result.update(weibull_scale_s=scale, weibull_shape=shape,
                      interval_method="Weibull distribution (5–95%; not empirically calibrated)",
                      forecast_method="window_weibull")
        try:
            lo = scale * (-math.log(0.95)) ** (1 / shape)
            hi = scale * (-math.log(0.05)) ** (1 / shape)
        except (OverflowError, ZeroDivisionError):
            lo = hi = float("nan")
        if np.isfinite([lo, hi]).all() and 0 <= lo <= hi:
            result.update(lower_rul_s=lo, upper_rul_s=hi)
    else:
        value = float(model.predicted_rul_s(x).reshape(-1)[0].cpu())
    result.update(predicted_rul_s=value, raw_rul_s=value)
    return result
