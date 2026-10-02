"""Causal equipment steps and inspectable terms from the saved reservoir."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from pdm.models.reservoir import recurrent_drive
from pdm.preprocessing import FEATURE_PIPELINE_VERSION, apply_preprocessor
from pdm.visualization.contributions import decompose_contributions
from pdm.visualization.trace import predict_with_trace


def simulate_step(features, unit_id, timestamp_s, model, prep, history_length, previous_trace=None) -> dict[str, Any]:
    """Score only this unit's observed prefix. Ground truth never enters inference."""
    prefix = features.loc[
        (features["unit_id"].astype(str) == str(unit_id))
        & (pd.to_numeric(features["timestamp_s"], errors="coerce") <= float(timestamp_s))
    ].sort_values("timestamp_s").copy()
    if hasattr(model, "encoder"):
        from pdm.visualization.recurrent_trace import recurrent_trace

        return recurrent_trace(prefix, model, prep, history_length)
    if getattr(model, "state_mode", None) == "continuous":
        return continuous_trace(prefix, model, prep, history_length, previous_trace=previous_trace)
    trace = predict_with_trace(
        prefix, str(unit_id), model, prep, history_length=history_length, device="cpu"
    )
    if trace["status"] != "predicted":
        return trace
    # Diagnostic decomposition only. States still come from the shared ESN kernel.
    states = torch.from_numpy(trace["states"])
    inputs = torch.from_numpy(trace["inputs"].copy())
    previous = torch.cat([torch.zeros_like(states[:1]), states[:-1]], dim=0)
    with torch.no_grad():
        trace["processing"] = {
            "input_drive": F.linear(inputs, model.W_in).numpy(),
            "recurrent_drive": recurrent_drive(previous, model.W_res).numpy(),
            "previous_state": previous.numpy(),
            "bias": model.b_res.detach().numpy().copy(),
            "leak": float(model.alpha),
        }
    return trace


def continuous_trace(prefix, model, prep, warmup, *, previous_trace=None, should_stop=None):
    """Exact chronological ESN states. Cached prefixes only accelerate append-only steps."""
    frame = prefix.sort_values("timestamp_s").copy().reset_index(drop=True)
    from pdm.visualization.trace import _display_from_raw, _empty_trace

    if frame.empty:
        return _empty_trace(model, list(model.node_order), 0)
    if prep.feature_pipeline_version != FEATURE_PIPELINE_VERSION:
        raise ValueError("Continuous forecasting requires the saved raw-first feature pipeline")
    if prep.dataset_id == "filters":
        from pdm.windows import recompute_filter_gap_before

        frame = recompute_filter_gap_before(frame, gap_multiplier=prep.gap_multiplier,
                                           sampling_interval_s=prep.sampling_interval_s, causal=True)
    if "unit_id" in frame and frame["unit_id"].astype(str).nunique() != 1:
        raise ValueError("Continuous history must belong to one bearing")
    ts = frame["timestamp_s"].to_numpy(dtype=float)
    if not np.isfinite(ts).all() or np.any(np.diff(ts) <= 0):
        raise ValueError("Measurement timestamps must be finite and strictly increasing")
    if "gap_before" in frame:
        gaps = np.flatnonzero(frame["gap_before"].fillna(False).to_numpy(dtype=bool))
        if len(gaps):
            frame = frame.iloc[int(gaps[-1]):].reset_index(drop=True)
            ts = frame["timestamp_s"].to_numpy(dtype=float)
    encoded = apply_preprocessor(prep, frame)
    inputs = encoded[prep.feature_names].to_numpy(dtype=np.float32, copy=True)
    if hasattr(model, "pool_index"):
        return _full_cns_trace(inputs, ts, model, warmup, previous_trace, should_stop=should_stop)
    previous_trace = previous_trace or {}
    old_inputs = previous_trace.get("inputs")
    old_ts = previous_trace.get("timestamps_s")
    old_n = len(old_inputs) if old_inputs is not None else 0
    identity = (id(model), model.W_in._version, model.W_res._version, model.b_res._version, float(model.alpha))
    reuse = (old_n > 0 and old_n < len(inputs) and old_ts is not None
             and np.array_equal(ts[:old_n], old_ts)
             and np.array_equal(inputs[:old_n], old_inputs)
             and previous_trace.get("model_identity") == identity)
    model.eval()
    device = model.W_in.device
    with torch.no_grad():
        x = torch.from_numpy(inputs).to(device)
        if reuse:
            cached = torch.from_numpy(previous_trace["states"]).to(device)
            fresh = model.forward_states(x[old_n:], x0=cached[-1])
            states = torch.cat([cached, fresh], dim=0)
        else:
            states = model.forward_states(x)
        raw = model.forward_raw(states, x)
        raw_rul = _display_from_raw(model, raw).detach().cpu().numpy()
        prev = states[-2:-1] if len(states) > 1 else torch.zeros_like(states[-1:])
        processing = {
            "input_drive": F.linear(x[-1:], model.W_in).cpu().numpy(),
            "recurrent_drive": recurrent_drive(prev, model.W_res).cpu().numpy(),
            "previous_state": prev.cpu().numpy(),
            "bias": model.b_res.detach().cpu().numpy().copy(), "leak": float(model.alpha),
        }
    states_np = states.cpu().numpy()
    contributions = decompose_contributions(states_np, inputs, model.readout)
    contributions["raw"] = raw.cpu().numpy()
    ready = len(frame) >= int(warmup)
    return {
        "status": "predicted" if ready else "Collecting history",
        "predicted_rul_s": float(raw_rul[-1]) if ready else None,
        "raw_prediction": raw[-1].cpu().numpy(), "raw_rul_s": raw_rul,
        "states": states_np, "inputs": inputs, "processing": processing,
        "contributions": contributions, "timestamps_s": ts,
        "frame_map": [{"timestamp_s": float(t), "frame_index": i} for i, t in enumerate(ts)],
        "node_order": list(model.node_order), "n_history": len(frame),
        "valid_history_reason": "" if ready else "insufficient_length",
        "model_identity": identity,
    }


def neuron_details(trace, model, prep, node_index: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Actual last-step incoming products, with source IDs and signed model weights."""
    i = int(node_index)
    inputs = trace["inputs"][-1]
    previous = trace["processing"]["previous_state"][-1]
    win = model.W_in[i].detach().cpu().numpy()
    if model.W_res.layout == torch.sparse_csr:
        row = model.W_res.crow_indices()
        start, end = int(row[i]), int(row[i + 1])
        source = model.W_res.col_indices()[start:end].cpu().numpy()
        weights = model.W_res.values()[start:end].cpu().numpy()
    else:
        wres = model.W_res[i].detach().cpu().numpy()
        source = np.flatnonzero(wres)
        weights = wres[source]
    sensors = pd.DataFrame({
        "Feature": prep.feature_names,
        "Normalized input": inputs,
        "Input weight": win,
        "Input contribution": inputs * win,
    })
    neighbors = pd.DataFrame({
        "Source body ID": [model.node_order[j] for j in source],
        "Previous state": previous[source],
        "Recurrent weight": weights,
        "Recurrent contribution": previous[source] * weights,
    })
    if not neighbors.empty:
        order = neighbors["Recurrent contribution"].abs().sort_values(ascending=False).index
        neighbors = neighbors.loc[order].reset_index(drop=True)
    return sensors, neighbors


def _full_cns_trace(inputs, ts, model, warmup, previous_trace, *, should_stop=None):
    """Bounded state storage: full current/previous state, causal readouts and raster.

    Seeking reconstructs the causal prefix with the same kernel. Playback appends
    only the new measurements; all neurons compute on every measurement.
    """
    from pdm.visualization.trace import _display_from_raw

    old = previous_trace or {}
    old_n = len(old.get("inputs", []))
    identity = (id(model), model.W_in._version, model.W_res._version, model.b_res._version, float(model.alpha))
    reuse = (0 < old_n < len(inputs) and old.get("model_identity") == identity
             and np.array_equal(old.get("timestamps_s"), ts[:old_n])
             and np.array_equal(old.get("inputs"), inputs[:old_n]))
    raster_ids = np.linspace(0, model.n_nodes - 1, 120, dtype=np.int64)
    raw_rul = [old["raw_rul_s"]] if reuse else []
    raster = [old["activity_history"]] if reuse else []
    x0 = torch.from_numpy(old["states"][-1]) if reuse else None
    previous = x0 if x0 is not None else torch.zeros(model.n_nodes)
    model.eval()
    with torch.no_grad():
        for offset in range(old_n if reuse else 0, len(inputs), 32):
            if should_stop and should_stop():
                raise InterruptedError("Evaluation cancelled")
            x = torch.from_numpy(inputs[offset:offset + 32].copy())
            states = model.forward_states(x, x0=x0)
            previous = states[-2] if len(states) > 1 else (x0 if x0 is not None else previous)
            raw = model.forward_raw(states, x)
            raw_rul.append(_display_from_raw(model, raw).cpu().numpy())
            raster.append(states[:, raster_ids].numpy())
            x0 = states[-1].clone()
        last = x0.unsqueeze(0)
        last_input = torch.from_numpy(inputs[-1:].copy())
        processing = {"input_drive": F.linear(last_input, model.W_in).numpy(),
                      "recurrent_drive": recurrent_drive(previous.unsqueeze(0), model.W_res).numpy(),
                      "previous_state": previous.unsqueeze(0).numpy(),
                      "bias": model.b_res.numpy(), "leak": model.alpha}
        raw = model.forward_raw(last, last_input).numpy()
    history = np.concatenate(raster)[-120:]
    rul = np.concatenate(raw_rul)
    last_np = last.numpy()
    contributions = decompose_contributions(last_np, inputs[-1:], model.readout)
    contributions["raw"] = raw
    ready = len(inputs) >= int(warmup)
    return {"status": "predicted" if ready else "Collecting history",
            "predicted_rul_s": float(rul[-1]) if ready else None,
            "raw_prediction": raw[-1], "raw_rul_s": rul,
            "states": np.concatenate([previous.unsqueeze(0).numpy(), last_np]),
            "inputs": inputs, "processing": processing, "contributions": contributions,
            "timestamps_s": ts, "frame_map": [{"timestamp_s": float(ts[-1])}],
            "node_order": model.node_order, "n_history": len(inputs),
            "valid_history_reason": "" if ready else "insufficient_length", "model_identity": identity,
            "activity_history": history, "activity_history_ids": raster_ids.tolist(),
            "activity_history_timestamps_s": ts[-len(history):], "state_history_complete": False}


def window_forecast_history(measurements, model, prep, history_length, *, cached=None):
    """Complete causal chart prefix, independent of the user's seek path.

    Cache only predictions of earlier prefixes; the current point and every
    missing point use the ordinary Predictor used by evaluation and export.
    """
    from pdm.predict import Predictor
    from pdm.replay import ReplaySource

    source = ReplaySource(measurements, gap_multiplier=getattr(prep, "gap_multiplier", None),
                          sampling_interval_s=getattr(prep, "sampling_interval_s", None))
    saved = cached or {}
    predictor = Predictor(model, prep, history_length, device="cpu")
    encoded = apply_preprocessor(prep, source.measurements) if prep.feature_recipe != "base_v1" else None
    rows = []
    for index in range(len(source)):
        stamp = float(source.measurements.iloc[index].timestamp_s)
        result = saved.get(stamp)
        if result is None:
            point = (predictor.predict_encoded_prefix(source.prefix(index), encoded.iloc[:index + 1]) if encoded is not None
                     else predictor.predict_from_history(source.prefix(index)))
            result = {"timestamp_s": stamp, **point}
        rows.append(result)
    return pd.DataFrame(rows)


def _continuous_model_identity(model):
    """Same identity tuple ``continuous_trace`` / ``_full_cns_trace`` store on a trace."""
    return (id(model), model.W_in._version, model.W_res._version, model.b_res._version, float(model.alpha))


def _forecast_frame(measurements, prep):
    """One unit's causal prefix. Filters recompute gaps before any split; bearings do not."""
    frame = measurements.sort_values("timestamp_s").reset_index(drop=True).copy()
    if getattr(prep, "dataset_id", None) == "filters":
        from pdm.windows import recompute_filter_gap_before

        frame = recompute_filter_gap_before(
            frame,
            gap_multiplier=prep.gap_multiplier,
            sampling_interval_s=prep.sampling_interval_s,
            causal=True,
        )
    return frame


def _segment_bounds(frame):
    if "gap_before" in frame.columns:
        gaps = frame["gap_before"].fillna(False).to_numpy(dtype=bool)
    else:
        gaps = np.zeros(len(frame), dtype=bool)
    starts = np.unique(np.r_[0, np.flatnonzero(gaps)])
    return starts, np.r_[starts[1:], len(frame)]


def _trace_covers_segment(trace, segment_ts) -> bool:
    if not trace:
        return False
    raw = trace.get("raw_rul_s")
    stamps = trace.get("timestamps_s")
    if raw is None or stamps is None:
        return False
    raw = np.asarray(raw, dtype=float).reshape(-1)
    stamps = np.asarray(stamps, dtype=float).reshape(-1)
    return len(raw) == len(segment_ts) and np.array_equal(stamps, segment_ts)


def _lookup_cached_segment(cache, identity, segment_ts, prefix_end):
    """Ignore a different model, a stale identity, or a segment that runs past this prefix."""
    for entry in cache:
        if not isinstance(entry, dict) or entry.get("model_identity") != identity:
            continue
        stamps = entry.get("timestamps_s")
        raw = entry.get("raw_rul_s")
        if stamps is None or raw is None:
            continue
        stamps = np.asarray(stamps, dtype=float).reshape(-1)
        raw = np.asarray(raw, dtype=float).reshape(-1)
        if stamps.size == 0 or float(stamps[-1]) > prefix_end or np.any(stamps > prefix_end):
            continue
        if len(stamps) != len(segment_ts) or len(raw) != len(segment_ts):
            continue
        if np.array_equal(stamps, segment_ts):
            return raw
    return None


def _store_cached_segment(cache, identity, segment_ts, raw):
    entry = {
        "model_identity": identity,
        "start_timestamp_s": float(segment_ts[0]),
        "length": int(len(segment_ts)),
        "timestamps_s": np.array(segment_ts, dtype=float, copy=True),
        "raw_rul_s": np.array(raw, dtype=float, copy=True),
    }
    for index, old in enumerate(cache):
        if not isinstance(old, dict) or old.get("model_identity") != identity:
            continue
        old_ts = np.asarray(old.get("timestamps_s", []), dtype=float).reshape(-1)
        if len(old_ts) == len(segment_ts) and np.array_equal(old_ts, segment_ts):
            cache[index] = entry
            return
    cache.append(entry)


def _score_continuous_segment(frame, start, end, model, prep, history_length):
    """One ``continuous_trace`` for this segment.

    Pass the causal prefix through ``end``, not the bare slice. On filters,
    ``continuous_trace`` recomputes ``delta_t_s`` before it keeps the last segment;
    an isolated slice would zero that gap row and disagree with ``predict_from_history``.
    """
    traced = continuous_trace(frame.iloc[: int(end)], model, prep, history_length)
    raw = np.asarray(traced["raw_rul_s"], dtype=float).reshape(-1)
    segment_ts = frame["timestamp_s"].to_numpy(dtype=float)[int(start) : int(end)]
    if len(raw) != len(segment_ts) or not np.array_equal(
        np.asarray(traced["timestamps_s"], dtype=float).reshape(-1), segment_ts
    ):
        raise ValueError("Continuous forecast segment does not align with its timestamps")
    return raw


def continuous_forecast_history(
    measurements, model, prep, history_length, *, trace=None, cached_segments=None
):
    """One causal predicted-RUL row per observed measurement, warmup-masked per segment.

    The active segment is copied from ``trace`` when its timestamps match. Earlier
    segments are stored on ``cached_segments`` and are not scored again. No ground
    truth, event time, or interval profile enters this frame.
    """
    frame = _forecast_frame(measurements, prep)
    columns = ("timestamp_s", "raw_rul_s", "predicted_rul_s")
    if frame.empty:
        return pd.DataFrame({name: pd.Series(dtype=float) for name in columns})
    timestamps = frame["timestamp_s"].to_numpy(dtype=float)
    starts, ends = _segment_bounds(frame)
    raw = np.full(len(frame), np.nan, dtype=float)
    seen = np.empty(len(frame), dtype=int)
    identity = _continuous_model_identity(model)
    cache = [] if cached_segments is None else cached_segments
    prefix_end = float(timestamps[-1])
    last = len(starts) - 1
    for index, (start, end) in enumerate(zip(starts, ends, strict=True)):
        segment_ts = timestamps[int(start) : int(end)]
        active = index == last
        segment_raw = None
        if active and _trace_covers_segment(trace, segment_ts):
            segment_raw = np.asarray(trace["raw_rul_s"], dtype=float).reshape(-1)
        if segment_raw is None:
            segment_raw = _lookup_cached_segment(cache, identity, segment_ts, prefix_end)
        if segment_raw is None:
            segment_raw = _score_continuous_segment(frame, start, end, model, prep, history_length)
        raw[int(start) : int(end)] = segment_raw
        seen[int(start) : int(end)] = np.arange(1, int(end) - int(start) + 1)
        if not active:
            _store_cached_segment(cache, identity, segment_ts, segment_raw)
    warmup = int(history_length)
    return pd.DataFrame({
        "timestamp_s": timestamps,
        "raw_rul_s": raw,
        "predicted_rul_s": np.where(seen >= warmup, raw, np.nan),
    })


def equipment_forecast_frame(
    prefix, model, prep, history_length, *, profile, trace, window_cache=None, segment_cache=None,
):
    """Profile, window-reset, or continuous chart rows for one observed prefix.

    A loaded profile uses only the active trace arrays. Window-reset models read
    ``window_cache``. Continuous models without a profile read ``trace`` and
    ``segment_cache``. Ground truth is not an input.
    """
    if profile is not None:
        from pdm.forecasting import predict_failure_interval

        return predict_failure_interval(trace["timestamps_s"], trace["raw_rul_s"], profile)
    if getattr(model, "state_mode", None) != "continuous":
        return window_forecast_history(prefix, model, prep, history_length, cached=window_cache)
    return continuous_forecast_history(
        prefix, model, prep, history_length, trace=trace, cached_segments=segment_cache,
    )
