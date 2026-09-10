from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.preprocessing import StandardScaler

from pdm.features import apply_log1p_columns
from pdm.windows import FORBIDDEN_FEATURE_NAMES


@dataclass
class Preprocessor:
    feature_names: list[str]
    log1p_features: list[str]
    scaler_mean: list[float]
    scaler_scale: list[float]
    time_scale_s: float
    categorical_maps: dict[str, list[str]] = field(default_factory=dict)
    fill_values: dict[str, float] = field(default_factory=dict)

    def transform_frame(self, df):
        out = df.copy()
        apply_log1p_columns(out, self.log1p_features)
        for col in self.feature_names:
            if col not in out.columns:
                out[col] = self.fill_values.get(col, 0.0)
        arr = out[self.feature_names].to_numpy(dtype=np.float64)
        arr = np.where(np.isfinite(arr), arr, 0.0)
        mean = np.asarray(self.scaler_mean, dtype=np.float64)
        scale = np.asarray(self.scaler_scale, dtype=np.float64)
        scale = np.where(np.abs(scale) < 1e-12, 1.0, scale)
        out[self.feature_names] = (arr - mean) / scale
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_names": self.feature_names,
            "log1p_features": self.log1p_features,
            "scaler_mean": self.scaler_mean,
            "scaler_scale": self.scaler_scale,
            "time_scale_s": self.time_scale_s,
            "categorical_maps": self.categorical_maps,
            "fill_values": self.fill_values,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Preprocessor":
        return cls(
            feature_names=list(d["feature_names"]),
            log1p_features=list(d.get("log1p_features") or []),
            scaler_mean=[float(x) for x in d["scaler_mean"]],
            scaler_scale=[float(x) for x in d["scaler_scale"]],
            time_scale_s=float(d["time_scale_s"]),
            categorical_maps=d.get("categorical_maps") or {},
            fill_values={k: float(v) for k, v in (d.get("fill_values") or {}).items()},
        )


def _one_hot(df, column: str, categories: list[str] | None = None):
    series = df[column].astype(str)
    cats = categories or sorted(series.unique().tolist())
    for c in cats:
        df[f"{column}__{c}"] = (series == c).astype(np.float32)
    return df, cats


def bearings_feature_frame(raw_features, log1p_cols: list[str]):
    df = raw_features.copy()
    base = [
        "operating_age_s",
        "rpm",
        "load_kn",
        "horizontal_rms",
        "horizontal_std",
        "horizontal_abs_peak",
        "horizontal_peak_to_peak",
        "horizontal_crest_factor",
        "horizontal_kurtosis",
        "vertical_rms",
        "vertical_std",
        "vertical_abs_peak",
        "vertical_peak_to_peak",
        "vertical_crest_factor",
        "vertical_kurtosis",
        "horizontal_band_0",
        "horizontal_band_1",
        "horizontal_band_2",
        "horizontal_band_3",
        "vertical_band_0",
        "vertical_band_1",
        "vertical_band_2",
        "vertical_band_3",
    ]
    missing = [c for c in base if c not in df.columns]
    if missing:
        raise ValueError(f"Bearing features missing columns {missing}. Actual={list(df.columns)}")
    return df, base, log1p_cols


def filters_feature_frame(raw_features, train_mask, dust_categories: list[str] | None = None):
    df = raw_features.copy()
    df, cats = _one_hot(df, "dust", dust_categories)
    base = [
        "operating_age_s",
        "delta_t_s",
        "differential_pressure",
        "delta_pressure",
        "flow_rate",
        "dust_feed",
        *[f"dust__{c}" for c in cats],
    ]
    leak = [c for c in base if c.lower() in FORBIDDEN_FEATURE_NAMES]
    extra = [c for c in ("RUL", "event_time_s", "official_rul_at_prefix_end_s") if c in df.columns]
    if leak:
        raise ValueError(f"Refusing leak columns in filter features: {leak}")
    # unit_id may exist for grouping but is not a model input (not in `base`).
    if extra:
        pass
    return df, base, cats


def fit_preprocessor(
    dataset_id: str,
    features,
    units,
    split: dict,
    cfg: dict,
    windows_train=None,
) -> tuple[Preprocessor, Any]:
    train_ids = set(split["train"])
    train_feat = features[features["unit_id"].isin(train_ids)].copy()
    if train_feat.empty:
        raise ValueError("No training features to fit preprocessor")

    if dataset_id == "bearings":
        log1p_cols = list(cfg.get("features", {}).get("log1p_features") or [])
        feat_df, names, log1p_cols = bearings_feature_frame(features, log1p_cols)
        train_view = feat_df[feat_df["unit_id"].isin(train_ids)].copy()
        apply_log1p_columns(train_view, log1p_cols)
        cats: dict[str, list[str]] = {}
    else:
        feat_df, names, cats_list = filters_feature_frame(features, None, None)
        # refit one-hot using train categories only
        train_raw = features[features["unit_id"].isin(train_ids)]
        _, names, cats_list = filters_feature_frame(features, None, sorted(train_raw["dust"].astype(str).unique()))
        feat_df, names, cats_list = filters_feature_frame(features, None, cats_list)
        train_view = feat_df[feat_df["unit_id"].isin(train_ids)].copy()
        log1p_cols = []
        cats = {"dust": cats_list}

    arr = train_view[names].to_numpy(dtype=np.float64)
    arr = np.where(np.isfinite(arr), arr, np.nan)
    fill = {}
    for i, name in enumerate(names):
        col = arr[:, i]
        med = float(np.nanmedian(col)) if np.isfinite(col).any() else 0.0
        fill[name] = med
        col = np.where(np.isfinite(col), col, med)
        arr[:, i] = col
    scaler = StandardScaler()
    scaler.fit(arr)

    if dataset_id == "bearings":
        unit_meta = units.set_index("unit_id")
        ruls = []
        for uid, g in train_view.groupby("unit_id"):
            et = float(unit_meta.loc[uid]["event_time_s"])
            ts = g["timestamp_s"].to_numpy(dtype=np.float64)
            ruls.extend((et - ts)[ts < et].tolist())
        time_scale = float(np.median(ruls)) if ruls else 1.0
    else:
        unit_meta = units.set_index("unit_id")
        durs = []
        for uid, g in train_view.groupby("unit_id"):
            end = float(unit_meta.loc[uid]["observation_end_s"])
            ts = g["timestamp_s"].to_numpy(dtype=np.float64)
            d = end - ts
            durs.extend(d[d > 0].tolist())
        time_scale = float(np.median(durs)) if durs else 1.0
    time_scale = max(time_scale, 1e-6)

    prep = Preprocessor(
        feature_names=names,
        log1p_features=log1p_cols,
        scaler_mean=scaler.mean_.astype(float).tolist(),
        scaler_scale=scaler.scale_.astype(float).tolist(),
        time_scale_s=time_scale,
        categorical_maps=cats,
        fill_values=fill,
    )
    transformed = prep.transform_frame(feat_df)
    return prep, transformed
