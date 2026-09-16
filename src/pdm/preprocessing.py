from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from pdm.feature_recipes import enrich_features, recipe_hash, recipe_metadata, recipe_names
from pdm.features import apply_log1p_columns
from pdm.windows import FORBIDDEN_FEATURE_NAMES, filter_gap_params

FEATURE_PIPELINE_VERSION = "v2_raw_first"


def _optional_positive_float(value: Any) -> float | None:
    if value is None:
        return None
    out = float(value)
    if not np.isfinite(out) or out <= 0.0:
        return None
    return out


def categorical_maps_fingerprint(categorical_maps: dict[str, list[str]]) -> str:
    import hashlib
    import json

    payload = {k: sorted(v) for k, v in sorted(categorical_maps.items())}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def _assert_encoded_for_transform(prep: Preprocessor, df: pd.DataFrame) -> None:
    if prep.feature_pipeline_version != FEATURE_PIPELINE_VERSION:
        return
    if df.attrs.get("raw_features") is not False:
        raise ValueError(
            "transform_frame expects an encoded feature frame with raw_features=False; "
            "call raw_to_feature_frame first."
        )
    if "dust" in df.columns:
        dust_cats = prep.categorical_maps.get("dust", [])
        missing_dust = [f"dust__{c}" for c in dust_cats if f"dust__{c}" not in df.columns]
        if missing_dust:
            raise ValueError(
                f"transform_frame expects dust one-hot columns {missing_dust}; "
                "call raw_to_feature_frame first."
            )
    missing_features = [c for c in prep.feature_names if c not in df.columns]
    if missing_features:
        raise ValueError(
            f"transform_frame missing encoded feature columns {missing_features}; "
            "call raw_to_feature_frame first."
        )


def _assert_raw_measurement_input(df: pd.DataFrame, raw_features: bool | None) -> None:
    """Require explicit confirmation that rows are raw measurements, not encoded features."""
    attr = df.attrs.get("raw_features")
    if raw_features is False or attr is False:
        raise ValueError(
            "Expected raw measurement rows; input is already encoded "
            "(raw_features=False or df.attrs['raw_features'] is False)."
        )
    if raw_features is not True and attr is not True:
        raise ValueError(
            "Ambiguous feature input: pass raw_features=True or set "
            "df.attrs['raw_features']=True to confirm rows are raw measurements "
            "before categorical encoding and log1p."
        )


@dataclass
class Preprocessor:
    feature_names: list[str]
    log1p_features: list[str]
    scaler_mean: list[float]
    scaler_scale: list[float]
    time_scale_s: float
    categorical_maps: dict[str, list[str]] = field(default_factory=dict)
    fill_values: dict[str, float] = field(default_factory=dict)
    feature_pipeline_version: str = FEATURE_PIPELINE_VERSION
    dataset_id: str | None = None
    gap_multiplier: float | None = None
    sampling_interval_s: float | None = None
    feature_recipe: str = "base_v1"

    def transform_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.attrs.get("scaled") is True:
            raise ValueError(
                "transform_frame received already-scaled features; do not apply scaling twice."
            )
        _assert_encoded_for_transform(self, df)
        missing_fills = [col for col in self.feature_names if col not in self.fill_values]
        if missing_fills:
            raise ValueError(f"Preprocessor missing fill_values for features: {missing_fills}")
        out = df.copy()
        if self.feature_pipeline_version != FEATURE_PIPELINE_VERSION:
            apply_log1p_columns(out, self.log1p_features)
            for col in self.feature_names:
                if col not in out.columns:
                    out[col] = self.fill_values[col]
        mean = np.asarray(self.scaler_mean, dtype=np.float64)
        scale = np.asarray(self.scaler_scale, dtype=np.float64)
        scale = np.where(np.abs(scale) < 1e-12, 1.0, scale)
        fills = np.asarray(
            [self.fill_values[col] for col in self.feature_names],
            dtype=np.float64,
        )
        arr = out[self.feature_names].to_numpy(dtype=np.float64)
        arr = np.where(np.isfinite(arr), arr, fills)
        out[self.feature_names] = (arr - mean) / scale
        out.attrs["scaled"] = True
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
            "feature_pipeline_version": self.feature_pipeline_version,
            "dataset_id": self.dataset_id,
            "gap_multiplier": self.gap_multiplier,
            "sampling_interval_s": self.sampling_interval_s,
            "feature_recipe": self.feature_recipe,
            "feature_recipe_parameters": recipe_metadata(self.feature_recipe),
            "feature_recipe_hash": recipe_hash(self.feature_recipe),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Preprocessor":
        recipe = d.get("feature_recipe", "base_v1")
        if d.get("feature_recipe_hash") and d["feature_recipe_hash"] != recipe_hash(recipe):
            raise ValueError("Feature recipe changed since this checkpoint was trained")
        if d.get("feature_recipe_parameters") and d["feature_recipe_parameters"] != recipe_metadata(recipe):
            raise ValueError("Saved feature recipe parameters do not match their version")
        return cls(
            feature_names=list(d["feature_names"]),
            log1p_features=list(d.get("log1p_features") or []),
            scaler_mean=[float(x) for x in d["scaler_mean"]],
            scaler_scale=[float(x) for x in d["scaler_scale"]],
            time_scale_s=float(d["time_scale_s"]),
            categorical_maps=d.get("categorical_maps") or {},
            fill_values={k: float(v) for k, v in (d.get("fill_values") or {}).items()},
            feature_pipeline_version=str(d.get("feature_pipeline_version") or ""),
            dataset_id=d.get("dataset_id"),
            gap_multiplier=_optional_positive_float(d.get("gap_multiplier")),
            sampling_interval_s=_optional_positive_float(d.get("sampling_interval_s")),
            feature_recipe=recipe,
        )


def apply_preprocessor(
    prep: Preprocessor,
    features: pd.DataFrame,
    dataset_id: str | None = None,
) -> pd.DataFrame:
    """Encode raw measurements with a saved preprocessor, then scale. Train-only maps/scaler."""
    ds = dataset_id or prep.dataset_id
    if not ds:
        raise ValueError("dataset_id required to apply preprocessor")
    feat_df = raw_to_feature_frame(
        ds,
        features,
        prep.categorical_maps,
        prep.log1p_features,
        raw_features=True,
        feature_recipe=prep.feature_recipe,
    )
    return prep.transform_frame(feat_df)


def preprocessor_resume_mismatches(saved: Preprocessor, live: Preprocessor) -> list[str]:
    """Scaler / maps / time_scale_s / feature_pipeline_version vs a live fit. Not eval-method drift."""
    differing: list[str] = []
    if saved.feature_recipe != live.feature_recipe:
        differing.append("feature_recipe")
    if saved.feature_pipeline_version != live.feature_pipeline_version:
        differing.append("feature_pipeline_version")
    if saved.categorical_maps != live.categorical_maps:
        differing.append("categorical_maps")
    if not np.isclose(float(saved.time_scale_s), float(live.time_scale_s), rtol=0.0, atol=1e-9):
        differing.append("time_scale_s")
    saved_mean = np.asarray(saved.scaler_mean, dtype=np.float64)
    live_mean = np.asarray(live.scaler_mean, dtype=np.float64)
    saved_scale = np.asarray(saved.scaler_scale, dtype=np.float64)
    live_scale = np.asarray(live.scaler_scale, dtype=np.float64)
    scaler_ok = (
        saved_mean.shape == live_mean.shape
        and saved_scale.shape == live_scale.shape
        and np.allclose(saved_mean, live_mean, rtol=0.0, atol=1e-9, equal_nan=True)
        and np.allclose(saved_scale, live_scale, rtol=0.0, atol=1e-9, equal_nan=True)
    )
    if not scaler_ok:
        differing.append("scaler")
    if not _optional_floats_close(saved.gap_multiplier, live.gap_multiplier):
        differing.append("gap_multiplier")
    if not _optional_floats_close(saved.sampling_interval_s, live.sampling_interval_s):
        differing.append("sampling_interval_s")
    return differing


def _optional_floats_close(left: float | None, right: float | None) -> bool:
    if left is None and right is None:
        return True
    if left is None or right is None:
        return False
    return bool(np.isclose(float(left), float(right), rtol=0.0, atol=1e-9))


def _one_hot(df, column: str, categories: list[str] | None = None):
    """One-hot encode ``column`` using train-only ``categories``.

    ``categories=None`` infers uniques from the series (legacy). ``categories=[]``
    means no known train categories → no one-hot columns (all-zero if none created).
    Values not in ``categories`` map to all-zero columns (no extra bucket).
    """
    series = df[column].astype(str)
    if categories is None:
        cats = sorted(series.unique().tolist())
    else:
        cats = list(categories)
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
    extra = [
        c
        for c in (
            "RUL",
            "event_time_s",
            "official_rul_at_prefix_end_s",
            "official_rul_at_prefix_end_original",
        )
        if c in df.columns
    ]
    if leak:
        raise ValueError(f"Refusing leak columns in filter features: {leak}")
    # unit_id may exist for grouping but is not a model input (not in `base`).
    if extra:
        pass
    return df, base, cats


def raw_to_feature_frame(
    dataset_id: str,
    df: pd.DataFrame,
    categorical_maps: dict[str, list[str]],
    log1p_features: list[str] | None = None,
    *,
    raw_features: bool | None = None,
    feature_recipe: str = "base_v1",
) -> pd.DataFrame:
    """Raw measurement rows → encoded, log1p'd model feature columns (pre-scaling).

    Stages: categorical encoding → log1p → column selection in saved order.
    Unknown categorical values (e.g. unseen ``dust``) yield all-zero one-hot columns
    for every category in ``categorical_maps``; no implicit unknown bucket is added.

    Requires ``raw_features=True`` or ``df.attrs['raw_features']=True`` so already-encoded
    frames cannot be double-transformed silently.
    """
    _assert_raw_measurement_input(df, raw_features)
    df = enrich_features(df, dataset_id, feature_recipe)
    if dataset_id == "bearings":
        log1p_cols = list(log1p_features or [])
        feat_df, names, log1p_cols = bearings_feature_frame(df, log1p_cols)
    elif dataset_id == "filters":
        dust_cats = list(categorical_maps.get("dust", []))
        feat_df, names, _ = filters_feature_frame(df, None, dust_cats)
        log1p_cols = []
    else:
        raise ValueError(f"Unknown dataset_id for feature pipeline: {dataset_id}")

    apply_log1p_columns(feat_df, log1p_cols)
    names = names + recipe_names(dataset_id, feature_recipe)
    out = feat_df.copy()
    missing = [c for c in names if c not in out.columns]
    if missing:
        raise ValueError(f"Feature pipeline missing columns after encoding: {missing}")
    out.attrs["raw_features"] = False
    out.attrs["feature_pipeline_version"] = FEATURE_PIPELINE_VERSION
    return out


def fit_preprocessor(
    dataset_id: str,
    features,
    units,
    split: dict,
    cfg: dict,
    windows_train=None,
) -> tuple[Preprocessor, Any]:
    train_ids = set(split["train"])
    feature_recipe = cfg.get("feature_recipe", "base_v1")
    train_feat = features[features["unit_id"].isin(train_ids)].copy()
    if train_feat.empty:
        raise ValueError("No training features to fit preprocessor")

    if dataset_id == "bearings":
        log1p_cols = list(cfg.get("features", {}).get("log1p_features") or [])
        cats: dict[str, list[str]] = {}
        _, names, log1p_cols = bearings_feature_frame(features, log1p_cols)
    else:
        log1p_cols = []
        train_raw = features[features["unit_id"].isin(train_ids)]
        cats_list = sorted(train_raw["dust"].astype(str).unique())
        cats = {"dust": cats_list}
        _, names, _ = filters_feature_frame(features, None, cats_list)

    feat_df = raw_to_feature_frame(
        dataset_id,
        features,
        cats,
        log1p_cols,
        raw_features=True,
        feature_recipe=feature_recipe,
    )
    names = names + recipe_names(dataset_id, feature_recipe)
    train_view = feat_df[feat_df["unit_id"].isin(train_ids)].copy()
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

    gap_k: float | None = None
    gap_samp: float | None = None
    if dataset_id == "filters":
        gap_k, gap_samp = filter_gap_params(cfg)

    prep = Preprocessor(
        feature_names=names,
        log1p_features=log1p_cols,
        scaler_mean=scaler.mean_.astype(float).tolist(),
        scaler_scale=scaler.scale_.astype(float).tolist(),
        time_scale_s=time_scale,
        categorical_maps=cats,
        fill_values=fill,
        feature_pipeline_version=FEATURE_PIPELINE_VERSION,
        dataset_id=dataset_id,
        gap_multiplier=gap_k,
        sampling_interval_s=gap_samp,
        feature_recipe=feature_recipe,
    )
    transformed = prep.transform_frame(feat_df)
    return prep, transformed
