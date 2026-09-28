"""Standalone replay of filter pressure sensor zones from a prepared snapshot."""
from __future__ import annotations

import hashlib
import json
from html import escape
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from pdm.monitoring.filter_zones import FILTER_ZONE_POLICY
from pdm.paths import runs_root

ZONE_COLORS = {
    "green": "#2ecc71",
    "yellow": "#f1c40f",
    "red": "#e74c3c",
    "unknown": "#9aa8b8",
    "gray": "#9aa8b8",
}
ZONE_LABELS = {
    "green": "Below provisional pressure warning band",
    "yellow": "Provisional pressure warning band",
    "red": "Configured laboratory pressure limit reached",
    "unknown": "Assessment unavailable",
    "gray": "Assessment unavailable",
}


def _fingerprint_sha256(fingerprint: dict) -> str:
    return hashlib.sha256(json.dumps(fingerprint, sort_keys=True).encode("utf-8")).hexdigest()


def _read_current_filter_zone_labels(data: dict, artifact_root: Path | None = None) -> tuple[pd.DataFrame, dict, Path]:
    """Load only a complete label artifact bound to the currently prepared snapshot."""
    version = data.get("dataset_version")
    fingerprint = data.get("fingerprint") or {}
    if not version or not fingerprint:
        raise ValueError("The prepared filter snapshot has no versioned fingerprint. Re-run filter preparation.")
    root = Path(artifact_root) if artifact_root is not None else runs_root() / "_zones" / "filters" / "label_artifacts"
    if not root.is_dir():
        raise FileNotFoundError("No filter sensor-zone label artifact exists yet.")

    expected_fingerprint = _fingerprint_sha256(fingerprint)
    candidates = []
    for manifest_path in root.glob("*/manifest.json"):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        policy = manifest.get("policy") or {}
        try:
            thresholds_match = (float(policy.get("yellow_limit_pa", -1)) == 300.0
                                and float(policy.get("red_limit_pa", -1)) == 600.0)
        except (TypeError, ValueError):
            thresholds_match = False
        if (manifest.get("dataset_id") != "filters"
                or manifest.get("dataset_version") != version
                or manifest.get("dataset_fingerprint_sha256") != expected_fingerprint
                or policy.get("version") != FILTER_ZONE_POLICY["version"]
                or not thresholds_match):
            continue
        labels_path = manifest_path.parent / str(manifest.get("labels_file", "labels.csv"))
        try:
            if hashlib.sha256(labels_path.read_bytes()).hexdigest() != manifest.get("labels_sha256"):
                continue
            labels = pd.read_csv(labels_path)
        except (OSError, pd.errors.ParserError, UnicodeDecodeError):
            continue
        required = {"unit_id", "split", "timestamp_s", "differential_pressure_pa", "flow_rate_recorded",
                    "dust_feed_recorded", "sensor_zone", "display_zone", "quality_status"}
        if (not required.issubset(labels.columns) or len(labels) != int(manifest.get("row_count", -1))
                or labels.duplicated(["unit_id", "timestamp_s"]).any()
                or not set(labels["display_zone"].dropna()).issubset({"green", "yellow", "red", "gray"})):
            continue
        candidates.append((str(manifest.get("created_at", "")), labels, manifest, manifest_path.parent))
    if not candidates:
        raise FileNotFoundError(
            "No intact filter sensor-zone label artifact matches the latest prepared snapshot and current 300/600 Pa policy. "
            "Export one with `python -m pdm zones-labels --dataset filters`."
        )
    _, labels, manifest, directory = max(candidates, key=lambda item: item[0])
    expected_splits = {str(uid): name for name in ("train", "validation", "test")
                       for uid in (data.get("split") or {}).get(name, [])}
    observed = labels.groupby("unit_id")["split"].agg(lambda values: set(values.astype(str)))
    for uid, split_values in observed.items():
        if uid not in expected_splits or split_values != {expected_splits[uid]}:
            raise ValueError(f"Filter label artifact split does not match the prepared snapshot for unit {uid}.")
    labels = labels.copy()
    labels["zone"] = labels["display_zone"].replace({"gray": "unknown"})
    return labels.sort_values(["unit_id", "timestamp_s"], kind="stable"), manifest, directory


def _filter_history_chart(rows: pd.DataFrame, index: int) -> go.Figure:
    visible = rows.iloc[: index + 1]
    fig = go.Figure()
    for _, row in visible.iterrows():
        x0 = float(row["timestamp_s"])
        x1 = float(row["next_timestamp_s"])
        zone = str(row["zone"])
        fig.add_vrect(x0=x0, x1=x1, fillcolor=ZONE_COLORS[zone], opacity=0.16, line_width=0, layer="below")
    fig.add_trace(go.Scatter(
        x=visible["timestamp_s"], y=visible["differential_pressure_pa"], mode="lines+markers",
        name="Differential pressure", line=dict(color="#e8eef5", width=2), marker=dict(size=5),
        customdata=visible[["flow_rate_recorded", "dust_feed_recorded", "zone"]],
        hovertemplate="Time: %{x:.4g} s<br>Pressure: %{y:.1f} Pa<br>Flow: %{customdata[0]:.3g}<br>Feed: %{customdata[1]:.3g}<br>Zone: %{customdata[2]}<extra></extra>",
    ))
    selected = visible.iloc[-1]
    fig.add_trace(go.Scatter(x=[selected["timestamp_s"]], y=[selected["differential_pressure_pa"]],
                             mode="markers", name="Selected measurement",
                             marker=dict(color="#5bd3f5", size=12, line=dict(width=2, color="white"))))
    fig.add_hline(y=300, line=dict(color=ZONE_COLORS["yellow"], dash="dash"),
                  annotation_text="Provisional warning · 300 Pa", annotation_position="top left")
    fig.add_hline(y=600, line=dict(color=ZONE_COLORS["red"], dash="dash"),
                  annotation_text="Configured limit · 600 Pa", annotation_position="top left")
    fig.update_layout(height=430, margin=dict(l=10, r=10, t=30, b=10), template="plotly_dark",
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      xaxis_title="Time (s; source label: HSE Figure 6)",
                      yaxis_title="Differential pressure, Pa", legend=dict(orientation="h", y=-0.2))
    return fig


def screen_filter_health_zones() -> None:
    from pdm.data.prepare import load_processed

    st.subheader("Health zones · filters")
    st.caption(
        "Measured differential-pressure zones for filters. Green is below the provisional 300 Pa warning band; "
        "yellow is 300 to below 600 Pa; red starts at the configured 600 Pa laboratory limit. "
        "Non-red zones require usable pressure, positive flow, and finite feed. Unknown means the required sensor context is unavailable. "
        "Green means below the provisional pressure band, not confirmed healthy. No event-time or remaining-life model is used."
    )
    st.caption('Time values use seconds, following the HSE source paper axis label "Time / s" (Figure 6, page 5).')
    try:
        data = load_processed("filters")
    except (FileNotFoundError, ValueError, OSError) as exc:
        st.info(f"Prepare the filter dataset to open sensor-zone replay: {exc}")
        return
    if data.get("dataset_version") is None:
        st.error("The current filter data is not a versioned prepared snapshot, so its labels cannot be verified. Re-run filter preparation.")
        return
    try:
        labels, manifest, _ = _read_current_filter_zone_labels(data)
    except FileNotFoundError as exc:
        st.info(str(exc))
        return
    except ValueError as exc:
        st.error(f"The filter zone artifact could not be matched to this prepared snapshot: {exc}")
        return
    if labels.empty:
        st.info("The matching filter label artifact contains no row-admitted measurements.")
        return

    split_order = [name for name in ("test", "validation", "train")
                   if labels["split"].astype(str).eq(name).any()]
    split = st.selectbox("Split", split_order, key="filter_hz_split")
    subset = labels[labels["split"].astype(str).eq(split)]
    units = sorted(subset["unit_id"].astype(str).unique())
    uid = st.selectbox("Filter unit", units, key="filter_hz_unit")
    rows = subset[subset["unit_id"].astype(str).eq(uid)].sort_values("timestamp_s", kind="stable").copy()
    t = pd.to_numeric(rows["timestamp_s"], errors="coerce").to_numpy(float)
    next_t = list(t[1:]) + [float(t[-1] + (t[-1] - t[-2] if len(t) > 1 else 1.0))]
    rows["next_timestamp_s"] = next_t
    cursor_key = f"filter_hz_cursor:{split}:{uid}"
    st.session_state.setdefault(cursor_key, min(len(rows) - 1, 5))
    if len(rows) > 1:
        index = st.slider("Measurement", 0, len(rows) - 1, key=cursor_key)
    else:
        index = 0
        st.caption("This unit has one row-admitted measurement in the matching label artifact.")
    row = rows.iloc[index]
    zone = str(row["zone"])
    color = ZONE_COLORS[zone]
    st.markdown(
        f"<div style='border:2px solid {color};border-radius:12px;padding:14px 18px;background:{color}22;margin:.5rem 0 1rem'>"
        f"<div style='font-size:.8rem;letter-spacing:.1em;opacity:.8'>{escape(str(uid))} · {escape(split.upper())} · MEASUREMENT {index + 1} / {len(rows)}</div>"
        f"<div style='font-size:1.6rem;font-weight:700;color:{color}'>{escape(zone.upper())} · {escape(ZONE_LABELS[zone])}</div>"
        f"<div>Pressure {float(row['differential_pressure_pa']):.1f} Pa · Flow {float(row['flow_rate_recorded']):.3g} · Feed {float(row['dust_feed_recorded']):.3g}</div>"
        f"<div style='font-size:.85rem;opacity:.8'>Quality status: {escape(str(row['quality_status']))} · {escape(str(row.get('zone_reason', '')))}</div></div>",
        unsafe_allow_html=True,
    )
    st.plotly_chart(_filter_history_chart(rows, index), width="stretch", key=f"filter_hz_chart:{split}:{uid}")
    counts = manifest.get("class_counts_by_split", {}).get(split, {})
    st.caption(
        f"Prepared snapshot {escape(str(data['dataset_version']))} · label artifact {escape(str(manifest.get('artifact_id', 'unknown')))} · "
        f"split counts green {counts.get('green', 0)}, yellow {counts.get('yellow', 0)}, red {counts.get('red', 0)}, unknown {counts.get('unknown', 0)}. "
        "Thresholds are provisional sensor rules; red indicates an observed limit crossing, not a failure-time prediction."
    )
