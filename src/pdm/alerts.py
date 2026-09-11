from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from pdm.io_util import atomic_write_json, read_json


def _optional_horizon(value: Any) -> float | None:
    if value is None:
        return None
    v = float(value)
    return None if v <= 0.0 else v


ALERT_POLICY_FILENAME = "alert_policy.json"
ALERT_POLICY_SCHEMA_VERSION = "v1"
ALERT_OUTCOMES = ("timely", "too_early", "late", "miss", "insufficient_coverage")
ALERT_EPISODE_COLUMNS = (
    "run_id",
    "unit_id",
    "timestamp_s",
    "type",
    "predicted_rul_s",
    "warning_horizon_s",
    "H_trigger",
    "confirmation_count",
    "confirmation_delay_steps",
    "model_id",
    "message",
)
ALERT_STEP_COLUMNS = (
    "unit_id",
    "timestamp_s",
    "predicted_rul_s",
    "warning_active",
    "status",
)
_POLICY_HASH_FIELDS = (
    "schema_version",
    "H_trigger",
    "minimum_action_lead_time",
    "max_useful_horizon_s",
    "confirmation_count",
    "reset_factor",
)


@dataclass
class AlertEngine:
    warning_horizon_s: float
    confirmation_count: int = 3
    reset_factor: float = 1.2
    consecutive_warn: int = 0
    consecutive_clear: int = 0
    warning_active: bool = False
    episodes: list[dict[str, Any]] = field(default_factory=list)
    unit_id: str | None = None
    last_status: str = "Collecting history"

    def reset_unit(self, unit_id: str) -> None:
        self.unit_id = unit_id
        self.consecutive_warn = 0
        self.consecutive_clear = 0
        self.warning_active = False
        self.last_status = "Collecting history"

    @property
    def H_trigger(self) -> float:
        """Alias of `warning_horizon_s`: predicted RUL ≤ H opens a warning after K steps."""
        return float(self.warning_horizon_s)

    @property
    def confirmation_delay_steps(self) -> int:
        return max(self.confirmation_count - 1, 0)

    def update(
        self,
        *,
        timestamp_s: float,
        predicted_rul_s: float | None,
        collecting: bool = False,
        observed_limit: bool = False,
        model_id: str | None = None,
    ) -> dict[str, Any]:
        if observed_limit:
            self.last_status = "Observed limit reached"
            return self._snapshot(timestamp_s, predicted_rul_s, opened=None)

        if collecting:
            # Gap/warmup: same confirmation reset as reset_unit so a Warning
            # cannot survive a gap and reappear without K new confirms.
            self.last_status = "Collecting history"
            self.consecutive_warn = 0
            self.consecutive_clear = 0
            self.warning_active = False
            return self._snapshot(timestamp_s, predicted_rul_s, opened=None)

        if predicted_rul_s is None or not _finite(predicted_rul_s):
            self.last_status = "No valid prediction"
            self.consecutive_warn = 0
            return self._snapshot(timestamp_s, predicted_rul_s, opened=None)

        h = float(self.warning_horizon_s)
        k = int(self.confirmation_count)
        opened = None
        if predicted_rul_s <= h:
            self.consecutive_warn += 1
            self.consecutive_clear = 0
            if not self.warning_active and self.consecutive_warn >= k:
                self.warning_active = True
                # Confirmed time = this timestamp (K-th qualifying step), not the first trigger.
                opened = {
                    "type": "horizon_warning",
                    "unit_id": self.unit_id,
                    "timestamp_s": timestamp_s,
                    "predicted_rul_s": predicted_rul_s,
                    "warning_horizon_s": h,
                    "H_trigger": h,
                    "confirmation_count": k,
                    "confirmation_delay_steps": self.confirmation_delay_steps,
                    "model_id": model_id,
                    "message": "Predicted critical condition within the selected horizon",
                }
                self.episodes.append(opened)
        else:
            self.consecutive_warn = 0
            if predicted_rul_s > self.reset_factor * h:
                self.consecutive_clear += 1
            else:
                self.consecutive_clear = 0
            if self.warning_active and self.consecutive_clear >= k:
                self.warning_active = False

        self.last_status = "Warning" if self.warning_active else "No horizon alert"
        return self._snapshot(timestamp_s, predicted_rul_s, opened=opened)

    def _snapshot(self, timestamp_s: float, predicted_rul_s: float | None, opened) -> dict[str, Any]:
        return {
            "timestamp_s": timestamp_s,
            "predicted_rul_s": predicted_rul_s,
            "status": self.last_status,
            "warning_active": self.warning_active,
            "opened_episode": opened,
            "n_episodes": len(self.episodes),
        }


def _finite(x: float) -> bool:
    try:
        return x == x and abs(x) != float("inf")
    except Exception:
        return False


def classify_alert_timing(alert_time_s: float, event_time_s: float, horizon_s: float) -> str:
    """Legacy H-window labels. Prefer `classify_alert_outcome` (min lead time)."""
    if alert_time_s < event_time_s - horizon_s:
        return "early_relative_to_selected_horizon"
    if event_time_s - horizon_s <= alert_time_s < event_time_s:
        return "timely"
    return "not_early"


def confirmation_delay_steps(confirmation_count: int) -> int:
    return max(int(confirmation_count) - 1, 0)


def confirmed_lead_time_s(*, event_time_s: float, confirmed_alert_time_s: float) -> float:
    """`event_time - episode_open_time`. Episode open is the K-th qualifying step."""
    return float(event_time_s) - float(confirmed_alert_time_s)


def _row_predicted_rul(row: Mapping[str, Any]) -> float | None:
    raw = row.get("predicted_rul_s")
    if raw is None:
        return None
    try:
        v = float(raw)
    except (TypeError, ValueError):
        return None
    return v if _finite(v) else None


def _empty_alert_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    return (
        pd.DataFrame(columns=list(ALERT_EPISODE_COLUMNS)),
        pd.DataFrame(columns=list(ALERT_STEP_COLUMNS)),
    )


def alerts_from_predictions(
    pred: pd.DataFrame | None,
    policy: Mapping[str, Any] | AlertPolicy,
    *,
    run_id: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Re-run `AlertEngine` on frozen `predicted_rul_s`. Does not call the model.

    Non-finite RUL is treated as collecting/gap warmup so a warning cannot survive
    a hole in predictions without K new confirms. Confirmed time is the K-th step.
    """
    if pred is None:
        return _empty_alert_frames()
    frame = pred if isinstance(pred, pd.DataFrame) else pd.DataFrame(pred)
    if frame.empty or "unit_id" not in frame.columns or "timestamp_s" not in frame.columns:
        return _empty_alert_frames()

    d = _policy_mapping(policy)
    h = _h_trigger(d)
    k = int(d.get("confirmation_count", 3))
    reset = float(d.get("reset_factor", 1.2))
    default_run = run_id
    if default_run is None and "run_id" in frame.columns and len(frame):
        raw_run = frame["run_id"].iloc[0]
        if raw_run is not None and not (isinstance(raw_run, float) and not _finite(raw_run)):
            default_run = str(raw_run)

    episodes: list[dict[str, Any]] = []
    steps: list[dict[str, Any]] = []
    has_status = "alert_status" in frame.columns
    has_dp = "differential_pressure" in frame.columns
    for uid, group in frame.groupby("unit_id", sort=False):
        g = group.sort_values("timestamp_s", kind="mergesort")
        engine = AlertEngine(h, k, reset)
        engine.reset_unit(str(uid))
        for rec in g.to_dict(orient="records"):
            rul = _row_predicted_rul(rec)
            status = str(rec.get("alert_status") or "") if has_status else ""
            collecting = status == "Collecting history" or rul is None
            observed_limit = status == "Observed limit reached"
            if not observed_limit and has_dp:
                try:
                    observed_limit = float(rec["differential_pressure"]) > 600.0
                except (TypeError, ValueError):
                    observed_limit = False
            rid = rec.get("run_id", default_run)
            model_id = None
            if rid is not None and not (isinstance(rid, float) and not _finite(rid)):
                model_id = str(rid)
            snap = engine.update(
                timestamp_s=float(rec["timestamp_s"]),
                predicted_rul_s=rul,
                collecting=collecting,
                observed_limit=observed_limit,
                model_id=model_id,
            )
            steps.append(
                {
                    "unit_id": str(uid),
                    "timestamp_s": float(rec["timestamp_s"]),
                    "predicted_rul_s": rul,
                    "warning_active": bool(snap["warning_active"]),
                    "status": snap["status"],
                }
            )
            opened = snap["opened_episode"]
            if opened:
                episodes.append(
                    {
                        **opened,
                        "run_id": model_id,
                        "unit_id": str(uid),
                        "model_id": opened.get("model_id") or model_id,
                    }
                )

    ep_df = pd.DataFrame(episodes)
    if ep_df.empty:
        ep_df = pd.DataFrame(columns=list(ALERT_EPISODE_COLUMNS))
    else:
        for col in ALERT_EPISODE_COLUMNS:
            if col not in ep_df.columns:
                ep_df[col] = None
        ep_df = ep_df.loc[:, [c for c in ALERT_EPISODE_COLUMNS if c in ep_df.columns]]
    st_df = pd.DataFrame(steps)
    if st_df.empty:
        st_df = pd.DataFrame(columns=list(ALERT_STEP_COLUMNS))
    return ep_df, st_df


def _policy_mapping(policy: Mapping[str, Any] | AlertPolicy) -> dict[str, Any]:
    if isinstance(policy, AlertPolicy):
        return policy.to_dict()
    return dict(policy)


def _h_trigger(policy: Mapping[str, Any]) -> float:
    h = policy.get("H_trigger", policy.get("warning_horizon_s"))
    if h is None:
        raise ValueError("alert policy requires H_trigger (alias warning_horizon_s)")
    return float(h)


def _min_lead(policy: Mapping[str, Any]) -> float:
    v = policy.get("minimum_action_lead_time")
    if v is None:
        raise ValueError("alert policy requires minimum_action_lead_time")
    return float(v)


def coverage_horizon_s(policy: Mapping[str, Any] | AlertPolicy) -> float:
    """Lookback that must be observed to score miss vs insufficient_coverage."""
    d = _policy_mapping(policy)
    return max(_h_trigger(d), _min_lead(d))


def has_sufficient_coverage(
    observation_end_s: float,
    event_time_s: float,
    policy: Mapping[str, Any] | AlertPolicy,
) -> bool:
    return float(observation_end_s) >= float(event_time_s) - coverage_horizon_s(policy)


def classify_alert_outcome(
    *,
    event_time_s: float | None,
    observation_end_s: float,
    policy: Mapping[str, Any] | AlertPolicy,
    confirmed_alert_time_s: float | None = None,
) -> str | None:
    """Classify one confirmed episode (or no alert) against a frozen policy.

    Confirmed alert time is the timestamp when the episode opens (K-th step).
    `lead_time = event_time - confirmed_alert_time` therefore already includes the
    K-1 confirmation delay.

    - timely iff lead_time >= minimum_action_lead_time and alert_time < event_time
    - too_early: alert before event and lead_time > max_useful_horizon_s (if set)
    - late: alert at/after event, or before event with lead_time < minimum_action_lead_time
    - miss: no alert and the pre-event window was observed
    - insufficient_coverage: no alert and observation_end < event - coverage_horizon

    Returns None when there is no finite event to score.
    """
    if event_time_s is None or not _finite(float(event_time_s)):
        return None
    et = float(event_time_s)
    d = _policy_mapping(policy)
    min_lead = _min_lead(d)
    max_useful_f = _optional_horizon(d.get("max_useful_horizon_s"))

    if confirmed_alert_time_s is not None and _finite(float(confirmed_alert_time_s)):
        lead = confirmed_lead_time_s(
            event_time_s=et, confirmed_alert_time_s=float(confirmed_alert_time_s)
        )
        if lead <= 0.0:
            return "late"
        if max_useful_f is not None and lead > max_useful_f:
            return "too_early"
        if lead >= min_lead:
            return "timely"
        return "late"

    if has_sufficient_coverage(float(observation_end_s), et, d):
        return "miss"
    return "insufficient_coverage"


@dataclass(frozen=True)
class AlertPolicy:
    H_trigger: float
    minimum_action_lead_time: float
    confirmation_count: int = 3
    reset_factor: float = 1.2
    max_useful_horizon_s: float | None = None
    schema_version: str = ALERT_POLICY_SCHEMA_VERSION
    source: str = "config"

    @property
    def warning_horizon_s(self) -> float:
        return float(self.H_trigger)

    def to_dict(self) -> dict[str, Any]:
        h = float(self.H_trigger)
        payload: dict[str, Any] = {
            "schema_version": str(self.schema_version),
            "H_trigger": h,
            "warning_horizon_s": h,
            "minimum_action_lead_time": float(self.minimum_action_lead_time),
            "max_useful_horizon_s": _optional_horizon(self.max_useful_horizon_s),
            "confirmation_count": int(self.confirmation_count),
            "reset_factor": float(self.reset_factor),
            "source": str(self.source),
        }
        payload["policy_hash"] = alert_policy_hash(payload)
        return payload

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> AlertPolicy:
        h = _h_trigger(data)
        lead = data.get("minimum_action_lead_time")
        if lead is None:
            lead = default_minimum_action_lead_time(h, data)
        max_u = data.get("max_useful_horizon_s")
        return cls(
            H_trigger=h,
            minimum_action_lead_time=float(lead),
            confirmation_count=int(data.get("confirmation_count", 3)),
            reset_factor=float(data.get("reset_factor", 1.2)),
            max_useful_horizon_s=_optional_horizon(max_u),
            schema_version=str(data.get("schema_version") or ALERT_POLICY_SCHEMA_VERSION),
            source=str(data.get("source") or "config"),
        )


def default_minimum_action_lead_time(
    h_trigger: float, alerts_cfg: Mapping[str, Any] | None = None
) -> float:
    cfg = dict(alerts_cfg or {})
    if cfg.get("minimum_action_lead_time_s") is not None:
        return max(0.0, float(cfg["minimum_action_lead_time_s"]))
    frac = float(cfg.get("minimum_action_lead_time_fraction", 0.5))
    return max(0.0, float(h_trigger) * frac)


def default_max_useful_horizon_s(
    h_trigger: float, alerts_cfg: Mapping[str, Any] | None = None
) -> float | None:
    cfg = dict(alerts_cfg or {})
    if cfg.get("max_useful_horizon_s") is not None:
        return _optional_horizon(cfg["max_useful_horizon_s"])
    frac = cfg.get("max_useful_horizon_fraction")
    if frac is None:
        return None
    return _optional_horizon(float(h_trigger) * float(frac))


def canonical_alert_policy_fields(policy: Mapping[str, Any] | AlertPolicy) -> dict[str, Any]:
    d = _policy_mapping(policy)
    h = _h_trigger(d)
    max_u = d.get("max_useful_horizon_s")
    return {
        "schema_version": str(d.get("schema_version") or ALERT_POLICY_SCHEMA_VERSION),
        "H_trigger": h,
        "minimum_action_lead_time": float(d["minimum_action_lead_time"]),
        "max_useful_horizon_s": _optional_horizon(max_u),
        "confirmation_count": int(d.get("confirmation_count", 3)),
        "reset_factor": float(d.get("reset_factor", 1.2)),
    }


def alert_policy_hash(policy: Mapping[str, Any] | AlertPolicy) -> str:
    payload = {k: canonical_alert_policy_fields(policy)[k] for k in _POLICY_HASH_FIELDS}
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def alert_policy_eval_block(policy: Mapping[str, Any] | AlertPolicy) -> dict[str, Any]:
    """Fields 15b copies into `evaluation_config.json` (nested under `alert_policy`)."""
    fields = canonical_alert_policy_fields(policy)
    h = float(fields["H_trigger"])
    return {
        **fields,
        "warning_horizon_s": h,
        "K": int(fields["confirmation_count"]),
        "policy_hash": alert_policy_hash(fields),
    }


def copy_alert_policy_into_evaluation_config(
    evaluation_config: Mapping[str, Any],
    policy: Mapping[str, Any] | AlertPolicy | None = None,
    *,
    run_dir: Path | None = None,
    metrics_version: str | None = None,
) -> dict[str, Any]:
    """Return a new config with nested `alert_policy`. Does not write files or touch predictions."""
    if policy is None:
        if run_dir is None:
            raise ValueError("policy or run_dir required")
        loaded = load_alert_policy(run_dir)
        if loaded is None:
            raise FileNotFoundError(f"No {ALERT_POLICY_FILENAME} under {run_dir}")
        policy = loaded
    out = dict(evaluation_config)
    block = alert_policy_eval_block(policy)
    if metrics_version is not None:
        block["metrics_version"] = str(metrics_version)
    out["alert_policy"] = block
    return out


def build_alert_policy(
    *,
    H_trigger: float,
    minimum_action_lead_time: float | None = None,
    confirmation_count: int = 3,
    reset_factor: float = 1.2,
    max_useful_horizon_s: float | None = None,
    alerts_cfg: Mapping[str, Any] | None = None,
    source: str = "config",
    fill_max_useful_from_cfg: bool = True,
) -> dict[str, Any]:
    h = float(H_trigger)
    lead = (
        float(minimum_action_lead_time)
        if minimum_action_lead_time is not None
        else default_minimum_action_lead_time(h, alerts_cfg)
    )
    # Explicit None (frozen JSON null) must not be replaced by YAML.
    if fill_max_useful_from_cfg and max_useful_horizon_s is None and alerts_cfg is not None:
        max_useful_horizon_s = default_max_useful_horizon_s(h, alerts_cfg)
    return AlertPolicy(
        H_trigger=h,
        minimum_action_lead_time=lead,
        confirmation_count=int(confirmation_count),
        reset_factor=float(reset_factor),
        max_useful_horizon_s=_optional_horizon(max_useful_horizon_s),
        source=source,
    ).to_dict()


def alert_policy_path(run_dir: Path) -> Path:
    return Path(run_dir) / ALERT_POLICY_FILENAME


def load_alert_policy(run_dir: Path) -> dict[str, Any] | None:
    path = alert_policy_path(run_dir)
    if not path.exists():
        return None
    return AlertPolicy.from_mapping(read_json(path)).to_dict()


def save_alert_policy(
    run_dir: Path,
    policy: Mapping[str, Any] | AlertPolicy,
    *,
    overwrite: bool = True,
) -> dict[str, Any]:
    path = alert_policy_path(run_dir)
    if path.exists() and not overwrite:
        loaded = load_alert_policy(run_dir)
        if loaded is None:
            raise FileNotFoundError(path)
        return loaded
    payload = AlertPolicy.from_mapping(_policy_mapping(policy)).to_dict()
    payload["frozen_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    payload["policy_hash"] = alert_policy_hash(payload)
    atomic_write_json(path, payload)
    return payload


def ensure_alert_policy(run_dir: Path, policy: Mapping[str, Any] | AlertPolicy) -> dict[str, Any]:
    """Write `runs/<run_id>/alert_policy.json` only if missing. Never rewrites predictions."""
    return save_alert_policy(run_dir, policy, overwrite=False)


def resolve_alert_policy(
    run_dir: Path,
    *,
    alerts_cfg: Mapping[str, Any] | None = None,
    H_trigger: float | None = None,
    warning_horizon_s: float | None = None,
    confirmation_count: int | None = None,
    minimum_action_lead_time: float | None = None,
    max_useful_horizon_s: float | None = None,
    reset_factor: float | None = None,
    default_h_trigger: float | None = None,
    source: str = "evaluate",
) -> dict[str, Any]:
    """Merge explicit args over frozen run policy over YAML defaults. Does not write."""
    raw_path = alert_policy_path(run_dir)
    raw = read_json(raw_path) if raw_path.exists() else {}
    if not isinstance(raw, dict):
        raw = {}
    frozen = load_alert_policy(run_dir) or {}
    cfg = dict(alerts_cfg or {})
    h = H_trigger if H_trigger is not None else warning_horizon_s
    if h is None and frozen:
        h = frozen.get("H_trigger", frozen.get("warning_horizon_s"))
    if h is None:
        h = default_h_trigger
    if h is None:
        raise ValueError("H_trigger / warning_horizon_s is required to resolve an alert policy")
    k = confirmation_count
    if k is None:
        k = frozen.get("confirmation_count", cfg.get("confirmation_count", 3))
    reset = reset_factor
    if reset is None:
        reset = frozen.get("reset_factor", cfg.get("reset_factor", 1.2))
    lead = minimum_action_lead_time
    if lead is None:
        lead = frozen.get("minimum_action_lead_time")
    # Raw file key presence: JSON null is an explicit unset, not "use YAML".
    # load_alert_policy() always re-emits the key, so check the file, not the normalized dict.
    max_u = max_useful_horizon_s
    fill_max = True
    if max_useful_horizon_s is not None:
        fill_max = False
    elif "max_useful_horizon_s" in raw:
        max_u = raw.get("max_useful_horizon_s")
        fill_max = False
    return build_alert_policy(
        H_trigger=float(h),
        minimum_action_lead_time=None if lead is None else float(lead),
        confirmation_count=int(k),
        reset_factor=float(reset),
        max_useful_horizon_s=_optional_horizon(max_u),
        alerts_cfg=cfg,
        source=str(frozen.get("source") or source),
        fill_max_useful_from_cfg=fill_max,
    )
