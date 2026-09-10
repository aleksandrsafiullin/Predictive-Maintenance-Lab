from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


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
            self.last_status = "Collecting history"
            self.consecutive_warn = 0
            self.consecutive_clear = 0
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
                opened = {
                    "type": "horizon_warning",
                    "unit_id": self.unit_id,
                    "timestamp_s": timestamp_s,
                    "predicted_rul_s": predicted_rul_s,
                    "warning_horizon_s": h,
                    "confirmation_count": k,
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
    if alert_time_s < event_time_s - horizon_s:
        return "early_relative_to_selected_horizon"
    if event_time_s - horizon_s <= alert_time_s < event_time_s:
        return "timely"
    return "not_early"
