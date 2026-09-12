"""Predictive Maintenance Lab — XJTU-SY bearings and HSE filters."""

__version__ = "0.1.0"

DATASET_BEARINGS = "bearings"
DATASET_FILTERS = "filters"
VALID_DATASETS = (DATASET_BEARINGS, DATASET_FILTERS)

STATUSES = (
    "not_ready",
    "preparing",
    "ready",
    "training",
    "completed",
    "cancelled",
    "stopped",  # legacy status.json files only; new interrupts write cancelled
    "failed",
)
