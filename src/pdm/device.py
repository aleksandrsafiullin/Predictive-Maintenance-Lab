from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DeviceInfo:
    name: str
    torch_device: str
    fallback_reason: str | None = None


def resolve_device(preference: str = "auto") -> DeviceInfo:
    """CUDA, then MPS, then CPU. Failures fall back to CPU with a reason."""
    import torch

    pref = (preference or "auto").lower()
    if pref not in {"auto", "cpu", "cuda", "mps"}:
        pref = "auto"

    if pref == "cpu":
        return DeviceInfo("cpu", "cpu")

    if pref in {"auto", "cuda"}:
        try:
            if torch.cuda.is_available():
                torch.zeros(1, device="cuda")
                return DeviceInfo("cuda", "cuda")
            if pref == "cuda":
                return DeviceInfo("cpu", "cpu", "CUDA requested but torch.cuda.is_available() is False")
        except Exception as exc:  # noqa: BLE001
            if pref == "cuda":
                return DeviceInfo("cpu", "cpu", f"CUDA backend error, using CPU: {exc}")

    if pref in {"auto", "mps"}:
        try:
            mps = getattr(torch.backends, "mps", None)
            if mps is not None and mps.is_available():
                torch.zeros(1, device="mps")
                return DeviceInfo("mps", "mps")
            if pref == "mps":
                return DeviceInfo("cpu", "cpu", "MPS requested but not available")
        except Exception as exc:  # noqa: BLE001
            return DeviceInfo("cpu", "cpu", f"MPS backend error, using CPU: {exc}")

    return DeviceInfo("cpu", "cpu")
