from __future__ import annotations

from pathlib import Path

import streamlit.components.v1 as components

_COMPONENT_DIR = Path(__file__).parent / "frontend"
neural_activity_explorer = components.declare_component(
    "neural_activity_explorer",
    path=str(_COMPONENT_DIR),
)
