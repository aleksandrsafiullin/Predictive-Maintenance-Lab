"""Actual PyTorch recurrent states; no illustrative or generated activity."""
from __future__ import annotations

import numpy as np
import torch

from pdm.predict import model_forecast, prepare_history_window


def recurrent_trace(history, model, prep, history_length, device="cpu"):
    prepared = prepare_history_window(history, prep, history_length)
    result = {
        "status": "Collecting history", "predicted_rul_s": None,
        "n_history": prepared["n_history"], "valid_history_reason": prepared.get("valid_history_reason", ""),
        "states": np.zeros((0, model.n_nodes), dtype=np.float32), "inputs": np.zeros((0, len(prep.feature_names)), dtype=np.float32),
        "node_order": model.node_order, "frame_map": [], "architecture": model.architecture,
    }
    if not prepared["ok"]:
        return result
    model.eval()
    x = torch.from_numpy(np.array(prepared["inputs"], dtype=np.float32, copy=True, order="C")).unsqueeze(0).to(device)
    h = None
    hidden, cells = [], []
    with torch.no_grad():
        forecast = model_forecast(model, x)
        for i in range(x.shape[1]):
            _, h = model.encoder.rnn(x[:, i:i + 1], h)
            state = h[0] if isinstance(h, tuple) else h
            hidden.append(state[:, 0].reshape(-1).cpu().numpy())
            if isinstance(h, tuple):
                cells.append(h[1][:, 0].reshape(-1).cpu().numpy())
    ready = np.isfinite(forecast["predicted_rul_s"]) and forecast["predicted_rul_s"] >= 0
    result.update(forecast)
    result.update(status="predicted" if ready else "No valid prediction", states=np.asarray(hidden),
                  inputs=prepared["inputs"], cell_states=np.asarray(cells),
                  timestamps_s=prepared["window"].timestamp_s.to_numpy(),
                  frame_map=[{"timestamp_s": float(t)} for t in prepared["window"].timestamp_s])
    return result


def render_recurrent_trace(trace, model, prep):
    import plotly.graph_objects as go
    import streamlit as st
    from plotly.subplots import make_subplots

    st.markdown(f"**{model.architecture.upper()}** · {model.encoder.rnn.num_layers} layer(s) · "
                f"{model.encoder.rnn.hidden_size} hidden units per layer")
    st.caption("Sensor features → recurrent state → trained output head → remaining life")
    states = trace.get("states")
    if states is None or len(states) == 0:
        st.info("Collecting the saved model's history window. Actual state activity appears with the first forecast.")
        return
    has_cells = len(trace.get("cell_states", [])) > 0
    panels = 3 if has_cells else 2
    titles = ["Normalized sensor inputs", "Hidden state h"] + (["Cell memory c"] if has_cells else [])
    fig = make_subplots(rows=panels, cols=1, subplot_titles=titles, vertical_spacing=0.16)
    ts = trace["timestamps_s"]
    tensors = [trace["inputs"].T, states.T] + ([trace["cell_states"].T] if has_cells else [])
    for row, values in enumerate(tensors, 1):
        labels = prep.feature_names if row == 1 else model.node_order
        fig.add_trace(go.Heatmap(z=values, x=ts, y=labels, colorscale=[[0, "#61d8ee"], [0.5, "#15202c"], [1, "#f2be68"]],
                                zmid=0, showscale=False, hovertemplate="%{y}<br>Time %{x}<br>Value %{z:.5f}<extra></extra>"), row=row, col=1)
        fig.update_yaxes(showticklabels=False, row=row, col=1)
    fig.update_layout(height=495, paper_bgcolor="#101925", plot_bgcolor="#101925", font_color="#edf4fa", margin=dict(l=15, r=15, t=35, b=25))
    fig.update_xaxes(title_text="Measurement time (internal seconds)", row=panels, col=1)
    st.plotly_chart(fig, width="stretch", theme=None)
    st.caption("Actual state values; each prediction resets at the beginning of its saved history window. Hover to inspect a feature or hidden unit.")
