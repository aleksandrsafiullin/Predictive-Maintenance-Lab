"""Regression checks for the actual anatomical draw buffers, without a GPU."""
from __future__ import annotations

import json
import shutil
import subprocess

import numpy as np
import pytest

from pdm.paths import project_root


@pytest.fixture(scope="module")
def anatomy_draw_report():
    """Execute the shipped JS with real Three geometry and a stub GPU boundary."""
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is needed to execute the frontend renderer checks")
    frontend = project_root() / "src/pdm/visualization/component/frontend"
    script = r'''
const fs = require("fs"), vm = require("vm");
const frontend = process.argv[1];
const THREE = require(frontend + "/vendor/three.min.js");
// Only the GPU is stubbed. The production scene, attributes, camera and shaders run.
THREE.WebGLRenderer = class {
    setClearColor() {} setPixelRatio() {} setSize() {} render() {}
};
let source = fs.readFileSync(frontend + "/main.js", "utf8");
const end = source.indexOf('    canvas.addEventListener("mousedown"');
if (end < 0) throw Error("Frontend setup boundary missing");
source = source.slice(0, end) + `
    globalThis.probe = {
        applyArgs, initRenderer, sceneBounds, cameraPose, selectDrawnEdges,
        draw: function() {
            var values = sampleState(frameIndex);
            renderer.render(values, []); updateHud(values);
            var context = renderer.scene.children.filter(function (c) {
                return c.isPoints && c !== renderer.core && c !== renderer.idle;
            });
            return {
                visible: renderer.visIndex,
                positions: Array.from(renderer.core.geometry.attributes.position.array),
                sizes: Array.from(renderer.sizeAttr.array),
                colors: Array.from(renderer.colorAttr.array),
                contextPositions: context.map(c => Array.from(c.geometry.attributes.position.array)),
                contextColors: context.map(c => Array.from(c.geometry.attributes.color.array)),
                geometryId: renderer.core.geometry.uuid,
                normalBlend: renderer.core.material.blending === THREE.NormalBlending,
                contextGeometryIds: context.map(c => c.geometry.uuid),
                peak: renderer.peak,
                frame: frameIndex,
                hud: hudLine.textContent,
                hint: hudHint.textContent,
                banner: hudBanner.textContent,
                dataset: Object.assign({}, canvas.dataset)
            };
        }
    };
})();`;
const elements = {};
const sandbox = {THREE, document: {
    getElementById: id => elements[id] ||= {style: {}, dataset: {}, clientWidth: 640, clientHeight: 620}
}, window: {devicePixelRatio: 1}, console};
vm.createContext(sandbox); vm.runInContext(source, sandbox);
const p = sandbox.probe;
const nodes = Array.from({length: 1000}, (_, i) => String(1000000 + i));
const positions = Object.fromEntries(nodes.map((id, i) => [id, [i / 1000, Math.sin(i) / 2, Math.cos(i) / 10]]));
const args = {
    nodes, positions, states: [nodes.map((_, i) => i === 0 ? 0 : (i % 2 ? 1 : -1) * 1e-5)],
    context_positions: [[-2, -3, -0.4], [2, 3, 0.4], [0, 2, 0]],
    flags: {hull_mode: "malecns_anatomy", synchronized: true, activity_scale: 1,
        context_downsampled: true, context_caption: "Soma context downsampled for display",
        failure_window_s: [600, 900], forecast_label: "Empirical failure window", time_scale: 60, time_unit: "min"},
    frame_map: [{timestamp_s: 120}],
    edges: nodes.slice(1).map(id => ({src: nodes[0], dst: id}))
};
// Match postMessage delivery: every event supplies a fresh deserialized object.
function deliver() { p.applyArgs(JSON.parse(JSON.stringify(args))); }
deliver(); p.initRenderer();
const first = p.draw();
const bounds = p.sceneBounds();
const edgesDefault = p.selectDrawnEdges().length;
args.states = [nodes.map((_, i) => i % 2 ? -.8 : .8)];
args.frame_map = [{timestamp_s: 180}];
args.flags.activity_scale = .5;
deliver();
const second = p.draw();
args.flags.show_connections = true; deliver();
const selectedEdges = p.selectDrawnEdges().length / 2;
delete positions[nodes[12]]; deliver();
const missing = p.draw();
console.log(JSON.stringify({first, second, bounds, edgesDefault, selectedEdges, missing,
    sourcePositions: Object.values(positions).flat()}));
'''
    result = subprocess.run(
        [node, "-e", script, str(frontend)], text=True, capture_output=True,
        timeout=30, check=True,
    )
    return json.loads(result.stdout)


def test_all_1000_compute_cells_have_draw_buffers_even_at_zero_or_tiny_signal(anatomy_draw_report):
    first = anatomy_draw_report["first"]
    assert first["visible"] == list(range(1000))
    assert len(first["positions"]) == 3000
    assert len(first["sizes"]) == 1000
    assert min(first["sizes"]) > 0
    assert min(first["colors"]) > 0
    assert first["dataset"]["drawnNeurons"] == first["dataset"]["modelNeurons"] == "1000"


def test_live_signal_changes_compute_colors_but_never_anatomical_reference(anatomy_draw_report):
    first, second = anatomy_draw_report["first"], anatomy_draw_report["second"]
    assert first["contextPositions"] == second["contextPositions"]
    assert first["contextColors"] == second["contextColors"]
    assert first["positions"] == second["positions"]
    assert first["colors"] != second["colors"]
    assert first["normalBlend"] and second["normalBlend"]  # Dense cells preserve signed colours.
    assert second["colors"][0] > second["colors"][2]  # positive is amber
    assert second["colors"][3] < second["colors"][5]  # negative is cyan
    assert first["frame"] == second["frame"] == 0
    assert first["dataset"]["timestampS"] == "120"
    assert second["dataset"]["timestampS"] == "180"
    assert "Failure window 10.0–15.0 min" in second["hud"]
    assert "Continuous history" in second["hint"]
    assert second["banner"] == ""


def test_observation_updates_keep_gpu_geometry_and_refresh_signal_scale(anatomy_draw_report):
    first, second = anatomy_draw_report["first"], anatomy_draw_report["second"]
    assert first["geometryId"] == second["geometryId"]
    assert first["contextGeometryIds"] == second["contextGeometryIds"]
    assert first["dataset"]["geometryBuilds"] == second["dataset"]["geometryBuilds"]
    assert first["peak"] == 1 and second["peak"] == 0.5
    missing = anatomy_draw_report["missing"]
    assert missing["geometryId"] != second["geometryId"]
    assert int(missing["dataset"]["geometryBuilds"]) > int(second["dataset"]["geometryBuilds"])


def test_missing_body_is_not_fabricated_and_full_context_frames_brain(anatomy_draw_report):
    report = anatomy_draw_report
    missing = report["missing"]
    assert len(missing["visible"]) == 999
    assert 12 not in missing["visible"]
    assert missing["positions"] == pytest.approx(report["sourcePositions"])
    assert missing["dataset"]["drawnNeurons"] == "999"
    assert "999 / 1000 neurons" in missing["hud"]
    assert report["bounds"]["minY"] == -3
    assert report["bounds"]["maxY"] == 3
    assert report["edgesDefault"] == 0
    assert 0 < report["selectedEdges"] <= 350


def test_anatomy_payload_keeps_all_model_ids_states_and_coordinates():
    from pdm.connectome.anatomy import SomaTable
    from pdm.visualization.explorer import build_explorer_payload

    # Deliberately reverse large IDs: anatomical ordering must never reorder states.
    nodes = [str(2**53 + i) for i in range(1000, 0, -1)]
    soma_positions = {nid: [float(i), float(i % 11), float(i % 19)] for i, nid in enumerate(nodes)}
    soma_positions.update({str(i): [-float(i), float(i % 7), 0.0] for i in range(1, 101)})
    soma = SomaTable(soma_positions, {}, False, len(soma_positions))
    states = np.arange(2000, dtype=float).reshape(2, 1000) / 2000
    payload = build_explorer_payload(
        nodes=nodes, edges=[], positions={}, states=states,
        frame_map=[{"timestamp_s": 0}, {"timestamp_s": 60}],
        flags={"graph_mode": "real_connectome", "is_synthetic": False},
        soma=soma, n_model=1000, context_cap=25,
    )
    assert payload["nodes"] == nodes
    assert set(payload["positions"]) == set(nodes)
    assert payload["states"] == states.tolist()
    assert len(payload["context_positions"]) == 25
    assert payload["flags"]["n_model"] == 1000
    assert payload["flags"]["n_reservoir_visible"] == 1000
    assert payload["flags"]["n_unmatched_reservoir"] == 0
    # Shared affine normalization preserves relative anatomical distances.
    x = np.asarray([payload["positions"][nid][0] for nid in nodes])
    assert np.diff(x).min() > 0
    assert np.diff(x).max() - np.diff(x).min() < 2e-6
