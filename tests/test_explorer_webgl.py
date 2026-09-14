from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pytest

from pdm.paths import project_root

FRONTEND = project_root() / "src" / "pdm" / "visualization" / "component" / "frontend"
CDN_HOSTS = ("unpkg", "cdn.jsdelivr", "cdnjs", "googleapis")
SCRIPT_HTTPS_SRC = re.compile(
    r"""<script[^>]+(?:src|href)\s*=\s*["']https://""",
    re.IGNORECASE,
)
MODULE_HTTPS = re.compile(
    r"""(?:from|import)\s+["']https://""",
    re.IGNORECASE,
)
SCHEMATIC_IF = re.compile(
    r"""if\s*\(\s*(?:hullMode\(\)|(?:[\w.]+\.)?hull_mode)\s*===\s*["']schematic_cns["']\s*\)"""
)
ANATOMY_IF = re.compile(
    r"""(?:else\s+)?if\s*\(\s*(?:isAnatomy\(\)|hullMode\(\)\s*===\s*["']malecns_anatomy["'])\s*\)"""
)
CARTOON_CALL = re.compile(r"(?:this\s*\.\s*)?\b(fitNodesIntoHull|polarToCns)\s*\(")
CARTOON_DEFN = re.compile(r"function\s+(?:fitNodesIntoHull|polarToCns)\s*\(")
BLEND_FUNC = re.compile(
    r"blendFunc\s*\(\s*gl\.(\w+)\s*,\s*gl\.(\w+)\s*\)",
)
CONTEXT_DRAW = re.compile(r"drawArrays\s*\([^;]*this\.nContext")


def _main_js() -> str:
    return (FRONTEND / "main.js").read_text(encoding="utf-8")


def _write_feather(path: Path, frame: dict) -> Path:
    import pandas as pd

    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(frame).to_feather(path)
    return path


def _write_run_connectome(rdir: Path, *, nodes, layout, provenance, edges=None) -> Path:
    cdir = Path(rdir) / "connectome"
    cdir.mkdir(parents=True, exist_ok=True)
    (cdir / "graph.json").write_text(
        json.dumps(
            {
                "nodes": list(nodes),
                "node_order": list(nodes),
                "edges": list(edges or []),
            }
        ),
        encoding="utf-8",
    )
    (cdir / "layout.json").write_text(json.dumps(layout), encoding="utf-8")
    (cdir / "provenance.json").write_text(json.dumps(provenance), encoding="utf-8")
    return cdir


def _skip_js_string(src: str, i: int) -> int:
    q = src[i]
    i += 1
    while i < len(src):
        if src[i] == "\\":
            i += 2
            continue
        if src[i] == q:
            return i + 1
        i += 1
    return i


def _matching_brace_span(src: str, open_idx: int) -> tuple[int, int]:
    if open_idx >= len(src) or src[open_idx] != "{":
        raise AssertionError("expected '{'")
    depth = 0
    i = open_idx
    while i < len(src):
        c = src[i]
        if c in "\"'`":
            i = _skip_js_string(src, i)
            continue
        if c == "/" and i + 1 < len(src):
            nxt = src[i + 1]
            if nxt == "/":
                nl = src.find("\n", i)
                i = len(src) if nl < 0 else nl + 1
                continue
            if nxt == "*":
                end = src.find("*/", i + 2)
                i = len(src) if end < 0 else end + 2
                continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return open_idx, i
        i += 1
    raise AssertionError("unbalanced braces")


def _js_fn_body(src: str, sig: str) -> str:
    idx = src.find(sig)
    assert idx >= 0, f"missing {sig}"
    brace = src.find("{", idx)
    assert brace >= 0, f"no body for {sig}"
    lo, hi = _matching_brace_span(src, brace)
    return src[lo + 1 : hi]


def _block_after_condition(src: str, cond_end: int) -> tuple[str, int]:
    i = cond_end
    while i < len(src) and src[i] in " \t\r\n":
        i += 1
    assert i < len(src) and src[i] == "{", "schematic/anatomy guard must use a braced block"
    lo, hi = _matching_brace_span(src, i)
    return src[lo + 1 : hi], hi


def _blocks_and_rest(body: str, guard: re.Pattern[str]) -> tuple[list[str], str]:
    blocks: list[str] = []
    rest_parts: list[str] = []
    last = 0
    for m in guard.finditer(body):
        rest_parts.append(body[last : m.start()])
        block, hi = _block_after_condition(body, m.end())
        blocks.append(block)
        last = hi + 1
    rest_parts.append(body[last:])
    return blocks, "".join(rest_parts)


def _cartoon_calls(text: str) -> list[str]:
    hits: list[str] = []
    for line in text.splitlines():
        if CARTOON_DEFN.search(line):
            continue
        if CARTOON_CALL.search(line):
            hits.append(line.strip())
    return hits


def _named_calls(text: str, name: str) -> list[str]:
    call = re.compile(rf"(?:this\s*\.\s*)?\b{re.escape(name)}\s*\(")
    defn = re.compile(rf"function\s+{re.escape(name)}\s*\(")
    hits: list[str] = []
    for line in text.splitlines():
        if defn.search(line):
            continue
        if call.search(line):
            hits.append(line.strip())
    return hits


def _assert_cartoon_mapping_schematic_only(src: str) -> None:
    """Calls to fitNodesIntoHull / polarToCns must be on the schematic path only."""
    place = _js_fn_body(src, "function placeReservoirNodes")
    schematic_blocks, place_rest = _blocks_and_rest(place, SCHEMATIC_IF)
    assert schematic_blocks, "placeReservoirNodes must guard with hullMode() === schematic_cns"
    fit_in_schematic = [c for b in schematic_blocks for c in _named_calls(b, "fitNodesIntoHull")]
    assert len(fit_in_schematic) == 1, fit_in_schematic
    assert _named_calls(src, "fitNodesIntoHull") == fit_in_schematic
    assert not _cartoon_calls(place_rest)

    fit_body = _js_fn_body(src, "function fitNodesIntoHull")
    polar = _named_calls(src, "polarToCns")
    assert polar, "expected polarToCns( call"
    assert polar == _named_calls(fit_body, "polarToCns")

    leaked = _cartoon_calls(_js_fn_body(src, "function placeAnatomyNodes"))
    leaked.extend(_cartoon_calls(_js_fn_body(src, "function applyArgs(args)")))
    for sig in (
        "ThreeRenderer.prototype.rebuild = function ()",
        "WebGLRenderer.prototype.rebuild = function ()",
    ):
        body = _js_fn_body(src, sig)
        _, rebuild_rest = _blocks_and_rest(body, SCHEMATIC_IF)
        leaked.extend(_cartoon_calls(rebuild_rest))
        for block in _blocks_and_rest(body, ANATOMY_IF)[0]:
            leaked.extend(_cartoon_calls(block))
    assert leaked == [], "cartoon mapping on anatomy path: " + "; ".join(leaked)


def _is_additive_blend(src_factor: str, dst_factor: str) -> bool:
    return dst_factor == "ONE" and src_factor in {"SRC_ALPHA", "ONE"}


def _webgl_blend_segments(render_body: str) -> list[tuple[str | None, str]]:
    segments: list[tuple[str | None, str]] = []
    last = 0
    mode: str | None = None
    for m in BLEND_FUNC.finditer(render_body):
        segments.append((mode, render_body[last : m.start()]))
        mode = "additive" if _is_additive_blend(m.group(1), m.group(2)) else "normal"
        last = m.end()
    segments.append((mode, render_body[last:]))
    return segments


def _assert_webgl_context_nonadditive(src: str) -> None:
    """WebGLRenderer: context/idle use ONE_MINUS_SRC_ALPHA; SRC_ALPHA,ONE only on reservoir."""
    render = _js_fn_body(src, "WebGLRenderer.prototype.render = function")
    assert BLEND_FUNC.search(render), "WebGLRenderer.render must set blendFunc"
    segments = _webgl_blend_segments(render)
    context_draws = 0
    additive_segs = 0
    for mode, code in segments:
        ctx = bool(CONTEXT_DRAW.search(code) or ("this.nContext" in code and "drawArrays" in code))
        if ctx:
            context_draws += 1
            assert mode == "normal", "anatomy context draw uses additive blendFunc"
            assert "contextBuf" in code or "this.nContext" in code
        if "idleSizeBuf" in code and "drawArrays" in code:
            assert mode != "additive", "idle reservoir pass uses additive blendFunc"
        if mode == "additive":
            additive_segs += 1
            assert "this.nContext" not in code
            assert "contextBuf" not in code
            assert "idleSizeBuf" not in code
            assert "colorBuf" in code or "sizeBuf" in code
    assert context_draws >= 1, "WebGLRenderer.render must draw this.nContext"
    assert additive_segs >= 1, "expected SRC_ALPHA, ONE on the active reservoir pass"
    assert re.search(
        r"blendFunc\s*\(\s*gl\.SRC_ALPHA\s*,\s*gl\.ONE_MINUS_SRC_ALPHA\s*\)",
        render,
    )
    assert re.search(r"blendFunc\s*\(\s*gl\.SRC_ALPHA\s*,\s*gl\.ONE\s*\)", render)


def test_no_cdn_in_frontend():
    """No https:// script/module src in non-vendor frontend; Math.random absent from main.js."""
    html_files = list(FRONTEND.rglob("*.html"))
    js_files = list(FRONTEND.rglob("*.js"))
    assert html_files
    assert js_files
    assert (FRONTEND / "vendor" / "streamlit-component-lib.js").is_file()
    for path in html_files + js_files:
        text = path.read_text(encoding="utf-8")
        for host in CDN_HOSTS:
            assert host not in text, f"{path} contains CDN host {host}"
        if "vendor" not in path.parts:
            assert SCRIPT_HTTPS_SRC.search(text) is None, f"{path} has https script src"
            assert MODULE_HTTPS.search(text) is None, f"{path} has https module import"
            assert 'src="https://' not in text
            assert "src='https://" not in text
            assert "src=`https://" not in text
        else:
            assert SCRIPT_HTTPS_SRC.search(text) is None, f"{path} has https script src"
            assert 'src="https://' not in text
            assert "src='https://" not in text
    main_js = _main_js()
    assert "Math.random" not in main_js
    assert "requestAnimationFrame" in main_js
    html = (FRONTEND / "index.html").read_text(encoding="utf-8")
    assert "./vendor/three.min.js" in html
    assert "./vendor/streamlit-component-lib.js" in html
    assert "./main.js" in html
    assert "https://" not in html


def test_index_html_scripts_are_local_only():
    html = (FRONTEND / "index.html").read_text(encoding="utf-8")
    srcs = re.findall(r"""<script[^>]+src\s*=\s*["']([^"']+)["']""", html, flags=re.IGNORECASE)
    assert srcs == [
        "./vendor/streamlit-component-lib.js",
        "./vendor/three.min.js",
        "./main.js",
    ]


def test_main_js_anatomy_contract_strings():
    main_js = _main_js()
    assert "context_positions" in main_js
    assert "hull_polyline" in main_js
    assert "malecns_anatomy" in main_js
    assert "schematic_cns" in main_js
    assert "args.context_positions || []" in main_js
    assert "args.hull_polyline || []" in main_js
    assert 'return m ? String(m) : "schematic_cns"' in main_js
    assert "Synthetic test graph — not a biological connectome" in main_js
    assert "flags.anatomy_missing" in main_js
    assert "flags.context_downsampled" in main_js
    assert "flags.context_caption" in main_js
    assert "Play" in main_js
    assert "Pause" in main_js
    assert "requestAnimationFrame" in main_js


def test_cartoon_mapping_only_when_schematic_cns():
    """fitNodesIntoHull / polarToCns calls must sit inside a schematic_cns block."""
    _assert_cartoon_mapping_schematic_only(_main_js())


def test_cartoon_mapping_grep_rejects_nearby_schematic_return():
    """Nearby hullMode check is not a guard; this.fitNodesIntoHull must also fail."""
    leaky = """
    function polarToCns(x, y, i, n, maxR) { return [x, y, 0]; }
    function fitNodesIntoHull() { polarToCns(0, 0, 0, 1, 1); }
    function placeAnatomyNodes() { this.fitNodesIntoHull(); }
    function placeReservoirNodes() {
        if (hullMode() === "schematic_cns") {
            fitNodesIntoHull();
            return;
        }
        polarToCns(0, 0, 0, 1, 1);
    }
    function applyArgs(args) { polarToCns(1, 1, 0, 1, 1); }
    ThreeRenderer.prototype.rebuild = function () {
        if (hullMode() === "schematic_cns") { return; }
        if (isAnatomy()) { polarToCns(0, 0, 0, 1, 1); }
    }
    WebGLRenderer.prototype.rebuild = function () {}
    """
    with pytest.raises(AssertionError):
        _assert_cartoon_mapping_schematic_only(leaky)


def test_additive_reservoir_normal_context_blending():
    main_js = _main_js()
    assert "AdditiveBlending" in main_js
    assert "NormalBlending" in main_js
    assert "contextDustMaterial" in main_js
    dust = re.search(
        r"function contextDustMaterial\([\s\S]*?function ThreeRenderer",
        main_js,
    )
    assert dust is not None
    assert "NormalBlending" in dust.group(0)
    assert "AdditiveBlending" not in dust.group(0)
    assert re.search(
        r"additive \? THREE\.AdditiveBlending : THREE\.NormalBlending",
        main_js,
    )
    _assert_webgl_context_nonadditive(main_js)


def test_webgl_fallback_grep_rejects_additive_context():
    leaky = """
    WebGLRenderer.prototype.render = function (values, inRow) {
        gl.blendFunc(gl.SRC_ALPHA, gl.ONE);
        if (this.nContext) {
            gl.bindBuffer(gl.ARRAY_BUFFER, this.contextBuf);
            gl.drawArrays(gl.POINTS, 0, this.nContext);
        }
        gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
        gl.drawArrays(gl.POINTS, 0, n);
    }
    """
    with pytest.raises(AssertionError):
        _assert_webgl_context_nonadditive(leaky)


def test_anatomy_connections_are_optional_and_camera_can_enter_cloud():
    """A brain silhouette stays readable by default; real edges are opt-in."""
    src = _main_js()
    assert re.search(r"ANATOMY_EDGE_CAP\s*=\s*350", src)
    select = _js_fn_body(src, "function selectDrawnEdges")
    assert re.search(
        r"isAnatomy\(\)\s*&&\s*flags\.show_connections\s*!==\s*true",
        select,
    )
    limits = _js_fn_body(src, "function camDistLimits")
    assert re.search(r"isAnatomy\(\)[\s\S]*min:\s*0\.028", limits)
    assert re.search(r"min:\s*0\.28", limits)
    assert "0.028" in limits
    assert re.search(r"min:\s*0\.28\b", limits)


def test_auto_orbit_schematic_only():
    """Idle rotY increment is schematic-only; anatomy must not auto-orbit."""
    tick = _js_fn_body(_main_js(), "function tick")
    guard = re.search(
        r"if\s*\(\s*!dragging\s*&&\s*hullMode\(\)\s*===\s*[\"']schematic_cns[\"']\s*\)",
        tick,
    )
    assert guard, "auto-orbit must be guarded by hullMode() === schematic_cns"
    block, hi = _block_after_condition(tick, guard.end())
    assert re.search(r"rotY\s*\+=\s*0\.00032", block)
    rest = tick[: guard.start()] + tick[hi + 1 :]
    assert not re.search(r"rotY\s*\+=\s*0\.00032", rest)


def test_webgl_line_program_skips_point_coord():
    """LINES use a solid-color program; gl_PointCoord is undefined for lines on ANGLE."""
    src = _main_js()
    ctor = _js_fn_body(src, "function WebGLRenderer()")
    assert "this.lineProgram" in ctor
    fs = re.search(
        r"var lineFs\s*=\s*((?:\"[^\"]*\"\s*\+\s*)*\"[^\"]*\")\s*;",
        ctor,
    )
    assert fs, "WebGLRenderer must define lineFs"
    line_fs = fs.group(1)
    assert "gl_PointCoord" not in line_fs
    assert "discard" not in line_fs
    assert re.search(r"gl_FragColor\s*=\s*vec4\s*\(\s*vCol", line_fs.replace(" ", ""))
    assert "gl_PointCoord" in ctor
    render = _js_fn_body(src, "WebGLRenderer.prototype.render = function")
    assert "this.lineProgram" in render
    assert "bindLineProg" in render
    assert re.search(r"drawArrays\s*\(\s*gl\.LINES", render)
    line_draws = [
        m.start()
        for m in re.finditer(r"drawArrays\s*\(\s*gl\.LINES", render)
    ]
    assert line_draws
    for pos in line_draws:
        prefix = render[:pos]
        assert "bindLineProg" in prefix or "lineProgram" in prefix


def test_anatomy_skips_second_additive_halo():
    """Anatomy: one additive core. Schematic may keep a mild halo."""
    src = _main_js()
    rebuild = _js_fn_body(src, "ThreeRenderer.prototype.rebuild = function ()")
    assert re.search(r"if\s*\(\s*!isAnatomy\(\)\s*\)", rebuild)
    assert re.search(r"if\s*\(\s*this\.halo\s*&&\s*!isAnatomy\(\)\s*\)", rebuild)
    anatomy_blocks, _ = _blocks_and_rest(rebuild, ANATOMY_IF)
    for block in anatomy_blocks:
        assert "scene.add(this.halo)" not in block
        assert "AdditiveBlending" not in block
    render = _js_fn_body(src, "WebGLRenderer.prototype.render = function")
    assert re.search(r"if\s*\(\s*!anatomy\s*\)", render)
    add = render.find("gl.SRC_ALPHA, gl.ONE")
    assert add >= 0
    additive_tail = render[add:]
    assert re.search(r"if\s*\(\s*!anatomy\s*\)", additive_tail)


def test_payload_malecns_anatomy_from_tmp_somas(monkeypatch, tmp_path):
    """Fake somas in a tmp dir yield hull_mode=malecns_anatomy and non-empty context."""
    from pdm.connectome.anatomy import SOMA_ALLOWLIST, load_soma_table
    from pdm.visualization.explorer import build_explorer_payload
    from pdm.visualization.scene import build_anatomy_scene

    soma_dir = tmp_path / "somas"
    run_dir = tmp_path / "run"
    monkeypatch.setattr("pdm.connectome.anatomy.default_soma_dir", lambda: soma_dir)

    nodes = [str(1000 + i) for i in range(8)]
    extra = list(range(9000, 9012))
    body_ids = [int(n) for n in nodes] + extra
    _write_feather(
        soma_dir / SOMA_ALLOWLIST[0],
        {
            "bodyId": body_ids,
            "x": [float(i) for i in range(len(body_ids))],
            "y": [float(i) * 0.5 for i in range(len(body_ids))],
            "z": [1.0] * len(body_ids),
        },
    )
    layout = {
        "positions": {n: [0.0, 0.0, float(i)] for i, n in enumerate(nodes)},
        "node_order": nodes,
    }
    _write_run_connectome(
        run_dir,
        nodes=nodes,
        layout=layout,
        provenance={
            "graph_mode": "real_connectome",
            "is_synthetic": False,
            "n_nodes": 8,
        },
    )
    soma = load_soma_table()
    assert soma.n_points == len(body_ids)
    anatomy = build_anatomy_scene(
        scene={
            "nodes": nodes,
            "positions": layout["positions"],
            "graph_mode": "real_connectome",
            "is_synthetic": False,
        },
        soma=soma,
        n_model=None,
        graph_mode="real_connectome",
        is_synthetic=False,
        layout=layout,
        run_dir=run_dir,
    )
    assert anatomy["flags"]["hull_mode"] == "malecns_anatomy"
    assert anatomy["context_positions"]
    assert anatomy["flags"]["n_viz"] > 0
    payload = build_explorer_payload(
        nodes=nodes,
        edges=[],
        positions=layout["positions"],
        states=np.zeros((2, len(nodes))),
        frame_map=[],
        flags={"mode": "Overview", "graph_mode": "real_connectome", "is_synthetic": False},
        anatomy=anatomy,
        soma=soma,
        run_dir=run_dir,
        n_model=None,
    )
    flags = payload["flags"]
    assert flags["hull_mode"] == "malecns_anatomy"
    assert payload["context_positions"]
    assert flags["n_viz"] > 0
    assert flags["n_viz"] == flags["context_n"] == len(payload["context_positions"])
    assert flags["n_model"] == 8
    assert flags["n_model"] == json.loads((run_dir / "connectome" / "provenance.json").read_text())["n_nodes"]
    assert len(payload["states"][0]) == len(nodes)
    json.dumps(payload)
