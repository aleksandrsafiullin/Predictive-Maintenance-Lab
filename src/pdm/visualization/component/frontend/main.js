(function () {
    "use strict";

    var canvas = document.getElementById("canvas");
    var controls = document.getElementById("controls");
    var hudBanner = document.getElementById("hud-banner");
    var hudLine = document.getElementById("hud-line");
    var hudHint = document.getElementById("hud-hint");
    var hudLegend = document.getElementById("hud-legend");

    var nodes = [];
    var edges = [];
    var positions = {};
    var states = [];
    var inputs = [];
    var frameMap = [];
    var flags = {};
    var predictedRul = null;

    var playing = false;
    var frameIndex = 0;
    var speed = 2;
    var lastTs = 0;
    var accum = 0;
    var rotY = 0.72;
    var rotX = 0.28;
    var dragging = false;
    var lastX = 0;
    var lastY = 0;
    var camDist = 1;
    var renderer = null;
    var needsRender = true;
    var renderCount = 0;
    var placed = [];
    var contextPositions = [];
    var hullPolyline = [];
    var cachedBounds = null;
    var lastFitKey = "";
    var lastGeometryMode = "";
    var morphology = {};
    var morphologyXYZ = null;
    var morphologyOwners = null;
    var focusBrain = false;
    var activityHistory = [];
    var activityHistoryTimestamps = [];
    var classPalette = {};

    var playBtn;
    var pauseBtn;
    var frameSlider;
    var frameLabel;
    var speedSlider;
    var modeLabel;

    var ACTIVITY_T = 0.38;
    var CANVAS_H = 620;
    var CANVAS_H_COMPACT = 360;
    var FOV_ANATOMY = 42;
    var FOV_SCHEMATIC = 48;
    var ANATOMY_EDGE_CAP = 350;

    function hasThree() {
        return typeof THREE !== "undefined" && THREE.WebGLRenderer && THREE.BufferGeometry;
    }

    function lerp(a, b, t) {
        return a + (b - a) * t;
    }

    function clamp(v, lo, hi) {
        return Math.max(lo, Math.min(hi, v));
    }

    function nFrames() {
        return states && states.length ? states.length : 0;
    }

    function nNodes() {
        if (nodes && nodes.length) {
            return nodes.length;
        }
        if (states && states.length && states[0]) {
            return states[0].length;
        }
        return 0;
    }

    function nFeatures() {
        if (inputs && inputs.length && inputs[0]) {
            return inputs[0].length;
        }
        return 0;
    }

    function modeName() {
        return (flags && flags.mode) || "Overview";
    }

    function isCompact() {
        return Boolean(flags && flags.compact);
    }

    function canvasHeight() {
        if (document.fullscreenElement) return window.innerHeight;
        if (flags.synchronized) {
            return 570;
        }
        return isCompact() ? CANVAS_H_COMPACT : CANVAS_H;
    }

    function hullMode() {
        var m = flags && flags.hull_mode;
        return m ? String(m) : "schematic_cns";
    }

    function isAnatomy() {
        return hullMode() === "malecns_anatomy";
    }

    function asXYZ(p) {
        if (!p || !p.length) {
            return [0, 0, 0];
        }
        return [Number(p[0]) || 0, Number(p[1]) || 0, Number(p[2]) || 0];
    }

    function reservoirActive(v, peak) {
        var p = peak > 0 ? peak : 1;
        if (flags.synchronized) {
            return Math.abs(Number(v) || 0) > 1e-8;
        }
        return Math.abs(Number(v) || 0) >= ACTIVITY_T * p;
    }

    function contextSizeScale(nCtx) {
        return clamp(6.5 / Math.sqrt(Math.max(nCtx, 80)), 0.045, 0.42);
    }

    function frameBounds() {
        var n = Math.max(nFrames() - 1, 0);
        var mode = modeName();
        var start = 0;
        var end = n;
        if (mode === "Equipment replay") {
            var step = 0;
            if (flags && flags.replay_step != null) {
                step = parseInt(flags.replay_step, 10) || 0;
            }
            end = clamp(step, 0, n);
        } else if (mode === "Inside prediction window") {
            start = 0;
            end = n;
        } else if (mode === "Alert inspection") {
            var target = flags && flags.alert_timestamp_s;
            if (target != null && frameMap && frameMap.length) {
                var best = 0;
                var bestAbs = Infinity;
                for (var i = 0; i < frameMap.length; i++) {
                    var ts = frameMap[i].timestamp_s;
                    if (ts == null) {
                        continue;
                    }
                    var d = Math.abs(Number(ts) - Number(target));
                    if (d < bestAbs) {
                        bestAbs = d;
                        best = i;
                    }
                }
                start = clamp(best, 0, n);
                end = start;
            }
        }
        if (start > end) {
            start = end;
        }
        return { start: start, end: end };
    }

    function sampleRow(rows, t, width) {
        var n = rows && rows.length ? rows.length : 0;
        if (!n) {
            return [];
        }
        var i0 = clamp(Math.floor(t), 0, n - 1);
        var i1 = clamp(i0 + 1, 0, n - 1);
        var f = t - i0;
        var a = rows[i0] || [];
        var b = rows[i1] || a;
        var out = [];
        var m = Math.max(a.length, b.length, width || 0);
        for (var i = 0; i < m; i++) {
            out.push(lerp(Number(a[i] || 0), Number(b[i] || 0), f));
        }
        return out;
    }

    function sampleState(t) {
        return sampleRow(states, t, nNodes());
    }

    function sampleInputs(t) {
        return sampleRow(inputs, t, nFeatures());
    }

    function maxAbs(rows) {
        var m = 0;
        for (var t = 0; t < (rows || []).length; t++) {
            var row = rows[t] || [];
            for (var i = 0; i < row.length; i++) {
                var v = Math.abs(Number(row[i] || 0));
                if (v > m) {
                    m = v;
                }
            }
        }
        return m > 0 ? m : 1;
    }

    function nodeSizeScale() {
        if (flags.full_cns) return 0.07;
        var n = nNodes();
        if (isAnatomy() && placed && placed.length) {
            var vis = 0;
            for (var i = 0; i < placed.length; i++) {
                if (placed[i]) {
                    vis += 1;
                }
            }
            n = Math.max(vis, 8);
        }
        return clamp(20 / Math.sqrt(Math.max(n, 8)), 0.22, 1.05);
    }

    function colorFromValue(v, peak) {
        var t = clamp(Math.abs(Number(v) || 0) / peak, 0, 1);
        if (isAnatomy()) {
            // Signed values from the actual model. Zero remains a visible cell;
            // brightness is a display transfer function, never an activation.
            var amount = Math.sqrt(t);
            var target = Number(v) < 0 ? [0.38, 0.85, 0.93] : [0.95, 0.75, 0.41];
            return {
                r: lerp(0.30, target[0], amount),
                g: lerp(0.39, target[1], amount),
                b: lerp(0.45, target[2], amount),
                t: t
            };
        }
        t = t * t;
        var r;
        var g;
        var b;
        if (t < 0.5) {
            var u = t / 0.5;
            r = lerp(0.06, 0.95, u);
            g = lerp(0.14, 0.72, u);
            b = lerp(0.22, 0.38, u);
        } else {
            var u2 = (t - 0.5) / 0.5;
            r = lerp(0.95, 1.0, u2);
            g = lerp(0.72, 0.96, u2);
            b = lerp(0.38, 0.88, u2);
        }
        return { r: r, g: g, b: b, t: t };
    }

    function cellColor(index, value, peak) {
        if (flags.color_mode === "Cell classes") {
            var cls = (flags.class_ids || [])[index] || 0;
            if (!classPalette[cls]) {
                var color = new THREE.Color().setHSL(cls * 0.61803398875 % 1, 0.65, 0.62);
                classPalette[cls] = {r: color.r, g: color.g, b: color.b, t: 0.5};
            }
            return classPalette[cls];
        }
        return colorFromValue(value, peak);
    }

    function decodeBuffer(encoded, Type) {
        var raw = atob(encoded || "");
        var bytes = new Uint8Array(raw.length);
        for (var i = 0; i < raw.length; i++) bytes[i] = raw.charCodeAt(i);
        return new Type(bytes.buffer);
    }

    function drawActivityHistory() {
        var panel = document.getElementById("activity-history-panel");
        if (!panel) return;
        panel.style.display = flags.full_cns ? "block" : "none";
        if (!flags.full_cns) return;
        var historyCanvas = document.getElementById("activity-history");
        var width = Math.max(panel.clientWidth - 32, 100);
        var height = 130;
        historyCanvas.width = width * 2;
        historyCanvas.height = height * 2;
        historyCanvas.style.width = width + "px";
        var ctx = historyCanvas.getContext("2d");
        ctx.scale(2, 2);
        ctx.fillStyle = "#081019"; ctx.fillRect(0, 0, width, height);
        var n = activityHistory.length;
        var rows = n ? activityHistory[0].length : 0;
        for (var t = 0; t < n; t++) {
            for (var j = 0; j < rows; j++) {
                var v = Number(activityHistory[t][j]) || 0;
                var c = colorFromValue(v, 1);
                ctx.fillStyle = "rgb(" + Math.round(c.r*255) + "," + Math.round(c.g*255) + "," + Math.round(c.b*255) + ")";
                ctx.fillRect(t * width / 120, j * height / rows, Math.ceil(width / 120), Math.ceil(height / rows));
            }
        }
        var start = n ? activityHistoryTimestamps[0] / 60 : 0;
        var end = n ? activityHistoryTimestamps[n - 1] / 60 : 0;
        document.getElementById("history-caption").textContent = "120 selected neurons · actual states · " + start.toFixed(0) + "–" + end.toFixed(0) + " min · time →";
        historyCanvas.dataset.frames = String(n);
        historyCanvas.dataset.neurons = String(rows);
    }

    function rawXYZ(i) {
        var id = nodes[i] != null ? String(nodes[i]) : String(i);
        var p = positions[id];
        if (p && p.length >= 2) {
            return [Number(p[0]) || 0, Number(p[1]) || 0, Number(p[2]) || 0];
        }
        var n = Math.max(nNodes(), 1);
        var angle = (2 * Math.PI * i) / n;
        return [Math.cos(angle), Math.sin(angle), 0];
    }

    function polarToCns(x, y, i, n, maxR) {
        var ang = Math.atan2(y, x);
        if (!isFinite(ang)) {
            ang = (2 * Math.PI * i) / Math.max(n, 1);
        }
        var r = Math.sqrt(x * x + y * y) / (maxR > 1e-8 ? maxR : 1);
        r = clamp(r, 0, 1);
        var u = (ang + Math.PI) / (2 * Math.PI);
        var v = i / Math.max(n - 1, 1);
        var ap = clamp(0.28 * v + 0.72 * ((i * 0.6180339887) % 1), 0, 1);
        var lobeMix = u < 0.22 ? -1 : u > 0.78 ? 1 : 0;
        if (ap < 0.62 && lobeMix !== 0) {
            var la = u * 2 * Math.PI + i * 0.37;
            var lr = 0.18 + 0.28 * r;
            return [
                lobeMix * 1.18 + Math.cos(la) * lr,
                0.16 + Math.sin(la) * lr * 0.85,
                Math.sin(la * 0.7 + v) * 0.28
            ];
        }
        if (ap < 0.7) {
            var th = u * 2 * Math.PI;
            var ph = (v - 0.5) * Math.PI * 0.9;
            return [
                0.58 * Math.cos(th) * Math.cos(ph),
                0.1 + 0.4 * Math.sin(ph),
                0.36 * Math.sin(th) * Math.cos(ph)
            ];
        }
        var tt = (ap - 0.7) / 0.3;
        var rad = 0.26 * (1 - 0.55 * tt);
        var th2 = u * 2 * Math.PI;
        return [rad * Math.cos(th2), -0.48 - 1.42 * tt, rad * Math.sin(th2) * 0.7];
    }

    function fitNodesIntoHull() {
        var n = nNodes();
        var raw = [];
        var i;
        var minX = Infinity;
        var minY = Infinity;
        var minZ = Infinity;
        var maxX = -Infinity;
        var maxY = -Infinity;
        var maxZ = -Infinity;
        var maxR = 0.001;
        for (i = 0; i < n; i++) {
            var p = rawXYZ(i);
            raw.push(p);
            minX = Math.min(minX, p[0]);
            minY = Math.min(minY, p[1]);
            minZ = Math.min(minZ, p[2]);
            maxX = Math.max(maxX, p[0]);
            maxY = Math.max(maxY, p[1]);
            maxZ = Math.max(maxZ, p[2]);
            var rr = Math.sqrt(p[0] * p[0] + p[1] * p[1]);
            if (rr > maxR) {
                maxR = rr;
            }
        }
        var spanZ = maxZ - minZ;
        var spanXY = Math.max(maxX - minX, maxY - minY, 1e-6);
        var planar = !n || spanZ <= 0.08 * spanXY;
        placed = [];
        for (i = 0; i < n; i++) {
            var q = raw[i];
            if (planar) {
                placed.push(polarToCns(q[0], q[1], i, n, maxR));
            } else {
                var nx = spanXY ? (q[0] - minX) / spanXY : 0.5;
                var ny = spanXY ? (q[1] - minY) / spanXY : 0.5;
                var nz = spanZ > 1e-8 ? (q[2] - minZ) / spanZ : 0.5;
                placed.push([
                    lerp(-1.55, 1.55, clamp(nx, 0, 1)),
                    lerp(-1.95, 0.72, clamp(ny, 0, 1)),
                    lerp(-0.42, 0.42, clamp(nz, 0, 1))
                ]);
            }
        }
    }

    function placeAnatomyNodes() {
        var n = nNodes();
        placed = [];
        for (var i = 0; i < n; i++) {
            var id = nodes[i] != null ? String(nodes[i]) : String(i);
            var p = positions[id];
            if (p && p.length >= 2) {
                placed.push([Number(p[0]) || 0, Number(p[1]) || 0, Number(p[2]) || 0]);
            } else {
                placed.push(null);
            }
        }
    }

    function placeReservoirNodes() {
        if (hullMode() === "schematic_cns") {
            fitNodesIntoHull();
            return;
        }
        placeAnatomyNodes();
    }

    function nodeXYZ(i) {
        if (placed && i >= 0 && i < placed.length) {
            return placed[i] || null;
        }
        if (isAnatomy()) {
            return null;
        }
        return rawXYZ(i);
    }

    function accBounds(p, b) {
        if (!p) {
            return;
        }
        var x = Number(p[0]) || 0;
        var y = Number(p[1]) || 0;
        var z = Number(p[2]) || 0;
        b.minX = Math.min(b.minX, x);
        b.minY = Math.min(b.minY, y);
        b.minZ = Math.min(b.minZ, z);
        b.maxX = Math.max(b.maxX, x);
        b.maxY = Math.max(b.maxY, y);
        b.maxZ = Math.max(b.maxZ, z);
        b.any = true;
    }

    function computeSceneBounds() {
        var b = {
            minX: Infinity,
            minY: Infinity,
            minZ: Infinity,
            maxX: -Infinity,
            maxY: -Infinity,
            maxZ: -Infinity,
            any: false
        };
        var i;
        if (hullMode() === "schematic_cns") {
            accBounds([-1.7, 0.82, 0.44], b);
            accBounds([1.7, 0.82, 0.44], b);
            accBounds([0, -2.38, 0.22], b);
            accBounds([0, 0.74, 0.15], b);
            var n = nNodes();
            for (i = 0; i < n; i++) {
                accBounds(nodeXYZ(i), b);
            }
        } else if (flags.full_cns && focusBrain && flags.brain_bounds) {
            accBounds(flags.brain_bounds[0], b);
            accBounds(flags.brain_bounds[1], b);
        } else {
            var nA = nNodes();
            for (i = 0; i < nA; i++) {
                accBounds(nodeXYZ(i), b);
            }
            for (i = 0; i < contextPositions.length; i++) {
                accBounds(asXYZ(contextPositions[i]), b);
            }
            for (i = 0; i < hullPolyline.length; i++) {
                accBounds(asXYZ(hullPolyline[i]), b);
            }
        }
        if (!b.any) {
            return {
                minX: -1,
                minY: -1,
                minZ: -1,
                maxX: 1,
                maxY: 1,
                maxZ: 1,
                center: [0, 0, 0],
                hx: 1,
                hy: 1,
                hz: 1,
                radius: 1.6
            };
        }
        var cx = (b.minX + b.maxX) * 0.5;
        var cy = (b.minY + b.maxY) * 0.5;
        var cz = (b.minZ + b.maxZ) * 0.5;
        var hx = (b.maxX - b.minX) * 0.5;
        var hy = (b.maxY - b.minY) * 0.5;
        var hz = (b.maxZ - b.minZ) * 0.5;
        var radius = Math.sqrt(hx * hx + hy * hy + hz * hz);
        if (!isFinite(radius) || radius < 1e-4) {
            radius = 0.35;
        }
        return {
            minX: b.minX,
            minY: b.minY,
            minZ: b.minZ,
            maxX: b.maxX,
            maxY: b.maxY,
            maxZ: b.maxZ,
            center: [cx, cy, cz],
            hx: Math.max(hx, 1e-4),
            hy: Math.max(hy, 1e-4),
            hz: Math.max(hz, 1e-4),
            radius: radius
        };
    }

    function sceneBounds() {
        if (!cachedBounds) {
            cachedBounds = computeSceneBounds();
        }
        return cachedBounds;
    }

    function sceneLookAt() {
        if (hullMode() === "schematic_cns") {
            return [0, -0.35, 0];
        }
        return sceneBounds().center;
    }

    function extentRadius() {
        return sceneBounds().radius;
    }

    function camDistLimits() {
        if (isAnatomy()) {
            return { min: 0.028, max: 2.6 };
        }
        return { min: 0.28, max: 4.2 };
    }

    function anatomyAxes() {
        var b = sceneBounds();
        if (flags.full_cns) {
            return {thin: [0, -1, 0], tall: [0, 0, -1], right: [1, 0, 0], hThin: b.hy, hMid: b.hx, hTall: b.hz};
        }
        var axes = [
            { u: [1, 0, 0], h: b.hx },
            { u: [0, 1, 0], h: b.hy },
            { u: [0, 0, 1], h: b.hz }
        ];
        axes.sort(function (a, c) {
            return a.h - c.h;
        });
        var thin = axes[0].u;
        var tall = axes[2].u;
        var mean = 0;
        var n = 0;
        var i;
        for (i = 0; i < contextPositions.length; i++) {
            var p = asXYZ(contextPositions[i]);
            mean += p[0] * tall[0] + p[1] * tall[1] + p[2] * tall[2];
            n += 1;
        }
        if (n && mean < 0) {
            tall = [-tall[0], -tall[1], -tall[2]];
        }
        var right = [
            tall[1] * thin[2] - tall[2] * thin[1],
            tall[2] * thin[0] - tall[0] * thin[2],
            tall[0] * thin[1] - tall[1] * thin[0]
        ];
        var rl = Math.sqrt(right[0] * right[0] + right[1] * right[1] + right[2] * right[2]) || 1;
        right[0] /= rl;
        right[1] /= rl;
        right[2] /= rl;
        return { thin: thin, tall: tall, right: right, hThin: axes[0].h, hMid: axes[1].h, hTall: axes[2].h };
    }

    function fittedViewDistance(aspect) {
        var b = sceneBounds();
        var fovDeg = isAnatomy() ? FOV_ANATOMY : FOV_SCHEMATIC;
        // Leave room for the clock above and anatomy controls / legend below.
        var fill = flags.full_cns ? 0.64 : (isAnatomy() ? 0.84 : 0.7);
        var fovY = (fovDeg * Math.PI) / 180;
        var asp = Math.max(Number(aspect) || 1, 0.2);
        var halfY = Math.tan(fovY * 0.5);
        var halfX = halfY * asp;
        if (isAnatomy()) {
            var ax = anatomyAxes();
            var distY = ax.hTall / (halfY * fill);
            var distX = ax.hMid / (halfX * fill);
            return Math.max(distX, distY, 0.04) + ax.hThin * 0.22;
        }
        var distBox = Math.max(b.hy / (halfY * fill), b.hx / (halfX * fill));
        return Math.max(distBox, 0.04);
    }

    function cameraPose(aspect) {
        var dist = camDist * fittedViewDistance(aspect);
        dist = Math.max(dist, 0.02);
        var fov = isAnatomy() ? FOV_ANATOMY : FOV_SCHEMATIC;
        if (hullMode() === "schematic_cns") {
            return {
                eye: [
                    dist * Math.cos(rotY) * Math.cos(rotX),
                    dist * Math.sin(rotX) + 0.15,
                    dist * Math.sin(rotY) * Math.cos(rotX)
                ],
                target: [0, -0.35, 0],
                up: [0, 1, 0],
                dist: dist,
                fov: fov
            };
        }
        var target = sceneLookAt();
        var ax = anatomyAxes();
        if (flags.full_cns) {
            target = target.map(function (value, axis) {
                return value - ax.tall[axis] * ax.hTall * 0.15;
            });
        }
        var cy = Math.cos(rotY);
        var sy = Math.sin(rotY);
        var cx = Math.cos(rotX);
        var sx = Math.sin(rotX);
        var dx = ax.thin[0] * cy * cx + ax.right[0] * sy * cx + ax.tall[0] * sx;
        var dy = ax.thin[1] * cy * cx + ax.right[1] * sy * cx + ax.tall[1] * sx;
        var dz = ax.thin[2] * cy * cx + ax.right[2] * sy * cx + ax.tall[2] * sx;
        return {
            eye: [target[0] + dist * dx, target[1] + dist * dy, target[2] + dist * dz],
            target: target,
            up: ax.tall,
            dist: dist,
            fov: fov
        };
    }

    function selectDrawnEdges() {
        var out = [];
        if (!edges || !edges.length || !nodes.length) {
            return out;
        }
        if (isAnatomy() && flags.show_connections !== true) {
            return out;
        }
        var indexOf = {};
        var k;
        for (k = 0; k < nodes.length; k++) {
            indexOf[String(nodes[k])] = k;
        }
        var cap = isAnatomy() ? (flags.full_cns ? Number(flags.edge_cap || 12000) : ANATOMY_EDGE_CAP) : edges.length;
        var step = edges.length > cap ? Math.ceil(edges.length / cap) : 1;
        var e;
        for (e = 0; e < edges.length; e += step) {
            var src = indexOf[String(edges[e].src)];
            var dst = indexOf[String(edges[e].dst)];
            if (src == null || dst == null) {
                continue;
            }
            if (isAnatomy() && (!nodeXYZ(src) || !nodeXYZ(dst))) {
                continue;
            }
            out.push(src, dst);
            if (out.length / 2 >= cap) {
                break;
            }
        }
        return out;
    }

    function maybeResetCameraFit() {
        var key = hullMode() + "|" + nNodes() + "|" + contextPositions.length;
        if (key !== lastFitKey) {
            lastFitKey = key;
            camDist = 1;
            if (isAnatomy()) {
                rotY = 0.12;
                rotX = 0.06;
            } else {
                rotY = 0.72;
                rotX = 0.28;
            }
        }
        cachedBounds = null;
    }

    function predictedRulValue() {
        if (predictedRul == null) {
            if (flags && flags.stored_predicted_rul_s != null) {
                return Number(flags.stored_predicted_rul_s);
            }
            return null;
        }
        if (typeof predictedRul === "number") {
            return predictedRul;
        }
        if (predictedRul.length) {
            var idx = clamp(Math.floor(frameIndex), 0, predictedRul.length - 1);
            return Number(predictedRul[idx]);
        }
        return Number(predictedRul);
    }

    function currentTimestamp() {
        if (flags.synchronized && flags.now_timestamp_s != null) {
            return Number(flags.now_timestamp_s);
        }
        if (!frameMap || !frameMap.length) {
            return null;
        }
        var idx = clamp(Math.floor(frameIndex), 0, frameMap.length - 1);
        var ts = frameMap[idx].timestamp_s;
        return ts == null ? null : Number(ts);
    }

    function applyArgs(args) {
        needsRender = true;
        args = args || {};
        var nextFlags = args.flags || {};
        var nextInputs = args.inputs || [];
        var geometryMode = String(nextFlags.hull_mode || "schematic_cns") + "|" +
            Boolean(nextFlags.show_connections) + "|" + Boolean(nextFlags.show_anatomy_envelope) + "|" +
            (nextInputs[0] ? nextInputs[0].length : 0) + "|" + Boolean(nextFlags.show_morphology) + "|" +
            ((args.morphology || {}).version || "");
        var rebuild = geometryMode !== lastGeometryMode || !sameSceneGeometry(args);
        nodes = args.nodes || [];
        edges = args.edges || [];
        positions = args.positions || {};
        states = args.states_b64 ? [decodeBuffer(args.states_b64, Float32Array)] : args.states || [];
        inputs = args.inputs || [];
        frameMap = args.frame_map || [];
        flags = args.flags || {};
        if ((args.morphology || {}).version !== morphology.version) {
            morphology = args.morphology || {};
            morphologyXYZ = morphology.positions_b64 ? decodeBuffer(morphology.positions_b64, Float32Array) : null;
            morphologyOwners = morphology.owners_b64 ? decodeBuffer(morphology.owners_b64, Int32Array) : null;
        }
        activityHistory = args.activity_history || [];
        activityHistoryTimestamps = args.activity_history_timestamps_s || [];
        drawActivityHistory();
        var viewButtons = document.getElementById("anatomy-views");
        if (viewButtons) viewButtons.style.display = flags.full_cns ? "flex" : "none";
        if (flags.synchronized) {
            playing = false;
            frameIndex = 0;
        }
        controls.style.display = flags.synchronized ? "none" : "flex";
        predictedRul = args.predicted_rul_s != null ? args.predicted_rul_s : null;
        contextPositions = args.context_positions || [];
        hullPolyline = args.hull_polyline || [];
        if (rebuild) {
            placeReservoirNodes();
            maybeResetCameraFit();
        }
        lastGeometryMode = geometryMode;
        canvas.style.height = canvasHeight() + "px";
        var bounds = frameBounds();
        if (modeName() === "Alert inspection") {
            frameIndex = bounds.start;
        } else {
            frameIndex = clamp(frameIndex, bounds.start, bounds.end);
        }
        if (frameSlider) {
            frameSlider.min = String(bounds.start);
            frameSlider.max = String(Math.max(bounds.end, bounds.start));
            frameSlider.value = String(Math.floor(frameIndex));
        }
        if (modeLabel) {
            modeLabel.textContent = modeName();
        }
        updateFrameLabel();
        if (renderer && renderer.rebuild) {
            if (rebuild) {
                renderer.rebuild();
            } else {
                renderer.peak = flags.activity_scale || maxAbs(states);
                renderer.inPeak = maxAbs(inputs);
                renderer.resize();
            }
        }
        syncFrameHeight();
    }

    function sameXYZ(a, b) {
        if (!a || !b) { return a === b; }
        return Number(a[0]) === Number(b[0]) && Number(a[1]) === Number(b[1]) &&
            Number(a[2] || 0) === Number(b[2] || 0);
    }

    function sameXYZList(a, b) {
        if (a.length !== b.length) { return false; }
        for (var i = 0; i < a.length; i++) {
            if (!sameXYZ(a[i], b[i])) { return false; }
        }
        return true;
    }

    function sameSceneGeometry(args) {
        // Streamlit sends fresh objects for every observation. Compare coordinates
        // without allocating a JSON string or rebuilding 40k context GPU buffers.
        var nextNodes = args.nodes || [];
        var nextPositions = args.positions || {};
        var nextState = args.states && args.states[0];
        if (nextNodes.length !== nodes.length || (nextNodes.length || (nextState ? nextState.length : 0)) !== nNodes()) {
            return false;
        }
        for (var i = 0; i < nodes.length; i++) {
            var id = String(nodes[i]);
            if (id !== String(nextNodes[i]) || !sameXYZ(positions[id], nextPositions[id])) { return false; }
        }
        if (!sameXYZList(contextPositions, args.context_positions || []) ||
            !sameXYZList(hullPolyline, args.hull_polyline || [])) { return false; }
        // Hidden anatomical connections have no geometry, regardless of count.
        if (!isAnatomy() || flags.show_connections === true) {
            var nextEdges = args.edges || [];
            if (nextEdges.length !== edges.length) { return false; }
            for (var j = 0; j < edges.length; j++) {
                if (String(edges[j].src) !== String(nextEdges[j].src) ||
                    String(edges[j].dst) !== String(nextEdges[j].dst)) { return false; }
            }
        }
        return true;
    }

    function updateFrameLabel() {
        if (frameLabel) {
            frameLabel.textContent = "frame " + Math.floor(frameIndex) + " / " + Math.max(nFrames() - 1, 0);
        }
        if (frameSlider && document.activeElement !== frameSlider) {
            frameSlider.value = String(Math.floor(frameIndex));
        }
    }

    function updateHud(values) {
        var peak = renderer && renderer.peak ? renderer.peak : 1;
        var vis = renderer && renderer.visIndex;
        var nHud = vis ? vis.length : 0;
        var ts = currentTimestamp();
        var rul = predictedRulValue();
        var bits = flags.synchronized ? [] : ["frame " + Math.floor(frameIndex)];
        var timeScale = Number(flags.time_scale) > 0 ? Number(flags.time_scale) : 60;
        var timeUnit = String(flags.time_unit || "min");
        if (ts != null && isFinite(ts)) {
            bits.push("Now " + (ts / timeScale).toFixed(1) + " " + timeUnit);
        }
        var interval = flags.failure_window_s;
        if (flags.synchronized && interval && interval.length === 2 &&
            interval[0] != null && interval[1] != null &&
            isFinite(interval[0]) && isFinite(interval[1]) && Number(interval[0]) <= Number(interval[1])) {
            bits.push("Failure window " +
                (Number(interval[0]) / timeScale).toFixed(1) + "–" +
                (Number(interval[1]) / timeScale).toFixed(1) + " " + timeUnit);
        }
        if (!flags.synchronized && rul != null && isFinite(rul)) {
            bits.push("predicted RUL " + rul.toFixed(3) + "s");
        }
        bits.push(flags.full_cns ? nNodes().toLocaleString() + " computing · " + nHud.toLocaleString() + " positioned" : nHud + " / " + nNodes() + " neurons");
        if (hudLine) {
            hudLine.textContent = bits.join("  ·  ");
        }
        if (hudBanner) {
            var parts = [];
            if (flags && flags.is_synthetic) {
                parts.push("Synthetic test graph — not a biological connectome");
            }
            if (flags && flags.phase === "training") {
                parts.push("Live training — train-split window only");
            }
            if (flags && flags.downsampled) {
                parts.push("Activity downsampled for display");
            }
            if (flags && flags.anatomy_missing) {
                parts.push("Anatomical soma coordinates are missing; showing a labeled schematic.");
            }
            if (!flags.synchronized && flags.context_caption) {
                parts.push(String(flags.context_caption));
            } else if (!flags.synchronized && flags.context_downsampled) {
                parts.push("Soma context downsampled for display");
            }
            hudBanner.textContent = parts.join("  ·  ");
        }
        if (hudHint) {
            hudHint.textContent = flags.synchronized
                ? (flags.continuous_history === false ? "Recorded history" : "Continuous history") + " · Drag to rotate · wheel to zoom"
                : isAnatomy()
                ? "Drag to rotate · wheel to zoom into the cloud · Play runs in this view"
                : "Drag to rotate · wheel to zoom · Play runs in this view";
        }
        if (hudLegend) {
            hudLegend.style.display = isAnatomy() ? "block" : "none";
            hudLegend.textContent = String(flags.signal_label || "State") + " · cyan: negative · amber: positive · brightness: magnitude" +
                " · scale ±" + Number(peak).toPrecision(3) +
                (contextPositions.length ? "\nDim gray: anatomical reference only" : "");
            if (flags.full_cns) {
                hudLegend.textContent = flags.color_mode === "Cell classes"
                    ? "Color: annotated cell class · central brain, optic lobes and nerve cord"
                    : "State · cyan: negative · amber: positive · scale ±1";
                if (flags.show_morphology && morphology.n_neurons) hudLegend.textContent += "\n" + morphology.n_neurons + " reconstructed neuron arbors · official MaleCNS SWC";
            }
        }
        // Read-only observability for browser verification of the actual draw set.
        canvas.dataset.drawnNeurons = String(vis ? vis.length : 0);
        canvas.dataset.modelNeurons = String(nNodes());
        canvas.dataset.contextCells = String(contextPositions.length);
        canvas.dataset.timestampS = ts == null ? "" : String(ts);
        canvas.dataset.geometryBuilds = String(renderer && renderer.rebuildCount || 0);
        canvas.dataset.recurrentEdges = String(flags.n_edges || edges.length);
        canvas.dataset.synapses = String(flags.n_synapses || 0);
        canvas.dataset.drawnConnections = String(renderer && renderer.edgeIndex ? renderer.edgeIndex.length / 2 : 0);
        canvas.dataset.morphologyNeurons = String(flags.show_morphology ? morphology.n_neurons || 0 : 0);
    }

    function syncFrameHeight() {
        if (window.Streamlit && window.Streamlit.setFrameHeight) {
            if (flags.synchronized) {
                window.Streamlit.setFrameHeight(canvasHeight() + (flags.full_cns ? 211 : 0));
                return;
            }
            var h = Math.max(document.body.scrollHeight, canvasHeight() + 88);
            window.Streamlit.setFrameHeight(isCompact() ? Math.max(h, 460) : Math.max(h, 700));
        }
    }

    function buildControls() {
        controls.innerHTML = "";
        playBtn = document.createElement("button");
        playBtn.type = "button";
        playBtn.textContent = "Play";
        pauseBtn = document.createElement("button");
        pauseBtn.type = "button";
        pauseBtn.textContent = "Pause";
        frameSlider = document.createElement("input");
        frameSlider.type = "range";
        frameSlider.min = "0";
        frameSlider.max = "0";
        frameSlider.value = "0";
        frameLabel = document.createElement("span");
        var speedWrap = document.createElement("label");
        speedWrap.appendChild(document.createTextNode("Speed "));
        speedSlider = document.createElement("input");
        speedSlider.type = "range";
        speedSlider.min = "1";
        speedSlider.max = "10";
        speedSlider.value = String(speed);
        speedWrap.appendChild(speedSlider);
        modeLabel = document.createElement("span");
        controls.appendChild(playBtn);
        controls.appendChild(pauseBtn);
        controls.appendChild(frameSlider);
        controls.appendChild(frameLabel);
        controls.appendChild(speedWrap);
        controls.appendChild(modeLabel);

        playBtn.addEventListener("click", function () {
            playing = true;
            lastTs = 0;
        });
        pauseBtn.addEventListener("click", function () {
            playing = false;
        });
        frameSlider.addEventListener("input", function () {
            playing = false;
            needsRender = true;
            frameIndex = Number(frameSlider.value) || 0;
            updateFrameLabel();
        });
        speedSlider.addEventListener("input", function () {
            speed = Number(speedSlider.value) || 1;
        });
    }

    function fibonacciFill(n, rx, ry, rz, ox, oy, oz, out, start) {
        var golden = Math.PI * (3 - Math.sqrt(5));
        for (var i = 0; i < n; i++) {
            var y = 1 - (i / Math.max(n - 1, 1)) * 2;
            var rad = Math.sqrt(Math.max(0, 1 - y * y));
            var theta = golden * i;
            var k = (start + i) * 3;
            out[k] = ox + Math.cos(theta) * rad * rx;
            out[k + 1] = oy + y * ry;
            out[k + 2] = oz + Math.sin(theta) * rad * rz;
        }
        return start + n;
    }

    function buildHullPositions() {
        var nLobe = 280;
        var nBrain = 420;
        var nVnc = 480;
        var total = nLobe * 2 + nBrain + nVnc;
        var out = new Float32Array(total * 3);
        var s = 0;
        s = fibonacciFill(nLobe, 0.52, 0.5, 0.42, -1.18, 0.16, 0, out, s);
        s = fibonacciFill(nLobe, 0.52, 0.5, 0.42, 1.18, 0.16, 0, out, s);
        s = fibonacciFill(nBrain, 0.72, 0.5, 0.44, 0, 0.08, 0, out, s);
        var golden = Math.PI * (3 - Math.sqrt(5));
        for (var i = 0; i < nVnc; i++) {
            var t = i / Math.max(nVnc - 1, 1);
            var y = -0.42 - 1.58 * t;
            var rad = 0.3 * (1 - 0.58 * t);
            var theta = golden * i;
            var k = (s + i) * 3;
            out[k] = Math.cos(theta) * rad;
            out[k + 1] = y;
            out[k + 2] = Math.sin(theta) * rad * 0.62;
        }
        return out;
    }

    function buildHullWire() {
        var rings = [];
        function ring(cx, cy, cz, rx, rz, n) {
            var pts = [];
            for (var i = 0; i < n; i++) {
                var a = (2 * Math.PI * i) / n;
                pts.push([cx + Math.cos(a) * rx, cy, cz + Math.sin(a) * rz]);
            }
            for (var j = 0; j < n; j++) {
                var a0 = pts[j];
                var a1 = pts[(j + 1) % n];
                rings.push(a0[0], a0[1], a0[2], a1[0], a1[1], a1[2]);
            }
        }
        ring(-1.18, 0.16, 0, 0.52, 0.42, 48);
        ring(1.18, 0.16, 0, 0.52, 0.42, 48);
        ring(0, 0.08, 0, 0.72, 0.44, 56);
        ring(0, -0.9, 0, 0.22, 0.14, 32);
        ring(0, -1.6, 0, 0.16, 0.1, 28);
        rings.push(-1.18, 0.16, 0, -0.55, 0.12, 0);
        rings.push(1.18, 0.16, 0, 0.55, 0.12, 0);
        rings.push(0, -0.38, 0, 0, -2.02, 0);
        return new Float32Array(rings);
    }

    function sensorXYZ(i, n) {
        var u = n <= 1 ? 0.5 : i / (n - 1);
        return [lerp(-0.72, 0.72, u), -2.12, 0.22];
    }

    function makeLabelSprite(text, x, y, z) {
        var c = document.createElement("canvas");
        c.width = 256;
        c.height = 64;
        var ctx = c.getContext("2d");
        ctx.clearRect(0, 0, 256, 64);
        ctx.font = "22px sans-serif";
        ctx.fillStyle = "rgba(170, 196, 214, 0.72)";
        ctx.textAlign = "center";
        ctx.fillText(text, 128, 40);
        var tex = new THREE.CanvasTexture(c);
        var mat = new THREE.SpriteMaterial({
            map: tex,
            transparent: true,
            depthWrite: false,
            opacity: 0.7
        });
        var spr = new THREE.Sprite(mat);
        spr.position.set(x, y, z);
        spr.scale.set(1.15, 0.28, 1);
        return spr;
    }

    function hullLineVerts() {
        var pts = hullPolyline || [];
        if (pts.length < 2) {
            return new Float32Array(0);
        }
        var out = [];
        var i;
        for (i = 0; i < pts.length; i++) {
            var a = asXYZ(pts[i]);
            var b = asXYZ(pts[(i + 1) % pts.length]);
            out.push(a[0], a[1], a[2], b[0], b[1], b[2]);
        }
        return new Float32Array(out);
    }

    function pointShader(sizeMul, additive) {
        var falloff = additive ? "7.2" : "5.2";
        return new THREE.ShaderMaterial({
            uniforms: { uMul: { value: sizeMul } },
            vertexShader:
                "attribute float aSize; attribute vec3 color; varying vec3 vCol;" +
                "uniform float uMul;" +
                "void main(){ vCol=color; vec4 mv=modelViewMatrix*vec4(position,1.0);" +
                "gl_Position=projectionMatrix*mv;" +
                "gl_PointSize=clamp(aSize*uMul*(36.0/max(-mv.z,0.4)), " + (flags.full_cns ? "0.8, 4.0" : "2.2, 13.0") + "); }",
            fragmentShader:
                "varying vec3 vCol; void main(){ vec2 p=gl_PointCoord*2.0-1.0; float d=dot(p,p);" +
                "if(d>1.0) discard; if(dot(vCol,vCol)<0.00015) discard; float a=exp(-d*" +
                falloff +
                "); gl_FragColor=vec4(vCol,a); }",
            transparent: true,
            blending: additive ? THREE.AdditiveBlending : THREE.NormalBlending,
            depthWrite: false
        });
    }

    function contextDustMaterial(sizeMul) {
        return new THREE.ShaderMaterial({
            uniforms: { uMul: { value: sizeMul } },
            vertexShader:
                "attribute float aSize; attribute vec3 color; varying vec3 vCol;" +
                "uniform float uMul;" +
                "void main(){ vCol=color; vec4 mv=modelViewMatrix*vec4(position,1.0);" +
                "gl_Position=projectionMatrix*mv;" +
                "gl_PointSize=clamp(aSize*uMul*(36.0/max(-mv.z,0.4)), 1.0, 1.8); }",
            fragmentShader:
                "varying vec3 vCol; void main(){ vec2 p=gl_PointCoord*2.0-1.0; float d=dot(p,p);" +
                "if(d>1.0) discard; float a=exp(-d*3.5)*0.30; gl_FragColor=vec4(vCol,a); }",
            transparent: true,
            blending: THREE.NormalBlending,
            depthWrite: false
        });
    }

    function ThreeRenderer() {
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0x0b131d);
        this.camera = new THREE.PerspectiveCamera(FOV_SCHEMATIC, 1, 0.002, 40);
        this.gl = new THREE.WebGLRenderer({ canvas: canvas, antialias: true, alpha: false });
        this.gl.setClearColor(0x0b131d, 1);
        this.gl.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
        this.core = null;
        this.idle = null;
        this.halo = null;
        this.colorAttr = null;
        this.idleColorAttr = null;
        this.sizeAttr = null;
        this.idleSizeAttr = null;
        this.haloSizeAttr = null;
        this.lineColor = null;
        this.sensorColor = null;
        this.nSensor = 0;
        this.peak = 1;
        this.inPeak = 1;
        this.rebuild();
        this.resize();
    }

    ThreeRenderer.prototype.rebuild = function () {
        this.rebuildCount = (this.rebuildCount || 0) + 1;
        while (this.scene.children.length) {
            var child = this.scene.children[0];
            if (child.geometry && child.geometry.dispose) { child.geometry.dispose(); }
            if (child.material) {
                if (child.material.map && child.material.map.dispose) { child.material.map.dispose(); }
                if (child.material.dispose) { child.material.dispose(); }
            }
            this.scene.remove(child);
        }
        placeReservoirNodes();
        cachedBounds = null;
        this.visIndex = [];
        var visPos = [];
        var nAll = nNodes();
        var i;
        for (i = 0; i < nAll; i++) {
            var p = nodeXYZ(i);
            if (!p) {
                continue;
            }
            this.visIndex.push(i);
            visPos.push(p[0], p[1], p[2]);
        }
        var nv = this.visIndex.length;
        var pos = nv ? new Float32Array(visPos) : new Float32Array(3);
        var col = new Float32Array(Math.max(nv, 1) * 3);
        var idleCol = new Float32Array(Math.max(nv, 1) * 3);
        var sz = new Float32Array(Math.max(nv, 1));
        var idleSz = new Float32Array(Math.max(nv, 1));
        var idleRgb = isAnatomy() ? [0.22, 0.26, 0.30] : [0.055, 0.08, 0.11];
        for (i = 0; i < nv; i++) {
            idleCol[i * 3] = idleRgb[0];
            idleCol[i * 3 + 1] = idleRgb[1];
            idleCol[i * 3 + 2] = idleRgb[2];
            sz[i] = 0;
            idleSz[i] = isAnatomy() ? 0.9 : 1.25;
        }
        var posAttr = new THREE.BufferAttribute(pos, 3);
        this.colorAttr = new THREE.BufferAttribute(col, 3);
        this.idleColorAttr = new THREE.BufferAttribute(idleCol, 3);
        this.sizeAttr = new THREE.BufferAttribute(sz, 1);
        this.idleSizeAttr = new THREE.BufferAttribute(idleSz, 1);
        this.haloSizeAttr = null;
        this.halo = null;
        var idleGeom = new THREE.BufferGeometry();
        idleGeom.setAttribute("position", posAttr);
        idleGeom.setAttribute("color", this.idleColorAttr);
        idleGeom.setAttribute("aSize", this.idleSizeAttr);
        var geom = new THREE.BufferGeometry();
        geom.setAttribute("position", posAttr);
        geom.setAttribute("color", this.colorAttr);
        geom.setAttribute("aSize", this.sizeAttr);
        this.idle = new THREE.Points(idleGeom, pointShader(1.0, false));
        // Normal blending preserves signed colours where anatomical cells overlap.
        // Additive emission would turn dense groups into a white patch.
        this.core = new THREE.Points(geom, pointShader(1.0, !isAnatomy()));
        if (!isAnatomy()) {
            this.haloSizeAttr = new THREE.BufferAttribute(new Float32Array(Math.max(nv, 1)), 1);
            var haloGeom = new THREE.BufferGeometry();
            haloGeom.setAttribute("position", posAttr);
            haloGeom.setAttribute("color", this.colorAttr);
            haloGeom.setAttribute("aSize", this.haloSizeAttr);
            this.halo = new THREE.Points(haloGeom, pointShader(1.4, true));
        }
        if (nv) {
            this.scene.add(this.idle);
        }

        if (hullMode() === "schematic_cns") {
            var hullPos = buildHullPositions();
            var hg = new THREE.BufferGeometry();
            hg.setAttribute("position", new THREE.BufferAttribute(hullPos, 3));
            var hullMat = new THREE.PointsMaterial({
                color: 0x3a6a88,
                size: 0.022,
                sizeAttenuation: true,
                transparent: true,
                opacity: 0.55,
                depthWrite: false,
                blending: THREE.NormalBlending
            });
            this.scene.add(new THREE.Points(hg, hullMat));
            var wire = buildHullWire();
            var wg = new THREE.BufferGeometry();
            wg.setAttribute("position", new THREE.BufferAttribute(wire, 3));
            this.scene.add(
                new THREE.LineSegments(
                    wg,
                    new THREE.LineBasicMaterial({
                        color: 0x244058,
                        transparent: true,
                        opacity: 0.2,
                        blending: THREE.NormalBlending
                    })
                )
            );
            this.scene.add(makeLabelSprite("optic lobe", -1.18, 0.82, 0.2));
            this.scene.add(makeLabelSprite("optic lobe", 1.18, 0.82, 0.2));
            this.scene.add(makeLabelSprite("central brain", 0, 0.74, 0.15));
            this.scene.add(makeLabelSprite("VNC", 0.55, -1.35, 0.2));
        } else if (isAnatomy()) {
            var nCtx = contextPositions.length;
            if (nCtx) {
                var cpos = new Float32Array(nCtx * 3);
                var ccol = new Float32Array(nCtx * 3);
                var csz = new Float32Array(nCtx);
                var cScale = contextSizeScale(nCtx);
                for (i = 0; i < nCtx; i++) {
                    var cp = asXYZ(contextPositions[i]);
                    cpos[i * 3] = cp[0];
                    cpos[i * 3 + 1] = cp[1];
                    cpos[i * 3 + 2] = cp[2];
                    ccol[i * 3] = 0.31;
                    ccol[i * 3 + 1] = 0.43;
                    ccol[i * 3 + 2] = 0.54;
                    csz[i] = 3.1 * cScale;
                }
                var cg = new THREE.BufferGeometry();
                cg.setAttribute("position", new THREE.BufferAttribute(cpos, 3));
                cg.setAttribute("color", new THREE.BufferAttribute(ccol, 3));
                cg.setAttribute("aSize", new THREE.BufferAttribute(csz, 1));
                this.scene.add(new THREE.Points(cg, contextDustMaterial(0.85)));
            }
            // The soma cloud supplies the real silhouette. Its projected convex
            // hull is only a bounding envelope and is not a brain surface.
            var hullVerts = flags.show_anatomy_envelope ? hullLineVerts() : [];
            if (hullVerts.length) {
                var hlg = new THREE.BufferGeometry();
                hlg.setAttribute("position", new THREE.BufferAttribute(hullVerts, 3));
                this.scene.add(
                    new THREE.LineSegments(
                        hlg,
                        new THREE.LineBasicMaterial({
                            color: 0x8aa4b8,
                            transparent: true,
                            opacity: 0.1,
                            blending: THREE.NormalBlending,
                            depthWrite: false
                        })
                    )
                );
            }
        }

        this.edgeIndex = [];
        this.arborColor = null;
        if (flags.show_morphology && morphologyXYZ && morphologyOwners) {
            var arbor = new THREE.BufferGeometry();
            arbor.setAttribute("position", new THREE.BufferAttribute(morphologyXYZ, 3));
            this.arborColor = new THREE.BufferAttribute(new Float32Array(morphologyXYZ.length), 3);
            arbor.setAttribute("color", this.arborColor);
            this.scene.add(new THREE.LineSegments(arbor, new THREE.LineBasicMaterial({
                vertexColors: true, transparent: true, opacity: 0.60, depthWrite: false,
                blending: THREE.NormalBlending
            })));
        }
        this.lineColor = null;
        var drawn = selectDrawnEdges();
        if (drawn.length) {
            var linePos = [];
            var lineCol = [];
            for (var e = 0; e < drawn.length; e += 2) {
                var a = nodeXYZ(drawn[e]);
                var b = nodeXYZ(drawn[e + 1]);
                if (!a || !b) {
                    continue;
                }
                linePos.push(a[0], a[1], a[2], b[0], b[1], b[2]);
                lineCol.push(0.1, 0.14, 0.18, 0.1, 0.14, 0.18);
                this.edgeIndex.push(drawn[e], drawn[e + 1]);
            }
            if (linePos.length) {
                var lg = new THREE.BufferGeometry();
                lg.setAttribute("position", new THREE.BufferAttribute(new Float32Array(linePos), 3));
                this.lineColor = new THREE.BufferAttribute(new Float32Array(lineCol), 3);
                lg.setAttribute("color", this.lineColor);
                this.scene.add(
                    new THREE.LineSegments(
                        lg,
                        new THREE.LineBasicMaterial({
                            vertexColors: true,
                            transparent: true,
                            opacity: isAnatomy() ? 0.12 : 0.28,
                            blending: THREE.NormalBlending,
                            depthWrite: false
                        })
                    )
                );
            }
        }

        if (nv) {
            if (this.halo && !isAnatomy()) {
                this.scene.add(this.halo);
            }
            this.scene.add(this.core);
        }

        var nf = nFeatures();
        this.nSensor = 0;
        this.sensorColor = null;
        // Hide sensor strip when malecns_anatomy.
        if (hullMode() === "schematic_cns" && nf) {
            this.nSensor = nf;
            var sp = new Float32Array(nf * 3);
            var sc = new Float32Array(nf * 3);
            var ss = new Float32Array(nf);
            for (i = 0; i < nf; i++) {
                var s = sensorXYZ(i, nf);
                sp[i * 3] = s[0];
                sp[i * 3 + 1] = s[1];
                sp[i * 3 + 2] = s[2];
                ss[i] = 8;
            }
            var sg = new THREE.BufferGeometry();
            sg.setAttribute("position", new THREE.BufferAttribute(sp, 3));
            this.sensorColor = new THREE.BufferAttribute(sc, 3);
            sg.setAttribute("color", this.sensorColor);
            sg.setAttribute("aSize", new THREE.BufferAttribute(ss, 1));
            this.scene.add(new THREE.Points(sg, pointShader(1.15, false)));
            this.scene.add(makeLabelSprite("sensor window", 0, -2.38, 0.2));
        }

        this.peak = flags.activity_scale || maxAbs(states);
        this.inPeak = maxAbs(inputs);
        this.resize();
    };

    ThreeRenderer.prototype.resize = function () {
        var w = canvas.clientWidth || 640;
        var h = canvas.clientHeight || canvasHeight();
        this.gl.setSize(w, h, false);
        this.camera.aspect = w / Math.max(h, 1);
        this.camera.updateProjectionMatrix();
    };

    ThreeRenderer.prototype.render = function (values, inRow) {
        var vis = this.visIndex || [];
        var nv = vis.length;
        var i;
        var anatomy = isAnatomy();
        if (this.colorAttr && nv) {
            var nScale = nodeSizeScale();
            for (i = 0; i < nv; i++) {
                var ni = vis[i];
                var c = cellColor(ni, values[ni] || 0, this.peak);
                var active = anatomy || reservoirActive(values[ni] || 0, this.peak);
                this.colorAttr.array[i * 3] = active ? c.r : 0;
                this.colorAttr.array[i * 3 + 1] = active ? c.g : 0;
                this.colorAttr.array[i * 3 + 2] = active ? c.b : 0;
                if (this.sizeAttr) {
                    this.sizeAttr.array[i] = active
                        ? (anatomy ? lerp(0.9, 1.85, c.t) : lerp(1.45, 4.2, c.t)) * nScale
                        : 0;
                }
                if (this.idleSizeAttr) {
                    this.idleSizeAttr.array[i] = active ? 0 : (anatomy ? 0.82 : 1.1) * nScale;
                }
                if (this.idleColorAttr) {
                    this.idleColorAttr.array[i * 3] = anatomy ? 0.22 : 0.055;
                    this.idleColorAttr.array[i * 3 + 1] = anatomy ? 0.26 : 0.08;
                    this.idleColorAttr.array[i * 3 + 2] = anatomy ? 0.30 : 0.11;
                }
                if (this.haloSizeAttr) {
                    this.haloSizeAttr.array[i] = active
                        ? (anatomy ? lerp(0.95, 1.85, c.t) : lerp(1.5, 4.6, c.t)) * nScale
                        : 0;
                }
            }
            this.colorAttr.needsUpdate = true;
            if (this.sizeAttr) {
                this.sizeAttr.needsUpdate = true;
            }
            if (this.idleSizeAttr) {
                this.idleSizeAttr.needsUpdate = true;
            }
            if (this.idleColorAttr) {
                this.idleColorAttr.needsUpdate = true;
            }
            if (this.haloSizeAttr) {
                this.haloSizeAttr.needsUpdate = true;
            }
        }
        if (this.lineColor && this.edgeIndex && this.edgeIndex.length) {
            var thr = ACTIVITY_T * this.peak;
            for (i = 0; i < this.edgeIndex.length; i += 2) {
                var sa = Math.abs(Number(values[this.edgeIndex[i]] || 0));
                var sb = Math.abs(Number(values[this.edgeIndex[i + 1]] || 0));
                var hot = sa >= thr && sb >= thr;
                var r = hot ? (anatomy ? 0.42 : 0.72) : anatomy ? 0.1 : 0.14;
                var g = hot ? (anatomy ? 0.48 : 0.7) : anatomy ? 0.14 : 0.2;
                var b = hot ? (anatomy ? 0.52 : 0.48) : anatomy ? 0.18 : 0.28;
                var a = i * 3;
                this.lineColor.array[a] = r;
                this.lineColor.array[a + 1] = g;
                this.lineColor.array[a + 2] = b;
                this.lineColor.array[a + 3] = r;
                this.lineColor.array[a + 4] = g;
                this.lineColor.array[a + 5] = b;
            }
            this.lineColor.needsUpdate = true;
        }
        if (this.arborColor && morphologyOwners) {
            for (i = 0; i < morphologyOwners.length; i++) {
                var owner = morphologyOwners[i];
                var ac = cellColor(owner, values[owner] || 0, this.peak);
                var at = i * 6;
                this.arborColor.array[at] = this.arborColor.array[at + 3] = ac.r;
                this.arborColor.array[at + 1] = this.arborColor.array[at + 4] = ac.g;
                this.arborColor.array[at + 2] = this.arborColor.array[at + 5] = ac.b;
            }
            this.arborColor.needsUpdate = true;
        }
        if (this.sensorColor && inRow) {
            for (i = 0; i < this.nSensor; i++) {
                var ic = colorFromValue(inRow[i] || 0, this.inPeak);
                this.sensorColor.array[i * 3] = ic.r;
                this.sensorColor.array[i * 3 + 1] = ic.g;
                this.sensorColor.array[i * 3 + 2] = ic.b;
            }
            this.sensorColor.needsUpdate = true;
        }
        var pose = cameraPose(this.camera.aspect || 1);
        this.camera.fov = pose.fov;
        this.camera.near = clamp(pose.dist * 0.018, 0.0008, 0.08);
        this.camera.far = Math.max(pose.dist * 28, extentRadius() * 16, 10);
        this.camera.updateProjectionMatrix();
        this.camera.position.set(pose.eye[0], pose.eye[1], pose.eye[2]);
        this.camera.up.set(pose.up[0], pose.up[1], pose.up[2]);
        this.camera.lookAt(pose.target[0], pose.target[1], pose.target[2]);
        this.gl.render(this.scene, this.camera);
    };

    function WebGLRenderer() {
        this.gl = canvas.getContext("webgl") || canvas.getContext("experimental-webgl");
        this.program = null;
        this.lineProgram = null;
        this.buf = null;
        this.colorBuf = null;
        this.sizeBuf = null;
        this.lineBuf = null;
        this.lineColorBuf = null;
        this.hullBuf = null;
        this.hullLineBuf = null;
        this.contextBuf = null;
        this.idleSizeBuf = null;
        this.sensorBuf = null;
        this.sensorColorBuf = null;
        this.nPoints = 0;
        this.nLineVerts = 0;
        this.nHull = 0;
        this.nHullLine = 0;
        this.nContext = 0;
        this.nSensor = 0;
        this.peak = 1;
        this.inPeak = 1;
        this.edgeIndex = [];
        this.visIndex = [];
        if (!this.gl) {
            return;
        }
        var vs =
            "attribute vec3 aPos; attribute vec3 aCol; attribute float aSize;" +
            "uniform mat4 uMVP; uniform float uSizeMul; varying vec3 vCol;" +
            "void main(){ vCol=aCol; gl_Position=uMVP*vec4(aPos,1.0); gl_PointSize=aSize*uSizeMul; }";
        var fs =
            "precision mediump float; varying vec3 vCol;" +
            "void main(){ vec2 p=gl_PointCoord*2.0-1.0; float d=dot(p,p);" +
            "if(d>1.0) discard; if(dot(vCol,vCol)<0.00015) discard; gl_FragColor=vec4(vCol, exp(-d*5.4)); }";
        this.program = compileProgram(this.gl, vs, fs);
        var lineVs =
            "attribute vec3 aPos; attribute vec3 aCol;" +
            "uniform mat4 uMVP; varying vec3 vCol;" +
            "void main(){ vCol=aCol; gl_Position=uMVP*vec4(aPos,1.0); }";
        var lineFs =
            "precision mediump float; varying vec3 vCol; uniform float uAlpha;" +
            "void main(){ gl_FragColor=vec4(vCol, uAlpha); }";
        this.lineProgram = compileProgram(this.gl, lineVs, lineFs);
        this.buf = this.gl.createBuffer();
        this.colorBuf = this.gl.createBuffer();
        this.sizeBuf = this.gl.createBuffer();
        this.lineBuf = this.gl.createBuffer();
        this.lineColorBuf = this.gl.createBuffer();
        this.hullBuf = this.gl.createBuffer();
        this.hullLineBuf = this.gl.createBuffer();
        this.contextBuf = this.gl.createBuffer();
        this.idleSizeBuf = this.gl.createBuffer();
        this.sensorBuf = this.gl.createBuffer();
        this.sensorColorBuf = this.gl.createBuffer();
        this.rebuild();
    }

    function compileProgram(gl, vsSrc, fsSrc) {
        function sh(type, src) {
            var s = gl.createShader(type);
            gl.shaderSource(s, src);
            gl.compileShader(s);
            return s;
        }
        var p = gl.createProgram();
        gl.attachShader(p, sh(gl.VERTEX_SHADER, vsSrc));
        gl.attachShader(p, sh(gl.FRAGMENT_SHADER, fsSrc));
        gl.linkProgram(p);
        return p;
    }

    WebGLRenderer.prototype.rebuild = function () {
        if (!this.gl) {
            return;
        }
        this.rebuildCount = (this.rebuildCount || 0) + 1;
        placeReservoirNodes();
        cachedBounds = null;
        this.visIndex = [];
        var visPos = [];
        var nAll = nNodes();
        var i;
        for (i = 0; i < nAll; i++) {
            var p = nodeXYZ(i);
            if (!p) {
                continue;
            }
            this.visIndex.push(i);
            visPos.push(p[0], p[1], p[2]);
        }
        var n = this.visIndex.length;
        this.nPoints = n;
        var pos = n ? new Float32Array(visPos) : new Float32Array(0);
        this.gl.bindBuffer(this.gl.ARRAY_BUFFER, this.buf);
        this.gl.bufferData(this.gl.ARRAY_BUFFER, pos, this.gl.STATIC_DRAW);
        this.nHull = 0;
        this.nContext = 0;
        this.nHullLine = 0;
        if (hullMode() === "schematic_cns") {
            var hull = buildHullPositions();
            this.nHull = hull.length / 3;
            this.gl.bindBuffer(this.gl.ARRAY_BUFFER, this.hullBuf);
            this.gl.bufferData(this.gl.ARRAY_BUFFER, hull, this.gl.STATIC_DRAW);
        } else if (isAnatomy()) {
            var nCtx = contextPositions.length;
            if (nCtx) {
                var cpos = new Float32Array(nCtx * 3);
                for (i = 0; i < nCtx; i++) {
                    var cp = asXYZ(contextPositions[i]);
                    cpos[i * 3] = cp[0];
                    cpos[i * 3 + 1] = cp[1];
                    cpos[i * 3 + 2] = cp[2];
                }
                this.nContext = nCtx;
                this.gl.bindBuffer(this.gl.ARRAY_BUFFER, this.contextBuf);
                this.gl.bufferData(this.gl.ARRAY_BUFFER, cpos, this.gl.STATIC_DRAW);
            }
            var hullVerts = flags.show_anatomy_envelope ? hullLineVerts() : [];
            if (hullVerts.length) {
                this.nHullLine = hullVerts.length / 3;
                this.gl.bindBuffer(this.gl.ARRAY_BUFFER, this.hullLineBuf);
                this.gl.bufferData(this.gl.ARRAY_BUFFER, hullVerts, this.gl.STATIC_DRAW);
            }
        }
        var drawn = selectDrawnEdges();
        var linePos = [];
        this.edgeIndex = [];
        for (var e = 0; e < drawn.length; e += 2) {
            var a = nodeXYZ(drawn[e]);
            var b = nodeXYZ(drawn[e + 1]);
            if (!a || !b) {
                continue;
            }
            linePos.push(a[0], a[1], a[2], b[0], b[1], b[2]);
            this.edgeIndex.push(drawn[e], drawn[e + 1]);
        }
        this.nLineVerts = linePos.length / 3;
        this.gl.bindBuffer(this.gl.ARRAY_BUFFER, this.lineBuf);
        this.gl.bufferData(this.gl.ARRAY_BUFFER, new Float32Array(linePos), this.gl.STATIC_DRAW);
        var nf = nFeatures();
        this.nSensor = 0;
        // Hide sensor strip when malecns_anatomy.
        if (hullMode() === "schematic_cns" && nf) {
            this.nSensor = nf;
            var sp = new Float32Array(nf * 3);
            for (i = 0; i < nf; i++) {
                var s = sensorXYZ(i, nf);
                sp[i * 3] = s[0];
                sp[i * 3 + 1] = s[1];
                sp[i * 3 + 2] = s[2];
            }
            this.gl.bindBuffer(this.gl.ARRAY_BUFFER, this.sensorBuf);
            this.gl.bufferData(this.gl.ARRAY_BUFFER, sp, this.gl.STATIC_DRAW);
        }
        this.peak = flags.activity_scale || maxAbs(states);
        this.inPeak = maxAbs(inputs);
        this.resize();
    };

    WebGLRenderer.prototype.resize = function () {
        if (!this.gl) {
            return;
        }
        var w = canvas.clientWidth || 640;
        var h = canvas.clientHeight || canvasHeight();
        canvas.width = w;
        canvas.height = h;
        this.gl.viewport(0, 0, w, h);
        this._aspect = w / Math.max(h, 1);
    };

    function perspective(out, fov, aspect, near, far) {
        var f = 1.0 / Math.tan(fov / 2);
        out[0] = f / aspect;
        out[1] = 0;
        out[2] = 0;
        out[3] = 0;
        out[4] = 0;
        out[5] = f;
        out[6] = 0;
        out[7] = 0;
        out[8] = 0;
        out[9] = 0;
        out[10] = (far + near) / (near - far);
        out[11] = -1;
        out[12] = 0;
        out[13] = 0;
        out[14] = (2 * far * near) / (near - far);
        out[15] = 0;
    }

    function lookAt(out, eye, center, up) {
        var zx = eye[0] - center[0];
        var zy = eye[1] - center[1];
        var zz = eye[2] - center[2];
        var zl = Math.sqrt(zx * zx + zy * zy + zz * zz) || 1;
        zx /= zl;
        zy /= zl;
        zz /= zl;
        var ux = up && up.length ? Number(up[0]) || 0 : 0;
        var uy = up && up.length ? Number(up[1]) || 0 : 1;
        var uz = up && up.length ? Number(up[2]) || 0 : 0;
        if (ux === 0 && uy === 0 && uz === 0) {
            uy = 1;
        }
        var xx = uy * zz - uz * zy;
        var xy = uz * zx - ux * zz;
        var xz = ux * zy - uy * zx;
        var xl = Math.sqrt(xx * xx + xy * xy + xz * xz) || 1;
        xx /= xl;
        xy /= xl;
        xz /= xl;
        var yx = zy * xz - zz * xy;
        var yy = zz * xx - zx * xz;
        var yz = zx * xy - zy * xx;
        out[0] = xx;
        out[1] = yx;
        out[2] = zx;
        out[3] = 0;
        out[4] = xy;
        out[5] = yy;
        out[6] = zy;
        out[7] = 0;
        out[8] = xz;
        out[9] = yz;
        out[10] = zz;
        out[11] = 0;
        out[12] = -(xx * eye[0] + xy * eye[1] + xz * eye[2]);
        out[13] = -(yx * eye[0] + yy * eye[1] + yz * eye[2]);
        out[14] = -(zx * eye[0] + zy * eye[1] + zz * eye[2]);
        out[15] = 1;
    }

    function mul4(a, b) {
        var o = new Float32Array(16);
        for (var c = 0; c < 4; c++) {
            for (var r = 0; r < 4; r++) {
                o[c * 4 + r] =
                    a[0 * 4 + r] * b[c * 4 + 0] +
                    a[1 * 4 + r] * b[c * 4 + 1] +
                    a[2 * 4 + r] * b[c * 4 + 2] +
                    a[3 * 4 + r] * b[c * 4 + 3];
            }
        }
        return o;
    }

    WebGLRenderer.prototype.render = function (values, inRow) {
        var gl = this.gl;
        if (!gl || !this.program) {
            return;
        }
        var vis = this.visIndex || [];
        var n = this.nPoints;
        var col = new Float32Array(Math.max(n, 1) * 3);
        var sz = new Float32Array(Math.max(n, 1));
        var idleSz = new Float32Array(Math.max(n, 1));
        var i;
        var anatomy = isAnatomy();
        var nScale = nodeSizeScale();
        for (i = 0; i < n; i++) {
            var ni = vis[i];
            var c = colorFromValue(values[ni] || 0, this.peak);
            var active = anatomy || reservoirActive(values[ni] || 0, this.peak);
            col[i * 3] = active ? c.r : 0;
            col[i * 3 + 1] = active ? c.g : 0;
            col[i * 3 + 2] = active ? c.b : 0;
            sz[i] = active ? (anatomy ? lerp(3.2, 7.2, c.t) : lerp(4.4, 11, c.t)) * nScale : 0;
            idleSz[i] = active ? 0 : (anatomy ? 2.1 : 3.4) * nScale;
        }
        gl.bindBuffer(gl.ARRAY_BUFFER, this.colorBuf);
        gl.bufferData(gl.ARRAY_BUFFER, col, gl.DYNAMIC_DRAW);
        gl.bindBuffer(gl.ARRAY_BUFFER, this.sizeBuf);
        gl.bufferData(gl.ARRAY_BUFFER, sz, gl.DYNAMIC_DRAW);
        gl.bindBuffer(gl.ARRAY_BUFFER, this.idleSizeBuf);
        gl.bufferData(gl.ARRAY_BUFFER, idleSz, gl.DYNAMIC_DRAW);

        var pose = cameraPose(this._aspect || 1);
        var near = clamp(pose.dist * 0.018, 0.0008, 0.08);
        var far = Math.max(pose.dist * 28, extentRadius() * 16, 10);
        var proj = new Float32Array(16);
        var view = new Float32Array(16);
        perspective(proj, (pose.fov * Math.PI) / 180, this._aspect || 1, near, far);
        lookAt(view, pose.eye, pose.target, pose.up);
        var mvp = mul4(proj, view);

        gl.clearColor(0.0431, 0.0745, 0.1137, 1);
        gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
        gl.disable(gl.DEPTH_TEST);
        gl.enable(gl.BLEND);
        gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
        gl.useProgram(this.program);
        var uMVP = gl.getUniformLocation(this.program, "uMVP");
        var uSize = gl.getUniformLocation(this.program, "uSizeMul");
        gl.uniformMatrix4fv(uMVP, false, mvp);
        var aPos = gl.getAttribLocation(this.program, "aPos");
        var aCol = gl.getAttribLocation(this.program, "aCol");
        var aSize = gl.getAttribLocation(this.program, "aSize");
        var self = this;
        var lineProg = this.lineProgram;
        var bindLineProg = function (alpha) {
            if (!lineProg) {
                return false;
            }
            gl.useProgram(lineProg);
            gl.uniformMatrix4fv(gl.getUniformLocation(lineProg, "uMVP"), false, mvp);
            gl.uniform1f(gl.getUniformLocation(lineProg, "uAlpha"), alpha);
            return true;
        };
        var bindPointProg = function () {
            gl.useProgram(self.program);
            gl.uniformMatrix4fv(uMVP, false, mvp);
        };

        if (this.nContext) {
            gl.uniform1f(uSize, Math.max(1.15, contextSizeScale(this.nContext) * 26));
            gl.bindBuffer(gl.ARRAY_BUFFER, this.contextBuf);
            gl.enableVertexAttribArray(aPos);
            gl.vertexAttribPointer(aPos, 3, gl.FLOAT, false, 0, 0);
            gl.disableVertexAttribArray(aCol);
            gl.vertexAttrib3f(aCol, 0.18, 0.22, 0.27);
            gl.disableVertexAttribArray(aSize);
            gl.vertexAttrib1f(aSize, 2.0);
            gl.drawArrays(gl.POINTS, 0, this.nContext);
        }
        if (this.nHullLine && bindLineProg(0.18)) {
            var laPos = gl.getAttribLocation(lineProg, "aPos");
            var laCol = gl.getAttribLocation(lineProg, "aCol");
            gl.bindBuffer(gl.ARRAY_BUFFER, this.hullLineBuf);
            gl.enableVertexAttribArray(laPos);
            gl.vertexAttribPointer(laPos, 3, gl.FLOAT, false, 0, 0);
            gl.disableVertexAttribArray(laCol);
            gl.vertexAttrib3f(laCol, 0.28, 0.33, 0.38);
            gl.drawArrays(gl.LINES, 0, this.nHullLine);
            bindPointProg();
        }
        if (this.nHull) {
            gl.uniform1f(uSize, 1.0);
            gl.bindBuffer(gl.ARRAY_BUFFER, this.hullBuf);
            gl.enableVertexAttribArray(aPos);
            gl.vertexAttribPointer(aPos, 3, gl.FLOAT, false, 0, 0);
            gl.disableVertexAttribArray(aCol);
            gl.vertexAttrib3f(aCol, 0.12, 0.22, 0.3);
            gl.disableVertexAttribArray(aSize);
            gl.vertexAttrib1f(aSize, 3.0);
            gl.drawArrays(gl.POINTS, 0, this.nHull);
        }
        if (this.nLineVerts) {
            var lc = new Float32Array(this.nLineVerts * 3);
            var thr = ACTIVITY_T * this.peak;
            for (i = 0; i < this.edgeIndex.length; i += 2) {
                var sa = Math.abs(Number(values[this.edgeIndex[i]] || 0));
                var sb = Math.abs(Number(values[this.edgeIndex[i + 1]] || 0));
                var hot = sa >= thr && sb >= thr;
                var r = hot ? (anatomy ? 0.42 : 0.72) : anatomy ? 0.1 : 0.16;
                var g = hot ? (anatomy ? 0.48 : 0.7) : anatomy ? 0.14 : 0.22;
                var b = hot ? (anatomy ? 0.52 : 0.48) : anatomy ? 0.18 : 0.3;
                var a = i * 3;
                lc[a] = r;
                lc[a + 1] = g;
                lc[a + 2] = b;
                lc[a + 3] = r;
                lc[a + 4] = g;
                lc[a + 5] = b;
            }
            if (bindLineProg(anatomy ? 0.12 : 0.28)) {
                var eaPos = gl.getAttribLocation(lineProg, "aPos");
                var eaCol = gl.getAttribLocation(lineProg, "aCol");
                gl.bindBuffer(gl.ARRAY_BUFFER, this.lineBuf);
                gl.enableVertexAttribArray(eaPos);
                gl.vertexAttribPointer(eaPos, 3, gl.FLOAT, false, 0, 0);
                gl.bindBuffer(gl.ARRAY_BUFFER, this.lineColorBuf);
                gl.bufferData(gl.ARRAY_BUFFER, lc, gl.DYNAMIC_DRAW);
                gl.enableVertexAttribArray(eaCol);
                gl.vertexAttribPointer(eaCol, 3, gl.FLOAT, false, 0, 0);
                gl.drawArrays(gl.LINES, 0, this.nLineVerts);
                bindPointProg();
            }
        }
        gl.uniform1f(uSize, 1.0);
        gl.bindBuffer(gl.ARRAY_BUFFER, this.buf);
        gl.enableVertexAttribArray(aPos);
        gl.vertexAttribPointer(aPos, 3, gl.FLOAT, false, 0, 0);
        gl.disableVertexAttribArray(aCol);
        gl.vertexAttrib3f(aCol, anatomy ? 0.22 : 0.06, anatomy ? 0.26 : 0.09, anatomy ? 0.30 : 0.12);
        gl.bindBuffer(gl.ARRAY_BUFFER, this.idleSizeBuf);
        gl.enableVertexAttribArray(aSize);
        gl.vertexAttribPointer(aSize, 1, gl.FLOAT, false, 0, 0);
        gl.drawArrays(gl.POINTS, 0, n);

        if (anatomy) {
            gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
        } else {
            gl.blendFunc(gl.SRC_ALPHA, gl.ONE);
        }
        gl.uniform1f(uSize, anatomy ? 1.25 : 1.55);
        gl.bindBuffer(gl.ARRAY_BUFFER, this.buf);
        gl.enableVertexAttribArray(aPos);
        gl.vertexAttribPointer(aPos, 3, gl.FLOAT, false, 0, 0);
        gl.bindBuffer(gl.ARRAY_BUFFER, this.colorBuf);
        gl.enableVertexAttribArray(aCol);
        gl.vertexAttribPointer(aCol, 3, gl.FLOAT, false, 0, 0);
        gl.bindBuffer(gl.ARRAY_BUFFER, this.sizeBuf);
        gl.enableVertexAttribArray(aSize);
        gl.vertexAttribPointer(aSize, 1, gl.FLOAT, false, 0, 0);
        gl.drawArrays(gl.POINTS, 0, n);
        if (!anatomy) {
            gl.uniform1f(uSize, 1.0);
            gl.drawArrays(gl.POINTS, 0, n);
        }
        if (this.nSensor && inRow) {
            gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
            var sc = new Float32Array(this.nSensor * 3);
            var ssz = new Float32Array(this.nSensor);
            for (i = 0; i < this.nSensor; i++) {
                var ic = colorFromValue(inRow[i] || 0, this.inPeak);
                sc[i * 3] = ic.r;
                sc[i * 3 + 1] = ic.g;
                sc[i * 3 + 2] = ic.b;
                ssz[i] = 10;
            }
            gl.bindBuffer(gl.ARRAY_BUFFER, this.sensorBuf);
            gl.enableVertexAttribArray(aPos);
            gl.vertexAttribPointer(aPos, 3, gl.FLOAT, false, 0, 0);
            gl.bindBuffer(gl.ARRAY_BUFFER, this.sensorColorBuf);
            gl.bufferData(gl.ARRAY_BUFFER, sc, gl.DYNAMIC_DRAW);
            gl.enableVertexAttribArray(aCol);
            gl.vertexAttribPointer(aCol, 3, gl.FLOAT, false, 0, 0);
            gl.bindBuffer(gl.ARRAY_BUFFER, this.sizeBuf);
            gl.bufferData(gl.ARRAY_BUFFER, ssz, gl.DYNAMIC_DRAW);
            gl.enableVertexAttribArray(aSize);
            gl.vertexAttribPointer(aSize, 1, gl.FLOAT, false, 0, 0);
            gl.uniform1f(uSize, 1.2);
            gl.drawArrays(gl.POINTS, 0, this.nSensor);
        }
    };

    function initRenderer() {
        try {
            if (hasThree()) {
                renderer = new ThreeRenderer();
                return;
            }
        } catch (err) {
            renderer = null;
        }
        renderer = new WebGLRenderer();
    }

    function tick(ts) {
        requestAnimationFrame(tick);
        var previousFrame = frameIndex;
        var bounds = frameBounds();
        if (playing && nFrames() > 0 && bounds.end >= bounds.start) {
            if (!lastTs) {
                lastTs = ts;
            }
            var dt = ts - lastTs;
            lastTs = ts;
            accum += dt * speed;
            var frameMs = 120;
            while (accum >= frameMs) {
                accum -= frameMs;
                frameIndex += 1;
                if (frameIndex > bounds.end) {
                    frameIndex = bounds.start;
                }
            }
            updateFrameLabel();
        } else {
            lastTs = 0;
        }
        if (!dragging && hullMode() === "schematic_cns") {
            rotY += 0.00032;
            needsRender = true;
        }
        if (frameIndex !== previousFrame) needsRender = true;
        if (!needsRender) return;
        needsRender = false;
        var values = sampleState(frameIndex);
        var inRow = sampleInputs(frameIndex);
        if (renderer && renderer.render) {
            renderer.render(values, inRow);
            canvas.dataset.renderCount = String(++renderCount);
        }
        updateHud(values);
    }

    canvas.addEventListener("mousedown", function (ev) {
        dragging = true;
        lastX = ev.clientX;
        lastY = ev.clientY;
    });
    document.getElementById("view-brain").addEventListener("click", function () {
        focusBrain = true; cachedBounds = null; camDist = 1; rotY = 0.08; rotX = 0.04; needsRender = true;
    });
    document.getElementById("view-cns").addEventListener("click", function () {
        focusBrain = false; cachedBounds = null; camDist = 1; rotY = 0.08; rotX = 0.04; needsRender = true;
    });
    document.getElementById("view-fullscreen").addEventListener("click", function () {
        var stage = document.getElementById("stage");
        if (document.fullscreenElement) document.exitFullscreen();
        else if (stage.requestFullscreen) stage.requestFullscreen();
    });
    document.addEventListener("fullscreenchange", function () {
        if (renderer) renderer.resize();
        needsRender = true;
    });
    canvas.addEventListener("touchstart", function (ev) {
        if (ev.touches.length === 1) {
            dragging = true; lastX = ev.touches[0].clientX; lastY = ev.touches[0].clientY;
        }
    }, {passive: true});
    canvas.addEventListener("touchmove", function (ev) {
        if (ev.touches.length === 1 && dragging) {
            ev.preventDefault();
            rotY += (ev.touches[0].clientX - lastX) * 0.006;
            rotX = clamp(rotX + (ev.touches[0].clientY - lastY) * 0.006, -1.3, 1.3);
            lastX = ev.touches[0].clientX; lastY = ev.touches[0].clientY; needsRender = true;
        }
    }, {passive: false});
    canvas.addEventListener("touchend", function () { dragging = false; }, {passive: true});
    document.getElementById("reset-view").addEventListener("click", function () {
        needsRender = true;
        camDist = 1;
        rotY = isAnatomy() ? 0.12 : 0.72;
        rotX = isAnatomy() ? 0.06 : 0.28;
    });
    window.addEventListener("mouseup", function () {
        dragging = false;
    });
    window.addEventListener("mousemove", function (ev) {
        if (!dragging) {
            return;
        }
        rotY += (ev.clientX - lastX) * 0.01;
        needsRender = true;
        rotX = clamp(rotX + (ev.clientY - lastY) * 0.01, -1.15, 1.15);
        lastX = ev.clientX;
        lastY = ev.clientY;
    });
    canvas.addEventListener(
        "wheel",
        function (ev) {
            ev.preventDefault();
            var limits = camDistLimits();
            var next = camDist * (ev.deltaY > 0 ? 1.1 : 0.9);
            camDist = clamp(next, limits.min, limits.max);
            needsRender = true;
        },
        { passive: false }
    );
    window.addEventListener("resize", function () {
        needsRender = true;
        drawActivityHistory();
        if (renderer && renderer.resize) {
            renderer.resize();
        }
        syncFrameHeight();
    });

    buildControls();
    initRenderer();
    requestAnimationFrame(tick);

    if (window.Streamlit) {
        window.Streamlit.onRender(function (event) {
            applyArgs(event.args || {});
            syncFrameHeight();
        });
        window.Streamlit.setComponentReady();
        syncFrameHeight();
    }
})();
