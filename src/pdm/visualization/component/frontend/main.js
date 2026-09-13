(function () {
    "use strict";

    var canvas = document.getElementById("canvas");
    var controls = document.getElementById("controls");

    var nodes = [];
    var edges = [];
    var positions = {};
    var states = [];
    var frameMap = [];
    var flags = {};

    var playing = false;
    var frameIndex = 0;
    var speed = 2;
    var lastTs = 0;
    var accum = 0;
    var rotY = 0.55;
    var rotX = 0.35;
    var dragging = false;
    var lastX = 0;
    var lastY = 0;
    var renderer = null;

    var playBtn;
    var pauseBtn;
    var frameSlider;
    var frameLabel;
    var speedSlider;
    var modeLabel;

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

    function modeName() {
        return (flags && flags.mode) || "Overview";
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

    function sampleState(t) {
        var n = nFrames();
        if (!n) {
            return [];
        }
        var i0 = clamp(Math.floor(t), 0, n - 1);
        var i1 = clamp(i0 + 1, 0, n - 1);
        var f = t - i0;
        var a = states[i0] || [];
        var b = states[i1] || a;
        var out = [];
        var m = Math.max(a.length, b.length, nNodes());
        for (var i = 0; i < m; i++) {
            var va = Number(a[i] || 0);
            var vb = Number(b[i] || 0);
            out.push(lerp(va, vb, f));
        }
        return out;
    }

    function maxAbsState() {
        var m = 0;
        for (var t = 0; t < states.length; t++) {
            var row = states[t] || [];
            for (var i = 0; i < row.length; i++) {
                var v = Math.abs(Number(row[i] || 0));
                if (v > m) {
                    m = v;
                }
            }
        }
        return m > 0 ? m : 1;
    }

    function colorFromValue(v, peak) {
        var t = clamp(Math.abs(Number(v) || 0) / peak, 0, 1);
        // blue (inactive) → green (active), from provided state — not noise
        return {
            r: lerp(0.15, 0.2, t),
            g: lerp(0.35, 0.95, t),
            b: lerp(0.95, 0.25, t)
        };
    }

    function nodeXYZ(i) {
        var id = nodes[i] != null ? String(nodes[i]) : String(i);
        var p = positions[id];
        if (p && p.length >= 2) {
            return [Number(p[0]) || 0, Number(p[1]) || 0, Number(p[2]) || 0];
        }
        var n = Math.max(nNodes(), 1);
        var angle = (2 * Math.PI * i) / n;
        return [Math.cos(angle), Math.sin(angle), 0];
    }

    function extentRadius() {
        var maxR = 1;
        var n = nNodes();
        for (var i = 0; i < n; i++) {
            var p = nodeXYZ(i);
            var r = Math.sqrt(p[0] * p[0] + p[1] * p[1] + p[2] * p[2]);
            if (r > maxR) {
                maxR = r;
            }
        }
        return maxR;
    }

    function applyArgs(args) {
        args = args || {};
        nodes = args.nodes || [];
        edges = args.edges || [];
        positions = args.positions || {};
        states = args.states || [];
        frameMap = args.frame_map || [];
        flags = args.flags || {};
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
            renderer.rebuild();
        }
    }

    function updateFrameLabel() {
        if (frameLabel) {
            frameLabel.textContent = "frame " + Math.floor(frameIndex) + " / " + Math.max(nFrames() - 1, 0);
        }
        if (frameSlider && document.activeElement !== frameSlider) {
            frameSlider.value = String(Math.floor(frameIndex));
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
            frameIndex = Number(frameSlider.value) || 0;
            updateFrameLabel();
        });
        speedSlider.addEventListener("input", function () {
            speed = Number(speedSlider.value) || 1;
        });
    }

    function ThreeRenderer() {
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0x111111);
        this.camera = new THREE.PerspectiveCamera(55, 1, 0.01, 100);
        this.gl = new THREE.WebGLRenderer({ canvas: canvas, antialias: true });
        this.points = null;
        this.lines = null;
        this.colorAttr = null;
        this.peak = 1;
        this.rebuild();
        this.resize();
    }

    ThreeRenderer.prototype.rebuild = function () {
        while (this.scene.children.length) {
            this.scene.remove(this.scene.children[0]);
        }
        var n = nNodes();
        var pos = new Float32Array(n * 3);
        var col = new Float32Array(n * 3);
        for (var i = 0; i < n; i++) {
            var p = nodeXYZ(i);
            pos[i * 3] = p[0];
            pos[i * 3 + 1] = p[1];
            pos[i * 3 + 2] = p[2];
        }
        var geom = new THREE.BufferGeometry();
        geom.setAttribute("position", new THREE.BufferAttribute(pos, 3));
        this.colorAttr = new THREE.BufferAttribute(col, 3);
        geom.setAttribute("color", this.colorAttr);
        var mat = new THREE.PointsMaterial({ size: 0.12, vertexColors: true, sizeAttenuation: true });
        this.points = new THREE.Points(geom, mat);
        this.scene.add(this.points);

        if (edges && edges.length && nodes.length) {
            var indexOf = {};
            for (var k = 0; k < nodes.length; k++) {
                indexOf[String(nodes[k])] = k;
            }
            var linePos = [];
            for (var e = 0; e < edges.length; e++) {
                var src = indexOf[String(edges[e].src)];
                var dst = indexOf[String(edges[e].dst)];
                if (src == null || dst == null) {
                    continue;
                }
                var a = nodeXYZ(src);
                var b = nodeXYZ(dst);
                linePos.push(a[0], a[1], a[2], b[0], b[1], b[2]);
            }
            if (linePos.length) {
                var lg = new THREE.BufferGeometry();
                lg.setAttribute("position", new THREE.BufferAttribute(new Float32Array(linePos), 3));
                this.lines = new THREE.LineSegments(
                    lg,
                    new THREE.LineBasicMaterial({ color: 0x445566, transparent: true, opacity: 0.45 })
                );
                this.scene.add(this.lines);
            }
        }
        this.peak = maxAbsState();
        this.resize();
    };

    ThreeRenderer.prototype.resize = function () {
        var w = canvas.clientWidth || 640;
        var h = canvas.clientHeight || 400;
        this.gl.setSize(w, h, false);
        this.camera.aspect = w / Math.max(h, 1);
        this.camera.updateProjectionMatrix();
    };

    ThreeRenderer.prototype.render = function (values) {
        var n = nNodes();
        if (this.colorAttr) {
            for (var i = 0; i < n; i++) {
                var c = colorFromValue(values[i] || 0, this.peak);
                this.colorAttr.array[i * 3] = c.r;
                this.colorAttr.array[i * 3 + 1] = c.g;
                this.colorAttr.array[i * 3 + 2] = c.b;
            }
            this.colorAttr.needsUpdate = true;
        }
        var radius = extentRadius() * 2.6 + 1.2;
        this.camera.position.set(
            radius * Math.cos(rotY) * Math.cos(rotX),
            radius * Math.sin(rotX),
            radius * Math.sin(rotY) * Math.cos(rotX)
        );
        this.camera.lookAt(0, 0, 0);
        this.gl.render(this.scene, this.camera);
    };

    function WebGLRenderer() {
        this.gl = canvas.getContext("webgl") || canvas.getContext("experimental-webgl");
        this.program = null;
        this.buf = null;
        this.colorBuf = null;
        this.lineBuf = null;
        this.nPoints = 0;
        this.nLineVerts = 0;
        this.peak = 1;
        if (!this.gl) {
            return;
        }
        var vs =
            "attribute vec3 aPos; attribute vec3 aCol; uniform mat4 uMVP; varying vec3 vCol;" +
            "void main(){ vCol=aCol; gl_Position=uMVP*vec4(aPos,1.0); gl_PointSize=8.0; }";
        var fs =
            "precision mediump float; varying vec3 vCol; void main(){ gl_FragColor=vec4(vCol,1.0); }";
        this.program = compileProgram(this.gl, vs, fs);
        this.buf = this.gl.createBuffer();
        this.colorBuf = this.gl.createBuffer();
        this.lineBuf = this.gl.createBuffer();
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
        var n = nNodes();
        this.nPoints = n;
        var pos = new Float32Array(n * 3);
        for (var i = 0; i < n; i++) {
            var p = nodeXYZ(i);
            pos[i * 3] = p[0];
            pos[i * 3 + 1] = p[1];
            pos[i * 3 + 2] = p[2];
        }
        this.gl.bindBuffer(this.gl.ARRAY_BUFFER, this.buf);
        this.gl.bufferData(this.gl.ARRAY_BUFFER, pos, this.gl.STATIC_DRAW);
        var indexOf = {};
        for (var k = 0; k < nodes.length; k++) {
            indexOf[String(nodes[k])] = k;
        }
        var linePos = [];
        for (var e = 0; e < (edges || []).length; e++) {
            var src = indexOf[String(edges[e].src)];
            var dst = indexOf[String(edges[e].dst)];
            if (src == null || dst == null) {
                continue;
            }
            var a = nodeXYZ(src);
            var b = nodeXYZ(dst);
            linePos.push(a[0], a[1], a[2], b[0], b[1], b[2]);
        }
        this.nLineVerts = linePos.length / 3;
        this.gl.bindBuffer(this.gl.ARRAY_BUFFER, this.lineBuf);
        this.gl.bufferData(this.gl.ARRAY_BUFFER, new Float32Array(linePos), this.gl.STATIC_DRAW);
        this.peak = maxAbsState();
        this.resize();
    };

    WebGLRenderer.prototype.resize = function () {
        if (!this.gl) {
            return;
        }
        var w = canvas.clientWidth || 640;
        var h = canvas.clientHeight || 400;
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

    function lookAt(out, eye, center) {
        var zx = eye[0] - center[0];
        var zy = eye[1] - center[1];
        var zz = eye[2] - center[2];
        var zl = Math.sqrt(zx * zx + zy * zy + zz * zz) || 1;
        zx /= zl;
        zy /= zl;
        zz /= zl;
        var ux = 0;
        var uy = 1;
        var uz = 0;
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

    WebGLRenderer.prototype.render = function (values) {
        var gl = this.gl;
        if (!gl || !this.program) {
            return;
        }
        var n = this.nPoints;
        var col = new Float32Array(n * 3);
        for (var i = 0; i < n; i++) {
            var c = colorFromValue(values[i] || 0, this.peak);
            col[i * 3] = c.r;
            col[i * 3 + 1] = c.g;
            col[i * 3 + 2] = c.b;
        }
        gl.bindBuffer(gl.ARRAY_BUFFER, this.colorBuf);
        gl.bufferData(gl.ARRAY_BUFFER, col, gl.DYNAMIC_DRAW);

        var radius = extentRadius() * 2.6 + 1.2;
        var eye = [
            radius * Math.cos(rotY) * Math.cos(rotX),
            radius * Math.sin(rotX),
            radius * Math.sin(rotY) * Math.cos(rotX)
        ];
        var proj = new Float32Array(16);
        var view = new Float32Array(16);
        perspective(proj, (55 * Math.PI) / 180, this._aspect || 1, 0.01, 100);
        lookAt(view, eye, [0, 0, 0]);
        var mvp = mul4(proj, view);

        gl.clearColor(0.067, 0.067, 0.067, 1);
        gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
        gl.enable(gl.DEPTH_TEST);
        gl.useProgram(this.program);
        var uMVP = gl.getUniformLocation(this.program, "uMVP");
        gl.uniformMatrix4fv(uMVP, false, mvp);
        var aPos = gl.getAttribLocation(this.program, "aPos");
        var aCol = gl.getAttribLocation(this.program, "aCol");
        if (this.nLineVerts) {
            gl.bindBuffer(gl.ARRAY_BUFFER, this.lineBuf);
            gl.enableVertexAttribArray(aPos);
            gl.vertexAttribPointer(aPos, 3, gl.FLOAT, false, 0, 0);
            gl.disableVertexAttribArray(aCol);
            gl.vertexAttrib3f(aCol, 0.27, 0.33, 0.4);
            gl.drawArrays(gl.LINES, 0, this.nLineVerts);
        }
        gl.bindBuffer(gl.ARRAY_BUFFER, this.buf);
        gl.enableVertexAttribArray(aPos);
        gl.vertexAttribPointer(aPos, 3, gl.FLOAT, false, 0, 0);
        gl.bindBuffer(gl.ARRAY_BUFFER, this.colorBuf);
        gl.enableVertexAttribArray(aCol);
        gl.vertexAttribPointer(aCol, 3, gl.FLOAT, false, 0, 0);
        gl.drawArrays(gl.POINTS, 0, n);
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
        var values = sampleState(frameIndex);
        if (renderer && renderer.render) {
            renderer.render(values);
        }
    }

    canvas.addEventListener("mousedown", function (ev) {
        dragging = true;
        lastX = ev.clientX;
        lastY = ev.clientY;
    });
    window.addEventListener("mouseup", function () {
        dragging = false;
    });
    window.addEventListener("mousemove", function (ev) {
        if (!dragging) {
            return;
        }
        rotY += (ev.clientX - lastX) * 0.01;
        rotX = clamp(rotX + (ev.clientY - lastY) * 0.01, -1.2, 1.2);
        lastX = ev.clientX;
        lastY = ev.clientY;
    });
    window.addEventListener("resize", function () {
        if (renderer && renderer.resize) {
            renderer.resize();
        }
        if (window.Streamlit && window.Streamlit.setFrameHeight) {
            window.Streamlit.setFrameHeight(document.body.scrollHeight);
        }
    });

    buildControls();
    initRenderer();
    requestAnimationFrame(tick);

    if (window.Streamlit) {
        window.Streamlit.onRender(function (event) {
            applyArgs(event.args || {});
            if (window.Streamlit.setFrameHeight) {
                window.Streamlit.setFrameHeight(document.body.scrollHeight);
            }
        });
        window.Streamlit.setComponentReady();
        if (window.Streamlit.setFrameHeight) {
            window.Streamlit.setFrameHeight(document.body.scrollHeight);
        }
    }
})();
