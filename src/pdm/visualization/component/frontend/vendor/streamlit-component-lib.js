/**
 * Local Streamlit custom-component iframe bridge (API v1).
 *
 * The published streamlit-component-lib npm package is ESM/React-only and is
 * not usable from a static <script> tag. This file is a browser shim of the
 * same postMessage protocol (Apache-2.0, Streamlit Inc. / Snowflake Inc.).
 */
(function (root) {
    "use strict";

    function send(type, extra) {
        var msg = { isStreamlitMessage: true, type: type, apiVersion: 1 };
        if (extra) {
            Object.keys(extra).forEach(function (key) {
                msg[key] = extra[key];
            });
        }
        window.parent.postMessage(msg, "*");
    }

    var lastHeight = null;
    var renderListeners = [];
    var listening = false;

    function onMessage(event) {
        var data = event && event.data ? event.data : {};
        if (data.type === "streamlit:render") {
            renderListeners.forEach(function (cb) {
                cb(data);
            });
        }
    }

    function ensureListen() {
        if (!listening) {
            listening = true;
            window.addEventListener("message", onMessage);
        }
    }

    root.Streamlit = {
        RENDER_EVENT: "streamlit:render",
        setComponentReady: function () {
            ensureListen();
            send("streamlit:componentReady", { apiVersion: 1 });
        },
        setComponentValue: function (value) {
            send("streamlit:setComponentValue", { value: value, dataType: "json" });
        },
        setFrameHeight: function (height) {
            if (height === undefined || height === null) {
                height = document.body ? document.body.scrollHeight : 480;
            }
            if (height === lastHeight) {
                return;
            }
            lastHeight = height;
            send("streamlit:setFrameHeight", { height: height });
        },
        onRender: function (callback) {
            ensureListen();
            renderListeners.push(callback);
        }
    };
})(window);
