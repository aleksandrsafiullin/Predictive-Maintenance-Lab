from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

from pdm import __version__
from pdm.device import resolve_device
from pdm.io_util import atomic_write_json
from pdm.paths import data_processed, data_raw, project_root, runs_root, worker_dir
from pdm.worker import job_path, pid_path, request_stop, worker_alive


def _python() -> str:
    return sys.executable


def doctor() -> dict:
    import platform

    import networkx
    import numpy
    import pandas
    import plotly
    import pyarrow
    import pytest
    import sklearn
    import streamlit
    import torch
    import yaml

    from pdm.connectome.anatomy import find_soma_table_path
    from pdm.connectome.sources import default_malemcns_path
    from pdm.models import PDMNet

    info = resolve_device("auto")
    device = info.torch_device
    net = PDMNet(input_size=4, hidden_size=8, architecture="gru", head="rul").to(device)
    x = torch.randn(2, 5, 4, device=device)
    y = net(x)
    loss = y.sum()
    loss.backward()
    fwd_ok = bool(torch.isfinite(y).all().item())
    grad_ok = all(p.grad is not None for p in net.parameters() if p.requires_grad)
    soma_path = find_soma_table_path()

    writable = {}
    for name, path in {"raw": data_raw(), "processed": data_processed(), "runs": runs_root()}.items():
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".write_probe"
        try:
            probe.write_text("ok", encoding="utf-8")
            probe.unlink()
            writable[name] = True
        except Exception as exc:  # noqa: BLE001
            writable[name] = False
            writable[f"{name}_error"] = str(exc)

    report = {
        "pdm_version": __version__,
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "torch": torch.__version__,
        "numpy": numpy.__version__,
        "pandas": pandas.__version__,
        "sklearn": sklearn.__version__,
        "streamlit": streamlit.__version__,
        "plotly": plotly.__version__,
        "pyarrow": pyarrow.__version__,
        "pyyaml": yaml.__version__,
        "pytest": pytest.__version__,
        "networkx": networkx.__version__,
        "malemcns_present": default_malemcns_path().is_file(),
        "soma_table_present": soma_path is not None,
        "soma_table_path": str(soma_path) if soma_path is not None else None,
        "device": info.name,
        "device_fallback": info.fallback_reason,
        "forward_backward_ok": fwd_ok and grad_ok,
        "writable": writable,
        "data_raw_exists": data_raw().exists(),
        "data_processed_exists": data_processed().exists(),
    }
    print(json.dumps(report, indent=2))
    if not report["forward_backward_ok"]:
        raise SystemExit(2)
    return report


def spawn_worker(job: dict) -> subprocess.Popen:
    if worker_alive():
        raise RuntimeError("A heavy job is already running. Stop it before starting another.")
    worker_dir().mkdir(parents=True, exist_ok=True)
    atomic_write_json(job_path(), job)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(project_root() / "src") + os.pathsep + env.get("PYTHONPATH", "")
    env["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"
    proc = subprocess.Popen(
        [_python(), "-m", "pdm.worker", "--job-file", str(job_path())],
        cwd=str(project_root()),
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    pid_path().write_text(str(proc.pid), encoding="utf-8")
    return proc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="pdm", description="Predictive Maintenance Lab")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("doctor")
    p_dl = sub.add_parser("download")
    p_dl.add_argument("--dataset", required=True, choices=["bearings", "filters", "all"])
    p_dl.add_argument("--local-path", default=None)

    p_ins = sub.add_parser("inspect")
    p_ins.add_argument("--dataset", required=True, choices=["bearings", "filters"])

    p_pr = sub.add_parser("prepare")
    p_pr.add_argument("--dataset", required=True, choices=["bearings", "filters"])

    p_tr = sub.add_parser("train")
    p_tr.add_argument("--dataset", required=True, choices=["bearings", "filters"])
    p_tr.add_argument(
        "--arch",
        default="gru",
        choices=["gru", "lstm", "fly_connectome_reservoir", "random_reservoir"],
    )
    p_tr.add_argument("--epochs", type=int, default=None)
    p_tr.add_argument("--history", type=int, default=None)
    p_tr.add_argument("--smoke", action="store_true")
    p_tr.add_argument("--resume", default=None)
    p_tr.add_argument("--device", default="auto")
    p_tr.add_argument("--max-windows-per-unit", type=int, default=None)
    p_tr.add_argument("--n-nodes", type=int, default=None)
    p_tr.add_argument(
        "--graph-mode",
        default=None,
        choices=["synthetic_fixture", "real_connectome", "random_rewire"],
    )
    p_tr.add_argument("--readout", default=None, choices=["ridge", "gradient"])
    p_tr.add_argument("--seed", type=int, default=None)
    p_tr.add_argument("--events-only", action="store_true", help="Filters training ablation; validation remains complete")
    p_tr.add_argument(
        "--source-path",
        default=None,
        help="Local MaleCNS feather for --graph-mode real_connectome. Not required in CI.",
    )

    p_ev = sub.add_parser("evaluate")
    p_ev.add_argument("--dataset", required=True, choices=["bearings", "filters"])
    p_ev.add_argument("--run-id", required=True)
    p_ev.add_argument(
        "--split",
        choices=["validation", "test"],
        default="test",
        help="Evaluate mask split. Default: test.",
    )
    p_ev.add_argument(
        "--horizon-s",
        type=float,
        default=None,
        help="H_trigger in seconds. Any H/K flag sets policy_mode=research and never writes alert_policy.json.",
    )
    p_ev.add_argument(
        "--k",
        type=int,
        default=None,
        help="Confirmation count K. Any H/K flag sets policy_mode=research (never writes).",
    )
    p_ev.add_argument(
        "--min-action-lead-s",
        type=float,
        default=None,
        help="minimum_action_lead_time in seconds. Any H/K flag sets policy_mode=research (never writes).",
    )
    p_ev.add_argument(
        "--max-useful-horizon-s",
        type=float,
        default=None,
        help="Optional too_early cap. Any H/K flag sets policy_mode=research (never writes).",
    )
    p_ev.add_argument(
        "--force",
        action="store_true",
        help="Evaluate even if dataset/split fingerprints do not match the run snapshot (not used by the UI)",
    )
    p_ev.add_argument(
        "--with-trace",
        action="store_true",
        help=(
            "Also write reservoir traces under runs/<dataset>/<run>/traces/<unit>/. "
            "Off by default; does not change predictions.csv."
        ),
    )
    p_ev.add_argument(
        "--list",
        action="store_true",
        help="List evaluations for --run-id without creating a new evaluation",
    )

    sub.add_parser("app")
    sub.add_parser("stop")
    matrix = sub.add_parser("train-matrix")
    matrix.add_argument("--resume-batch", default=None)
    quality = sub.add_parser("quality")
    quality.add_argument("--dataset", required=True, choices=["bearings", "filters"])
    compare = sub.add_parser("compare")
    compare.add_argument("--dataset", required=True, choices=["bearings", "filters"])
    compare.add_argument("--evaluation", action="append", required=True, metavar="RUN_ID:EVAL_ID")

    args = parser.parse_args(argv)
    if args.cmd == "train-matrix":
        proc = spawn_worker({"kind": "train_matrix", "batch_id": args.resume_batch})
        print(json.dumps({"worker_pid": proc.pid, "status": "started"}))
        return 0
    if args.cmd == "quality":
        from pdm.data.prepare import load_processed

        print(json.dumps(load_processed(args.dataset)["report"].get("quality", {}), indent=2))
        return 0
    if args.cmd == "compare":
        from pdm.benchmark import compare_evaluations, load_evaluation, save_comparison

        result = compare_evaluations([load_evaluation(args.dataset, *pair.split(":", 1)) for pair in args.evaluation])
        print(result["table"].to_string(index=False))
        print(save_comparison(result))
        return 0
    if args.cmd == "doctor":
        doctor()
        return 0
    if args.cmd == "download":
        from pdm.data.download import download_dataset

        targets = ["bearings", "filters"] if args.dataset == "all" else [args.dataset]
        for ds in targets:
            rec = download_dataset(ds, local_path=args.local_path)
            print(json.dumps({"dataset": ds, "n_files": len(rec.get("files") or [])}, indent=2))
        return 0
    if args.cmd == "inspect":
        from pdm.data.prepare import inspect_dataset

        print(json.dumps(inspect_dataset(args.dataset), indent=2, default=str))
        return 0
    if args.cmd == "prepare":
        from pdm.data.prepare import prepare_dataset

        rec = prepare_dataset(args.dataset, progress=lambda s, i: print(s, i, flush=True))
        print(json.dumps({"dir": str(rec["dir"]), "split": rec["split"]}, indent=2, default=str))
        return 0
    if args.cmd == "train":
        from pdm.train import run_training

        rec = run_training(
            args.dataset,
            architecture=args.arch,
            max_epochs=args.epochs,
            history_length=args.history,
            smoke=args.smoke,
            resume_run_id=args.resume,
            device_pref=args.device,
            log=lambda m: print(m, flush=True),
            max_windows_per_unit=args.max_windows_per_unit,
            n_nodes=args.n_nodes,
            graph_mode=args.graph_mode,
            readout=args.readout,
            source_path=args.source_path,
            seed=args.seed,
            events_only=args.events_only,
        )
        print(json.dumps(rec, indent=2, default=str))
        return 0
    if args.cmd == "evaluate":
        from pdm.evaluate import evaluate_run, policy_mode_from_hk_overrides
        from pdm.experiments import list_evaluations, run_dir

        if args.list:
            print(json.dumps(list_evaluations(run_dir(args.dataset, args.run_id)), indent=2, default=str))
            return 0
        rec = evaluate_run(
            args.dataset,
            args.run_id,
            split_name=args.split,
            policy_mode=policy_mode_from_hk_overrides(
                warning_horizon_s=args.horizon_s,
                confirmation_count=args.k,
                minimum_action_lead_time=args.min_action_lead_s,
                max_useful_horizon_s=args.max_useful_horizon_s,
            ),
            warning_horizon_s=args.horizon_s,
            confirmation_count=args.k,
            minimum_action_lead_time=args.min_action_lead_s,
            max_useful_horizon_s=args.max_useful_horizon_s,
            force=bool(args.force),
            with_trace=bool(args.with_trace),
        )
        print(json.dumps(rec, indent=2, default=str))
        return 0
    if args.cmd == "stop":
        request_stop()
        print("Stop requested")
        return 0
    if args.cmd == "app":
        root = project_root()
        cmd = [
            _python(),
            "-m",
            "streamlit",
            "run",
            str(root / "src" / "pdm" / "app.py"),
            "--server.address",
            "127.0.0.1",
            "--server.port",
            "8501",
            "--browser.gatherUsageStats",
            "false",
        ]
        env = os.environ.copy()
        env["PYTHONPATH"] = str(root / "src") + os.pathsep + env.get("PYTHONPATH", "")
        env["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"
        return subprocess.call(cmd, cwd=str(root), env=env)
    return 1
