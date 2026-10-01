"""Hyperparameter sweep: GRU/LSTM × datasets × key training dims.

Runs the training_v2 pipeline with an abbreviated protocol (max_epochs=30,
patience=10) so the sweep finishes in minutes on CPU.  Results are written to
docs/hyperparameter_sweep_report.md and runs/ (one sub-directory per config).

Usage (all datasets present):
    python3 scripts/hparam_sweep.py

Usage (filters only):
    python3 scripts/hparam_sweep.py --datasets filters

Usage (dry-run — list configs only):
    python3 scripts/hparam_sweep.py --dry-run
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any

# Ensure src is on sys.path when run as a script.
_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root / "src"))

from pdm.paths import dataset_runs  # noqa: E402

# ---------------------------------------------------------------------------
# Config definitions
# ---------------------------------------------------------------------------

SWEEP_PROTOCOL_OVERRIDES: dict[str, Any] = {
    "max_epochs": 30,
    "min_epochs": 8,
    "patience": 10,
}


def make_protocol(dataset_id: str, *, mode: str = "adaptive",
                  learning_rate: float = 0.001,
                  sampling: str = "unit_replacement",
                  near_weight: float = 0.0,
                  feature_recipe: str = "base_v1") -> dict:
    from pdm.training_protocol import protocol as base_protocol
    p = base_protocol(dataset_id, mode=mode, learning_rate=learning_rate,
                      sampling=sampling, near_weight=near_weight,
                      feature_recipe=feature_recipe)
    p.update(SWEEP_PROTOCOL_OVERRIDES)
    return p


def build_configs(datasets: list[str]) -> list[dict]:
    configs = []

    for ds in datasets:
        # 1. GRU baseline (adaptive, base_v1, lr=0.001, unit_replacement)
        configs.append({
            "label": f"{ds}/gru/base_v1/lr0.001/unit_repl",
            "dataset_id": ds,
            "architecture": "gru",
            "protocol": make_protocol(ds, learning_rate=0.001,
                                      sampling="unit_replacement",
                                      feature_recipe="base_v1"),
        })

        # 2. LSTM, same otherwise
        configs.append({
            "label": f"{ds}/lstm/base_v1/lr0.001/unit_repl",
            "dataset_id": ds,
            "architecture": "lstm",
            "protocol": make_protocol(ds, learning_rate=0.001,
                                      sampling="unit_replacement",
                                      feature_recipe="base_v1"),
        })

        # 3. GRU + degradation_v1
        configs.append({
            "label": f"{ds}/gru/degradation_v1/lr0.001/unit_repl",
            "dataset_id": ds,
            "architecture": "gru",
            "protocol": make_protocol(ds, learning_rate=0.001,
                                      sampling="unit_replacement",
                                      feature_recipe="degradation_v1"),
        })

        # 4. GRU + multiscale_trend_v2
        configs.append({
            "label": f"{ds}/gru/multiscale_trend_v2/lr0.001/unit_repl",
            "dataset_id": ds,
            "architecture": "gru",
            "protocol": make_protocol(ds, learning_rate=0.001,
                                      sampling="unit_replacement",
                                      feature_recipe="multiscale_trend_v2"),
        })

        # 5. GRU + multiscale_no_age_v2
        configs.append({
            "label": f"{ds}/gru/multiscale_no_age_v2/lr0.001/unit_repl",
            "dataset_id": ds,
            "architecture": "gru",
            "protocol": make_protocol(ds, learning_rate=0.001,
                                      sampling="unit_replacement",
                                      feature_recipe="multiscale_no_age_v2"),
        })

        # 6. GRU + lower LR
        configs.append({
            "label": f"{ds}/gru/base_v1/lr0.0003/unit_repl",
            "dataset_id": ds,
            "architecture": "gru",
            "protocol": make_protocol(ds, learning_rate=0.0003,
                                      sampling="unit_replacement",
                                      feature_recipe="base_v1"),
        })

        # 7. GRU + full_pass sampling
        configs.append({
            "label": f"{ds}/gru/base_v1/lr0.001/full_pass",
            "dataset_id": ds,
            "architecture": "gru",
            "protocol": make_protocol(ds, learning_rate=0.001,
                                      sampling="full_pass",
                                      feature_recipe="base_v1"),
        })

        # 8. GRU + full_pass + degradation_v1
        configs.append({
            "label": f"{ds}/gru/degradation_v1/lr0.001/full_pass",
            "dataset_id": ds,
            "architecture": "gru",
            "protocol": make_protocol(ds, learning_rate=0.001,
                                      sampling="full_pass",
                                      feature_recipe="degradation_v1"),
        })

        # 9. LSTM + degradation_v1 + full_pass
        configs.append({
            "label": f"{ds}/lstm/degradation_v1/lr0.001/full_pass",
            "dataset_id": ds,
            "architecture": "lstm",
            "protocol": make_protocol(ds, learning_rate=0.001,
                                      sampling="full_pass",
                                      feature_recipe="degradation_v1"),
        })

        # 10. Bearings-only: near_weight=0.5 + full_pass (near-event weighting)
        if ds == "bearings":
            configs.append({
                "label": f"{ds}/gru/base_v1/lr0.001/full_pass_near0.5",
                "dataset_id": ds,
                "architecture": "gru",
                "protocol": make_protocol(ds, learning_rate=0.001,
                                          sampling="full_pass",
                                          feature_recipe="base_v1",
                                          near_weight=0.5),
            })
            configs.append({
                "label": f"{ds}/gru/degradation_v1/lr0.001/full_pass_near0.5",
                "dataset_id": ds,
                "architecture": "gru",
                "protocol": make_protocol(ds, learning_rate=0.001,
                                          sampling="full_pass",
                                          feature_recipe="degradation_v1",
                                          near_weight=0.5),
            })

    return configs


# ---------------------------------------------------------------------------
# Single training run
# ---------------------------------------------------------------------------

def run_one(cfg: dict, results_dir: Path) -> dict:
    """Train one config and return a result dict."""
    label = cfg["label"]
    dataset_id = cfg["dataset_id"]
    architecture = cfg["architecture"]
    training_protocol = cfg["protocol"]

    # Thread count: train_v2 forces 1; set explicitly to avoid inheriting parent's count.
    import torch
    torch.set_num_threads(1)

    t0 = time.monotonic()
    result: dict[str, Any] = {
        "label": label,
        "dataset_id": dataset_id,
        "architecture": architecture,
        "feature_recipe": training_protocol["feature_recipe"],
        "learning_rate": training_protocol["learning_rate"],
        "sampling": training_protocol["sampling"],
        "near_weight": training_protocol.get("near_weight", 0.0),
        "max_epochs_limit": training_protocol["max_epochs"],
        "status": "pending",
    }

    try:
        from pdm.train import run_training

        log_lines: list[str] = []

        def log(msg: str) -> None:
            log_lines.append(msg)
            print(f"[{label}] {msg}", flush=True)

        rec = run_training(
            dataset_id,
            architecture=architecture,
            training_protocol=training_protocol,
            device_pref="auto",
            log=log,
        )

        elapsed = time.monotonic() - t0
        run_id = rec["run_id"]
        best_epoch = rec.get("best_epoch", 0)
        best_metric = rec.get("best_metric", float("nan"))

        # Load full validation metrics from the run directory.
        val_metrics: dict[str, Any] = {}
        val_path = dataset_runs(dataset_id) / run_id / "validation_metrics.json"
        if val_path.exists():
            val_metrics = json.loads(val_path.read_text())

        result.update({
            "status": rec.get("status", "completed"),
            "run_id": run_id,
            "best_epoch": best_epoch,
            "best_metric": round(best_metric, 6),
            "elapsed_s": round(elapsed, 1),
            "selection_metric_name": val_metrics.get("selection_metric_name", ""),
            "selection_metric_unit": val_metrics.get("selection_metric_unit", ""),
            "val_mae_events": round(val_metrics.get("last", {}).get("val_mae_events") or float("nan"), 4)
                              if val_metrics else float("nan"),
            "n_val_event_units": val_metrics.get("last", {}).get("n_val_event_units"),
            "log_tail": log_lines[-3:],
        })
        print(f"[{label}] DONE  best_epoch={best_epoch}  best_metric={best_metric:.6f}  elapsed={elapsed:.1f}s",
              flush=True)

    except Exception as exc:
        elapsed = time.monotonic() - t0
        result.update({
            "status": "failed",
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "elapsed_s": round(elapsed, 1),
        })
        print(f"[{label}] FAILED: {exc}", flush=True)

    # Persist individual result.
    safe_label = label.replace("/", "__")
    result_path = results_dir / f"{safe_label}.json"
    result_path.write_text(json.dumps(result, indent=2, default=str))
    return result


# ---------------------------------------------------------------------------
# Parallel sweep runner
# ---------------------------------------------------------------------------

def run_sweep(configs: list[dict], workers: int, results_dir: Path) -> list[dict]:
    results_dir.mkdir(parents=True, exist_ok=True)
    all_results: list[dict] = []

    print(f"\nRunning {len(configs)} configs with {workers} parallel workers…\n")
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(run_one, cfg, results_dir): cfg for cfg in configs}
        for future in concurrent.futures.as_completed(futures):
            cfg = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                result = {
                    "label": cfg["label"],
                    "status": "failed",
                    "error": str(exc),
                }
            all_results.append(result)

    # Sort by dataset_id, then by best_metric (lower is better).
    all_results.sort(key=lambda r: (
        r.get("dataset_id", ""),
        r.get("best_metric", float("inf")) if r.get("status") != "failed" else float("inf"),
    ))
    return all_results


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def format_metric(val: Any, unit: str) -> str:
    if val is None or (isinstance(val, float) and (val != val)):  # nan check
        return "—"
    if unit == "seconds":
        v = float(val)
        return f"{v:.1f} s ({v / 60:.2f} min)"
    if unit == "NLL":
        return f"{float(val):.6f}"
    return str(val)


def build_report(results: list[dict], datasets_used: list[str],
                 sweep_protocol: dict, report_path: Path) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime())
    lines: list[str] = [
        "# Hyperparameter Sweep Report",
        "",
        f"_Generated: {ts}_",
        "",
        "## Overview",
        "",
        "Abbreviated training-v2 sweep across GRU / LSTM × feature recipes × learning rates × sampling strategies.",
        f"Each run uses `max_epochs={sweep_protocol['max_epochs']}`, `patience={sweep_protocol['patience']}`,",
        f"`min_epochs={sweep_protocol['min_epochs']}` (adaptive early stopping).",
        "Results are **exploratory** — not a full-length study benchmark.",
        "For a rigorous benchmark use `python3 -m pdm training-study`.",
        "",
        "### Datasets",
    ]
    for ds in datasets_used:
        if ds == "bearings":
            lines.append("- **bearings** (XJTU-SY): RUL prediction. Selection metric = near-30-min equal-weight unit MAE (seconds, lower is better).")
        else:
            lines.append("- **filters** (HSE): Censored survival. Selection metric = unit-equal val Weibull NLL (lower is better).")
    lines.append("")

    for ds in datasets_used:
        ds_results = [r for r in results if r.get("dataset_id") == ds]
        if not ds_results:
            continue

        sel_unit = ds_results[0].get("selection_metric_unit", "")
        metric_label = "val MAE (s)" if ds == "bearings" else "val NLL"
        lines += [
            f"## {ds.title()} Results",
            "",
            f"Selection metric: **{metric_label}** (lower is better).",
            "",
        ]

        # Winner table
        ok = [r for r in ds_results if r.get("status") == "completed"]
        failed = [r for r in ds_results if r.get("status") == "failed"]

        if ok:
            ok_sorted = sorted(ok, key=lambda r: r.get("best_metric", float("inf")))
            header = "| Rank | Config label | Arch | Feature recipe | LR | Sampling | Near-wt | Best epoch | " + metric_label + " | Elapsed |"
            sep = "|------|-------------|------|---------------|-----|----------|---------|------------|" + "-" * (len(metric_label) + 2) + "|---------|"
            lines += [header, sep]
            for rank, r in enumerate(ok_sorted, 1):
                metric_str = format_metric(r.get("best_metric"), sel_unit)
                elapsed = r.get("elapsed_s", 0)
                elapsed_str = f"{elapsed:.0f}s" if elapsed < 60 else f"{elapsed/60:.1f}min"
                label = r["label"].replace(f"{ds}/", "")
                lines.append(
                    f"| {rank} | `{label}` | {r.get('architecture','')} | {r.get('feature_recipe','')} "
                    f"| {r.get('learning_rate','')} | {r.get('sampling','')} "
                    f"| {r.get('near_weight', 0.0)} | {r.get('best_epoch','')} "
                    f"| {metric_str} | {elapsed_str} |"
                )

            winner = ok_sorted[0]
            lines += [
                "",
                f"### Winner — {ds.title()}",
                "",
                f"**Config:** `{winner['label']}`",
                "",
                "| Field | Value |",
                "|-------|-------|",
                f"| Architecture | {winner.get('architecture','')} |",
                f"| Feature recipe | {winner.get('feature_recipe','')} |",
                f"| Learning rate | {winner.get('learning_rate','')} |",
                f"| Sampling | {winner.get('sampling','')} |",
                f"| Near weight | {winner.get('near_weight', 0.0)} |",
                f"| Best epoch | {winner.get('best_epoch','')} |",
                f"| {metric_label} | {format_metric(winner.get('best_metric'), sel_unit)} |",
                f"| Run id | `{winner.get('run_id','')}` |",
                "",
                "**Reproduce command:**",
                "",
                "```bash",
            ]
            arch = winner.get("architecture", "gru")
            recipe = winner.get("feature_recipe", "base_v1")
            lr = winner.get("learning_rate", 0.001)
            samp = winner.get("sampling", "unit_replacement")
            nw = winner.get("near_weight", 0.0)
            protocol_flag = "--protocol adaptive"
            nw_flag = f" --near-weight {nw}" if nw > 0 else ""
            lines.append(
                f"python3 -m pdm train --dataset {ds} --arch {arch} {protocol_flag} "
                f"--feature-recipe {recipe} --learning-rate {lr} --sampling {samp}{nw_flag}"
            )
            lines.append("```")
            lines.append("")

        if failed:
            lines += ["### Failed runs", ""]
            for r in failed:
                lines.append(f"- `{r.get('label','')}`: {r.get('error','')}")
            lines.append("")

    # Limitations section
    lines += [
        "## Limitations and Notes",
        "",
        f"1. **Epoch budget**: Each run capped at `max_epochs={sweep_protocol['max_epochs']}` (production study uses 100).",
        "   Rankings may shift with longer training, especially for LSTM which often converges slower.",
        "2. **Single seed**: All runs use seed=42. Add `--seed 43 --seed 44` confirmation runs before wiring a default.",
        "3. **Reservoir architectures** (`fly_connectome_reservoir`, `random_reservoir`) require MaleCNS feather",
        "   (`malemcns_present: false` on this VM) and were not included in the sweep.",
        "4. **No test evaluation**: Metrics here are validation split only. Run `python3 -m pdm evaluate` with",
        "   `--split test` on the winner to get held-out test scores.",
        "5. **Filter censoring**: Only 5 observed 600 Pa events in the training set; NLL is the primary metric.",
        "   Val MAE is reported only on the small number of validation units that did cross 600 Pa.",
        "6. **CPU-only**: `train_v2` sets `torch.set_num_threads(1)` for reproducibility. Times on GPU will be faster.",
        "",
        "## Raw Results JSON",
        "",
        f"Individual result files are stored under `{report_path.parent.name}/sweep_results/`.",
        "Run `python3 -m pdm evaluate --dataset <ds> --run-id <run_id>` on any run to get a full evaluation.",
    ]

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nReport written to {report_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Hyperparameter sweep for PDM training")
    parser.add_argument("--datasets", nargs="+", choices=["bearings", "filters", "both"],
                        default=["filters"], help="Datasets to sweep (default: filters)")
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of parallel training processes (default: 4)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print configs and exit without training")
    parser.add_argument("--results-dir", default=None,
                        help="Directory for per-config JSON results")
    args = parser.parse_args()

    datasets: list[str] = []
    if "both" in args.datasets:
        datasets = ["bearings", "filters"]
    else:
        datasets = list(args.datasets)

    # Check which datasets are actually ready.
    ready: list[str] = []
    for ds in datasets:
        try:
            from pdm.data.prepare import load_processed
            load_processed(ds)
            ready.append(ds)
            print(f"Dataset {ds!r}: READY")
        except Exception as exc:
            print(f"Dataset {ds!r}: NOT READY ({exc}) — skipping")

    if not ready:
        print("No datasets are ready. Run `python3 -m pdm prepare --dataset <ds>` first.")
        return 1

    configs = build_configs(ready)
    print(f"\n{len(configs)} configs for datasets {ready}:")
    for c in configs:
        p = c["protocol"]
        print(f"  {c['label']:60s}  epochs≤{p['max_epochs']}  patience={p['patience']}")

    if args.dry_run:
        return 0

    results_dir = Path(args.results_dir) if args.results_dir else (
        Path(__file__).parent.parent / "docs" / "sweep_results"
    )
    all_results = run_sweep(configs, workers=min(args.workers, len(configs)), results_dir=results_dir)

    report_path = Path(__file__).parent.parent / "docs" / "hyperparameter_sweep_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    build_report(all_results, ready, SWEEP_PROTOCOL_OVERRIDES, report_path)

    # Summary to stdout
    print("\n=== FINAL RANKINGS ===")
    for ds in ready:
        ds_ok = sorted(
            [r for r in all_results if r.get("dataset_id") == ds and r.get("status") == "completed"],
            key=lambda r: r.get("best_metric", float("inf")),
        )
        print(f"\n{ds.upper()}:")
        for rank, r in enumerate(ds_ok, 1):
            print(f"  #{rank}  {r['label']:60s}  {r.get('best_metric', float('nan')):.6f}  epoch={r.get('best_epoch')}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
