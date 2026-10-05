"""Summarize saved Train/Validation experiments without accessing sensor targets."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from pdm.io_util import atomic_write_json


def main():
    root = Path("output/bearings-learned-funnel-20261002")
    rows = []
    for directory in sorted((root / "candidates").iterdir()):
        frozen_path = directory / "frozen_candidate.json"
        if not frozen_path.is_file():
            continue
        frozen = json.loads(frozen_path.read_text())
        config = frozen["config"]
        metrics_path = directory / "validation_metrics.json"
        metrics = json.loads(metrics_path.read_text()) if metrics_path.is_file() else None
        status_path = directory / "status.json"
        status = json.loads(status_path.read_text()) if status_path.is_file() else {}
        failure_path = directory / "failure.json"
        engine = frozen.get("engine_id")
        candidate_config = root / f"candidate-{directory.name}.json"
        if engine is None and candidate_config.is_file():
            engine = json.loads(candidate_config.read_text())["engine_id"]
        row = {
            "candidate": directory.name,
            "engine": engine,
            "stage": "failed" if failure_path.is_file() else status.get("stage", "unknown"),
            "evaluation_partition": "Validation" if metrics else None,
            "test_access": frozen["test_access"],
            "test_feedback": False,
            "feature_decode_policy": "Parquet-filtered Train/Validation only" if frozen.get("feature_read_scope") else
                "Whole verified snapshot decoded; Test rows removed before preprocessing/fitting/selection/evaluation",
            "config": config,
            "released_to_application": False,
            "coverage_guarantee": False,
        }
        if metrics:
            score = next((h for h in metrics["horizons"] if h["horizon_s"] == 1800), None)
            row["validation_30min"] = score
            if score:
                row["provisional_funnel_gate_passed"] = (
                    score["whole_path_coverage"] >= .9 and score["mean_width_g"] <= 1.5
                    and score["interval_score_skill"] > 0
                )
        rows.append(row)
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scope": "Train-only fitting, reused Validation selection/evaluation; Test targets never used for fitting, selection or diagnostics",
        "comparable_architectures": False,
        "comparison_limitation": "Candidates differ in input features, objective, parameterization and window caps",
        "quality_targets": {
            "nominal_whole_path_coverage": .9,
            "provisional_mean_30min_width_g": 1.5,
            "width_target_source": "internal engineering comparison, not user requirement",
            "warning_policy": "probability>=.5, unconditional finite90% corridor<=900s, observed conservative lead>=600s",
        },
        "application_release": "none; forecast and warning quality remain unresolved",
        "candidates": rows,
    }
    atomic_write_json(root / "candidate_register.json", summary)
    lines = [
        "# Bearings learned trajectory experiments", "", summary["scope"], "",
        "No candidate has been released. A trained distribution and passing engineering checks do not establish useful forecast or warning quality.", "",
        "Configurations differ; this table cannot isolate an architecture effect. The 1.5g width cap is an internal provisional comparison target, not a requirement supplied by the user.", "",
        "Historical candidate loaders decoded the whole verified snapshot, then removed Test rows before all preprocessing, fitting, selection and evaluation. Future candidate loading filters Train/Validation at Parquet decoding. Older Validation reports have a generic reused-Test status-label error; their actual input/evaluation partition is Validation. Original saved reports are retained unchanged.", "",
        "| Candidate | Engine | 30min whole-path coverage | Mean width (g) | Interval-score skill | Useful RED events |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        score = row.get("validation_30min")
        if score:
            cells = [f"{100 * score['whole_path_coverage']:.2f}%", f"{score['mean_width_g']:.3f}",
                     f"{score['interval_score_skill']:.3f}", str(score["events_with_useful_warning"])]
        else:
            cells = [row["stage"], "—", "—", "—"]
        lines.append(f"| {row['candidate']} | {row['engine']} | {' | '.join(cells)} |")
    lines += ["", "Full MaleCNS actual saved graph/reload checks are recorded separately in `qa/actual_full_learned_report.md`. They establish engineering integrity, not forecast quality.", "",
              "The GRU08 path-prefix ablation reduced width but did not resolve transition misses. The GRU09 capacity ablation and GRU10 event-prefix ablation also failed the fixed criteria. Controlled comparisons are recorded in `review/horizon_capacity_ablation_results.md`. GRU11 physical-group batches reduced selected-stage clipping without improving practical quality (`review/balanced_batch_ablation_results.md`). GRU12 separate-phase covariance improved the measured 30-minute tradeoff but failed practical criteria, with a covariance-capacity confound (`review/phase_covariance_ablation_results.md`). GRU13 faster event-head optimization narrowed bands with severe undercoverage and zero useful warnings (`review/event_head_rate_ablation_results.md`). GRU14 finite-horizon timing components and separate horizon survival retained approximately the GRU12 path tradeoff and zero useful warnings (`review/finite_horizon_event_ablation_results.md`). The completed fixed event-only diagnostic (`review/event_only_report.md`) still produces zero useful warnings at its Validation-selected checkpoint; it does not establish sensor identifiability or impossibility. Test remains outside selection."]
    lines += ["", "GRU15 changes only training/selection path draws from 32 to 128 relative to GRU14, retaining 256 inference paths and identical source/population controls. Its common 30-minute inference coverage increased to 79.20%, with mean width 4.144g and zero useful warnings; the fixed criteria still fail (`review/training_sample_count_ablation_results.md`). Raw joint selection scores use different finite-sample band statistics and are not comparable quality scores."]
    lines += ["", "GRU16 expands GRU15 to three timing-associated path heads (2,452,738 parameters), retaining its sample counts and data controls. Coverage fell to 65.65%, mean width increased to 4.354g, and useful RED warnings remained zero; the combined family/capacity intervention failed the fixed criteria (`review/coupled_path_timing_ablation_results.md`). GRU17 therefore uses GRU15 as its single-path control for a variable causal-context comparison with unchanged early-origin eligibility."]
    lines += ["", "GRU17 retains every early origin and the GRU15 scaler anchors/common inference draw identity while extending causal recurrent context from eight to at most 32 observations, with the same 833,458 parameters. Coverage fell to 64.49%, width narrowed to 3.597g and useful warnings remained zero (`review/variable_causal_context_ablation_results.md`); it fails the unchanged criteria. The Train-only temporal noise diagnostic supports a later post-factor regularization experiment, not a quality claim or data-impossibility conclusion (`review/temporal_path_noise_interpretation.md`)."]
    (root / "candidate_study.md").write_text("\n".join(lines) + "\n")
    print(f"Summarized {len(rows)} saved candidates; no sensor targets accessed")


if __name__ == "__main__":
    main()
