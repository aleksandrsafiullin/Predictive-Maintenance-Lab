"""Source/task capabilities and lazy display providers for shared project pages."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from pdm.probabilistic import workflow as flow
from pdm.probabilistic.contract import SPLITS

LEGACY_PARTS = (("train","Training Data"),("validation","Validation Data"),("test","Testing Data"))
CALIBRATED_PARTS = (("train","Training Data"),("validation","Validation Data"),("calibration","Calibration Data"),("test","Testing Data"))


def import_profile(project):
    return {"task":flow.TASK,"parts":CALIBRATED_PARTS,"calibrated":True,"signal_editable":True,
            "defaults":{"train":55,"validation":15,"calibration":15,"test":15}}


def signal_defaults(project):
    if project.get("active_snapshot_id"):
        from pdm.project_snapshot import load_project_limits, project_snapshot
        view = project_snapshot(project["project_id"])
        schema = view["schema"]
        if not schema.get("cadence_s"):
            from pdm.project_snapshot import limits_features
            times = limits_features(project["project_id"],view["snapshot_id"]).groupby("unit_id").timestamp_s.diff()
            schema = {**schema,"cadence_s":float(times[times > 0].median()) if (times > 0).any() else 1.}
        return {**schema,"thresholds":load_project_limits(project["project_id"],view["snapshot_id"]) or schema.get("thresholds",{})}
    defaults = {"xjtu_bearings":("combined_rms","Combined max-axis RMS","g",60.),
                "hse_filters":("differential_pressure","Differential pressure","Pa",.1),
                "synthetic_sanity":("vibration_rms_g","Vibration RMS","g",60.),
                "synthetic_benchmark":("vibration_rms_g","Vibration RMS","g",60.)}
    column,label,unit,cadence = defaults.get(project["source_kind"],("signal","Signal","unit",1.))
    return {"signal_column":column,"signal_label":label,"signal_unit":unit,"cadence_s":cadence}


def source_defaults(project):
    source = project.get("source_manifest") or {}
    if source.get("source_plan"):
        return source["source_plan"]
    if project["source_kind"] not in flow.SYNTHETIC_KINDS:
        return {}
    root = Path(source.get("source_root") or flow.DEFAULT_SOURCE)
    if (root/"data").is_dir():
        root = root/"data"
    suite = flow.SYNTHETIC_KINDS[project["source_kind"]]
    return {"primary":{"mode":"folder","path":str(root/suite/"sensor_csv"/"train")},
            **{f"{part}_mode":"folder" for part in SPLITS[1:]},
            **{part:{"mode":"folder","path":str(root/suite/"sensor_csv"/part)} for part in SPLITS[1:]},
            "weights":{part:value/100 for part,value in import_profile(project)["defaults"].items()},"seed":42,
            "manifest_path":source.get("manifest_path") or str(flow.DEFAULT_ARCHIVE/"data"/"dataset_manifest.json")}


class SensorQualityProvider:
    """One lazy observed-display contract for sensor and historical snapshots."""
    capabilities = {"events":True,"membership":False,"thresholds":True,"legacy_admission":False,"preserve_order":True,"table":True}

    def __init__(self,project):
        from pdm.project_snapshot import project_snapshot
        self.project = project
        self.view = project_snapshot(project["project_id"])
        self.snapshot = self.view.get("sensor_snapshot")
        self.capabilities = {**self.capabilities,"membership":True}

    @property
    def notice(self):
        if not self.snapshot:
            return "Historical three-role data · reimport four independent roles for calibrated training."
        status = self.snapshot["evaluation_status"]
        return "Original distribution · exposed reference data." if status == "reference_exposed" else "Custom distribution · development data; source already exposed." if status == "development_reference_exposed" else "Development data · user-supplied sources; independent quality has not been established."

    def metadata(self,part):
        if self.snapshot:
            report = self.snapshot["admission"].get(part,{})
            return {"units":report.get("units",0),"rows":report.get("accepted_rows",0),"gaps":report.get("gaps",0),"missing_signal":report.get("rejected_rows",0),"time_start":None}
        from pdm.project_quality_ui import part_summary
        return part_summary(self.view["features"],self.view["split"],part)

    def unit_ids(self,part):
        return self.view["split"].get(part,[])

    def frame(self,part,unit_id):
        from pdm.project_snapshot import role_frame
        return role_frame(self.view,part,unit_id)

    def details(self,part,unit_id):
        from pdm.project_snapshot import descriptive_frame
        return descriptive_frame(self.view,part,unit_id)

    def limits_features(self):
        from pdm.project_snapshot import limits_features
        return limits_features(self.project["project_id"],self.view["snapshot_id"])

    def ready(self):
        if not self.snapshot or not self.unit_ids("calibration"):
            return False,"Reimport four independent roles: this historical snapshot has no Calibration data."
        reports = self.snapshot["admission"]
        ready = all(reports[part]["common_history_origins"] > 0 for part in ("train","validation"))
        return ready,"Training and Validation each need a continuous valid history of 60 observations."


def quality_provider(project):
    return SensorQualityProvider(project)


ENGINE_LABELS = {"persistence":"Persistence","local_trend":"Local trend","quantile_boosting":"Quantile boosting","gru":"GRU"}


def run_label(row):
    cfg = row["config"]
    engine = "Boosting" if cfg.get("forecast_mode") == "bounded_trend_v1" and row["engine_id"] == "quantile_boosting" else ENGINE_LABELS[row["engine_id"]]
    label = f"{engine} · {cfg['history_length']} observations"
    if row["engine_id"] == "gru":
        label += f" · {cfg['hidden_size']} hidden units"
    if cfg.get("forecast_mode") == "bounded_trend_v1":
        label += " · Trend corridor"
    from datetime import datetime
    created = datetime.fromisoformat(row["created_at"].replace("Z","+00:00")).isoformat(sep=" ",timespec="milliseconds")
    return f"{label} · seed {cfg['seed']} · {created[:-6]}"


def historical_signal_metadata(project_id, run_id):
    """Verify saved metadata/bytes without admitting a numerical replay contract."""
    from pdm.data.project_prepare import load_snapshot
    from pdm.io_util import read_json, sha256_file
    from pdm.projects import project_store
    from pdm.signal_training import _digest
    directory = project_store().run_path(project_id,run_id)
    if directory.is_symlink() or (directory/"manifest.json").is_symlink():
        raise ValueError("Historical saved model path is unsafe")
    row = read_json(directory/"manifest.json")
    if (row.get("project_id"),row.get("run_id"),row.get("task"),row.get("status")) != (project_id,run_id,"signal_forecast","completed"):
        raise ValueError("Historical saved model identity does not match this project")
    artifacts = row.get("artifacts",{})
    if not artifacts or row.get("artifact") not in artifacts or "training_contract.json" not in artifacts:
        raise ValueError("Historical saved model artifacts are missing")
    for name,digest in artifacts.items():
        path = directory/name
        if Path(name).name != name or path.is_symlink() or sha256_file(path) != digest:
            raise ValueError("Historical saved model artifact integrity mismatch")
    contract = read_json(directory/"training_contract.json")
    for key in ("project_id","snapshot_id","engine_id","params","schema","snapshot_fingerprint_sha256","scaler"):
        if contract.get(key) != row.get(key):
            raise ValueError(f"Historical saved model {key} disagrees with its training contract")
    for key in ("corridor_contract","funnel","connectome"):
        if row.get(key) != contract.get(key):
            raise ValueError(f"Historical saved model {key} disagrees with its training contract")
    snapshot = load_snapshot(project_id,row["snapshot_id"])
    if _digest(snapshot["fingerprint"]) != row.get("snapshot_fingerprint_sha256") or snapshot["schema"] != row.get("schema"):
        raise ValueError("Historical saved model snapshot binding mismatch")
    return row


class CalibratedTaskProvider:
    """Backend-only capabilities for the actual common Training/Results pages."""
    task = flow.TASK
    labels = {**ENGINE_LABELS,"quantile_boosting":"Boosting"}
    description = "Trend corridor predicts a moving center from observed history. Its full relative width is fixed at the saved budget; independent Calibration measures containment without widening bounds or guaranteeing coverage."
    history_max = 60
    fixed_span = False
    span_options = [300.,900.,1800.,3600.]
    display_anchor = False

    def __init__(self,project):
        self.project = project
        self.project_id = project["project_id"]
        self.quality = SensorQualityProvider(project)
        self.view = self.quality.view
        self.active_snapshot = self.quality.snapshot
        self._horizon_profile = None
        self._horizon_error = None
        # Navigation and unrelated pages need only metadata. Train capacity is
        # derived lazily by Training/defaults; Results uses the selected run.
        cfg = self.active_snapshot["config"] if self.active_snapshot else {
            "cadence_s":self.view["schema"].get("cadence_s") or 1.,
            "report_horizons":[5,15,30,60]}
        self.span_options = [cfg["cadence_s"]*h for h in cfg["report_horizons"]]
        self.full_span_supported = False
        self.full_span_default = False

    def runs(self,historical=False):
        self.run_errors = []
        rows = flow.list_runs(self.project_id,current_snapshot=not historical)
        # Listing alone never admits an unverifiable saved model.
        for row in rows:
            flow.load_run(self.project_id,row["run_id"],require_current=not historical)
            flow.snapshot_for_project(self.project_id,row["snapshot_id"])
        if historical:
            from pdm.red_entry_training import list_red_entry_runs, load_red_entry_run
            from pdm.signal_training import list_project_runs, load_signal_run
            for row in list_project_runs(self.project_id)+list_red_entry_runs(self.project_id):
                try:
                    if row["task"] == "signal_forecast":
                        row = historical_signal_metadata(self.project_id,row["run_id"])
                    from pdm.project_snapshot import project_snapshot
                    project_snapshot(self.project_id,row["snapshot_id"])
                except (OSError,ValueError,KeyError,RuntimeError) as exc:
                    self.run_errors.append(str(exc))
                    continue
                try:
                    (load_red_entry_run if row["task"] == "red_entry" else load_signal_run)(self.project_id,row["run_id"])
                except (OSError,ValueError,KeyError,RuntimeError) as exc:
                    if row["task"] == "signal_forecast":
                        version = (row.get("corridor_contract") or {}).get("version")
                        detail = f"Saved corridor version {version}: {exc}" if version is not None else str(exc)
                        rows.append({**row,"replay_status":"unavailable","replay_error":detail})
                    else:
                        self.run_errors.append(str(exc))
                else:
                    rows.append(row)
        return rows

    def engines(self):
        ready,reason = self.quality.ready()
        if ready and self.active_snapshot["config"].get("observed_profile"):
            from pdm.data.project_import import validate_absolute_thresholds
            from pdm.project_snapshot import load_project_limits
            try:
                validate_absolute_thresholds(load_project_limits(self.project_id,self.view["snapshot_id"]) or self.active_snapshot["config"].get("imported_rule"))
            except ValueError:
                ready,reason = False,"Save absolute Yellow and Red limits on Data Quality before training."
        if ready and self.horizon_profile() is None:
            ready,reason = False,self._horizon_error
        return [{"engine_id":engine,"label":label,"available":ready,"reason":None if ready else reason} for engine,label in self.labels.items()]

    def horizon_profile(self):
        """Choose a new grid from admitted Train observations only."""
        if self._horizon_profile is None and self._horizon_error is None and self.active_snapshot:
            from pdm.probabilistic.data import load_split
            from pdm.probabilistic.horizons import train_horizon_profile
            try:
                self._horizon_profile = train_horizon_profile(load_split(self.active_snapshot,"train"),self.active_snapshot["config"])
            except ValueError as exc:
                self._horizon_error = str(exc)
        return self._horizon_profile

    @property
    def training_span_max_s(self):
        profile = self.horizon_profile()
        return profile["span_s"] if profile else 0.

    @property
    def horizon_notice(self):
        profile = self.horizon_profile()
        if not profile:
            return ("Forecast span unavailable: no future target horizon is supported. " + self._horizon_error
                    if self._horizon_error else "Reimport four roles before choosing a calibrated forecast horizon.")
        return (f"Train supports up to {profile['horizon']} direct future observations "
                f"({profile['span_s']/60:g} minutes). Longer Validation endpoints may have limited or unknown support.")

    def defaults(self,engine):
        from pdm.probabilistic.contract import default_config
        cfg = default_config(**{**(self.active_snapshot["config"] if self.active_snapshot else {}),"engine_id":engine})
        if not self.active_snapshot:
            schema = self.view["schema"]
            frame = self.quality.limits_features()
            times = frame.groupby("unit_id").timestamp_s.diff()
            cadence = float(times[times > 0].median()) if (times > 0).any() else 1.
            valid = frame.signal[np.isfinite(frame.signal) & frame.signal.gt(0)]
            floor = .02*float(valid.median()) if not valid.empty else 1.
            cfg = default_config(**{**cfg,"target":schema["signal_column"],"schema":["unit_id","timestamp_s",schema["signal_column"]],"unit":schema["signal_unit"],"cadence_s":cadence,"horizons_s":[cadence*h for h in range(1,61)],"scale_floor":floor})
        profile = self.horizon_profile()
        if profile:
            from pdm.probabilistic.contract import trend_config
            cfg = trend_config(cfg,profile["horizon"])
        return {**cfg,"epochs":cfg["max_epochs"],"max_iter":cfg["boosting_max_iter"]}

    def training_job(self,engine,params):
        from pdm.probabilistic.contract import trend_config
        profile = self.horizon_profile()
        if not profile:
            raise ValueError(self.horizon_notice)
        cadence = self.active_snapshot["config"]["cadence_s"]
        requested = float(params["horizons_s"][-1])/cadence
        H = int(round(requested))
        if not np.isclose(requested,H,atol=1e-6,rtol=0) or not 1 <= H <= profile["horizon"]:
            raise ValueError("Forecast span must stay within the admitted Train horizon capacity")
        grid = trend_config(self.active_snapshot["config"],H)
        if "training_population_protocol" in params:
            grid = trend_config({**grid,"training_population_protocol":params["training_population_protocol"]},H)
        translated = {"history_length":params["history_length"],"seed":params["seed"],
                      **{key:grid[key] for key in ("horizon_protocol","max_horizon","horizons_s","report_horizons")},
                      **{key:grid[key] for key in ("forecast_mode","center_protocol","corridor_protocol","training_population_protocol") if key in grid},
                      **{key:grid[key] for key in ("quantiles","nominal","output_kind","output_slots","coverage_guarantee")},
                      "decision_quality_policy":flow.TREND_QUALITY_POLICY,
                      "training_horizon_profile":profile,
                      **{key:params[key] for key in ("width_budget","scale_floor") if key in params}}
        from pdm.probabilistic.contract import canonical_hash
        from pdm.project_snapshot import load_project_limits
        rule = load_project_limits(self.project_id,self.view["snapshot_id"])
        if rule or self.active_snapshot and self.active_snapshot["config"].get("observed_profile"):
            from pdm.data.project_import import validate_absolute_thresholds
            rule = validate_absolute_thresholds(rule or self.active_snapshot["config"].get("imported_rule"))
            translated.update(zone_rule=rule,zone_rule_hash=canonical_hash(rule))
        if engine == "gru":
            translated.update(hidden_size=params["hidden_size"],batch_size=params["batch_size"],max_epochs=params["epochs"],learning_rate=params["learning_rate"])
        elif engine == "quantile_boosting":
            translated["boosting_max_iter"] = params["max_iter"]
        return {"task":self.task,"snapshot_id":self.project["active_snapshot_id"],"engine_id":engine,"params":translated}

    def training_summary(self,run):
        loss = run["training_summary"].get("best_validation_loss")
        bounded = run["config"].get("forecast_mode") == "bounded_trend_v1"
        metrics = [("Validation center loss" if bounded else "Validation loss","—" if loss is None else f"{loss:.6f}"),("Best epoch",run["training_summary"].get("best_epoch") or "—")]
        if bounded:
            controls = run.get("validation_controls",{})
            containment = controls.get("all_validation",{}).get("point_coverage")
            status = controls.get("usefulness_diagnostic",{}).get("status")
            metrics.extend([("Validation point containment","—" if containment is None else f"{containment:.1%}"),
                            ("Useful forecast","Target met · exploratory" if status == "validation_target_met" else "Unknown" if status == "validation_quality_unknown" or status is None else "Not demonstrated")])
        support = run["training_summary"].get("support_by_lead")
        if support:
            metrics.extend((f"{part.title()} units · final lead",support[part]["physical_unit_count"][-1]) for part in ("train","validation"))
        return metrics

    def training_actions(self,run):
        rows = flow.list_calibrations(self.project_id,run["run_id"])
        bounded = run["config"].get("forecast_mode") == "bounded_trend_v1"
        caption = ("Calibrate on the independent Calibration split. Model weights and median forecasts stay fixed. "
                   "Each saved forecast prefix has its own eligible population and finite-rank correction; nominal 90% where available.")
        if bounded:
            caption = "Assess empirical point containment on independent Calibration data. Center and fixed-width bounds remain unchanged; this is not a nominal coverage guarantee."
        support = run["training_summary"].get("support_by_lead")
        if support and not support["validation"]["target_count"][-1]:
            caption += " The final lead has no Validation targets; its validation accuracy is unknown."
        count = rows[-1]["summary"].get("n_by_horizon",{}).get(str(run["config"]["max_horizon"]),rows[-1]["n"]) if rows else None
        if rows and bounded:
            from pdm.probabilistic.corridor import corridor_prefix
            projection = corridor_prefix(rows[-1]["summary"],run["config"]["max_horizon"],run["config"])
            count = projection["n"]
            if count == 0:
                caption += " The saved endpoint has no Calibration observations; tail quality remains unknown."
        if rows and not bounded and rows[-1]["summary"]["c_path"].get(str(run["config"]["max_horizon"])) is None:
            caption += " Calibration is insufficient at the saved maximum; that band remains diagnostic."
        return [{"title":"Calibration","caption":caption,"metrics":[("Calibration units · saved maximum",count)] if rows else [],
                 "actions":[{"label":"Assess saved corridor" if bounded else "Calibrate saved model","key":f"calibrate:{self.project_id}:{run['run_id']}","kind":"probabilistic_calibrate","params":{"run_id":run["run_id"]}}]}]

    @staticmethod
    def run_label(row):
        if row.get("task") != flow.TASK:
            from pdm.project_results_ui import _run_label
            return _run_label(row)+" · historical"+(" · replay unavailable" if row.get("replay_status") == "unavailable" else "")
        return run_label(row)

    def load_run(self,run_id):
        from pdm.red_entry_training import list_red_entry_runs, load_red_entry_run
        from pdm.signal_training import list_project_runs, load_signal_run
        for row in list_project_runs(self.project_id)+list_red_entry_runs(self.project_id):
            if row["run_id"] == run_id:
                return (load_red_entry_run if row["task"] == "red_entry" else load_signal_run)(self.project_id,run_id)
        return flow.load_run(self.project_id,run_id,require_current=False)[0]

    def run_view(self,run):
        if run.get("task") != flow.TASK:
            from pdm.project_snapshot import project_snapshot
            view = project_snapshot(self.project_id,run["snapshot_id"])
            params = run.get("params",{})
            saved_spans = list(params.get("horizons_s") or [1.])
            self.span_editable = params.get("forecast_mode") in {"learned_joint_trajectories","bounded_trend_corridor"}
            self.span_options = saved_spans if self.span_editable else [max(saved_spans)]
            self.full_span_supported = False
            self.full_span_default = False
            return {**view,"schema":run.get("schema",view["schema"])}
        snapshot = flow.snapshot_for_project(self.project_id,run["snapshot_id"])
        cfg = run["config"]
        self.span_options = [cfg["cadence_s"]*h for h in cfg["report_horizons"]]
        self.full_span_supported = True
        self.full_span_default = cfg.get("horizon_protocol") == "dense-v2"
        self.span_editable = True
        return {"project_id":self.project_id,"snapshot_id":snapshot["snapshot_id"],"split":{p:snapshot["split_manifest"][p]["units"] for p in SPLITS},
                "schema":{"signal_column":cfg["target"],"signal_label":cfg.get("signal_label",{"vibration_rms_g":"Vibration RMS"}.get(cfg["target"],cfg["target"])),"signal_unit":cfg["unit"],"thresholds":cfg.get("zone_rule",{"mode":"absolute","direction":cfg.get("threshold_direction","above"),"yellow":cfg["yellow"],"red":cfg["red"]})},"sensor_snapshot":snapshot}

    def calibrations(self,run):
        if run.get("task") != flow.TASK:
            return []
        return flow.list_calibrations(self.project_id,run["run_id"],require_current=False)

    def unit_view(self,view,unit_id):
        if "sensor_snapshot" not in view:
            return view
        from pdm.project_snapshot import role_frame
        return {**view,"features":role_frame(view,"test",unit_id)}

    def replay_defaults(self,run):
        if run.get("task") != flow.TASK:
            return {"cursor":0,"playing":False}
        return {"cursor":run["config"]["history_length"]-1,"playing":False}

    def forecast(self,run_id,unit_id,origin_s,*,prediction_horizon_s=1800.,calibration_id=None,pointwise=False,should_stop=None,progress_cb=None):
        run = self.load_run(run_id)
        if run.get("task") != flow.TASK:
            from pdm.signal_inference import forecast_prefix
            params = run["params"]
            selected_span = prediction_horizon_s
            if params.get("forecast_mode") not in {"learned_joint_trajectories","bounded_trend_corridor"}:
                if not np.isclose(prediction_horizon_s,max(params["horizons_s"]),atol=1e-6,rtol=0):
                    raise ValueError("Historical model uses its fixed saved forecast grid")
                selected_span = None
            return forecast_prefix(self.project_id,run_id,unit_id,origin_s,prediction_horizon_s=selected_span,thresholds=run.get("schema",{}).get("thresholds"),should_stop=should_stop,progress_cb=progress_cb)
        from pdm.probabilistic.evaluation import red_window
        saved = flow.save_forecast(self.project_id,run_id,calibration_id,unit_id,origin_s,should_stop=should_stop)
        manifest,forecast = flow.load_forecast(self.project_id,run_id,saved["forecast_id"])
        cfg = forecast["config"]
        from pdm.probabilistic.calibration import band_availability, calibration_prefix
        H = int(round(prediction_horizon_s/cfg["cadence_s"]))
        if not 1 <= H <= cfg["max_horizon"] or not np.isclose(prediction_horizon_s,H*cfg["cadence_s"],atol=1e-6,rtol=0):
            raise ValueError("Forecast span must use the directly saved observation grid")
        calibrator = flow.load_calibration(self.project_id,run_id,calibration_id,require_current=False)[1] if calibration_id else None
        decision = cfg.get("forecast_mode") == "bounded_trend_v1"
        if decision:
            from pdm.probabilistic.corridor import corridor_prefix, decision_band
            prefix = corridor_prefix(calibrator,H,cfg)
            band = decision_band(np.asarray(forecast["decision_outputs"]),cfg,H,assessment=calibrator,model_hash=run["model_hash"])
        else:
            prefix = calibration_prefix(calibrator,H,cfg)
            band = forecast["bands"][str(prefix["calibration_H"])]
        trajectory_lower = np.asarray(band["trajectory_lower"]).reshape(-1)[:H]
        trajectory_upper = np.asarray(band["trajectory_upper"]).reshape(-1)[:H]
        lower = np.asarray(band["pointwise_lower"]).reshape(-1)[:H] if pointwise else trajectory_lower
        upper = np.asarray(band["pointwise_upper"]).reshape(-1)[:H] if pointwise else trajectory_upper
        median = np.asarray(band["center"] if decision else band["q50"]).reshape(-1)[:H]
        event = red_window({"trajectory_lower":trajectory_lower,"trajectory_upper":trajectory_upper},forecast["past_status"],red=cfg["red"],cadence_s=cfg["cadence_s"],direction=cfg.get("threshold_direction","above"))
        availability = band_availability({**band,"trajectory_lower":trajectory_lower,"trajectory_upper":trajectory_upper},cfg)
        status = event["status"]
        messages = {"already_observed":"The red limit was already observed in the visible history.","first_event_unknown":"First red entry is unknown because the observed history contains a gap.","unavailable_forecast":"Red-entry geometry is unavailable at this time.","no_crossing_in_horizon":"No red-limit crossing within the selected forecast band."}
        message = messages.get(status)
        if not message:
            start = event["earliest_s"]/60
            message = f"Possible first red entry from {start:g} minutes; its latest time is outside this horizon." if status == "open_right" else f"First red-entry geometry: {start:g}–{event['latest_s']/60:g} minutes from Now."
        if decision:
            message = "Decision corridor geometry · no actionable warning. " + message
        elif pointwise:
            message = "Pointwise diagnostic view · no actionable warning. " + message
        elif not availability["accepted"] and status not in {"already_observed","first_event_unknown","unavailable_forecast"}:
            message = "Diagnostic threshold geometry · no actionable warning. " + message
        entry = {"status":"unavailable"}
        if not decision and not pointwise and availability["accepted"] and status == "bounded":
            entry = {"status":"available","earliest_s":origin_s+event["earliest_s"],"latest_s":origin_s+event["latest_s"]}
        context = {"n":band["n"],"width":float(np.mean(trajectory_upper-trajectory_lower)),"wide":availability["wide_interval"],"calibrated":band.get("calibrated"),"red_message":message,"pointwise":pointwise and not decision,"scope":band.get("scope"),"display_H":H,"calibration_H":prefix.get("calibration_H"),"conservative_prefix":prefix["conservative_prefix"],"rank":prefix.get("rank")}
        if decision:
            context.update(output_kind="decision_corridor",coverage_guarantee=False,nominal=None,
                           width_budget=cfg["width_budget"],empirically_assessed=band["empirically_assessed"],
                           empirical_support=prefix,quality_status=run.get("validation_controls",{}).get("usefulness_diagnostic",{}).get("status"),quality_accepted=False)
        return {"project_id":self.project_id,"run_id":run_id,"snapshot_id":manifest["snapshot_id"],"as_of_s":origin_s,
                "display_anchor":False,"points":[{"target_time_s":origin_s+cfg["cadence_s"]*(i+1),"value":float(median[i]),"lower":float(lower[i]),"upper":float(upper[i])} for i in range(H)],
                "thresholds":{"status":"available","direction":cfg.get("threshold_direction","above"),"yellow":cfg["yellow"],"red":cfg["red"]},"red_entry_corridor":entry,
                "funnel":{"mode":"bounded_trend_v1" if decision else "calibrated_signal_trajectory","issued_horizon_s":prediction_horizon_s},
                "band_context":context,
                "saved_forecast":forecast,"forecast_id":saved["forecast_id"]}

    @staticmethod
    def resolve_span(run,selection,remaining_s):
        """Recording extent is a UI limit; it never supplies predictor inputs."""
        cfg = run["config"]
        requested_s = max(0.,float(remaining_s)) if selection == "full_observation" else float(selection)
        if not np.isfinite(requested_s) or requested_s < 0:
            raise ValueError("Forecast display duration must be finite and nonnegative")
        cadence = cfg["cadence_s"]
        requested_H = max(0,int(np.floor((requested_s+cfg.get("clock_tolerance_s",1e-6))/cadence)))
        H = min(requested_H,cfg["max_horizon"])
        return {"requested_H":requested_H,"display_H":H,"display_span_s":H*cadence,
                "history_only":H == 0,"clipped":requested_H > cfg["max_horizon"],
                "saved_span_s":cfg["max_horizon"]*cadence}

    def test_scope(self,run):
        if run.get("task") != flow.TASK:
            return "Historical Test · saved task"
        snapshot = flow.snapshot_for_project(self.project_id,run["snapshot_id"])
        if run["config"].get("zone_rule_hash"):
            cfg = snapshot["config"]
            original = {"mode":"absolute","direction":cfg.get("threshold_direction","above"),"yellow":cfg["yellow"],"red":cfg["red"]}
            if run["config"]["zone_rule"] != original:
                return "Development Test · custom saved limits; source already exposed" if snapshot["evaluation_status"] != "development_user_supplied" else "Development Test · custom saved limits"
        return "Exposed reference Test" if snapshot["evaluation_status"] == "reference_exposed" else "Development Test · source already exposed" if snapshot["evaluation_status"] == "development_reference_exposed" else "Development Test · user-supplied data"

    def external_action(self,run,calibration_id,source,*,current):
        """Expose one external action; admission follows its verified physical contract."""
        from pdm.io_util import sha256_file
        from pdm.probabilistic.data import _load_manifest, _source_root
        action = {"label":"Evaluate Stress","key":f"evaluate:{self.project_id}:{run['run_id']}:stress",
                  "kind":"probabilistic_evaluate","disabled":True,"reason":None}
        if not current:
            action["reason"] = "External evaluation is unavailable for a saved model from an earlier active dataset; its saved replay remains available."
            return action
        root = source.get("source_root") or str(flow.DEFAULT_SOURCE)
        manifest_path = Path(source.get("manifest_path") or flow.DEFAULT_ARCHIVE/"data"/"dataset_manifest.json")
        try:
            cfg = run["config"]
            manifest = _load_manifest(_source_root(root),manifest_path,cfg)
            declared = manifest["splits"].get("03_stress",{})
            relative = "03_stress/sensor_csv/test/measurements.csv"
            path = _source_root(root)/relative
            if set(declared) != {"test"} or not path.is_file() or sha256_file(path) != manifest["file_sha256"].get(relative):
                raise ValueError("Missing or unverified external Test-only sensor source")
        except (OSError,ValueError,KeyError,TypeError) as exc:
            action["reason"] = "External evaluation is unavailable: a verified Test-only source must match the saved signal, physical unit and observation cadence. " + str(exc)
            return action
        metadata = manifest_path.parent/"units_EVALUATION_ONLY.csv"
        expected = manifest["file_sha256"].get("units_EVALUATION_ONLY.csv")
        metadata_path = str(metadata) if expected and metadata.is_file() and sha256_file(metadata) == expected else None
        action.update(disabled=False,params={"run_id":run["run_id"],"calibration_id":calibration_id,"protocol":"external",
                                            "source_root":str(root),"manifest_path":str(manifest_path),"suite":"03_stress","metadata_path":metadata_path})
        return action

    def result_sections(self,run,calibration_id,H=30):
        if run.get("task") != flow.TASK:
            return [{"title":"Calibration and evaluation","caption":"Historical saved task · replay is preserved. Reimport four independent roles to train and calibrate a new signal forecast.","actions":[]}]
        import json
        reports = flow.list_evaluations(self.project_id,run["run_id"],require_current=False)
        sections = []
        selected_snapshot = flow.snapshot_for_project(self.project_id,run["snapshot_id"])
        current = run["snapshot_id"] == self.project.get("active_snapshot_id")
        source = self.project.get("source_manifest") or {}
        metadata = source.get("metadata_path")
        if not metadata:
            candidate = Path(source.get("manifest_path") or flow.DEFAULT_ARCHIVE/"data"/"dataset_manifest.json").parent/"units_EVALUATION_ONLY.csv"
            metadata = str(candidate) if candidate.is_file() else None
        if selected_snapshot["evaluation_status"] != "reference_exposed":
            metadata = None  # Original-Test labels cannot describe reassigned development units.
        actions = [{"label":label,"key":f"evaluate:{self.project_id}:{run['run_id']}:{protocol}","kind":"probabilistic_evaluate","disabled":not current,
                    "params":{"run_id":run["run_id"],"calibration_id":calibration_id,"protocol":protocol,"metadata_path":metadata}}
                   for protocol,label in (("reference","Evaluate Test"),("rolling","Evaluate replay"))]
        external = self.external_action(run,calibration_id,source,current=current)
        actions.append(external)
        sections.append({"title":"Evaluation","caption":"Saved bundle is frozen before evaluation. All forecasts, including wide bands, remain in the reports. Historical bundles are replay-only; evaluation actions apply to the current dataset."+(" Mechanism slices are unavailable for development units without verified evaluator metadata." if selected_snapshot["evaluation_status"] != "reference_exposed" else "")+(" " + external["reason"] if external["reason"] else ""),"actions":actions})
        downloads = []
        for report in reports:
            scope = self.test_scope(run) if report["protocol"] == "reference" else "Sequential replay" if report["protocol"] == "rolling" else "Stress" if report["suite"] == "03_stress" else "Fresh confirmation"
            downloads.append({"label":f"Download {scope} summary","data":json.dumps(report["summary"],indent=2,default=str),"filename":f"{scope.lower().replace(' ','-')}.json","key":f"report:{report['evaluation_id']}"})
            for name in ("metrics_by_horizon.csv","metrics_by_lead.csv","slices.csv","rolling_table.csv"):
                path = Path(report["directory"])/name
                if path.exists():
                    downloads.append({"label":f"Download {scope} · {name.replace('_',' ')}","data":path.read_bytes(),"filename":name,"key":f"report:{report['evaluation_id']}:{name}"})
        sections.append({"title":"Detailed reports","downloads":downloads})
        return sections

    def _measured_metric_row(self,run,calibration_id,H):
        if run.get("task") != flow.TASK:
            return None
        import pandas as pd
        reports = [r for r in flow.list_evaluations(self.project_id,run["run_id"],require_current=False) if r["protocol"] == "reference" and r.get("calibration_id") == calibration_id]
        if not reports:
            return None
        table = pd.read_csv(Path(reports[-1]["directory"])/"metrics_by_horizon.csv").set_index("H")
        if H <= 0:
            H = run["config"]["max_horizon"]  # recorded endpoint does not remove run-level quality
        available = sorted(int(h) for h in table.index if int(h) >= H)
        if not available:
            return None
        measured_H = available[0]
        return measured_H,table.loc[measured_H]

    @staticmethod
    def _measured_coverage_available(row):
        return np.isfinite(row.whole_path_coverage) and ("calibrated" not in row or str(row.calibrated).lower() == "true")

    def measured_summary(self,run,calibration_id,H):
        selected = self._measured_metric_row(run,calibration_id,H)
        if selected is None:
            return []
        measured_H,row = selected
        if run["config"].get("forecast_mode") == "bounded_trend_v1":
            containment = f"{row.point_coverage:.1%}" if np.isfinite(row.point_coverage) else "—"
            error = f"{row.mae:.4f} {run['config']['unit']}" if np.isfinite(row.mae) else "—"
            return [("Supported Test units",int(row.supported_units)),(f"Point containment · saved {measured_H*run['config']['cadence_s']/60:g} min",containment),("Center mean absolute error",error)]
        coverage = f"{row.whole_path_coverage:.1%}" if self._measured_coverage_available(row) else "—"
        error = f"{row.mae:.4f} {run['config']['unit']}" if np.isfinite(row.mae) else "—"
        return [("Independent Test units",int(row.units)),(f"Whole-path coverage · saved {measured_H*run['config']['cadence_s']/60:g} min",coverage),("Mean absolute error",error)]

    def measured_summary_caption(self,run,calibration_id,H):
        """Explain the selected saved metric's support without changing its scope."""
        selected = self._measured_metric_row(run,calibration_id,H)
        if selected is None:
            return ""
        _,row = selected
        if run["config"].get("forecast_mode") == "bounded_trend_v1":
            miss = f"{row.miss_rate:.1%}" if np.isfinite(row.miss_rate) else "unavailable"
            return (f"Point containment and center MAE use {int(row.known_targets):,} supported observations across {int(row.supported_units)} of {int(row.units)} issued units, balanced by physical unit. "
                    f"Miss rate: {miss}. Unsupported observations remain unknown; whole-path and RED geometry are separate diagnostics.")
        def count(key):
            value = row.get(key)
            return int(value) if value is not None and np.isfinite(value) else None
        complete,units,partial,no_future = (count(key) for key in ("complete_paths","n","partial_paths","no_future_paths"))
        known = self._measured_coverage_available(row)
        if all(value is not None for value in (complete,units,partial,no_future)):
            population = f"{complete:,} complete trajectories from {units:,} independent units; {partial:,} partial and {no_future:,} no-future paths"
            caption = f"Whole-path coverage uses {population} are excluded." if known else f"Whole-path coverage unavailable: {population}."
            if not known and complete > 0 and "calibrated" in row and str(row.calibrated).lower() != "true":
                caption += " This saved prefix has no calibrated interval."
        else:
            caption = "Complete-path support is not recorded in this saved report." if known else "Whole-path coverage unavailable; complete-path support is not recorded in this saved report."
        targets,supported,issued = (count(key) for key in ("known_targets","supported_units","units"))
        if all(value is not None for value in (targets,supported,issued)):
            caption += f" MAE uses {targets:,} supported observations across {supported:,} of {issued:,} issued units, balanced by physical unit."
        else:
            caption += " MAE uses supported future observations, balanced by physical unit; detailed support counts are not recorded."
        return caption


def task_provider(project):
    return CalibratedTaskProvider(project)
