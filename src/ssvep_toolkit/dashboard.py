"""Portable, dependency-free scientific dashboard generated from the registry."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import time
from urllib.parse import quote

from ssvep_toolkit.registry import ExperimentRegistry
from ssvep_toolkit.progress import latest_progress


METRIC_META = {
    "accuracy": ("Accuracy", "%", 100.0),
    "fused_accuracy": ("Fused accuracy", "%", 100.0),
    "rotating_branch_dropout_accuracy": ("Branch-dropout accuracy", "%", 100.0),
    "feature_count_noise_accuracy": ("Count-noise accuracy", "%", 100.0),
    "robustness_accuracy_drop": ("Robustness accuracy loss", "percentage points", 100.0),
    "neural_window_itr_bits_per_minute": ("Neural-window ITR", "bits/min", 1.0),
    "practical_itr_bits_per_minute": ("Practical ITR", "bits/min", 1.0),
    "decision_seconds": ("Decision window", "s", 1.0),
    "onset_latency_seconds": ("Onset latency", "s", 1.0),
    "practical_overhead_seconds": ("Practical overhead", "s", 1.0),
    "mean_spikes_per_trial": ("Mean spike cost", "spikes/trial", 1.0),
    "spikes_per_correct_selection": ("Spike cost per correct selection", "spikes/correct selection", 1.0),
    "elapsed_seconds": ("Processing time", "s", 1.0),
    "correct": ("Correct trials", "trials", 1.0),
    "trials": ("Trials", "trials", 1.0),
    "classes": ("Class count", "classes", 1.0),
    "optimization_boundary_hit_fraction": ("Encoder parameter boundary-hit rate", "%", 100.0),
    "optimization_l2_boundary_hit_fraction": ("Ridge boundary-hit rate", "%", 100.0),
}


def render_dashboard(database: str | Path, output: str | Path, *,
                     example_directory: str | Path | None = None,
                     experiment_directory: str | Path | None = None) -> Path:
    target = Path(output).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    registry = ExperimentRegistry(database).initialize()
    snapshot = registry.dashboard_snapshot()
    if experiment_directory is None:
        experiment_directory = Path(database).resolve().parent.parent / "experiments"
    data = _compact_snapshot(snapshot)
    data["examples"] = _load_examples(example_directory, relative_to=target.parent)
    data["experiment_progress"] = _experiment_progress(experiment_directory)
    data["encoder_comparison_results"] = _encoder_comparison_results(experiment_directory)
    metric_meta = {
        name: {"label": label, "unit": unit, "scale": scale}
        for name, (label, unit, scale) in METRIC_META.items()
    }
    for metric in data["metrics"]:
        name = str(metric["name"])
        if name.startswith("boundary_rate:") and name not in metric_meta:
            parameter = name.split(":", 1)[1].replace(".", " · ").replace("_", " ")
            metric_meta[name] = {
                "label": f"Boundary selection rate · {parameter}", "unit": "%", "scale": 100.0,
            }
        elif name.startswith("ablation_accuracy:") and name not in metric_meta:
            ablation = name.split(":", 1)[1].replace(":", " · ").replace("_", " ")
            metric_meta[name] = {
                "label": f"Ablation accuracy · {ablation}", "unit": "%", "scale": 100.0,
            }
    data["metric_meta"] = metric_meta
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"), allow_nan=False).replace("</", "<\\/")
    target.write_text(_DASHBOARD_TEMPLATE.replace("__PAYLOAD__", payload), encoding="utf-8")
    return target


def _compact_snapshot(snapshot: dict[str, object]) -> dict[str, object]:
    studies = list(snapshot.get("studies", []))
    study_names = {int(study["id"]): str(study["name"]) for study in studies}
    runs = []
    run_lookup = {}
    for source in snapshot.get("runs", []):
        run_key = str(source["run_key"])
        filter_mode = "causal" if "causal" in run_key.lower() else "offline" if "offline" in run_key.lower() else "unspecified"
        item = {
            "id": int(source["id"]), "study_id": int(source["study_id"]),
            "study": study_names.get(int(source["study_id"]), f"Study {source['study_id']}"),
            "run_key": run_key, "status": source["status"], "subject": source.get("subject_id"),
            "classes": source.get("class_count"), "encoder": source.get("encoder"),
            "filter": filter_mode, "started": source.get("started_utc"), "finished": source.get("finished_utc"),
        }
        runs.append(item); run_lookup[item["id"]] = item
    metrics = []
    for source in snapshot.get("metrics", []):
        value = float(source["value"])
        run = run_lookup.get(int(source["run_id"]))
        if run is None or not math.isfinite(value):
            continue
        metrics.append({
            "run_id": int(source["run_id"]), "name": str(source["name"]), "value": round(value, 10),
            "split": source.get("split") or "unspecified", "step": source.get("step"),
            "study_id": run["study_id"], "study": run["study"], "subject": run["subject"],
            "classes": run["classes"], "encoder": run["encoder"], "filter": run["filter"],
        })
    return {
        "summary": snapshot.get("summary", {}),
        "studies": [{"id": int(item["id"]), "name": item["name"], "created": item.get("created_utc")} for item in studies],
        "runs": runs, "metrics": metrics,
    }


def _load_examples(directory: str | Path | None, *, relative_to: Path,
                   maximum_examples: int = 100, maximum_embedded_traces: int = 20) -> list[dict[str, object]]:
    if directory is None:
        return []
    root = Path(directory)
    if not root.exists():
        return []
    examples = []
    metadata_files = sorted(root.glob("*.json"), key=lambda path: path.stat().st_mtime, reverse=True)[:maximum_examples]
    for index, metadata in enumerate(metadata_files):
        try:
            item = json.loads(metadata.read_text(encoding="utf-8"))
            image = metadata.with_name(str(item["image"])).resolve()
            if not image.exists():
                continue
            relative = Path(os.path.relpath(image, relative_to)).as_posix()
            item["image_url"] = quote(relative, safe="/.:..")
            item["image_bytes"] = image.stat().st_size
            item["units"] = {
                "time": "ms", "raw_uV": "µV", "filtered_uV": "µV", "delta_change_uV": "µV",
                "rf_u": "dimensionless state", "rf_v": "dimensionless state",
                "lif_membrane": "µV-equivalent membrane state", "spikes": "event time (ms)",
            }
            trace_file = metadata.with_suffix(".npz")
            if trace_file.exists() and index < maximum_embedded_traces:
                item["traces"] = _load_trace_preview(trace_file)
            examples.append(item)
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return examples


def _load_trace_preview(path: Path, maximum_points: int = 1200) -> dict[str, object]:
    import numpy as np

    with np.load(path, allow_pickle=False) as data:
        time_ms = np.asarray(data["time_ms"], dtype=float)
        if time_ms.size > maximum_points:
            indices = np.linspace(0, time_ms.size - 1, maximum_points).astype(int)
        else:
            indices = np.arange(time_ms.size)
        result: dict[str, object] = {"time_ms": np.round(time_ms[indices], 4).tolist()}
        for key in ("raw_uV", "filtered_uV", "rf_u", "rf_v", "delta_change_uV", "lif_membrane"):
            if key in data:
                result[key] = np.round(np.asarray(data[key], dtype=float)[indices], 6).tolist()
        for key in ("rf_spikes", "delta_up", "delta_down", "lif_spikes"):
            if key in data:
                spike_indices = np.asarray(data[key], dtype=int)
                spike_indices = spike_indices[(spike_indices >= 0) & (spike_indices < time_ms.size)]
                result[f"{key}_ms"] = np.round(time_ms[spike_indices], 4).tolist()
        return result


def _experiment_progress(directory: str | Path | None) -> list[dict[str, object]]:
    if directory is None:
        return []
    root = Path(directory)
    if not root.exists():
        return []
    planned = {
        "resonate_and_fire_decision_endpoints": 450,
        "resonate_and_fire_fused_reference_search": 150,
        "individual_spike_encoders": 300,
        "nested_multi_encoder_harmonic_aware": 120,
        "nested_multi_encoder_4c16c_confirmatory_v2": 120,
    }
    directories = [path for path in root.iterdir() if path.is_dir() and path.name != "gui_runs"]
    gui_root = root / "gui_runs"
    if gui_root.exists():
        directories.extend(path for path in gui_root.iterdir() if path.is_dir())
    rows = []
    for experiment in directories:
        files = tuple(path for path in experiment.rglob("*") if path.is_file())
        checkpoints = len(tuple((experiment / "checkpoints").glob("*.npz")))
        bytes_total = sum(path.stat().st_size for path in files)
        updated = max((path.stat().st_mtime for path in files), default=experiment.stat().st_mtime)
        expected = planned.get(experiment.name)
        if experiment.name in {"single_encoder_8class", "rf_bank_8class"}:
            try:
                plan = json.loads((experiment / "study_plan.json").read_text(encoding="utf-8"))
                configuration = plan["config"]
                active_encoders = sum(bool(configuration.get(name, {}).get("enabled", True)) for name in ("resonate_fire", "lif"))
                expected = len(configuration["study"]["subjects"]) * active_encoders
            except (OSError, ValueError, KeyError, TypeError):
                expected = None
        current = latest_progress(experiment / "progress.jsonl")
        advanced_cell_progress = (
            current is not None and current.get("cell_current") is not None and current.get("cell_total")
        )
        if advanced_cell_progress:
            # The advanced runner records the fractional in-cell position in
            # ``current``. Convert that to an honest study-wide ratio.
            cell_fraction = float(current["cell_current"]) / float(current["cell_total"])
            completed_cells = math.floor(float(current.get("current", 0.0)))
            fraction = (completed_cells + cell_fraction) / float(current["total"])
        else:
            cell_fraction = None
            completed_cells = int(current.get("current", 0)) if current and current.get("current") is not None else None
            fraction = current.get("fraction") if current else None
        pid = current.get("pid") if current else None
        # Dashboard refreshes can run in a process sandbox which cannot probe
        # the worker PID. A freshly updated running journal is therefore also
        # a reliable local liveness signal; old interrupted journals stay
        # visibly resumable rather than being mislabeled as live.
        encoder_run_active = bool(expected and checkpoints < expected and time.time() - updated < 180.0)
        active = encoder_run_active or bool(
            current and current.get("status") == "running"
            and (_pid_is_running(pid) or time.time() - updated < 180.0)
        )
        rows.append({
            "name": experiment.name, "checkpoints": checkpoints,
            "planned": expected or (int(current["total"]) if current and current.get("total") else None),
            "size_mb": round(bytes_total / (1024 * 1024), 2), "updated_epoch": round(updated, 3),
            "phase": current.get("phase") if current else None,
            "message": current.get("message") if current else None,
            "fraction": fraction,
            "cell_fraction": cell_fraction,
            "cell_current": current.get("cell_current") if current else None,
            "cell_total": current.get("cell_total") if current else None,
            "completed_cells": completed_cells,
            "eta_seconds": current.get("eta_seconds") if current else None,
            "status": current.get("status") if current else ("completed" if expected and checkpoints >= expected else "running" if encoder_run_active else "pending"),
            "active": active,
            "pid": pid,
        })
    return sorted(rows, key=lambda item: item["updated_epoch"], reverse=True)[:20]


def _encoder_comparison_results(directory: str | Path | None) -> list[dict[str, object]]:
    """Summarize subject-wise held-block results from the 8-class studies."""
    if directory is None:
        return []
    import numpy as np
    rows = []
    for name in ("single_encoder_8class", "rf_bank_8class"):
        checkpoint_dir = Path(directory) / name / "checkpoints"
        for path in sorted(checkpoint_dir.glob("*.npz")) if checkpoint_dir.exists() else ():
            try:
                with np.load(path, allow_pickle=False) as payload:
                    parameters = tuple(str(item) for item in payload["parameter_names"])
                    selected = np.asarray(payload["selected_parameters_per_block"], dtype=float)
                    rows.append({
                        "study": name, "encoder": str(payload["encoder"]), "subject": int(payload["subject_id"]),
                        "accuracy": round(100 * float(payload["accuracy"]), 2),
                        "candidates": int(len(payload["parameter_grid"])), "parameters": ", ".join(parameters),
                        "selected_summary": np.round(np.median(selected, axis=0), 5).tolist(),
                        "updated_epoch": round(path.stat().st_mtime, 3),
                    })
            except (OSError, ValueError, KeyError, TypeError):
                continue
    return sorted(rows, key=lambda item: (item["study"], item["encoder"], item["subject"]))


def _pid_is_running(pid: object) -> bool:
    """Best-effort local check, so interrupted studies are not shown as live."""
    try:
        candidate = int(pid)
        if candidate <= 0:
            return False
        os.kill(candidate, 0)
        return True
    except (TypeError, ValueError, OSError):
        return False


_DASHBOARD_TEMPLATE = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>SSVEP Experiment Dashboard</title>
<style>
:root{--ink:#15202b;--muted:#5e6f80;--line:#d7e0e8;--paper:#fff;--bg:#f3f6f9;--blue:#1769aa;--green:#16856b;--orange:#b56a18;--red:#b63e4a;--soft:#eaf1f7}
@media(prefers-color-scheme:dark){:root{--ink:#e8eef4;--muted:#aab8c5;--line:#344454;--paper:#16212b;--bg:#0f171f;--blue:#65aee8;--green:#55c2a4;--orange:#e0a253;--red:#e47c86;--soft:#223240}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 system-ui,sans-serif}header{padding:22px 28px 16px;background:var(--paper);border-bottom:1px solid var(--line)}h1{margin:0 0 4px;font-size:24px}h2{margin:0 0 12px;font-size:18px}h3{margin:0 0 8px;font-size:15px}p{margin:0;color:var(--muted)}main{padding:18px 28px 28px;display:grid;gap:16px}.panel,.card{background:var(--paper);border:1px solid var(--line);border-radius:10px}.panel{padding:16px;overflow:hidden}.cards{display:grid;grid-template-columns:repeat(3,minmax(150px,1fr));gap:12px}.card{padding:14px}.value{font-size:26px;font-weight:600}.label,.note{color:var(--muted)}.controls{display:flex;gap:9px;flex-wrap:wrap;align-items:end}.control{display:grid;gap:3px}.control label{font-size:12px;color:var(--muted)}input,select,button{font:inherit;padding:7px 9px;border:1px solid var(--line);border-radius:6px;background:var(--paper);color:var(--ink)}button{cursor:pointer}.warning{padding:9px 11px;border-left:4px solid var(--orange);background:color-mix(in srgb,var(--orange) 10%,var(--paper));margin-top:10px}.grid2{display:grid;grid-template-columns:minmax(0,1.35fr) minmax(320px,.65fr);gap:16px}.chart{width:100%;min-height:310px}.chart svg{display:block;width:100%;height:auto;max-height:360px}.axis{stroke:var(--muted);stroke-width:1}.grid{stroke:var(--line);stroke-width:1}.tick{fill:var(--muted);font-size:12px}.mark-causal{fill:var(--green)}.mark-offline{fill:var(--blue)}.mark-unspecified{fill:var(--orange)}.legend{display:flex;gap:15px;color:var(--muted);margin-top:4px}.swatch{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:5px}.table-wrap{overflow:auto;max-height:360px}table{border-collapse:collapse;width:100%;white-space:nowrap}th,td{padding:7px 9px;border-bottom:1px solid var(--line);text-align:left}th{font-size:11px;color:var(--muted);text-transform:uppercase;letter-spacing:.04em;position:sticky;top:0;background:var(--paper)}td.num{text-align:right;font-variant-numeric:tabular-nums}.badge{display:inline-block;padding:2px 7px;border-radius:999px;background:var(--soft)}.split-outer_test{color:var(--green)}.split-apparent_same_data{color:var(--orange)}.example-layout{display:grid;grid-template-columns:minmax(0,1fr) minmax(300px,.85fr);gap:16px;align-items:start}.example-image{display:block;width:100%;height:auto;max-height:560px;object-fit:contain;background:var(--paper)}.image-frame{border:1px solid var(--line);border-radius:8px;overflow:hidden}.caption{margin-top:7px;color:var(--muted)}.pager{display:flex;gap:8px;align-items:center;margin-top:9px}.empty{padding:28px;text-align:center;color:var(--muted)}code{color:var(--ink)}
.facets{display:grid;grid-template-columns:repeat(3,minmax(220px,1fr));gap:18px}.facet h3{margin-bottom:2px}.facet svg{display:block;width:100%;height:auto}.facet .range{color:var(--muted);font-size:12px}
.live-progress{display:grid;gap:12px}.live-progress .live-row{padding:12px;border:1px solid var(--line);border-radius:8px;background:color-mix(in srgb,var(--blue) 5%,var(--paper))}.live-progress .live-head{display:flex;justify-content:space-between;gap:12px;align-items:baseline;flex-wrap:wrap}.progress-track{height:11px;border-radius:99px;background:var(--soft);overflow:hidden;margin-top:8px}.progress-fill{height:100%;background:linear-gradient(90deg,var(--blue),var(--green));min-width:2px}.progress-detail{display:flex;gap:14px;flex-wrap:wrap;margin-top:7px;color:var(--muted);font-size:12px}.live-state{color:var(--green);font-weight:650}.pending-state{color:var(--orange);font-weight:650}
@media(max-width:900px){.grid2,.example-layout{grid-template-columns:1fr}.cards,.facets{grid-template-columns:1fr}.panel{overflow:auto}main,header{padding-left:13px;padding-right:13px}}
</style></head><body>
<header><h1>SSVEP Experiment Dashboard</h1><p>Results, parameter optimization, accuracy, information-transfer rate, latency, spike cost, robustness, runtime, and neuron-state plots.</p></header>
<main>
<section class="panel"><div class="controls">
<div class="control"><label for="study">Study</label><select id="study"></select></div>
<div class="control"><label for="split">Evaluation split</label><select id="split"></select></div>
<div class="control"><label for="subject">Subject</label><select id="subject"></select></div>
<div class="control"><label for="classes">Classes</label><select id="classes"></select></div>
<div class="control"><label for="filter">Filter mode</label><select id="filter"></select></div>
</div><div id="split-warning" class="warning"></div></section>
<section class="panel" id="live-progress-panel"><h2>Live and pending experiments</h2><div id="live-progress" class="live-progress"></div></section>
<section class="panel"><h2>8-class encoder comparison</h2><p class="note">Subject-wise accuracy evaluated on held-out stimulus blocks.</p><div id="encoder-results"></div></section>
<section class="cards" id="cards"></section>
<section class="panel"><h2>All selected metrics at a glance</h2><div id="metric-facets" class="facets"></div></section>
<section class="panel"><div class="controls"><div class="control"><label for="metric">Plotted metric</label><select id="metric"></select></div></div><div id="metric-chart" class="chart"></div><div class="legend"><span><i class="swatch" style="background:var(--green)"></i>Causal</span><span><i class="swatch" style="background:var(--blue)"></i>Offline</span><span><i class="swatch" style="background:var(--orange)"></i>Unspecified</span></div></section>
<div class="grid2"><section class="panel"><h2>Metrics for the selected evaluation split</h2><div class="table-wrap"><table><thead><tr><th>Metric</th><th>Unit</th><th>N</th><th>Median</th><th>Minimum</th><th>Maximum</th></tr></thead><tbody id="metric-summary"></tbody></table></div></section>
<section class="panel"><h2>Recent experiment storage and progress</h2><div class="table-wrap"><table><thead><tr><th>Experiment</th><th>Checkpoints</th><th>Current phase</th><th>ETA</th><th>Size</th><th>Updated</th></tr></thead><tbody id="progress"></tbody></table></div></section></div>
<section class="panel"><h2>Signal, internal state, and spikes</h2><div class="controls"><div class="control"><label for="example">Segment</label><select id="example"></select></div><div class="control"><label for="trace">Trace</label><select id="trace"></select></div></div><div class="example-layout"><div><div id="trace-chart" class="chart"></div><div id="trace-caption" class="caption"></div></div><div id="example-view"></div></div></section>
<section class="panel"><h2>Runs</h2><div class="controls"><div class="control"><label for="search">Search</label><input id="search" placeholder="Run, encoder, or study"></div></div><div class="table-wrap"><table><thead><tr><th>Study</th><th>Subject</th><th>Classes</th><th>Filter</th><th>Encoder</th><th>Accuracy</th><th>Practical ITR</th><th>Status</th></tr></thead><tbody id="runs"></tbody></table></div><div class="pager"><button id="prev">Previous</button><span id="page"></span><button id="next">Next</button></div></section>
</main>
<script id="dashboard-data" type="application/json">__PAYLOAD__</script><script>
const D=JSON.parse(document.getElementById('dashboard-data').textContent),q=s=>document.querySelector(s),esc=x=>String(x??'–').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const meta=name=>D.metric_meta[name]||{label:name.replaceAll('_',' '),unit:'',scale:1};
const median=a=>{const b=[...a].sort((x,y)=>x-y),n=b.length;return n?b.length%2?b[(n-1)/2]:(b[n/2-1]+b[n/2])/2:NaN};
const fmt=(v,name)=>{if(!Number.isFinite(v))return '–';const m=meta(name),x=v*m.scale;return `${Math.abs(x)>=100?x.toFixed(1):Math.abs(x)>=10?x.toFixed(2):x.toFixed(3)}${m.unit==='%'?'%':''}`};
const options=(values,label='All')=>`<option value="">${label}</option>`+[...new Set(values.filter(x=>x!==null&&x!==undefined))].map(x=>`<option value="${esc(x)}">${esc(x)}</option>`).join('');
q('#study').innerHTML=options(D.studies.map(x=>x.id),'All studies');D.studies.forEach(s=>{const o=[...q('#study').options].find(x=>x.value===String(s.id));if(o)o.textContent=s.name});
const splitOrder=['outer_test','inner_validation','outer_test_perturbed','apparent_same_data'];q('#split').innerHTML=splitOrder.filter(x=>D.metrics.some(m=>m.split===x)).map(x=>`<option value="${x}">${x==='outer_test'?'Nested outer test':x==='inner_validation'?'Inner validation':x==='outer_test_perturbed'?'Perturbed outer test':'Same-data score'}</option>`).join('');
q('#subject').innerHTML=options(D.runs.map(x=>x.subject),'All subjects');q('#classes').innerHTML=options(D.runs.map(x=>x.classes),'All class counts');q('#filter').innerHTML=options(['causal','offline','unspecified'],'All filter modes');
let page=0,pageSize=50;
function baseMatch(x){return(!q('#study').value||String(x.study_id)===q('#study').value)&&(!q('#subject').value||String(x.subject)===q('#subject').value)&&(!q('#classes').value||String(x.classes)===q('#classes').value)&&(!q('#filter').value||x.filter===q('#filter').value)}
function filteredMetrics(ignoreName=false){const split=q('#split').value,name=q('#metric').value;return D.metrics.filter(m=>m.split===split&&baseMatch(m)&&(ignoreName||m.name===name))}
function updateMetricOptions(){const names=[...new Set(filteredMetrics(true).map(x=>x.name))].sort((a,b)=>meta(a).label.localeCompare(meta(b).label)),old=q('#metric').value;q('#metric').innerHTML=names.map(x=>`<option value="${x}">${esc(meta(x).label)} · ${esc(meta(x).unit)}</option>`).join('');q('#metric').value=names.includes(old)?old:names.includes('accuracy')?'accuracy':names[0]||''}
function renderWarning(){const split=q('#split').value;q('#split-warning').innerHTML=split==='apparent_same_data'?'<strong>Same-data score:</strong> parameter selection and scoring reused the same trials; use this only to inspect the parameter search.':split==='inner_validation'?'<strong>Inner-fold summary:</strong> boundary rates and selection stability describe the parameter search; they are not held-out accuracy.':split==='outer_test_perturbed'?'<strong>Perturbed held-out trials:</strong> compare this result with the matching unperturbed held-out result.':'<strong>Held-out result:</strong> outer-block trials were not used for parameter or ridge selection.'}
function renderCards(){const rows=filteredMetrics(true),runIds=new Set(rows.map(x=>x.run_id)),acc=rows.filter(x=>x.name==='accuracy').map(x=>x.value),itr=rows.filter(x=>x.name==='practical_itr_bits_per_minute').map(x=>x.value);q('#cards').innerHTML=[['Runs',runIds.size,'selected split'],['Median accuracy',acc.length?fmt(median(acc),'accuracy'):'–',`${acc.length} values`],['Best practical ITR',itr.length?fmt(Math.max(...itr),'practical_itr_bits_per_minute'):'–','bits/min including overhead']].map(x=>`<article class="card"><div class="value">${x[1]}</div><div class="label">${x[0]} · ${x[2]}</div></article>`).join('')}
function renderMetricChart(){const name=q('#metric').value,rows=filteredMetrics(),box=q('#metric-chart');if(!rows.length){box.innerHTML='<div class="empty">No values for this selection.</div>';return}const m=meta(name),values=rows.map(x=>x.value*m.scale),classes=rows.map(x=>Number(x.classes)).filter(Number.isFinite),useClasses=new Set(classes).size>1,xvals=rows.map((r,i)=>useClasses?Number(r.classes):(Number(r.subject)||i+1)),xmin=Math.min(...xvals),xmax=Math.max(...xvals),ymin=Math.min(0,...values),ymax=Math.max(...values),ys=ymax-ymin||1,xs=xmax-xmin||1,W=1000,H=330,L=78,R=22,T=22,B=55;let grid='';for(let i=0;i<=5;i++){const y=T+(H-T-B)*i/5,v=ymax-ys*i/5;grid+=`<line class="grid" x1="${L}" y1="${y}" x2="${W-R}" y2="${y}"/><text class="tick" x="${L-8}" y="${y+4}" text-anchor="end">${v.toFixed(Math.abs(v)>=100?0:2)}</text>`}const marks=rows.map((r,i)=>{const x=L+(W-L-R)*(xvals[i]-xmin)/xs+(rows.length>1?((Number(r.subject)||i)%7-3)*2:0),y=T+(H-T-B)*(ymax-values[i])/ys,mode=r.filter||'unspecified',suffix=m.unit==='%'?'':` ${m.unit}`;return`<circle class="mark-${mode}" cx="${x}" cy="${y}" r="5"><title>${esc(r.study)} · S${esc(r.subject)} · ${esc(r.classes)} classes · ${mode} · ${fmt(r.value,name)}${esc(suffix)}</title></circle>`}).join('');box.innerHTML=`<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(m.label)} plot">${grid}<line class="axis" x1="${L}" y1="${H-B}" x2="${W-R}" y2="${H-B}"/><line class="axis" x1="${L}" y1="${T}" x2="${L}" y2="${H-B}"/>${marks}<text class="tick" x="${(L+W-R)/2}" y="${H-12}" text-anchor="middle">${useClasses?'Class count':'Subject'}</text><text class="tick" transform="translate(18 ${(T+H-B)/2}) rotate(-90)" text-anchor="middle">${esc(m.label)} (${esc(m.unit)})</text></svg>`}
function renderMetricSummary(){const rows=filteredMetrics(true),names=[...new Set(rows.map(x=>x.name))].sort((a,b)=>meta(a).label.localeCompare(meta(b).label));q('#metric-summary').innerHTML=names.map(name=>{const vals=rows.filter(x=>x.name===name).map(x=>x.value),m=meta(name);return`<tr><td>${esc(m.label)}</td><td>${esc(m.unit)}</td><td class="num">${vals.length}</td><td class="num">${fmt(median(vals),name)}</td><td class="num">${fmt(Math.min(...vals),name)}</td><td class="num">${fmt(Math.max(...vals),name)}</td></tr>`}).join('')||'<tr><td colspan="6">No metrics.</td></tr>'}
function progressPct(value){return Number.isFinite(value)?Math.max(0,Math.min(100,value*100)):null}
function renderLiveProgress(){const rows=D.experiment_progress.filter(x=>x.active||(x.status==='running'&&x.planned&&x.checkpoints<x.planned)),box=q('#live-progress');if(!rows.length){box.innerHTML='<div class="empty">No experiment is running. Completed results are shown below.</div>';return}box.innerHTML=rows.map(x=>{const pct=progressPct(x.fraction),cell=Number.isFinite(x.cell_fraction)?progressPct(x.cell_fraction):null,state=x.active?'Running now':'Pending / resumable',stateClass=x.active?'live-state':'pending-state',cells=x.planned?`${x.completed_cells??x.checkpoints}/${x.planned} study cells`:`${x.checkpoints} checkpoints`,cellText=cell===null?'':` · current cell ${x.cell_current}/${x.cell_total} (${cell.toFixed(1)}%)`;return`<article class="live-row"><div class="live-head"><strong>${esc(x.name)}</strong><span class="${stateClass}">${state}</span></div><div class="progress-track" role="progressbar" aria-label="${esc(x.name)} study progress" aria-valuemin="0" aria-valuemax="100" aria-valuenow="${pct??0}"><div class="progress-fill" style="width:${pct??0}%"></div></div><div class="progress-detail"><span><strong>${pct===null?'Progress unavailable':pct.toFixed(2)+'%'}</strong> · ${cells}${cellText}</span><span>${esc(x.phase||'idle')} · ${esc(x.message||'')}</span><span>updated ${new Date(x.updated_epoch*1000).toLocaleTimeString()}</span></div>${cell===null?'':`<div class="progress-track" role="progressbar" aria-label="${esc(x.name)} current cell progress" aria-valuemin="0" aria-valuemax="100" aria-valuenow="${cell.toFixed(1)}"><div class="progress-fill" style="width:${cell}%"></div></div>`}</article>`}).join('')}
function renderProgress(){q('#progress').innerHTML=D.experiment_progress.slice(0,10).map(x=>{const pct=Number.isFinite(x.fraction)?` · ${(100*x.fraction).toFixed(2)}%`:'',eta=Number.isFinite(x.eta_seconds)?x.eta_seconds<60?`${x.eta_seconds.toFixed(0)} s`:`${(x.eta_seconds/60).toFixed(1)} min`:'–';return`<tr><td>${esc(x.name)}</td><td class="num">${x.checkpoints}${x.planned?`/${x.planned}`:''}</td><td><span class="badge">${esc(x.phase||x.status||'idle')}${pct}</span><br><span class="note">${esc(x.message||'')}</span></td><td class="num">${eta}</td><td class="num">${x.size_mb.toFixed(2)} MB</td><td>${new Date(x.updated_epoch*1000).toLocaleString()}</td></tr>`}).join('')}
const traceLabels={raw_uV:'Raw EEG',filtered_uV:'Band-pass filtered EEG',rf_u:'R&F u state',rf_v:'R&F v state',delta_change_uV:'Successive sample change',lif_membrane:'LIF membrane state'};
function setExample(){const item=D.examples[Number(q('#example').value)||0];if(!item){q('#example-view').innerHTML='<div class="empty">No generated examples.</div>';q('#trace-chart').innerHTML='';return}const keys=Object.keys(item.traces||{}).filter(k=>traceLabels[k]);q('#trace').innerHTML=keys.map(k=>`<option value="${k}">${traceLabels[k]} · ${item.units[k]}</option>`).join('');renderTrace();q('#example-view').innerHTML=`<div class="image-frame"><a href="${item.image_url}" target="_blank"><img class="example-image" src="${item.image_url}" alt="Signal, neuron state, threshold and spikes for S${item.subject}, ${item.frequency_hz} Hz, ${item.electrode}, block ${item.block}"></a></div><div class="caption">Full-resolution PNG · ${(item.image_bytes/1048576).toFixed(2)} MB · ${item.sampling_rate_hz} samples/s · time in ms · EEG amplitudes in µV · band ${Math.max(.1,item.frequency_hz-item.filter_half_width_hz)}–${item.frequency_hz+item.filter_half_width_hz} Hz (order ${item.filter_order}). Click image for native resolution.</div>`}
function renderTrace(){const item=D.examples[Number(q('#example').value)||0],key=q('#trace').value,trace=item?.traces;if(!trace||!trace[key]){q('#trace-chart').innerHTML='<div class="empty">No trace data.</div>';return}const t=trace.time_ms,y=trace[key],xmin=Math.min(...t),xmax=Math.max(...t),ymin=Math.min(...y),ymax=Math.max(...y),xs=xmax-xmin||1,ys=ymax-ymin||1,W=1000,H=330,L=78,R=22,T=22,B=55,pts=t.map((x,i)=>`${L+(W-L-R)*(x-xmin)/xs},${T+(H-T-B)*(ymax-y[i])/ys}`).join(' ');let spikeKey=key.startsWith('rf_')?'rf_spikes_ms':key==='delta_change_uV'?'delta_up_ms':key==='lif_membrane'?'lif_spikes_ms':null,spikes=spikeKey?(trace[spikeKey]||[]):[],spikeLines=spikes.map(x=>{const px=L+(W-L-R)*(x-xmin)/xs;return`<line x1="${px}" y1="${T}" x2="${px}" y2="${H-B}" stroke="var(--red)" stroke-width="1" opacity=".45"/>`}).join('');q('#trace-chart').innerHTML=`<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${traceLabels[key]} over time"><line class="axis" x1="${L}" y1="${H-B}" x2="${W-R}" y2="${H-B}"/><line class="axis" x1="${L}" y1="${T}" x2="${L}" y2="${H-B}"/>${spikeLines}<polyline points="${pts}" fill="none" stroke="var(--blue)" stroke-width="1.5"/><text class="tick" x="${(L+W-R)/2}" y="${H-12}" text-anchor="middle">Time after segment start (ms)</text><text class="tick" transform="translate(18 ${(T+H-B)/2}) rotate(-90)" text-anchor="middle">${traceLabels[key]} (${item.units[key]})</text><text class="tick" x="${L}" y="${T-7}">${ymin.toFixed(3)} to ${ymax.toFixed(3)} ${item.units[key]} · ${spikes.length} spike events overlaid</text></svg>`;q('#trace-caption').textContent=`S${item.subject} · ${item.frequency_hz} Hz stimulus class · ${item.electrode} · block ${item.block} · ${item.duration_ms} ms · ${item.sampling_rate_hz} samples/s.`}
q('#example').innerHTML=D.examples.map((x,i)=>`<option value="${i}">S${x.subject} · ${x.frequency_hz} Hz · ${x.electrode} · block ${x.block}</option>`).join('');q('#example').onchange=setExample;q('#trace').onchange=renderTrace;setExample();
function renderRuns(){const term=q('#search').value.toLowerCase(),selectedMetrics=D.metrics.filter(m=>m.split===q('#split').value),splitRuns=new Set(selectedMetrics.map(m=>m.run_id)),metricByRun={};selectedMetrics.filter(m=>m.name==='accuracy'||m.name==='practical_itr_bits_per_minute').forEach(m=>{(metricByRun[m.run_id]??={})[m.name]=m.value});const rows=D.runs.filter(r=>splitRuns.has(r.id)&&baseMatch(r)&&(!term||`${r.run_key} ${r.encoder||''} ${r.study}`.toLowerCase().includes(term))),pages=Math.max(1,Math.ceil(rows.length/pageSize));page=Math.min(page,pages-1);q('#runs').innerHTML=rows.slice(page*pageSize,(page+1)*pageSize).map(r=>{const mm=metricByRun[r.id]||{};return`<tr><td>${esc(r.study)}</td><td>${esc(r.subject)}</td><td>${esc(r.classes)}</td><td>${esc(r.filter)}</td><td>${esc(r.encoder)}</td><td class="num">${fmt(mm.accuracy,'accuracy')}</td><td class="num">${fmt(mm.practical_itr_bits_per_minute,'practical_itr_bits_per_minute')}</td><td><span class="badge">${esc(r.status)}</span></td></tr>`}).join('');q('#page').textContent=`Page ${page+1}/${pages} · ${rows.length} runs`;q('#prev').disabled=page===0;q('#next').disabled=page>=pages-1}
function renderFacets(){const rows=filteredMetrics(true),names=[...new Set(rows.map(x=>x.name))].sort((a,b)=>meta(a).label.localeCompare(meta(b).label));q('#metric-facets').innerHTML=names.map(name=>{const m=meta(name),suffix=m.unit==='%'?'':` ${m.unit}`,vals=rows.filter(x=>x.name===name).map(x=>x.value*m.scale),lo=Math.min(...vals),hi=Math.max(...vals),span=hi-lo||1,points=vals.map((v,i)=>`<circle cx="${16+268*(v-lo)/span}" cy="${22+(i%3-1)*4}" r="4" fill="var(--blue)" opacity=".75"><title>${v.toFixed(3)}${esc(suffix)}</title></circle>`).join('');return`<div class="facet"><h3>${esc(m.label)}</h3><div class="range">${fmt(lo/m.scale,name)} to ${fmt(hi/m.scale,name)}${esc(suffix)} · N=${vals.length}</div><svg viewBox="0 0 300 45" role="img" aria-label="${esc(m.label)} distribution"><line x1="16" y1="22" x2="284" y2="22" stroke="var(--line)"/>${points}</svg></div>`}).join('')||'<div class="empty">No metrics in this split.</div>'}
function renderEncoderResults(){const rows=D.encoder_comparison_results,box=q('#encoder-results');if(!rows.length){box.innerHTML='<div class="empty">No encoder results are available yet.</div>';return}box.innerHTML=`<div class="table-wrap"><table><thead><tr><th>Study</th><th>Encoder</th><th>Subject</th><th>Accuracy</th><th>Settings evaluated</th><th>Median selected parameters</th></tr></thead><tbody>${rows.map(x=>`<tr><td>${esc(x.study)}</td><td>${esc(x.encoder)}</td><td class="num">${x.subject}</td><td class="num">${x.accuracy.toFixed(2)}%</td><td class="num">${x.candidates}</td><td>${esc(x.selected_summary.join(', '))}</td></tr>`).join('')}</tbody></table></div>`}
function updateAll(){renderWarning();updateMetricOptions();renderLiveProgress();renderEncoderResults();renderCards();renderFacets();renderMetricChart();renderMetricSummary();renderProgress();page=0;renderRuns()}
['#study','#split','#subject','#classes','#filter'].forEach(s=>q(s).onchange=updateAll);q('#metric').onchange=renderMetricChart;q('#search').oninput=()=>{page=0;renderRuns()};q('#prev').onclick=()=>{page--;renderRuns()};q('#next').onclick=()=>{page++;renderRuns()};updateAll();
</script></body></html>'''
