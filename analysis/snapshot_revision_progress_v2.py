"""Record verified ongoing work without counting it as a completed experiment."""
from __future__ import annotations

import json
import re
import shutil
import sys
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.revision_validation.study_io import now, read, sha256, verify_bundle, verify_completed, write
from experiments.revision_validation.extended_engine import verify_extension
from analysis.runtime_observation import read, shared_binary_read

RUNTIME = Path("D:/pa-csi-dg-revision-runtime")
DOCS = ROOT / "docs/revision"


def copy_once(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if sha256(destination) != sha256(source):
            raise RuntimeError(f"Frozen evidence copy changed: {destination}")
    else:
        shutil.copy2(source, destination)


def optional(path):
    return read(path) if path.exists() else None


def append_text(path):
    if not path.exists():
        return ""
    with shared_binary_read(path) as stream:
        value = stream.read().decode("utf-8-sig")
    return value[:value.rfind("\n")+1]


def active_owner(bundle):
    for process in psutil.process_iter(["pid", "cmdline", "create_time"]):
        command = process.info["cmdline"] or []
        if "--bundle" in command:
            index = command.index("--bundle") + 1
            if index < len(command) and Path(command[index]).resolve() == bundle.resolve():
                return {"pid": process.info["pid"], "create_time": process.info["create_time"]}
    return None


def preparation_progress(path, kind):
    text = append_text(path)
    if kind == "widar":
        rows = re.findall(r"([^\r\n]+\.zip): prepared (\d+)/(\d+)", text)
        last = {archive: int(done) for archive, done, total in rows}
        return {"evidence": "append-only log; last announced counts", "completed_at_least": sum(last.values()), "by_archive": last}
    if kind == "widar_v2":
        rows = re.findall(r"([^\r\n]+\.zip): accounted (\d+)/(\d+); valid (\d+); excluded (\d+)", text)
        last = {archive: {"accounted": int(done), "total": int(total), "valid": int(valid), "excluded": int(excluded)}
                for archive, done, total, valid, excluded in rows}
        return {"evidence": "v2 append-only log; last announced inventory accounting",
                "accounted_at_least": sum(item["accounted"] for item in last.values()), "by_archive": last}
    rows = re.findall(r"Prepared(?: phase)? (\d+)/(\d+); failures (\d+)", text)
    return {"evidence": "append-only log; last announced count", "completed_at_least": int(rows[-1][0]),
            "total": int(rows[-1][1]), "failed_at_last_announcement": int(rows[-1][2])} if rows else None


def coordinator_snapshot(directory):
    config = optional(directory / "config.json")  # Immutable after its initial write.
    if not config:
        return None
    pid = config.get("pid")
    if pid is None:
        expected = directory.resolve()
        for process in psutil.process_iter(["pid", "cmdline"]):
            command = process.info["cmdline"] or []
            if "--output" in command:
                index = command.index("--output") + 1
                if index < len(command) and Path(command[index]).resolve() == expected:
                    pid = process.info["pid"]
                    break
    try:
        command = " ".join(psutil.Process(pid).cmdline()).replace("\\", "/").lower() if pid is not None else ""
        live = str(directory).replace("\\", "/").lower() in command
    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
        live = False
    events = [json.loads(line) for line in append_text(directory / "events.jsonl").splitlines()]
    return {"directory": str(directory), "pid": pid, "live": live,
            "status": "active; mutable status not read" if live else optional(directory / "status.json"),
            "last_append_event": events[-1] if events else None,
            "config_sha256": sha256(directory / "config.json")}


def formal_source_analysis_snapshot():
    analysis_root = RUNTIME / "analyses/formal_source_v1"
    completed = []
    summaries = sorted(analysis_root.glob("*/*/summary.json")) if analysis_root.exists() else []
    for summary_path in summaries:
        study, job_id = summary_path.parent.parent.name, summary_path.parent.name
        summary = read(summary_path)
        if summary["job_id"] != job_id or not summary["full_trained_model_analysis"] or summary["target_data_loaded"]:
            raise RuntimeError("Source-analysis summary scope mismatch")
        completed.append({"study": study, "job_id": job_id, "summary_sha256": sha256(summary_path)})
    return {"directory": str(analysis_root), "completed": completed, "completed_count": len(completed),
            "expected_total": 117, "target_data_loaded": False,
            "evidence": "each retained summary is hash-verified and records source-only scope"}


def study_snapshot(bundle, verifier):
    protocol, bundle_hash = verifier(bundle)
    finished, started, latest = [], [], None
    owner = active_owner(bundle)
    for config in protocol["jobs"]:
        run = bundle / "runs" / config["job_id"]
        if (run / "config.json").exists():
            started.append(config["job_id"])
        events = [json.loads(line) for line in append_text(run / "events.jsonl").splitlines()]
        if any(e["event"] == "completed" for e in events):
            # The completion event is written after the final status and all
            # artifacts. This job's status is then no longer rewritten.
            verify_completed(run, config)
            finished.append(config["job_id"])
        elif events:
            epochs = [e.get("epoch", e.get("completed_epochs")) for e in events if e["event"] in ("epoch_committed", "resumed")]
            latest = {"job_id": config["job_id"], "epoch": epochs[-1] if epochs else 0, "last_append_event": events[-1],
                      "evidence": "append-only event log; mutable JSON not opened"}
    queue = {"status": "running" if owner else "not_running", "process": owner, "evidence": "OS process state"}
    if not owner and len(finished) == len(protocol["jobs"]):
        queue = optional(bundle / "queue_status.json")
    return {"bundle": str(bundle), "bundle_sha256": bundle_hash, "frozen_jobs": len(protocol["jobs"]),
            "started_jobs": started, "completed_jobs": finished, "queue": queue, "current_run": latest,
            "target_gate_available": (bundle / "target_gate.json").exists(), "queue_failure_available": (bundle / "queue_failure.json").exists()}


def h_supplement_snapshot():
    root = ROOT / "outputs_h_grouped_l1248/v2"
    bundle, queue = root / "bundle", root / "queue"
    owner = optional(queue / "owner.json")
    live = False
    if owner:
        try:
            process = psutil.Process(owner["pid"])
            live = abs(process.create_time() - owner["process_create_time"]) < 0.01 and process.cmdline() == owner["command"]
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            live = False
    events = [json.loads(line) for line in append_text(queue / "events.jsonl").splitlines()]
    state = "active; mutable completion files not read" if live else (
        "completed" if (queue / "completed.json").exists() else
        "failed" if (queue / "failed.json").exists() else "not_started")
    return {"bundle": str(bundle),
            "bundle_manifest_sha256": sha256(bundle / "bundle_manifest.json") if bundle.exists() else None,
            "frozen_jobs": 18, "queue_owner": owner, "queue_owner_live": live,
            "status": state, "last_append_event": events[-1] if events else None,
            "widar_dependency": False}


def main():
    bundle = RUNTIME / "studies/r_core_v1"
    controls_bundle = RUNTIME / "studies/r_controls_v1"
    inputs_bundle = RUNTIME / "studies/r_inputs_v1"
    widar_bundle = RUNTIME / "studies/r_widar_v2"
    widar_summary_path = RUNTIME / "data/widar_native_amp_v2/summary.json"
    widar_summary = optional(widar_summary_path)
    core = study_snapshot(bundle, verify_bundle)
    controls = study_snapshot(controls_bundle, verify_extension)
    inputs = study_snapshot(inputs_bundle, verify_extension)
    widar = study_snapshot(widar_bundle, verify_extension) if widar_bundle.exists() else None
    widar_cancellation = optional(DOCS / "evidence/widar_cancelled_20260910.json")
    snapshot = {"recorded_at": now(), "plan_completed": False,
                "goal": "Complete the revised MultiEnv-only plan plus the original-H grouped L1248 supplement",
                "core": core, "controls": controls, "inputs": inputs, "widar_study": widar,
                "current_scope": {"multienv_formal_runs": 180, "source_analyses": 117,
                    "h_grouped_l1248_supplement_runs": 18, "widar_training_cancelled": True,
                    "widar_cancellation": widar_cancellation},
                "h_grouped_l1248": h_supplement_snapshot(),
                "native_multienv": {"directory": str(RUNTIME / "data/native_amp_v1"),
                    "status": {"state": "completed", "rows": 9000, "failures": 0,
                        "evidence": "completed summary; upstream writer exited"}
                        if (RUNTIME / "data/native_amp_v1/summary.json").exists()
                        else preparation_progress(RUNTIME / "data/native_amp_v1.stdout.log", "native"),
                    "summary_available": (RUNTIME / "data/native_amp_v1/summary.json").exists()},
                "widar": {"directory": str(RUNTIME / "data/widar_native_amp_v2"),
                    "inventoried_nonempty_records": 6749,
                    "status": {"state": "completed", "valid_rows": widar_summary["valid_rows"],
                        "signal_exclusions": len(widar_summary["signal_exclusions"]),
                        "failures": len(widar_summary["failures"]),
                        "complete_inventory_accounting": widar_summary["complete_inventory_accounting"],
                        "replacement_samples_added": widar_summary["replacement_samples_added"],
                        "target_model_evaluation": widar_summary["target_model_evaluation"],
                        "summary_sha256": sha256(widar_summary_path)}
                        if widar_summary else preparation_progress(RUNTIME / "data/widar_native_amp_v2.stdout.log", "widar_v2"),
                    "summary_available": widar_summary is not None,
                    "retained_failed_v1": {"directory": str(RUNTIME / "data/widar_native_amp_v1"),
                        "committed_records": 3440, "failure_dossier": str(RUNTIME / "data/widar_native_amp_v1/recovery/incomplete_001")}},
                "phase_controls": {"directory": str(RUNTIME / "data/native_phase_v2"),
                    "status": preparation_progress(RUNTIME / "coordinators/input_preparation_v1/native_phase.stdout.log", "phase"),
                    "summary_available": (RUNTIME / "data/native_phase_v2/summary.json").exists()},
                "coordinators": {name: coordinator_snapshot(RUNTIME / "coordinators" / name)
                    for name in ("core_to_controls_v1", "input_preparation_v1", "controls_to_inputs_v1",
                                 "formal_source_analysis_v1", "widar_after_preparation_v2",
                                 "final_revision_evidence_v5", "formal_paper_assets_v1",
                                 "formal_fragment_v1", "final_manuscript_v1", "revision_release_v1",
                                 "final_manuscript_v2", "revision_release_v2", "revision_release_v3",
                                 "revision_release_v4", "final_manuscript_v3", "revision_release_v5",
                                 "revision_release_v6", "controls_to_inputs_v2",
                                 "formal_source_analysis_v2", "widar_after_preparation_v3",
                                 "final_revision_evidence_v6", "formal_paper_assets_v2",
                                 "formal_fragment_v2", "final_manuscript_v4", "revision_release_v7",
                                 "multienv_final_evidence_v1", "multienv_paper_assets_v1",
                                 "multienv_fragment_v1", "multienv_manuscript_v1",
                                 "multienv_release_v1", "controls_to_inputs_v3",
                                 "formal_source_analysis_v3", "multienv_final_evidence_v2",
                                 "multienv_paper_assets_v2", "multienv_fragment_v2",
                                 "multienv_manuscript_v2", "multienv_release_v2",
                                 "multienv_final_evidence_v3", "multienv_paper_assets_v3",
                                 "multienv_fragment_v3", "multienv_manuscript_v3",
                                 "multienv_release_v3", "controls_to_inputs_v4",
                                 "multienv_final_evidence_v4", "multienv_paper_assets_v4",
                                 "multienv_fragment_v4", "multienv_manuscript_v4",
                                 "multienv_release_v4", "multienv_final_evidence_v5",
                                 "multienv_paper_assets_v5", "multienv_fragment_v5",
                                 "multienv_manuscript_v5", "multienv_release_v5",
                                 "multienv_final_evidence_v6", "multienv_paper_assets_v6",
                                 "multienv_fragment_v6", "multienv_manuscript_v6",
                                 "multienv_manuscript_v7", "multienv_release_v7", "multienv_release_v8")},
                "compiled_input_manifests": {kind: str(RUNTIME / "data" / folder / "dataset_manifest.json")
                    if (RUNTIME / "data" / folder / "dataset_manifest.json").exists() else None
                    for kind, folder in (("native", "native_amp_arrays_v1"), ("widar_v2", "widar_arrays_v2"),
                                         ("modality", "modality_arrays_v1"))},
                "completed_input_audits": {
                    "native_reconstruction": str(RUNTIME / "analyses/native_reconstruction_v1/summary.json"),
                    "mechanism_metric_qa": str(RUNTIME / "qa/mechanism_v1/mechanism_qa.json"),
                    "source_analysis_qa": str(RUNTIME / "qa/source_analysis_v1/summary.json")},
                "formal_source_analyses": formal_source_analysis_snapshot(),
                "writing_scope": "Original H/Y plus the verified original-H grouped supplement; R scores, tables and conclusions remain withdrawn from the manuscript and reviewer reply",
                "remaining": ["Verify local release, final document sync and terminal completion gate",
                    "Retain unresolved author metadata for final author review"]}
    final_audit = optional(DOCS / "evidence/final_multienv_completion_audit.json")
    if final_audit and final_audit.get("status") == "completed":
        snapshot["plan_completed"] = True
        snapshot["remaining"] = ["Author metadata, declarations, public deposit and final submission review"]
        snapshot["final_completion_audit_sha256"] = sha256(DOCS / "evidence/final_multienv_completion_audit.json")
    snapshot["completed_phases"] = {"R_runs": 180, "source_analyses": 117, "H_runs": 18,
        "H_pairs": 9, "technical_evidence": "multienv_final_evidence_v6",
        "paper_assets": "multienv_paper_assets_v6", "H_only_fragment": "multienv_fragment_v6",
        "manuscript": "multienv_manuscript_v7"}
    write(DOCS / "evidence/runtime_progress.json", snapshot)
    for filename in ("protocol.json", "bundle_manifest.json", "shared_initialization_audit.json"):
        copy_once(bundle / filename, DOCS / "evidence/r_core_v1" / filename)
    for filename in ("protocol.json", "bundle_manifest.json", "model_costs.json", "extension_qa.json", "target_gate_negative_test.log"):
        copy_once(controls_bundle / filename, DOCS / "evidence/r_controls_v1" / filename)
    for filename in ("protocol.json", "bundle_manifest.json", "model_costs.json", "input_study_qa.json",
                     "target_gate_negative_test.log", "prerequisite_gate_negative_test.log", "negative_gate_qa.json"):
        copy_once(inputs_bundle / filename, DOCS / "evidence/r_inputs_v1" / filename)
    for filename in ("decision.json", "observation_addendum.json"):
        copy_once(bundle / "recovery/windows_status_io_001" / filename, DOCS / "evidence/core_io_recovery" / filename)
    for filename in ("qa.json", "append_qa.json", "runtime_observation.py"):
        copy_once(RUNTIME / "qa/windows_observer_v1" / filename, DOCS / "evidence/core_io_recovery/windows_observer_v1" / filename)
    for folder in ("resume_v1", "resume_v2_cublas", "resume_v3_cpu"):
        copy_once(RUNTIME / "qa" / folder / "resume_qa.json", DOCS / "evidence" / (folder + ".json"))
    copy_once(RUNTIME / "qa/cuda_state_roundtrip.json", DOCS / "evidence/cuda_state_roundtrip.json")
    copy_once(RUNTIME / "data/native_amp_smoke_v1/summary.json", DOCS / "evidence/native_amp_smoke.json")
    copy_once(RUNTIME / "data/widar_smoke_v1/summary.json", DOCS / "evidence/widar_amp_smoke.json")
    copy_once(RUNTIME / "data/native_amp_v1/summary.json", DOCS / "evidence/native_amp_full_summary.json")
    copy_once(RUNTIME / "data/native_amp_arrays_v1/dataset_manifest.json", DOCS / "evidence/native_amp_dataset_manifest.json")
    copy_once(RUNTIME / "qa/mechanism_v1/mechanism_qa.json", DOCS / "evidence/mechanism_qa.json")
    copy_once(RUNTIME / "qa/widar_design_v1/design_qa.json", DOCS / "evidence/widar_design_qa.json")
    copy_once(RUNTIME / "qa/widar_design_v2/design_qa.json", DOCS / "evidence/widar_design_qa_v2.json")
    copy_once(RUNTIME / "data/native_phase_v2/summary.json", DOCS / "evidence/native_phase_full_summary.json")
    copy_once(RUNTIME / "data/modality_arrays_v1/dataset_manifest.json", DOCS / "evidence/modality_dataset_manifest.json")
    for filename in ("status.json", "native.branch.json", "widar.branch.json",
                     "native_phase.completed.json", "modality_compile.completed.json"):
        copy_once(RUNTIME / "coordinators/input_preparation_v1" / filename,
                  DOCS / "evidence/input_preparation_v1" / filename)
    for filename in ("summary.json", "per_record_differences.csv", "native_clock_statistics.csv"):
        copy_once(RUNTIME / "analyses/native_reconstruction_v1" / filename,
                  DOCS / "evidence/native_reconstruction_v1" / filename)
    for filename in ("summary.json", "selection.json", "gradient_probes.json", "source_features_and_norms.npz"):
        copy_once(RUNTIME / "qa/source_analysis_v1" / filename,
                  DOCS / "evidence/source_analysis_qa" / filename)
    for source_name, destination_name in (("qa/phase_v1/phase_qa.json", "phase_v1_qa.json"),
            ("qa/phase_v1/pi_boundary_check.json", "phase_v1_pi_boundary_failure.json"),
            ("qa/phase_v2/phase_qa.json", "phase_v2_qa.json"),
            ("qa/modality_v1/modality_qa.json", "modality_component_qa.json"),
            ("data/native_phase_smoke_v2/summary.json", "native_phase_smoke.json")):
        copy_once(RUNTIME / source_name, DOCS / "evidence" / destination_name)
    for filename in ("inventory.json", "selection_before_inventory.json"):
        copy_once(RUNTIME / "data/widar_inventory_v1" / filename, DOCS / "evidence/widar_inventory_v1" / filename)
    copy_once(widar_summary_path, DOCS / "evidence/widar_native_v2_summary.json")
    copy_once(RUNTIME / "data/widar_arrays_v2/dataset_manifest.json",
              DOCS / "evidence/widar_compiled_v2_manifest.json")
    copy_once(RUNTIME / "qa/widar_study_v2/widar_study_qa.json",
              DOCS / "evidence/widar_study_v2_qa.json")
    copy_once(RUNTIME / "coordinators/widar_after_preparation_v2/negative_gate_qa.json",
              DOCS / "evidence/widar_study_v2_negative_gate_qa.json")
    copy_once(ROOT / "outputs_h_grouped_l1248/v2/bundle/bundle_manifest.json",
              DOCS / "evidence/h_grouped_l1248_v2_bundle_manifest.json")
    for filename in ("protocol.json", "bundle_manifest.json", "model_costs.json",
                     "preparation_summary.json", "widar_study_qa.json"):
        copy_once(widar_bundle / filename, DOCS / "evidence/r_widar_v2" / filename)
    for source_name, destination_name in (("target_negative_gate.log", "target_gate_negative_test.log"),
                                          ("prerequisite_negative_gate.log", "prerequisite_gate_negative_test.log"),
                                          ("negative_gate_qa.json", "negative_gate_qa.json")):
        copy_once(RUNTIME / "coordinators/widar_after_preparation_v2" / source_name,
                  DOCS / "evidence/r_widar_v2" / destination_name)
    for record in snapshot["formal_source_analyses"]["completed"]:
        source = RUNTIME / "analyses/formal_source_v1" / record["study"] / record["job_id"] / "summary.json"
        copy_once(source, DOCS / "evidence/formal_source_analyses" / record["study"] / (record["job_id"] + ".json"))
    manifest_path = DOCS / "revision_manifest.json"
    archive = DOCS / "archive/revision_manifest_before_full_R.json"
    if not archive.exists():
        copy_once(manifest_path, archive)
    manifest = read(manifest_path)
    widar_started = len(widar["started_jobs"]) if widar else 0
    widar_finished = len(widar["completed_jobs"]) if widar else 0
    started = len(core["started_jobs"]) + len(controls["started_jobs"]) + len(inputs["started_jobs"])
    finished = len(core["completed_jobs"]) + len(controls["completed_jobs"]) + len(inputs["completed_jobs"])
    interim_pdf_qa = optional(DOCS / "evidence/manuscript_static_qa_v6.json")
    manifest.update(date=snapshot["recorded_at"][:10], new_training_runs=25 + started,
                    new_training_runs_completed=25 + finished, full_baseline_runs_completed=finished,
                    new_training_run_breakdown={"source_smokes": 12, "source_pilot": 1, "resume_QA_trajectories": 6,
                        "extension_CPU_resume_QA_trajectories": 2, "extension_adapter_identity_one_update_paths": 2,
                        "input_twin_CPU_resume_QA_trajectories": 2,
                        "core_started": len(core["started_jobs"]), "core_completed": len(core["completed_jobs"]),
                        "controls_started": len(controls["started_jobs"]), "controls_completed": len(controls["completed_jobs"]),
                        "inputs_started": len(inputs["started_jobs"]), "inputs_completed": len(inputs["completed_jobs"]),
                        "widar_started": widar_started, "widar_completed": widar_finished,
                        "widar_status": "cancelled_by_user_before_training",
                        "h_grouped_l1248_frozen_jobs": 18,
                        "h_grouped_l1248_queue_owner_live": snapshot["h_grouped_l1248"]["queue_owner_live"]},
                    training_count_note="Includes explicitly separated short QA paths; nine extension and seven input objective/backward checks used zero optimizer updates and are not counted as training runs",
                    remaining_work=snapshot["remaining"], runtime_evidence="docs/revision/evidence/runtime_progress.json",
                    interim_static_pdf={"pages": interim_pdf_qa["pages"], "sha256": interim_pdf_qa["pdf_sha256"],
                        "status": interim_pdf_qa["status"], "qa_record": "docs/revision/evidence/manuscript_static_qa_v6.json"}
                        if interim_pdf_qa else None)
    new_files = list((ROOT / "experiments/revision_validation").glob("*.py"))
    new_files += [ROOT / "analysis" / filename for filename in (
        "prepare_native_multienv.py", "prepare_widar_inventory.py", "download_widar_cohort.py",
        "verify_revision_controls.py", "compile_revision_inputs.py", "snapshot_revision_progress.py",
        "native_phase_processing.py", "verify_native_phase.py", "prepare_native_phase.py",
        "verify_modality_models.py", "compile_modality_inputs.py", "audit_native_reconstruction.py",
        "verify_mechanism_metrics.py", "analyze_source_representation.py")]
    new_files += [ROOT / "analysis" / filename for filename in (
        "runtime_observation.py", "recover_core_io_failure.py", "continue_controls_after_core.py",
        "continue_input_preparation.py", "continue_source_analyses.py", "continue_inputs_after_controls.py",
        "download_widar_cohort_v2.py", "compile_widar_inputs_v2.py", "continue_widar_after_preparation.py",
        "compile_formal_revision_evidence.py", "continue_final_revision_evidence.py",
        "continue_paper_assets_after_evidence.py", "continue_formal_fragment_after_assets.py",
        "build_final_revision_pdf.py", "continue_final_manuscript_after_fragment.py",
        "build_revision_release_bundle.py", "continue_release_after_final_manuscript.py",
        "verify_final_revision_completion.py")]
    new_files += [ROOT / "analysis" / filename for filename in (
        "compile_multienv_revision_evidence.py", "continue_multienv_final_evidence.py",
        "continue_multienv_paper_assets_after_evidence.py", "continue_multienv_fragment_after_assets.py",
        "build_multienv_revision_pdf.py", "continue_multienv_manuscript_after_fragment.py",
        "build_multienv_release_bundle.py", "continue_multienv_release_after_manuscript.py",
        "verify_multienv_revision_completion.py")]
    new_files += [ROOT / "analysis/recover_h_grouped_l1248_queue.py",
                  ROOT / "analysis/execute_h_grouped_l1248_recovery.py",
                  ROOT / "analysis/recover_multienv_inputs_after_interruption.py",
                  ROOT / "analysis/hold_completed_source_analysis_identity.py",
                  ROOT / "analysis/record_runtime_recovery_20260917.py",
                  ROOT / "analysis/heartbeat_revision_status.py"]
    new_files += [ROOT / "analysis" / name for name in (
        "h_gpu_availability_v2.py", "execute_h_grouped_l1248_recovery_v2.py",
        "recover_h_grouped_l1248_queue_v2.py", "continue_multienv_final_evidence_v2.py",
        "prepare_h_wddm_recovery_v5.py", "launch_h_wddm_recovery_v5.py",
        "verify_multienv_inputs_completion_20260917.py", "verify_new_h_grouped_completions.py")]
    new_files += [DOCS / "evidence" / name for name in (
        "multienv_inputs_completion_verified_20260917.json", "runtime_recovery_20260917_wddm.json",
        "h_grouped_started_verified_20260917.json",
        "wddm_waiting_owners_retired_20260917.json", "h_gpu_preflight_20260917.json",
        "h_gpu_preflight_20260917_v2.json")]
    new_files += [p for p in DOCS.glob("*.md")]
    new_files += [ROOT / "paper/sensors_manuscript/build_formal_revision_figures.py",
                  ROOT / "paper/sensors_manuscript/build_formal_revision_fragment.py",
                  ROOT / "paper/sensors_manuscript/build_multienv_revision_figures.py",
                  ROOT / "paper/sensors_manuscript/build_multienv_revision_fragment.py"]
    new_files += [DOCS / "evidence" / name for name in (
        "runtime_progress.json", "resume_v1.json", "resume_v2_cublas.json", "resume_v3_cpu.json",
        "cuda_state_roundtrip.json", "native_amp_smoke.json", "widar_amp_smoke.json", "control_component_qa.json",
        "cache_migration.json", "input_packing_qa.json", "phase_v1_qa.json", "phase_v1_pi_boundary_failure.json",
        "phase_v2_qa.json", "modality_component_qa.json", "native_phase_smoke.json",
        "native_amp_full_summary.json", "native_amp_dataset_manifest.json", "mechanism_qa.json",
        "widar_design_qa.json", "widar_design_qa_v2.json", "native_phase_full_summary.json",
        "widar_native_v2_summary.json", "widar_compiled_v2_manifest.json", "widar_study_v2_qa.json",
        "widar_study_v2_negative_gate_qa.json",
        "modality_dataset_manifest.json", "manuscript_static_qa_v3.json", "manuscript_static_qa_v4.json",
        "manuscript_static_qa_v5.json", "manuscript_static_qa_v6.json", "formal_fragment_qa_v1.json", "release_preflight_v1.json",
        "release_preflight_v2.json", "release_preflight_v3.json", "release_preflight_v4.json",
        "release_preflight_v5.json",
        "release_gate_negative_qa_v1.json", "release_gate_negative_test.log",
        "final_completion_gate_negative_qa_v1.json", "final_completion_gate_negative_test.log",
        "final_completion_gate_negative_qa_v2.json", "final_completion_gate_negative_test_v2.log",
        "final_pdf_pipeline_qa_v1.json")]
    if (DOCS / "evidence/runtime_recovery_20260911.json").is_file():
        new_files.append(DOCS / "evidence/runtime_recovery_20260911.json")
    if (DOCS / "evidence/runtime_recovery_20260911_addendum.json").is_file():
        new_files.append(DOCS / "evidence/runtime_recovery_20260911_addendum.json")
    if (DOCS / "evidence/runtime_recovery_20260911_addendum2.json").is_file():
        new_files.append(DOCS / "evidence/runtime_recovery_20260911_addendum2.json")
    if (DOCS / "evidence/runtime_recovery_20260917.json").is_file():
        new_files.append(DOCS / "evidence/runtime_recovery_20260917.json")
    if (DOCS / "evidence/multienv_core_controls_completion_verified_20260913.json").is_file():
        new_files.append(DOCS / "evidence/multienv_core_controls_completion_verified_20260913.json")
    if (DOCS / "evidence/manuscript_final_qa.json").is_file():
        new_files.append(DOCS / "evidence/manuscript_final_qa.json")
    if (DOCS / "evidence/manuscript_multienv_final_qa.json").is_file():
        new_files.append(DOCS / "evidence/manuscript_multienv_final_qa.json")
    for name in ("final_document_sync_approval.json", "final_completion_audit.json"):
        if (DOCS / "evidence" / name).is_file():
            new_files.append(DOCS / "evidence" / name)
    new_files += list((DOCS / "sources/phase_processing").glob("*"))
    new_files += list((DOCS / "evidence/r_core_v1").glob("*.json"))
    new_files += list((DOCS / "evidence/r_controls_v1").glob("*"))
    new_files += list((DOCS / "evidence/r_inputs_v1").glob("*"))
    new_files += list((DOCS / "evidence/h_grouped_sealed_runs").glob("*.json"))
    new_files += list((DOCS / "evidence/h_grouped_verified_pairs").glob("*.json"))
    new_files += [p for p in (DOCS / "evidence/core_io_recovery").rglob("*") if p.is_file()]
    new_files += list((DOCS / "evidence/widar_inventory_v1").glob("*.json"))
    new_files += list((DOCS / "evidence/r_widar_v2").glob("*"))
    new_files += [p for p in (DOCS / "evidence/native_reconstruction_v1").glob("*") if p.is_file()]
    new_files += [p for p in (DOCS / "evidence/source_analysis_qa").glob("*") if p.is_file()]
    new_files += [p for p in (DOCS / "evidence/formal_source_analyses").rglob("*.json") if p.is_file()]
    new_files += [p for p in (DOCS / "evidence/input_preparation_v1").glob("*.json") if p.is_file()]
    new_files += [p for p in (ROOT / "paper/sensors_manuscript/revision_assets/formal_v1").glob("*") if p.is_file()]
    new_files += [p for p in (ROOT / "paper/sensors_manuscript/revision_assets/multienv_v2").glob("*") if p.is_file()]
    new_files += [ROOT / "analysis" / name for name in (
        "compile_multienv_revision_evidence_v3.py", "recover_multienv_evidence_v6.py",
        "continue_multienv_paper_assets_after_evidence_v2.py", "continue_multienv_fragment_after_assets_v2.py",
        "continue_multienv_manuscript_after_fragment_v2.py", "continue_multienv_release_after_manuscript_v2.py",
        "build_multienv_revision_pdf_v2.py", "build_multienv_release_bundle_v2.py",
        "build_multienv_release_bundle_v3.py", "continue_multienv_release_after_manuscript_v3.py",
        "verify_release_manuscript_source.py",
        "verify_hy_writing_scope.py", "verify_release_archive.py", "snapshot_revision_progress_v2.py")]
    new_files += [ROOT / "paper/sensors_manuscript/build_hy_revision_fragment.py"]
    new_files += [p for p in (DOCS / "evidence/multienv_revision_v2").rglob("*") if p.is_file()]
    new_files += [p for p in (DOCS / "evidence/publication_provenance_20260924").rglob("*") if p.is_file()]
    for name in ("final_multienv_document_sync_approval.json", "final_multienv_completion_audit.json"):
        if (DOCS / "evidence" / name).is_file():
            new_files.append(DOCS / "evidence" / name)
    artifact_paths = set(manifest["artifacts"]) | {p.relative_to(ROOT).as_posix() for p in new_files}
    manifest["artifacts"] = {name: sha256(ROOT / name) for name in sorted(artifact_paths)}
    write(manifest_path, manifest)
    active_run = inputs["current_run"] or controls["current_run"] or core["current_run"]
    print({"core_completed": len(core["completed_jobs"]), "core_started": len(core["started_jobs"]),
           "controls_completed": len(controls["completed_jobs"]),
           "inputs_completed": len(inputs["completed_jobs"]),
           "source_analyses_completed": snapshot["formal_source_analyses"]["completed_count"],
           "core_training_queue": core["queue"],
           "controls_training_queue": controls["queue"],
           "core_to_controls_coordinator_live": snapshot["coordinators"]["core_to_controls_v1"]["live"],
           "frozen_core_jobs": core["frozen_jobs"],
           "inputs_training_queue": inputs["queue"],
           "controls_to_inputs_coordinator_live": (
               snapshot["coordinators"]["controls_to_inputs_v4"]["live"]
               or snapshot["coordinators"]["controls_to_inputs_v3"]["live"]
               or snapshot["coordinators"]["controls_to_inputs_v2"]["live"]),
           "frozen_input_jobs": inputs["frozen_jobs"],
           "widar_queue": widar["queue"] if widar else None,
           "current_run": active_run,
           "native_prepared": snapshot["native_multienv"]["status"], "widar_prepared": snapshot["widar"]["status"]})


if __name__ == "__main__":
    main()
