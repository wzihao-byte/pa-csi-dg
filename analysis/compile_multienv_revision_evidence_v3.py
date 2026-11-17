"""Compile the revised MultiEnv-only evidence plus the distinct H supplement.

Widar preparation/freeze records are retained separately as cancelled work and
are never accepted here as performance evidence.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(os.environ.get("PA_CSI_DG_ROOT", Path(__file__).resolve().parents[1])).resolve()
sys.path.insert(0, str(ROOT))

from analysis.compile_formal_revision_evidence import (
    latex_table,
    resource_rows_for,
    summarize_gradient_arms,
    summarize_gradient_runs,
    summarize_mechanisms,
    summarize_residual_arms,
    verify_source_analyses,
    verify_study,
    write_csv,
)
from experiments.revision_validation.extended_engine import verify_extension
from experiments.revision_validation.study_io import now, read, sha256, verify_bundle, write


MULTIENV_STUDIES = (
    ("core", "r_core_v1", verify_bundle, 36),
    ("controls", "r_controls_v1", verify_extension, 81),
    ("inputs", "r_inputs_v1", verify_extension, 63),
)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def verify_h_supplement(root: Path):
    bundle, queue = root / "bundle", root / "queue"
    owner = read(queue / "owner.json")
    import psutil
    try:
        live = abs(psutil.Process(owner["pid"]).create_time() - owner["process_create_time"]) < 0.01
    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
        live = False
    require(not live, "H supplement queue owner is still active")
    require(not (queue / "failed.json").exists(), "H supplement retained a failure")
    completed = read(queue / "completed.json")
    require(completed.get("status") == "completed" and completed.get("jobs") == 18,
            "H supplement queue is incomplete")
    if owner.get("superseded_owner"):
        superseded = Path(owner["superseded_owner"])
        require(superseded.is_file() and sha256(superseded) == owner.get("superseded_owner_sha256"),
                "H recovery predecessor record differs")
        require(owner.get("widar_dependency") is False and
                owner.get("training_started_at_recovery") is False,
                "H recovery did not preserve the pre-training/Widar-free boundary")
        require(owner.get("code_sha256") == sha256(ROOT / "analysis/recover_h_grouped_l1248_queue_v2.py"),
                "H recovery queue code differs from its owner record")
        require(completed.get("owner_sha256") == sha256(queue / "owner.json") and
                completed.get("execution_wrapper_sha256") ==
                sha256(ROOT / "analysis/execute_h_grouped_l1248_recovery_v2.py"),
                "H recovery completion provenance differs")

    recovery_path = ROOT / "docs/revision/evidence/runtime_recovery_20260917_wddm.json"
    require(sha256(recovery_path) == "9f1506a88d6fff7c35f2f433ef17da9e364034df48be3c33b1a434bebd809f98",
            "WDDM operational recovery record differs")
    recovery = read(recovery_path)
    for name, digest in recovery["code_sha256"].items():
        require(sha256(ROOT / "analysis" / name) == digest, f"Recovery code differs: {name}")
    require(recovery["current_h_owner_sha256"] == sha256(queue / "owner.json"), "Recovered owner differs")
    require(owner["gpu_guard_sha256"] == recovery["code_sha256"]["h_gpu_availability_v2.py"], "GPU gate differs")
    require(owner["execution_wrapper_sha256"] == completed["execution_wrapper_sha256"], "Wrapper owner/completion differ")
    require(owner["completed_input_verification_sha256"] == recovery["input_completion_verification_sha256"], "Input recovery binding differs")

    manifest = read(bundle / "bundle_manifest.json")
    for relative, expected in manifest["files_sha256"].items():
        require(sha256(bundle / relative) == expected, f"H bundle member changed: {relative}")
    for path, expected in manifest["external_code_sha256"].items():
        require(sha256(Path(path)) == expected, f"H historical dependency changed: {path}")
    approval = read(root / "qa/approval.json")
    require(approval.get("status") == "passed" and
            approval.get("bundle_manifest_sha256") == sha256(bundle / "bundle_manifest.json"),
            "H supplement QA approval is not bound to this bundle")

    dry = subprocess.run(
        [sys.executable, "-B", "-X", "utf8", str(bundle / "code/run.py"),
         "--dry-run", "--bundle", str(bundle)],
        cwd=bundle / "code", capture_output=True, text=True,
        creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
    )
    require(dry.returncode == 0, "Frozen H runner could not revalidate all sealed attempts")
    dry_report = json.loads(dry.stdout)
    require(dry_report.get("status") == "dry_run_passed" and len(dry_report.get("plan", [])) == 18,
            "Frozen H runner returned the wrong completed grid")
    require(all(row.get("action") == "skip" for row in dry_report["plan"]),
            "At least one H supplement job is not sealed and complete")

    result_path = root / "factorial_results.json"
    result = read(result_path)
    require(result.get("status") == "complete" and result.get("job_count") == 18,
            "H result grid is incomplete")
    require(completed.get("results_sha256") == sha256(result_path), "H queue/result binding differs")
    require(result.get("bundle_manifest_sha256") == sha256(bundle / "bundle_manifest.json"),
            "H results belong to another bundle")
    if owner.get("superseded_owner"):
        invocations = [path for path in (root / "runner_invocations").glob("*/completed.json")
                       if read(path).get("results_sha256") == sha256(result_path)]
        require(len(invocations) == 1, "Expected one completed recovered H invocation")
        invocation = read(invocations[0])
        require(invocation.get("queue_owner_sha256") == sha256(queue / "owner.json") and
                invocation.get("execution_wrapper_sha256") == completed.get("execution_wrapper_sha256"),
                "Recovered H invocation provenance differs")
    registry = read(bundle / "registry.json")
    expected = {(row["target"], row["seed"], row["arm"], row["job_id"]) for row in registry["jobs"]}
    observed = {(row["target_env"], row["seed"], row["arm"], row["job_id"]) for row in result["rows"]}
    require(observed == expected and len(observed) == 18, "H result identities differ from the registry")
    for row in result["rows"]:
        attempt = Path(row["attempt_dir"])
        require(sha256(attempt / "attempt_completed.json") == row["completion_manifest_sha256"],
                f"H completion changed: {row['job_id']}")
        require(sha256(attempt / "payload/run/metrics.json") == row["metrics_file_sha256"],
                f"H metrics changed: {row['job_id']}")

    analyses = []
    for path in (root / "analysis").glob("*/completed.json"):
        value = read(path)
        if value.get("status") == "complete" and value.get("results_sha256") == sha256(result_path):
            analyses.append(path.parent)
    require(len(analyses) == 1, "Expected one completed H analysis bound to the final results")
    analysis = analyses[0]
    completion = read(analysis / "completed.json")
    for name, expected_hash in completion["files_sha256"].items():
        require(sha256(analysis / name) == expected_hash, f"H analysis artifact changed: {name}")
    summary = read(analysis / "results.json")
    require(summary.get("formal_runs") == 18 and summary.get("population_inference") is False,
            "H paired analysis has the wrong scope")
    return {"root": root, "bundle": bundle, "manifest": manifest, "approval": approval,
            "result_path": result_path, "result": result, "analysis": analysis,
            "analysis_summary": summary, "dry_report": dry_report,
            "bundle_manifest_sha256": sha256(bundle / "bundle_manifest.json"),
            "qa_approval_sha256": sha256(root / "qa/approval.json"),
            "queue_completion_sha256": sha256(queue / "completed.json"),
            "results_sha256": sha256(result_path),
            "analysis_completion_sha256": sha256(analysis / "completed.json")}


def h_latex_table(rows):
    lookup = {(r["target"], r["metric"], r["arm"]): r for r in rows}
    lines = [r"\begin{table*}[tbp]",
             r"\caption{Original H L1248 with grouped source train/validation membership.\label{tab:hgrouped}}",
             r"\centering\small", r"\begin{tabular}{llrrr}", r"\toprule",
             r"Target & Metric & H10 & H11 & H11$-$H10 \\", r"\midrule"]
    labels = {"accuracy": "Accuracy", "f1_macro": "Macro-F1"}
    for target in ("E1", "E2", "E3", "E1-E3_equal_weight"):
        for metric in ("accuracy", "f1_macro"):
            values = [lookup[(target, metric, arm)] for arm in ("H10", "H11", "H11_minus_H10")]
            cells = [f"{v['mean_percent_or_pp']:.2f} ({v['seed_sd']:.2f})" for v in values]
            target_label = target.replace("_", r"\_")
            lines.append(f"{target_label} & {labels[metric]} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}",
              r"\begin{flushleft}\footnotesize Values are percentages and paired percentage-point differences; parentheses are sample SD across three training seeds on one fixed grouped source partition. They are not population intervals.\end{flushleft}",
              r"\end{table*}", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, default=Path("D:/pa-csi-dg-revision-runtime"))
    parser.add_argument("--h-root", type=Path, default=ROOT / "outputs_h_grouped_l1248/v2")
    parser.add_argument("--source-completion", type=Path,
                        default=Path("D:/pa-csi-dg-revision-runtime/coordinators/formal_source_analysis_v3/completion.json"))
    parser.add_argument("--output", type=Path,
                        default=ROOT / "docs/revision/evidence/multienv_revision_v2")
    args = parser.parse_args()
    runtime, output = args.runtime.resolve(), args.output.resolve()
    if output.exists():
        raise FileExistsError("Preserve the existing current-scope evidence version")

    bundles = {}
    for short, folder, verifier, count in MULTIENV_STUDIES:
        path = runtime / "studies" / folder
        protocol, digest, statuses, gate, summary, summary_hash = verify_study(path, verifier, count)
        bundles[short] = {"path": path, "protocol": protocol, "bundle_sha256": digest,
                          "statuses": statuses, "target_gate": gate, "summary": summary,
                          "evaluation_summary_sha256": summary_hash}
    source_completion = args.source_completion.resolve()
    source_rows, gradient_rows, residual_rows, source_hashes, source_completion_hash = verify_source_analyses(
        runtime / "analyses/formal_source_v1", bundles, source_completion)
    h = verify_h_supplement(args.h_root.resolve())

    environment_rows, effect_rows, run_rows, resource_rows = [], [], [], []
    for study, details in bundles.items():
        environment_rows += [{"study": study, **row} for row in details["summary"]["environment_summary"]]
        effect_rows += [{"study": study, **row} for row in details["summary"].get("paired_effects", [])]
        configs = {config["job_id"]: config for config in details["protocol"]["jobs"]}
        core_parameters = (read(details["path"] / "shared_initialization_audit.json")["total_parameters"]
                           if study == "core" else {})
        for row in details["summary"]["runs"]:
            status, config = details["statuses"][row["job_id"]], configs[row["job_id"]]
            prefix = f"evaluation/{row['job_id']}"
            run_rows.append({"study": study, "job_id": row["job_id"], "arm": row["arm"],
                "target": row["target"], "seed": row["seed"], "macro_f1": row["metrics"]["macro_f1"],
                "accuracy": row["metrics"]["accuracy"], "source_macro_f1": row["source_metrics"]["macro_f1"],
                "best_epoch": status["best_epoch"], "optimizer_updates": status["optimizer_updates"],
                "elapsed_seconds": status["elapsed_seconds"],
                "peak_cuda_allocated_bytes": status["peak_cuda_allocated_bytes_this_process"],
                "parameters": config.get("total_parameters", core_parameters.get(row["arm"])),
                "checkpoint_sha256": row["checkpoint_sha256"], "source_status_sha256": status["status_sha256"],
                "result_sha256": details["summary"]["artifacts_sha256"][prefix + "/result.json"],
                "predictions_sha256": details["summary"]["artifacts_sha256"][prefix + "/predictions.json"]})
        resource_rows += resource_rows_for(study, details)
    require(len(run_rows) == 180 and len(source_rows) == 117, "Current MultiEnv evidence counts differ")

    output.mkdir(parents=True)
    write_csv(output / "multienv_formal_runs.csv", run_rows, list(run_rows[0]))
    write_csv(output / "environment_summary.csv", environment_rows, list(environment_rows[0]))
    write_csv(output / "paired_effects.csv", effect_rows, list(effect_rows[0]))
    mechanism_rows = summarize_mechanisms(source_rows)
    gradient_run_rows = summarize_gradient_runs(gradient_rows)
    gradient_arm_rows = summarize_gradient_arms(gradient_run_rows)
    residual_arm_rows = summarize_residual_arms(residual_rows)
    write_csv(output / "source_mechanism_runs.csv", source_rows, list(source_rows[0]))
    write_csv(output / "source_mechanism_summary.csv", mechanism_rows, list(mechanism_rows[0]))
    write_csv(output / "source_gradient_probes.csv", gradient_rows, list(gradient_rows[0]))
    write_csv(output / "source_gradient_run_summary.csv", gradient_run_rows, list(gradient_run_rows[0]))
    write_csv(output / "source_gradient_arm_summary.csv", gradient_arm_rows, list(gradient_arm_rows[0]))
    write_csv(output / "source_residual_runs.csv", residual_rows, list(residual_rows[0]))
    write_csv(output / "source_residual_arm_summary.csv", residual_arm_rows, list(residual_arm_rows[0]))
    write_csv(output / "resource_summary.csv", resource_rows, list(resource_rows[0]))
    for name in ("paired_seed_summary.csv", "historical_comparison.csv", "results.json"):
        shutil.copy2(h["analysis"] / name, output / ("h_" + name))
    shutil.copy2(h["result_path"], output / "h_grouped_runs.json")
    for study, caption, label in (
        ("core", "Frozen common-protocol R factorial on the three MultiEnv development environments.", "tab:formalcore"),
        ("controls", "Frozen capacity, mask, temporal-intervention, and same-backbone DG controls on MultiEnv.", "tab:formalcontrols"),
        ("inputs", "Frozen input-processing and modality sensitivity study on MultiEnv.", "tab:formalinputs"),
    ):
        (output / f"{study}_table.tex").write_text(
            latex_table(study, bundles[study]["summary"], caption, label), encoding="utf-8")
    with (h["analysis"] / "paired_seed_summary.csv").open(encoding="utf-8") as stream:
        h_summary_rows = list(csv.DictReader(stream))
    for row in h_summary_rows:
        for field in ("mean_percent_or_pp", "seed_sd", "seed_42", "seed_52", "seed_62"):
            row[field] = float(row[field])
    (output / "h_grouped_table.tex").write_text(h_latex_table(h_summary_rows), encoding="utf-8")

    evidence = {"completed_at": now(), "schema": "multienv-revision-evidence/v3",
        "writing_scope": "Original H/Y plus verified grouped-H; R retained as technical audit only",
        "wddm_recovery_sha256": sha256(ROOT / "docs/revision/evidence/runtime_recovery_20260917_wddm.json"),
        "all_current_scope_studies_completed": True, "population_inference_performed": False,
        "independent_cross_dataset_confirmation_claimed": False, "widar_training_cancelled_by_user": True,
        "multienv_target_gates_verified": True, "multienv_formal_run_count": len(run_rows),
        "formal_source_analysis_count": len(source_rows), "h_supplement_run_count": 18,
        "distinct_protocol_total_training_runs": len(run_rows) + 18,
        "studies": {study: {"bundle": str(details["path"]), "bundle_sha256": details["bundle_sha256"],
            "job_count": len(details["protocol"]["jobs"]),
            "evaluation_summary_sha256": details["evaluation_summary_sha256"],
            "equal_environment_macro_f1": details["summary"]["equal_environment_macro_f1"],
            "inference_scope": details["summary"]["inference_scope"]} for study, details in bundles.items()},
        "h_grouped_l1248": {"bundle": str(h["bundle"]), "bundle_manifest_sha256": h["bundle_manifest_sha256"],
            "qa_approval_sha256": h["qa_approval_sha256"], "queue_completion_sha256": h["queue_completion_sha256"],
            "results_sha256": h["results_sha256"], "analysis_completion_sha256": h["analysis_completion_sha256"],
            "job_count": 18, "population_inference": False},
        "formal_source_analysis_completion_sha256": source_completion_hash,
        "formal_source_analysis_summary_sha256": source_hashes,
        "limitations": ["E1-E3 were inspected during development and remain development environments.",
            "Seed SD describes optimization variation on one fixed partition, not population uncertainty.",
            "The H supplement changes both source train and validation membership relative to historical H.",
            "The two-arm H supplement tests the L1248 SupCon increment and is not a full factorial interaction.",
            "Widar training was cancelled by the user; its preparation and frozen bundle are not performance evidence."]}
    generated = [path for path in output.iterdir() if path.is_file()]
    evidence["artifacts_sha256"] = {path.name: sha256(path) for path in generated}
    write(output / "verified_multienv_results.json", evidence)
    write(output / "manifest.json", {"files_sha256": {path.name: sha256(path)
          for path in output.iterdir() if path.is_file()}})
    print({"multienv_runs": len(run_rows), "source_analyses": len(source_rows),
           "h_supplement_runs": 18, "output": str(output), "verified": True})


if __name__ == "__main__":
    main()
