"""Verify every formal revision study and emit immutable paper-ready evidence.

This script is intentionally unusable while any source runner remains active or
before every deferred target gate and formal source analysis has completed.
It reports descriptive environment means, seed SDs, and paired differences; it
does not create population confidence intervals or p-values.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.revision_validation.extended_engine import verify_extension
from experiments.revision_validation.study_io import (
    bound_path,
    now,
    read,
    sha256,
    verify_bundle,
    verify_completed,
    write,
)


STUDIES = (
    ("core", "r_core_v1", verify_bundle, 36),
    ("controls", "r_controls_v1", verify_extension, 81),
    ("inputs", "r_inputs_v1", verify_extension, 63),
    ("widar", "r_widar_v2", verify_extension, 36),
)


def active_owner(bundle: Path):
    for process in psutil.process_iter(["pid", "cmdline", "create_time"]):
        command = process.info["cmdline"] or []
        if "--bundle" not in command:
            continue
        index = command.index("--bundle") + 1
        if index < len(command) and Path(command[index]).resolve() == bundle.resolve():
            return {"pid": process.info["pid"], "create_time": process.info["create_time"]}
    return None


def verify_study(bundle: Path, verifier, expected_jobs: int):
    owner = active_owner(bundle)
    if owner:
        raise RuntimeError(f"Formal runner remains active for {bundle}: {owner}")
    protocol, digest = verifier(bundle)
    if len(protocol["jobs"]) != expected_jobs:
        raise RuntimeError(f"Wrong job count for {bundle}")
    if (bundle / "queue_failure.json").exists():
        raise RuntimeError(f"Unresolved formal failure in {bundle}")
    queue = read(bundle / "queue_status.json")
    expected_queue_status = "source_training_completed" if verifier is verify_bundle else "completed"
    if queue.get("status") != expected_queue_status:
        raise RuntimeError(f"Formal queue is not complete: {bundle}")
    statuses = {}
    for config in protocol["jobs"]:
        status = verify_completed(bundle / "runs" / config["job_id"], config)
        statuses[config["job_id"]] = {
            "status_sha256": sha256(bundle / "runs" / config["job_id"] / "status.json"),
            "best_epoch": status["best_epoch"],
            "elapsed_seconds": status["elapsed_seconds"],
            "peak_cuda_allocated_bytes_this_process": status["peak_cuda_allocated_bytes_this_process"],
            "optimizer_updates": status["optimizer_updates"],
        }
    gate = read(bundle / "target_gate.json")
    if gate.get("bundle_sha256") != digest:
        raise RuntimeError(f"Target gate is not bound to {bundle}")
    gated = gate.get("verified_source_jobs")
    if gated is None and gate.get("all_source_jobs_completed") == expected_jobs:
        gated = [config["job_id"] for config in protocol["jobs"]]
    if set(gated or []) != {config["job_id"] for config in protocol["jobs"]}:
        raise RuntimeError(f"Target gate does not cover every source job in {bundle}")
    summary_path = bundle / "evaluation" / "summary.json"
    summary = read(summary_path)
    if {row["job_id"] for row in summary["runs"]} != {config["job_id"] for config in protocol["jobs"]}:
        raise RuntimeError(f"Evaluation summary does not cover every job in {bundle}")
    for relative, expected in summary["artifacts_sha256"].items():
        if sha256(bound_path(bundle, relative)) != expected:
            raise RuntimeError(f"Evaluation artifact differs: {bundle / relative}")
    if queue.get("evaluation_summary_sha256") and queue["evaluation_summary_sha256"] != sha256(summary_path):
        raise RuntimeError(f"Queue/evaluation binding differs in {bundle}")
    return protocol, digest, statuses, gate, summary, sha256(summary_path)


def verify_source_analyses(directory: Path, bundles: dict, completion_path: Path | None = None):
    completion_path = (completion_path or
                       directory.parent.parent / "coordinators/formal_source_analysis_v1/completion.json")
    completion = read(completion_path)
    if completion.get("target_data_loaded") or completion.get("completed_count") != 117:
        raise RuntimeError("Formal source analysis is incomplete or has the wrong scope")
    expected = []
    job_configs = {}
    for study in ("core", "controls"):
        protocol = bundles[study]["protocol"]
        for config in protocol["jobs"]:
            key = ("r_core_v1" if study == "core" else "r_controls_v1") + "/" + config["job_id"]
            expected.append(key)
            job_configs[key] = config
    if set(completion["completed"]) != set(expected):
        raise RuntimeError("Formal source-analysis job identities differ")
    rows, gradient_rows, residual_rows, hashes = [], [], [], {}
    for key in sorted(expected):
        study_name, job_id = key.split("/", 1)
        path = directory / study_name / job_id / "summary.json"
        if sha256(path) != completion["completed"][key]:
            raise RuntimeError(f"Source-analysis completion hash differs for {key}")
        summary = read(path)
        if summary["job_id"] != job_id or not summary["full_trained_model_analysis"] or summary["target_data_loaded"]:
            raise RuntimeError(f"Source-analysis scope differs for {key}")
        if summary["source_prediction_differences_from_recorded_backend"] != 0:
            raise RuntimeError(f"Source predictions were not reproduced for {key}")
        for filename, expected_hash in summary["artifacts_sha256"].items():
            if sha256(path.parent / filename) != expected_hash:
                raise RuntimeError(f"Source-analysis artifact differs for {key}/{filename}")
        config = job_configs[key]
        feature = summary["scatter"]["feature"]["raw_feature"]
        rows.append({
            "study": "core" if study_name == "r_core_v1" else "controls",
            "job_id": job_id,
            "target": config["target_name_only"],
            "seed": config["seed"],
            "arm": config.get("control_id", config.get("arm")),
            "selected_epoch": summary["selected_epoch"],
            "source_environment_probe_balanced_accuracy": summary["environment_probe"]["balanced_accuracy"],
            "feature_between_to_within_class_ratio": feature["between_to_within_class_ratio"],
            "feature_environment_scatter_within_class": feature["environment_scatter_within_class"],
            "feature_equal_class_domain_squared_distance": feature["equal_class_mean_domain_squared_distance"],
            "gradient_probe_batches": summary["gradient_probe_batches"],
        })
        for path_name, values in summary["residual_summary"].items():
            residual_rows.append({"study": rows[-1]["study"], "job_id": job_id,
                                  "target": config["target_name_only"], "seed": config["seed"],
                                  "arm": rows[-1]["arm"], "path": path_name, **values})
        probes = read(path.parent / "gradient_probes.json")
        if len(probes) != summary["gradient_probe_batches"]:
            raise RuntimeError(f"Gradient-probe count differs for {key}")
        for probe in probes:
            for group, values in probe["alignment"].items():
                if not isinstance(values, dict):
                    continue
                gradient_rows.append({"study": rows[-1]["study"], "job_id": job_id,
                    "target": config["target_name_only"], "seed": config["seed"], "arm": rows[-1]["arm"],
                    "batch_index": probe["batch_index"], "regularizer_family": probe["regularizer_family"],
                    "parameter_group": group, "primary_gradient_norm": values.get("primary_gradient_norm"),
                    "regularizer_gradient_norm": values.get("regularizer_gradient_norm"),
                    "cosine": values.get("cosine"),
                    "weighted_regularizer_to_primary_norm_ratio": values.get("weighted_regularizer_to_primary_norm_ratio"),
                    "regularizer_weight": values.get("regularizer_weight"),
                    "parameter_count": values.get("parameter_count")})
        hashes[key] = sha256(path)
    return rows, gradient_rows, residual_rows, hashes, sha256(completion_path)


def write_csv(path: Path, rows: list[dict], fields: list[str]):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def format_cell(value, sd):
    return f"{100 * value:.2f} ({100 * sd:.2f})"


def latex_table(study: str, summary: dict, caption: str, label: str):
    environments = sorted({row["target"] for row in summary["environment_summary"]})
    arms = sorted({row["arm"] for row in summary["environment_summary"]})
    lookup = {(row["arm"], row["target"]): row for row in summary["environment_summary"]}
    lines = [
        r"\begin{table*}[tbp]",
        rf"\caption{{{caption}\label{{{label}}}}}",
        r"\centering\small",
        r"\begin{tabular}{l" + "c" * (len(environments) + 1) + "}",
        r"\toprule",
        "Arm & " + " & ".join(environments) + r" & Equal-environment mean \\",
        r"\midrule",
    ]
    for arm in arms:
        cells = [format_cell(lookup[(arm, env)]["macro_f1_mean"], lookup[(arm, env)]["seed_sample_sd"])
                 for env in environments]
        lines.append(arm.replace("_", r"\_") + " & " + " & ".join(cells) +
                     f" & {100 * summary['equal_environment_macro_f1'][arm]:.2f}" + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}",
                  r"\begin{flushleft}\footnotesize Values are macro-F1 percentages; parentheses contain the sample SD across the three fixed seeds within each environment. No population confidence interval is implied.\end{flushleft}",
                  r"\end{table*}", ""])
    return "\n".join(lines)


def summarize_mechanisms(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["study"], row["arm"])].append(row)
    output = []
    for (study, arm), records in sorted(groups.items()):
        for field in ("source_environment_probe_balanced_accuracy", "feature_between_to_within_class_ratio",
                      "feature_environment_scatter_within_class", "feature_equal_class_domain_squared_distance"):
            values = np.asarray([row[field] for row in records], dtype=float)
            output.append({"study": study, "arm": arm, "metric": field, "formal_runs": len(values),
                           "run_mean": float(values.mean()), "run_sample_sd": float(values.std(ddof=1)),
                           "run_min": float(values.min()), "run_max": float(values.max())})
    return output


def summarize_gradient_runs(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["study"], row["job_id"], row["target"], row["seed"], row["arm"],
                row["regularizer_family"], row["parameter_group"])].append(row)
    output = []
    for key, records in sorted(groups.items()):
        cosine = np.asarray([row["cosine"] for row in records if row["cosine"] is not None], dtype=float)
        ratio = np.asarray([row["weighted_regularizer_to_primary_norm_ratio"] for row in records
                            if row["weighted_regularizer_to_primary_norm_ratio"] is not None], dtype=float)
        output.append(dict(zip(("study", "job_id", "target", "seed", "arm", "regularizer_family", "parameter_group"), key),
            fixed_batches=len(records), cosine_median=float(np.median(cosine)) if len(cosine) else None,
            cosine_negative_fraction=float(np.mean(cosine < 0)) if len(cosine) else None,
            weighted_ratio_median=float(np.median(ratio)) if len(ratio) else None))
    return output


def summarize_gradient_arms(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["study"], row["arm"], row["regularizer_family"], row["parameter_group"])].append(row)
    output = []
    for key, records in sorted(groups.items()):
        medians = np.asarray([row["cosine_median"] for row in records if row["cosine_median"] is not None], dtype=float)
        negatives = np.asarray([row["cosine_negative_fraction"] for row in records
                                if row["cosine_negative_fraction"] is not None], dtype=float)
        ratios = np.asarray([row["weighted_ratio_median"] for row in records if row["weighted_ratio_median"] is not None], dtype=float)
        output.append(dict(zip(("study", "arm", "regularizer_family", "parameter_group"), key),
            formal_runs=len(records), cosine_run_median_min=float(medians.min()) if len(medians) else None,
            cosine_run_median_max=float(medians.max()) if len(medians) else None,
            negative_fraction_run_min=float(negatives.min()) if len(negatives) else None,
            negative_fraction_run_max=float(negatives.max()) if len(negatives) else None,
            weighted_ratio_run_median_min=float(ratios.min()) if len(ratios) else None,
            weighted_ratio_run_median_max=float(ratios.max()) if len(ratios) else None))
    return output


def summarize_residual_arms(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["study"], row["arm"], row["path"])].append(row)
    output = []
    for key, records in sorted(groups.items()):
        medians = np.asarray([row["nonzero_global_ratios_median"] for row in records
                              if row["nonzero_global_ratios_median"] is not None], dtype=float)
        p95 = np.asarray([row["nonzero_global_ratios_p95"] for row in records
                          if row["nonzero_global_ratios_p95"] is not None], dtype=float)
        output.append(dict(zip(("study", "arm", "path"), key), formal_runs=len(records),
            total_validation_rows=sum(row["rows"] for row in records),
            total_near_zero_global_rows=sum(row["near_zero_global_rows"] for row in records),
            nonzero_ratio_run_median_min=float(medians.min()) if len(medians) else None,
            nonzero_ratio_run_median_max=float(medians.max()) if len(medians) else None,
            nonzero_ratio_run_p95_min=float(p95.min()) if len(p95) else None,
            nonzero_ratio_run_p95_max=float(p95.max()) if len(p95) else None))
    return output


def resource_rows_for(study, details):
    protocol, bundle = details["protocol"], details["path"]
    configurations = {config["job_id"]: config for config in protocol["jobs"]}
    costs = read(bundle / "model_costs.json") if (bundle / "model_costs.json").exists() else {}
    if study == "core":
        parameters = read(bundle / "shared_initialization_audit.json")["total_parameters"]
    else:
        parameters = {}
    groups = defaultdict(list)
    for job_id, status in details["statuses"].items():
        config = configurations[job_id]
        arm = config.get("control_id", config.get("arm"))
        groups[arm].append(status)
    rows = []
    for arm, records in sorted(groups.items()):
        elapsed = np.asarray([record["elapsed_seconds"] for record in records], dtype=float) / 3600
        peak = np.asarray([record["peak_cuda_allocated_bytes_this_process"] for record in records], dtype=float) / 2**20
        cost = costs.get(arm, {})
        rows.append({
            "study": study,
            "arm": arm,
            "formal_runs": len(records),
            "elapsed_hours_mean": float(elapsed.mean()),
            "elapsed_hours_sample_sd": float(elapsed.std(ddof=1)),
            "elapsed_hours_min": float(elapsed.min()),
            "elapsed_hours_max": float(elapsed.max()),
            "peak_cuda_allocated_mib_max": float(peak.max()),
            "parameters": cost.get("parameters", cost.get("total_parameters", parameters.get(arm))),
            "counted_inference_macs": cost.get("counted_inference_macs", {}).get("total_counted_macs"),
            "mac_counting_scope": cost.get("counted_inference_macs", {}).get("excludes"),
        })
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, default=Path("D:/pa-csi-dg-revision-runtime"))
    parser.add_argument("--output", type=Path, default=ROOT / "docs/revision/evidence/formal_revision_v1")
    args = parser.parse_args()
    runtime, output = args.runtime.resolve(), args.output.resolve()
    if output.exists():
        raise FileExistsError("Preserve the existing formal evidence version")

    bundles = {}
    for short, folder, verifier, count in STUDIES:
        path = runtime / "studies" / folder
        protocol, digest, statuses, gate, summary, summary_hash = verify_study(path, verifier, count)
        bundles[short] = {"path": path, "protocol": protocol, "bundle_sha256": digest,
                          "statuses": statuses, "target_gate": gate, "summary": summary,
                          "evaluation_summary_sha256": summary_hash}
    source_rows, gradient_rows, residual_rows, source_hashes, _ = verify_source_analyses(
        runtime / "analyses/formal_source_v1", bundles)

    environment_rows, effect_rows, run_rows, resource_rows = [], [], [], []
    for study, details in bundles.items():
        for row in details["summary"]["environment_summary"]:
            environment_rows.append({"study": study, **row})
        for row in details["summary"].get("paired_effects", []):
            effect_rows.append({"study": study, **row})
        configs = {config["job_id"]: config for config in details["protocol"]["jobs"]}
        core_parameters = (read(details["path"] / "shared_initialization_audit.json")["total_parameters"]
                           if study == "core" else {})
        for row in details["summary"]["runs"]:
            status = details["statuses"][row["job_id"]]
            config = configs[row["job_id"]]
            prefix = f"evaluation/{row['job_id']}"
            run_rows.append({"study": study, "job_id": row["job_id"], "arm": row["arm"],
                             "target": row["target"], "seed": row["seed"],
                             "macro_f1": row["metrics"]["macro_f1"],
                             "accuracy": row["metrics"]["accuracy"],
                             **{f"class_f1_{index}": value for index, value in enumerate(row["metrics"]["class_f1"])},
                             "source_macro_f1": row["source_metrics"]["macro_f1"],
                             "best_epoch": status["best_epoch"],
                             "optimizer_updates": status["optimizer_updates"],
                             "elapsed_seconds": status["elapsed_seconds"],
                             "peak_cuda_allocated_bytes": status["peak_cuda_allocated_bytes_this_process"],
                             "parameters": config.get("total_parameters", core_parameters.get(row["arm"])),
                             "checkpoint_sha256": row["checkpoint_sha256"],
                             "source_status_sha256": status["status_sha256"],
                             "result_sha256": details["summary"]["artifacts_sha256"][prefix + "/result.json"],
                             "predictions_sha256": details["summary"]["artifacts_sha256"][prefix + "/predictions.json"]})
        resource_rows.extend(resource_rows_for(study, details))
    mechanism_rows = summarize_mechanisms(source_rows)
    gradient_run_rows = summarize_gradient_runs(gradient_rows)
    gradient_arm_rows = summarize_gradient_arms(gradient_run_rows)
    residual_arm_rows = summarize_residual_arms(residual_rows)
    output.mkdir(parents=True)
    write_csv(output / "formal_runs.csv", run_rows,
              ["study", "job_id", "arm", "target", "seed", "macro_f1", "accuracy",
               "class_f1_0", "class_f1_1", "class_f1_2", "class_f1_3", "class_f1_4", "class_f1_5",
               "source_macro_f1", "best_epoch", "optimizer_updates", "elapsed_seconds",
               "peak_cuda_allocated_bytes", "parameters", "checkpoint_sha256", "source_status_sha256",
               "result_sha256", "predictions_sha256"])
    write_csv(output / "environment_summary.csv", environment_rows,
              ["study", "target", "arm", "seeds", "macro_f1_mean", "seed_sample_sd"])
    write_csv(output / "paired_effects.csv", effect_rows,
              ["study", "target", "effect", "treatment", "reference", "reference_study",
               "interpretation", "paired_seeds", "paired_seed_pp", "mean_pp", "seed_sample_sd_pp"])
    write_csv(output / "source_mechanism_runs.csv", source_rows,
              ["study", "job_id", "target", "seed", "arm", "selected_epoch",
               "source_environment_probe_balanced_accuracy", "feature_between_to_within_class_ratio",
               "feature_environment_scatter_within_class", "feature_equal_class_domain_squared_distance",
               "gradient_probe_batches"])
    write_csv(output / "source_mechanism_summary.csv", mechanism_rows,
              ["study", "arm", "metric", "formal_runs", "run_mean", "run_sample_sd", "run_min", "run_max"])
    write_csv(output / "source_gradient_probes.csv", gradient_rows,
              ["study", "job_id", "target", "seed", "arm", "batch_index", "regularizer_family",
               "parameter_group", "primary_gradient_norm", "regularizer_gradient_norm", "cosine",
               "weighted_regularizer_to_primary_norm_ratio", "regularizer_weight", "parameter_count"])
    write_csv(output / "source_gradient_run_summary.csv", gradient_run_rows,
              ["study", "job_id", "target", "seed", "arm", "regularizer_family", "parameter_group",
               "fixed_batches", "cosine_median", "cosine_negative_fraction", "weighted_ratio_median"])
    write_csv(output / "source_gradient_arm_summary.csv", gradient_arm_rows,
              ["study", "arm", "regularizer_family", "parameter_group", "formal_runs",
               "cosine_run_median_min", "cosine_run_median_max", "negative_fraction_run_min",
               "negative_fraction_run_max", "weighted_ratio_run_median_min", "weighted_ratio_run_median_max"])
    write_csv(output / "source_residual_runs.csv", residual_rows,
              ["study", "job_id", "target", "seed", "arm", "path", "rows", "near_zero_global_rows",
               "all_ratios_median", "all_ratios_p95", "nonzero_global_ratios_median", "nonzero_global_ratios_p95"])
    write_csv(output / "source_residual_arm_summary.csv", residual_arm_rows,
              ["study", "arm", "path", "formal_runs", "total_validation_rows", "total_near_zero_global_rows",
               "nonzero_ratio_run_median_min", "nonzero_ratio_run_median_max",
               "nonzero_ratio_run_p95_min", "nonzero_ratio_run_p95_max"])
    write_csv(output / "resource_summary.csv", resource_rows,
              ["study", "arm", "formal_runs", "elapsed_hours_mean", "elapsed_hours_sample_sd",
               "elapsed_hours_min", "elapsed_hours_max", "peak_cuda_allocated_mib_max", "parameters",
               "counted_inference_macs", "mac_counting_scope"])

    captions = {
        "core": ("Frozen common-protocol R factorial on the three MultiEnv development environments.", "tab:formalcore"),
        "controls": ("Frozen capacity, mask, temporal-intervention, and same-backbone DG controls on MultiEnv.", "tab:formalcontrols"),
        "inputs": ("Frozen input-processing and modality sensitivity study on MultiEnv.", "tab:formalinputs"),
        "widar": ("Predeclared within-Widar retraining factorial across three public-data rooms.", "tab:formalwidar"),
    }
    for study, (caption, label) in captions.items():
        (output / f"{study}_table.tex").write_text(
            latex_table(study, bundles[study]["summary"], caption, label), encoding="utf-8")

    evidence = {
        "completed_at": now(),
        "schema": "formal-revision-evidence/v1",
        "all_formal_studies_completed": True,
        "population_inference_performed": False,
        "target_gates_verified": True,
        "formal_run_count": len(run_rows),
        "formal_source_analysis_count": len(source_rows),
        "formal_source_gradient_probe_row_count": len(gradient_rows),
        "formal_source_residual_path_run_count": len(residual_rows),
        "studies": {study: {
            "bundle": str(details["path"]),
            "bundle_sha256": details["bundle_sha256"],
            "job_count": len(details["protocol"]["jobs"]),
            "evaluation_summary_sha256": details["evaluation_summary_sha256"],
            "equal_environment_macro_f1": details["summary"]["equal_environment_macro_f1"],
            "inference_scope": details["summary"]["inference_scope"],
        } for study, details in bundles.items()},
        "formal_source_analysis_summary_sha256": source_hashes,
        "limitations": [
            "E1-E3 were inspected during development and remain development environments.",
            "Seed SD quantifies optimization variation within an environment and is not an environment-population interval.",
            "Widar uses within-dataset retraining, has different gesture classes, and confounds room with subject composition.",
            "Native-time and phase variants are processing sensitivity analyses, not complete physical calibration.",
        ],
    }
    generated = [path for path in output.iterdir() if path.is_file()]
    evidence["artifacts_sha256"] = {path.name: sha256(path) for path in generated}
    write(output / "verified_formal_results.json", evidence)
    final_files = [path for path in output.iterdir() if path.is_file()]
    write(output / "manifest.json", {"files_sha256": {path.name: sha256(path) for path in final_files}})
    print({"formal_runs": len(run_rows), "formal_source_analyses": len(source_rows),
           "output": str(output), "verified": True})


if __name__ == "__main__":
    main()
