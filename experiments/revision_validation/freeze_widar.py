"""Freeze the external Widar room factorial after full preparation and source QA."""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import torch

from experiments.revision_validation.experiment_design import build_experiment_model
from experiments.revision_validation.extended_engine import verify_extension
from experiments.revision_validation.freeze_controls import EXTENSION_CODE, ROOT
from experiments.revision_validation.models import count_matmul_conv_macs
from experiments.revision_validation.study_io import bound_path, now, read, sha256, write
from experiments.revision_validation.widar_study_design import WIDAR_ARMS, WIDAR_SEEDS, widar_config


WIDAR_CODE = EXTENSION_CODE + [
    "analysis/prepare_widar_inventory.py", "analysis/download_widar_cohort_v2.py", "analysis/compile_widar_inputs_v2.py",
] + ["experiments/revision_validation/" + name for name in (
    "input_study_engine.py", "widar_study_design.py", "widar_study_qa.py", "freeze_widar.py", "widar_study_engine.py")]


def check_widar_dataset(directory, verify_files=True):
    directory = Path(directory).resolve()
    manifest = read(directory / "dataset_manifest.json")
    if not manifest.get("complete") or manifest.get("variants") != ["native_amplitude"] or manifest.get("rows", 0) <= 0:
        raise RuntimeError("Widar study requires the full compiled native-amplitude cohort")
    if set(manifest.get("room_counts", {})) != {"W1", "W2", "W3"} or len(manifest.get("subjects", [])) != 9:
        raise RuntimeError("Widar compiled room/subject structure differs from the predeclared cohort")
    if verify_files:
        for relative, expected in manifest["artifacts_sha256"].items():
            if sha256(bound_path(directory, relative)) != expected:
                raise RuntimeError("Widar compiled artifact hash mismatch")
    return sha256(directory / "dataset_manifest.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--input-bundle", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--preparation-summary", type=Path, required=True)
    parser.add_argument("--qa-report", type=Path, required=True)
    args = parser.parse_args()
    bundle, dependency, data = args.bundle.resolve(), args.input_bundle.resolve(), args.data.resolve()
    if bundle.exists():
        raise FileExistsError("Use a new Widar study version; preserve every existing freeze")
    _, dependency_hash = verify_extension(dependency)
    dataset_hash = check_widar_dataset(data)
    qa = read(args.qa_report)
    if not qa.get("passed") or qa.get("target_data_loaded") or qa.get("target_performance_or_model_selection_used"):
        raise RuntimeError("Complete source-only Widar QA must pass")
    if qa.get("compiled_dataset_manifest_sha256") != dataset_hash:
        raise RuntimeError("Widar compiled data differ from source QA")
    for name, digest in qa["code_sha256"].items():
        if sha256(ROOT / name) != digest:
            raise RuntimeError(f"Widar code changed after QA: {name}")
    manifest_source = data / "native_amplitude/manifests"
    expected_names = {f"{prefix}_{room}.json" for prefix in ("source", "deferred_target") for room in WIDAR_SEEDS}
    manifest_paths = list(manifest_source.glob("*.json"))
    if {path.name for path in manifest_paths} != expected_names:
        raise RuntimeError("Unexpected Widar manifest set")
    for relative in WIDAR_CODE:
        destination = bundle / "code" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, destination)
    for path in manifest_paths:
        destination = bundle / "manifests" / path.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
    for source, name in ((args.selection, "selection_before_inventory.json"), (args.inventory, "inventory.json"),
                         (args.preparation_summary, "preparation_summary.json"), (args.qa_report, "widar_study_qa.json")):
        shutil.copy2(source, bundle / name)
    jobs = [widar_config(room, arm, seed, bundle / "manifests" / f"source_{room}.json",
                         bundle / "manifests" / f"deferred_target_{room}.json")
            for room, seeds in WIDAR_SEEDS.items() for seed in seeds for arm in WIDAR_ARMS]
    costs = {}
    for arm in WIDAR_ARMS:
        config = next(job for job in jobs if job["control_id"] == arm)
        model = build_experiment_model(config)
        costs[arm] = {"parameters": config["total_parameters"],
                      "counted_inference_macs": count_matmul_conv_macs(model, torch.zeros(1, 850, 90))}
    write(bundle / "model_costs.json", costs)
    comparisons = [
        {"treatment": "R10", "reference": "R00", "reference_study": "extension", "interpretation": "TPP effect under CE"},
        {"treatment": "R11", "reference": "R01", "reference_study": "extension", "interpretation": "TPP effect under fixed SupCon"},
        {"treatment": "R01", "reference": "R00", "reference_study": "extension", "interpretation": "Fixed SupCon effect under global pooling"},
        {"treatment": "R11", "reference": "R10", "reference_study": "extension", "interpretation": "Fixed SupCon effect under TPP"},
    ]
    dataset_manifest = read(data / "dataset_manifest.json")
    protocol = {"schema": "revision-extension-study/v1", "study_kind": "external_widar_room_factorial",
        "study_id": bundle.name, "frozen_at": now(), "entry_module": "experiments.revision_validation.widar_study_engine",
        "jobs": jobs, "job_count": len(jobs), "controls": list(WIDAR_ARMS), "comparisons": comparisons,
        "prerequisites": [{"bundle": str(dependency), "bundle_sha256": dependency_hash, "kind": "extension"}],
        "compiled_input": {"directory": str(data), "dataset_manifest_sha256": dataset_hash,
                           "rows": dataset_manifest["rows"], "room_counts": dataset_manifest["room_counts"],
                           "subjects": dataset_manifest["subjects"]},
        "cohort_binding": {"selection_sha256": sha256(args.selection), "inventory_sha256": sha256(args.inventory),
                           "preparation_summary_sha256": sha256(args.preparation_summary)},
        "training_budget": "200 epochs x 300 updates, batch 16, same R architecture/Adam/LR schedule and fixed SupCon strength; zero new search",
        "selection": "Earliest maximum source validation macro-F1; subject-grouped validation declared before signal preparation",
        "target_gate": "All 36 source jobs complete and verified before any Widar target-room signal evaluation",
        "inference_scope": "Independent public-dataset method replication after within-Widar retraining; not direct MultiEnv label transfer or a population room effect",
        "known_limitations": ["Widar gesture classes differ from MultiEnv activities",
                              "Selected rooms contain disjoint subjects, so room and subject composition are confounded",
                              "One public cohort and three rooms do not identify population-level environmental effects",
                              "Native-time interpolation is an input convention, not complete physical calibration"],
        "hyperparameter_search_trials": 0,
        "stopping": "No performance early stop or outcome-based seed/file additions; retain and diagnose failures; resume only verified committed states"}
    write(bundle / "protocol.json", protocol)
    files = [path for path in bundle.rglob("*") if path.is_file()]
    write(bundle / "bundle_manifest.json", {"frozen_at": now(),
          "files_sha256": {path.relative_to(bundle).as_posix(): sha256(path) for path in files}})
    verified, digest = verify_extension(bundle)
    print(f"Frozen {len(verified['jobs'])} Widar jobs; bundle SHA256 {digest}")


if __name__ == "__main__":
    main()
