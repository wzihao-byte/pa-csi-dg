"""Deferred evaluation and descriptive paired summaries for extension studies."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from experiments.revision_validation.experiment_data import VerifiedInputs
from experiments.revision_validation.experiment_design import build_experiment_model, score_partition
from experiments.revision_validation.evaluate_study import metrics_for
from experiments.revision_validation.extended_engine import verify_extension
from experiments.revision_validation.study_engine import runtime_settings
from experiments.revision_validation.study_io import bound_path, now, read, sha256, value_hash, verify_completed, write


class VerifiedTarget(VerifiedInputs):
    def __init__(self, manifest_path, modalities):
        self.path = Path(manifest_path)
        self.manifest = read(self.path)
        self.target = self.manifest["outer_target"]
        self.modalities = tuple(modalities)
        if set(self.manifest["files"]) != {self.target}:
            raise RuntimeError("Target manifest must contain exactly its target environment")
        self.arrays, self.opened_paths = {self.target: {}}, []
        files = self.manifest["files"][self.target]
        for kind in self.modalities + ("label",):
            specification = files["data_files"][kind]
            path = Path(specification["path"])
            if sha256(path) != files["actual_file_sha256"][kind]:
                raise RuntimeError("Deferred target file hash mismatch")
            array = np.load(path, mmap_mode="r", allow_pickle=False)
            if list(array.shape) != specification["shape"] or str(array.dtype) != specification["dtype"]:
                raise RuntimeError("Deferred target shape/dtype mismatch")
            self.opened_paths.append(str(path))
            if kind == "label":
                labels = np.asarray(array).reshape(-1)
            else:
                if array.shape[1:] != (850, 90):
                    raise RuntimeError("Unexpected target input dimensions")
                self.arrays[self.target][kind] = array
        rows = self.manifest["test"]
        if len({r["sample_id"] for r in rows}) != len(rows) or {r["label"] for r in rows} != set(range(6)):
            raise RuntimeError("Duplicate target IDs or missing class")
        if any(r["environment"] != self.target or int(labels[r["index"]]) != r["label"] for r in rows):
            raise RuntimeError("Target row/label mismatch")


def verify_source_run(run, config):
    status = verify_completed(run, config)
    source = read(config["source_manifest_path"])
    predictions = read(run / "selected_source_predictions.json")
    if len(predictions) != len(source["val"]):
        raise RuntimeError("Wrong source prediction count")
    for actual, predicted in zip(source["val"], predictions):
        if actual["label"] != predicted["true"] or any(actual[k] != predicted[k] for k in ("sample_id", "subject_id", "acquisition_id")):
            raise RuntimeError("Saved source prediction identity/label mismatch")
    if metrics_for(predictions) != read(run / "history.json")[status["best_epoch"] - 1]["source_validation"]:
        raise RuntimeError("Saved source metrics do not reconstruct")
    return status


def evaluate(bundle):
    bundle = Path(bundle).resolve()
    protocol, bundle_hash = verify_extension(bundle)
    if Path(__file__).resolve().parents[2] != bundle / "code":
        raise RuntimeError("Use the frozen extension evaluator")
    # Check every source result before even constructing a target data loader.
    for config in protocol["jobs"]:
        verify_source_run(bundle / "runs" / config["job_id"], config)
    runtime = runtime_settings()
    if any(config["expected_runtime"] != runtime for config in protocol["jobs"]):
        raise RuntimeError("Frozen evaluation runtime differs")
    write(bundle / "target_gate.json", {"passed_at": now(), "bundle_sha256": bundle_hash,
          "verified_source_jobs": [c["job_id"] for c in protocol["jobs"]],
          "source_status_sha256": {c["job_id"]: sha256(bundle / "runs" / c["job_id"] / "status.json") for c in protocol["jobs"]}})
    results = []
    for config in protocol["jobs"]:
        run, output = bundle / "runs" / config["job_id"], bundle / "evaluation" / config["job_id"]
        source = VerifiedInputs(config["source_manifest_path"], config["input_modalities"])
        model = build_experiment_model(config).to(config["device"])
        model.load_state_dict(torch.load(run / "best_model.pt", map_location="cpu", weights_only=True), strict=True)
        source_metrics, source_predictions = score_partition(model, source, config)
        if source_predictions != read(run / "selected_source_predictions.json"):
            raise RuntimeError("Source-best weights do not reproduce stored source predictions")
        target_path = Path(config["target_manifest_path"]) if "target_manifest_path" in config else bundle / "manifests" / f"deferred_target_{config['target_name_only']}.json"
        if "target_manifest_path" in config and sha256(target_path) != config["target_manifest_sha256"]:
            raise RuntimeError("Configured target manifest hash changed")
        target_metadata = read(target_path)
        for key in ("sample_id", "subject_id", "acquisition_id"):
            source_ids = {r[key] for part in ("train", "val") for r in source.manifest[part]}
            if source_ids & {r[key] for r in target_metadata["test"]}:
                raise RuntimeError(f"Source/target identity overlap: {key}")
        target = VerifiedTarget(target_path, config["input_modalities"])
        metrics, predictions = score_partition(model, target, config, "test")
        output.mkdir(parents=True, exist_ok=True)
        result = {"job_id": config["job_id"], "arm": config["control_id"], "target": config["target_name_only"],
                  "seed": config["seed"], "metrics": metrics, "source_metrics": source_metrics,
                  "source_predictions_reproduced": True, "config_value_sha256": value_hash(config),
                  "checkpoint_sha256": sha256(run / "best_model.pt"), "sample_count": len(predictions),
                  "input_modalities": config["input_modalities"], "experiment": config["experiment"]}
        if (output / "result.json").exists():
            if read(output / "result.json") != result or read(output / "predictions.json") != predictions:
                raise RuntimeError("Repeat evaluation differs; retain the original evidence and investigate")
        else:
            write(output / "predictions.json", predictions)
            write(output / "result.json", result)
        results.append(result)
        print(f"{config['job_id']}: target macro-F1 {metrics['macro_f1']:.6f}", flush=True)
    environment_summary = []
    for target in sorted({r["target"] for r in results}):
        for arm in sorted({r["arm"] for r in results}):
            records = sorted([r for r in results if r["target"] == target and r["arm"] == arm], key=lambda r: r["seed"])
            values = [r["metrics"]["macro_f1"] for r in records]
            environment_summary.append({"target": target, "arm": arm, "seeds": [r["seed"] for r in records],
                                        "macro_f1_mean": float(np.mean(values)), "seed_sample_sd": float(np.std(values, ddof=1))})
    reference_path = protocol.get("core_reference_bundle", protocol.get("source_training_dependency"))
    core = []
    if any(c["reference_study"] == "core" for c in protocol.get("comparisons", [])):
        if not reference_path:
            raise RuntimeError("Paired core comparisons require their frozen reference bundle")
        from experiments.revision_validation.study_io import verify_bundle
        reference_bundle = Path(reference_path)
        _, reference_hash = verify_bundle(reference_bundle)
        expected_hash = protocol.get("core_reference_bundle_sha256", protocol.get("dependency_bundle_sha256"))
        if reference_hash != expected_hash:
            raise RuntimeError("Reference core bundle differs from its declared hash")
        reference_summary = read(reference_bundle / "evaluation/summary.json")
        for relative, expected in reference_summary["artifacts_sha256"].items():
            if sha256(bound_path(reference_bundle, relative)) != expected:
                raise RuntimeError("Reference core evaluation artifact differs")
        core = reference_summary["runs"]
    paired_effects = []
    for comparison in protocol.get("comparisons", []):
        reference = core if comparison["reference_study"] == "core" else results
        for target in sorted({r["target"] for r in results}):
            treatment = {r["seed"]: r for r in results if r["arm"] == comparison["treatment"] and r["target"] == target}
            control = {r["seed"]: r for r in reference if r["arm"] == comparison["reference"] and r["target"] == target}
            if not treatment or set(treatment) != set(control):
                raise RuntimeError("Paired comparison seeds do not match")
            values = [100 * (treatment[s]["metrics"]["macro_f1"] - control[s]["metrics"]["macro_f1"]) for s in sorted(treatment)]
            paired_effects.append({**comparison, "target": target, "paired_seeds": sorted(treatment),
                                   "paired_seed_pp": values, "mean_pp": float(np.mean(values)), "seed_sample_sd_pp": float(np.std(values, ddof=1))})
    summary = {"runs": results, "environment_summary": environment_summary, "paired_effects": paired_effects,
               "equal_environment_macro_f1": {arm: float(np.mean([r["macro_f1_mean"] for r in environment_summary if r["arm"] == arm])) for arm in sorted({r["arm"] for r in results})},
               "inference_scope": protocol["inference_scope"],
               "artifacts_sha256": {p.relative_to(bundle).as_posix(): sha256(p) for p in (bundle / "evaluation").glob("*/*.json")}}
    summary_path = bundle / "evaluation/summary.json"
    if summary_path.exists():
        previous = read(summary_path)
        previous.pop("completed_at")
        if previous != summary:
            raise RuntimeError("Completed evaluation summary differs")
    else:
        write(summary_path, {"completed_at": now(), **summary})
    return sha256(summary_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    evaluate(parser.parse_args().bundle)
