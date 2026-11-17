"""Deferred evaluation: all frozen source jobs must pass before target loading."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from experiments.revision_validation.models import build_model
from experiments.revision_validation.source_data import VerifiedSources
from experiments.revision_validation.study_engine import runtime_settings
from experiments.revision_validation.train_source import score_source
from experiments.revision_validation.study_io import (
    bound_path, now, read, sha256, value_hash, verify_bundle, verify_completed, write,
)


def metrics_for(predictions):
    confusion = np.zeros((6, 6), dtype=np.int64)
    for row in predictions:
        confusion[row["true"], row["predicted"]] += 1
    diagonal = confusion.diagonal()
    denominator = confusion.sum(0) + confusion.sum(1)
    f1 = np.divide(2 * diagonal, denominator, out=np.zeros(6), where=denominator > 0)
    return {"macro_f1": float(f1.mean()), "class_f1": f1.tolist(),
            "accuracy": float(diagonal.sum() / confusion.sum()), "confusion_matrix": confusion.tolist()}


def verify_source_predictions(directory, config):
    manifest = read(config["source_manifest_path"])
    predictions = read(directory / "selected_source_predictions.json")
    if len(predictions) != len(manifest["val"]):
        raise RuntimeError("Incomplete source predictions")
    for actual, predicted in zip(manifest["val"], predictions):
        if any(actual[k] != predicted[k] for k in ("sample_id", "subject_id", "acquisition_id")):
            raise RuntimeError("Source prediction identity mismatch")
        if actual["label"] != predicted["true"]:
            raise RuntimeError("Source prediction label mismatch")
    status = verify_completed(directory, config)
    selected = read(directory / "history.json")[status["best_epoch"] - 1]
    if metrics_for(predictions) != selected["source_validation"]:
        raise RuntimeError("Source metrics do not reconstruct from predictions")


def evaluate(bundle):
    protocol, bundle_hash = verify_bundle(bundle)
    if Path(__file__).resolve().parents[2] != bundle / "code":
        raise RuntimeError("Use the frozen evaluator snapshot")
    # This entire loop finishes before parsing or loading any target data.
    for config in protocol["jobs"]:
        run = bound_path(bundle, "runs/" + config["job_id"])
        verify_source_predictions(run, config)
    runtime = runtime_settings()
    if any(runtime != config["expected_runtime"] for config in protocol["jobs"]):
        raise RuntimeError("Evaluation runtime differs from the frozen training runtime")
    write(bundle / "target_gate.json", {"passed_at": now(), "bundle_sha256": bundle_hash,
          "all_source_jobs_completed": len(protocol["jobs"]), "source_prediction_metrics_verified": True,
          "source_status_sha256": {c["job_id"]: sha256(bundle / "runs" / c["job_id"] / "status.json")
                                   for c in protocol["jobs"]}})
    results = []
    for config in protocol["jobs"]:
        run = bundle / "runs" / config["job_id"]
        output = bundle / "evaluation" / config["job_id"]
        source = VerifiedSources(config["source_manifest_path"])
        model = build_model(config["arm"], config["seed"]).to(config["device"])
        model.load_state_dict(torch.load(run / "best_model.pt", map_location="cpu", weights_only=True), strict=True)
        source_metrics, source_predictions = score_source(model, source, config["device"])
        if source_predictions != read(run / "selected_source_predictions.json"):
            raise RuntimeError("Restored source-best checkpoint does not reproduce saved predictions")
        target = config["target_name_only"]
        manifest = read(bundle / "manifests" / f"deferred_target_{target}.json")
        if manifest["outer_target"] != target or set(manifest["files"]) != {target}:
            raise RuntimeError("Wrong target manifest")
        specification = manifest["files"][target]
        for kind in ("amp", "label"):
            if sha256(specification["data_files"][kind]["path"]) != specification["actual_file_sha256"][kind]:
                raise RuntimeError(f"Target {kind} hash mismatch")
        array = np.load(specification["data_files"]["amp"]["path"], mmap_mode="r", allow_pickle=False)
        labels = np.load(specification["data_files"]["label"]["path"], allow_pickle=False).reshape(-1)
        if any(int(labels[r["index"]]) != r["label"] for r in manifest["test"]):
            raise RuntimeError("Target manifest labels disagree with verified file")
        predictions = []
        model.eval()
        with torch.no_grad():
            for start in range(0, len(manifest["test"]), 32):
                rows = manifest["test"][start:start + 32]
                amplitude = np.stack([array[r["index"]] for r in rows]).astype(np.float32)
                logits = model(torch.from_numpy(amplitude).to(config["device"]), project=False)["logits"]
                if not torch.isfinite(logits).all():
                    raise FloatingPointError("Nonfinite target logits")
                predicted = logits.argmax(1).cpu().tolist()
                predictions.extend({"sample_id": r["sample_id"], "subject_id": r["subject_id"],
                    "acquisition_id": r["acquisition_id"], "true": r["label"], "predicted": p}
                    for r, p in zip(rows, predicted))
        output.mkdir(parents=True, exist_ok=True)
        result = {"job_id": config["job_id"], "arm": config["arm"], "target": target, "seed": config["seed"],
                  "metrics": metrics_for(predictions), "source_predictions_reproduced": True,
                  "source_metrics": source_metrics, "checkpoint_sha256": sha256(run / "best_model.pt"),
                  "config_value_sha256": value_hash(config), "development_environments": True,
                  "sample_count": len(predictions), "phase_loaded": False}
        if (output / "result.json").exists():
            previous = read(output / "result.json")
            if previous != result or read(output / "predictions.json") != predictions:
                raise RuntimeError("Repeat evaluation differs; preserve existing evidence and investigate")
        else:
            write(output / "predictions.json", predictions)
            write(output / "result.json", result)
        results.append(result)
        print(f"{config['job_id']}: target macro-F1 {result['metrics']['macro_f1']:.6f}", flush=True)
    summaries, effects = [], []
    for target in ("E1", "E2", "E3"):
        by_arm = {arm: sorted([r for r in results if r["target"] == target and r["arm"] == arm], key=lambda r: r["seed"])
                  for arm in ("R00", "R01", "R10", "R11")}
        for arm, records in by_arm.items():
            values = [r["metrics"]["macro_f1"] for r in records]
            summaries.append({"target": target, "arm": arm, "seeds": [r["seed"] for r in records],
                              "macro_f1_mean": float(np.mean(values)), "seed_sample_sd": float(np.std(values, ddof=1))})
        vectors = {arm: np.array([r["metrics"]["macro_f1"] for r in records]) for arm, records in by_arm.items()}
        for name, weights in {"TPP_at_CE": {"R10": 1, "R00": -1}, "TPP_at_SC": {"R11": 1, "R01": -1},
                              "SC_at_global": {"R01": 1, "R00": -1}, "SC_at_TPP": {"R11": 1, "R10": -1},
                              "interaction": {"R11": 1, "R10": -1, "R01": -1, "R00": 1}}.items():
            differences = sum(weight * vectors[arm] for arm, weight in weights.items()) * 100
            effects.append({"target": target, "effect": name, "paired_seed_pp": differences.tolist(),
                            "mean_pp": float(differences.mean()), "seed_sample_sd_pp": float(differences.std(ddof=1))})
    write(bundle / "evaluation" / "summary.json", {"evaluated_at": now(), "runs": results,
          "environment_summary": summaries, "paired_effects": effects,
          "equal_environment_macro_f1": {arm: float(np.mean([r["macro_f1_mean"] for r in summaries if r["arm"] == arm]))
                                         for arm in ("R00", "R01", "R10", "R11")},
          "inference_scope": "Three development environments; seed SD describes optimization variation, no population p/CI",
          "artifacts_sha256": {p.relative_to(bundle).as_posix(): sha256(p)
                                for p in (bundle / "evaluation").glob("*/*.json")}})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    evaluate(parser.parse_args().bundle.resolve())
