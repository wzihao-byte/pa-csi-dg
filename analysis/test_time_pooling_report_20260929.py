"""CPU-only synthetic evidence checks for the complete 54-run aggregator."""
from __future__ import annotations

import argparse
from pathlib import Path
import tempfile

from experiments.revision_validation.study_io import read, sha256, value_hash, write
from experiments.revision_validation.time_pooling_report import _metrics, summarize, SEEDS


def fixture(root):
    roots = {role: root / role for role in ("core", "inputs", "new")}
    arms = {"core": ("R00", "R10"), "inputs": ("NATIVE_MERGE", "NATIVE_LINEAR"),
            "new": ("GLOBAL_NATIVE_MERGE", "GLOBAL_NATIVE_LINEAR")}
    correct = {"R00": 1200, "R10": 1800, "NATIVE_MERGE": 1900, "NATIVE_LINEAR": 2200,
               "GLOBAL_NATIVE_MERGE": 2000, "GLOBAL_NATIVE_LINEAR": 2100}
    for role, bundle in roots.items():
        frozen = {}
        jobs, results = [], []
        for target, seeds in SEEDS.items():
            target_rows = [{"sample_id": f"{target}:target:{i}", "subject_id": f"{target}:target:subject:{i // 100}",
                            "acquisition_id": f"{target}:target:acq:{i}", "label": i % 6} for i in range(3000)]
            source_rows = [{"sample_id": f"{target}:source:{i}", "subject_id": f"{target}:source:subject:{i}",
                            "acquisition_id": f"{target}:source:acq:{i}", "label": i} for i in range(6)]
            source_path = bundle / "manifests" / f"source_{target}.json"
            target_path = bundle / "manifests" / f"deferred_target_{target}.json"
            write(source_path, {"train": [], "val": source_rows})
            write(target_path, {"test": target_rows})
            frozen[source_path.relative_to(bundle).as_posix()] = sha256(source_path)
            frozen[target_path.relative_to(bundle).as_posix()] = sha256(target_path)
            source_predictions = [{"sample_id": row["sample_id"], "subject_id": row["subject_id"],
                                   "acquisition_id": row["acquisition_id"], "true": row["label"],
                                   "predicted": row["label"]} for row in source_rows]
            source_metrics = _metrics(source_predictions)
            for arm in arms[role]:
                for seed in seeds:
                    job_id = f"{target}_{arm}_{seed}"
                    config = {"job_id": job_id, "target_name_only": target, "seed": seed,
                              "arm": arm if role == "core" else "R00", "epochs": 2, "updates_per_epoch": 1,
                              "source_manifest_path": str(source_path), "source_manifest_sha256": sha256(source_path)}
                    if role != "core":
                        config.update(control_id=arm, target_manifest_path=str(target_path),
                                      target_manifest_sha256=sha256(target_path))
                    run = bundle / "runs" / job_id
                    write(run / "config.json", config)
                    write(run / "history.json", [{"source_validation": source_metrics}] * 2)
                    write(run / "selected_source_predictions.json", source_predictions)
                    (run / "best_model.pt").write_bytes(b"synthetic checkpoint: never loaded as a model")
                    artifacts = {name: sha256(run / name) for name in
                                 ("config.json", "history.json", "selected_source_predictions.json", "best_model.pt")}
                    write(run / "status.json", {"status": "completed", "config_value_sha256": value_hash(config),
                          "optimizer_updates": 2, "best_epoch": 1, "best_source_macro_f1": 1.0,
                          "target_data_loaded": False, "artifacts_sha256": artifacts})
                    count_correct = correct[arm] + (seed % 10 - 1) * 6
                    predictions = [{"sample_id": row["sample_id"], "subject_id": row["subject_id"],
                                    "acquisition_id": row["acquisition_id"], "true": row["label"],
                                    "predicted": row["label"] if i < count_correct else (row["label"] + 1) % 6}
                                   for i, row in enumerate(target_rows)]
                    result = {"job_id": job_id, "arm": arm, "target": target, "seed": seed,
                              "config_value_sha256": value_hash(config), "checkpoint_sha256": sha256(run / "best_model.pt"),
                              "source_predictions_reproduced": True, "sample_count": 3000,
                              "source_metrics": source_metrics, "metrics": _metrics(predictions)}
                    write(bundle / "evaluation" / job_id / "result.json", result)
                    write(bundle / "evaluation" / job_id / "predictions.json", predictions)
                    jobs.append(config)
                    results.append(result)
        protocol = {"schema": "revision-core-study/v1" if role == "core" else "revision-extension-study/v1", "jobs": jobs}
        if role == "new":
            protocol["references"] = [{"role": refrole, "bundle": str(roots[refrole]),
                                       "bundle_sha256": sha256(roots[refrole] / "bundle_manifest.json"),
                                       "evaluation_summary_sha256": sha256(roots[refrole] / "evaluation/summary.json")}
                                      for refrole in ("core", "inputs")]
        write(bundle / "protocol.json", protocol)
        frozen["protocol.json"] = sha256(bundle / "protocol.json")
        write(bundle / "bundle_manifest.json", {"files_sha256": frozen})
        write(bundle / "target_gate.json", {"bundle_sha256": sha256(bundle / "bundle_manifest.json"),
              "source_status_sha256": {config["job_id"]: sha256(bundle / "runs" / config["job_id"] / "status.json") for config in jobs}})
        write(bundle / "evaluation/summary.json", {"runs": results,
              "artifacts_sha256": {path.relative_to(bundle).as_posix(): sha256(path) for path in (bundle / "evaluation").glob("*/*.json")}})
    return roots["new"]


def must_fail(action, match):
    try:
        action()
    except RuntimeError as error:
        assert match in str(error), (match, str(error))
        return
    raise AssertionError(f"Expected rejection: {match}")


def test_report(root):
    bundle = fixture(root)
    first_hash = summarize(bundle)
    assert summarize(bundle) == first_hash
    report = read(bundle / "comparison/report.json")
    assert len(report["runs"]) == 54 and len(report["cells"]) == 24
    assert len(report["paired_gains"]) == 27 and len(report["gain_changes"]) == 27
    assert len(report["paired_gain_summary"]) == len(report["gain_change_summary"]) == 12
    b = [row for row in report["paired_gain_summary"] if row["pipeline"] == "B" and row["target"] == "ALL"][0]
    assert b["negative_pairs"] == b["n"] == 9
    gains = {(row["pipeline"], row["target"], row["seed"]): row["effect_pp"] for row in report["paired_gains"]}
    for row in report["gain_changes"]:
        before, after = row["transition"].split("_to_")
        assert row["effect_pp"] == gains[after, row["target"], row["seed"]] - gains[before, row["target"], row["seed"]]
    summary_path = bundle / "evaluation/summary.json"
    original = read(summary_path)
    incomplete = {**original, "runs": original["runs"][:-1]}
    write(summary_path, incomplete)
    must_fail(lambda: summarize(bundle), "Evaluation does not cover every frozen source job")
    write(summary_path, original)
    first_id = original["runs"][0]["job_id"]
    prediction_path = bundle / "evaluation" / first_id / "predictions.json"
    predictions = read(prediction_path)
    original_bytes = prediction_path.read_bytes()
    prediction_path.write_bytes(original_bytes + b"\n")
    must_fail(lambda: summarize(bundle), "Evaluation artifact changed")
    prediction_path.write_bytes(original_bytes)
    changed_predictions = [dict(row) for row in predictions]
    changed_predictions[0]["subject_id"] = "wrong-subject"
    write(prediction_path, changed_predictions)
    rehashed = {**original, "artifacts_sha256": dict(original["artifacts_sha256"])}
    rehashed["artifacts_sha256"][prediction_path.relative_to(bundle).as_posix()] = sha256(prediction_path)
    write(summary_path, rehashed)
    must_fail(lambda: summarize(bundle), "Target predictions and manifest identities/labels differ")
    prediction_path.write_bytes(original_bytes)
    write(summary_path, original)
    result_path = bundle / "evaluation" / first_id / "result.json"
    result = read(result_path)
    changed_result = {**result, "metrics": {**result["metrics"], "macro_f1": 0.999}}
    write(result_path, changed_result)
    changed_summary = {**original, "runs": [changed_result, *original["runs"][1:]],
                       "artifacts_sha256": dict(original["artifacts_sha256"])}
    changed_summary["artifacts_sha256"][result_path.relative_to(bundle).as_posix()] = sha256(result_path)
    write(summary_path, changed_summary)
    must_fail(lambda: summarize(bundle), "Target metrics do not reconstruct")
    write(result_path, result)
    write(summary_path, original)
    assert summarize(bundle) == first_hash
    return {"passed": True, "synthetic_run_count": 54, "prediction_count": 162000,
            "checks": ["complete six-cell matrix", "deterministic repeat", "retains all nine negative B gains",
                       "difference-of-differences algebra", "rejects incomplete evaluation", "rejects modified artifact",
                       "rejects target identity mismatch", "rejects incorrect stored metrics"],
            "report_sha256": first_hash}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--temp-parent", type=Path, default=Path("D:/pa-csi-dg-revision-runtime/qa"))
    args = parser.parse_args()
    args.temp_parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="time_pooling_report_", dir=args.temp_parent) as temporary:
        result = test_report(Path(temporary))
    if args.output:
        write(args.output, result)
    print(result)
