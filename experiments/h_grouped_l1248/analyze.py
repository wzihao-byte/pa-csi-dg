"""Descriptive paired-seed analysis; never select settings from target results."""
from __future__ import annotations

import csv
import statistics
from pathlib import Path

from common import ARMS, OUTPUT, SEEDS, TARGETS, read, require, sha, verify_bundle, write_new


def analyze(bundle, result_path):
    verify_bundle(bundle)
    result = read(result_path)
    require(result["status"] == "complete" and result["job_count"] == 18, "All 18 formal runs are required")
    require(result["bundle_manifest_sha256"] == sha(bundle / "bundle_manifest.json"), "Results belong to another bundle")
    rows = {(r["target_env"], r["seed"], r["arm"]): r for r in result["rows"]}
    require(set(rows) == {(e, s, a) for e in TARGETS for s in SEEDS for a in ARMS}, "Incomplete or duplicate result grid")
    for row in rows.values():
        attempt = Path(row["attempt_dir"])
        require(sha(attempt / "attempt_completed.json") == row["completion_manifest_sha256"], "Completion changed")
        require(sha(attempt / "payload/run/metrics.json") == row["metrics_file_sha256"], "Metrics changed")
    historical = {}
    for reference in read(bundle / "historical_references.json")["references"]:
        attempt = Path(reference["historical_attempt"])
        require(sha(attempt / "attempt_completed.json") == reference["historical_completion_sha256"], "Historical completion changed")
        path = attempt / "payload/run/metrics.json"
        require(sha(path) == reference["historical_metrics_sha256"], "Historical metrics changed")
        historical[reference["job_id"]] = read(path)["test_metrics"]
    summary = []
    for metric in ("accuracy", "f1_macro"):
        for target in (*TARGETS, "E1-E3_equal_weight"):
            environments = TARGETS if target == "E1-E3_equal_weight" else (target,)
            values = {arm: [statistics.mean(rows[(e, s, arm)]["test_metrics"][metric] for e in environments) * 100 for s in SEEDS] for arm in ARMS}
            delta = [b - a for a, b in zip(values["H10"], values["H11"])]
            for arm in (*ARMS, "H11_minus_H10"):
                scores = delta if arm == "H11_minus_H10" else values[arm]
                summary.append({"target": target, "metric": metric, "arm": arm,
                                "mean_percent_or_pp": statistics.mean(scores), "seed_sd": statistics.stdev(scores),
                                "seed_42": scores[0], "seed_52": scores[1], "seed_62": scores[2]})
    comparisons = [{"target": e, "seed": s, "arm": a, "metric": m,
                    "grouped_percent": 100 * r["test_metrics"][m],
                    "historical_percent": 100 * historical[r["job_id"]][m],
                    "grouped_minus_historical_pp": 100 * (r["test_metrics"][m] - historical[r["job_id"]][m])}
                   for (e, s, a), r in rows.items() for m in ("accuracy", "f1_macro")]
    destination = OUTPUT / "analysis" / sha(result_path)[:16]
    destination.mkdir(parents=True, exist_ok=True)
    if (destination / "completed.json").exists():
        record = read(destination / "completed.json")
        require(record["results_sha256"] == sha(result_path), "Analysis completion drift")
        for name, expected in record["files_sha256"].items():
            require(sha(destination / name) == expected, "Analysis artifact changed")
        return destination
    for filename, records in (("paired_seed_summary.csv", summary), ("historical_comparison.csv", comparisons)):
        with (destination / filename).open("x", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(records[0]))
            writer.writeheader(); writer.writerows(records)
    write_new(destination / "results.json", {"status": "complete", "formal_runs": 18,
              "results_sha256": sha(result_path), "summary": summary, "historical_comparison": comparisons,
              "population_inference": False, "target_used_previously_for_development": True,
              "interpretation": "H11-H10 isolates the original SupCon increment with L1248 fixed under grouped source validation. Historical differences assess the combined change in source training and validation membership. Three training seeds share one fixed source partition; they are not independent subjects, environments, or partition replications.",
              "limits": "Does not estimate a four-arm factorial interaction, rule out every capacity explanation, or establish independent target confirmation."})
    lines = ["# Original H L1248: grouped source validation", "", "All 18 frozen runs completed. Values are percentages; differences are percentage points. SD is across three training seeds on one fixed partition.", "", "| Target | Metric | H10 mean ± SD | H11 mean ± SD | Paired H11−H10 mean ± SD |", "|---|---|---:|---:|---:|"]
    for target in (*TARGETS, "E1-E3_equal_weight"):
        for metric in ("accuracy", "f1_macro"):
            cells = [next(r for r in summary if r["target"] == target and r["metric"] == metric and r["arm"] == arm) for arm in (*ARMS, "H11_minus_H10")]
            lines.append(f"| {target} | {metric} | " + " | ".join(f"{r['mean_percent_or_pp']:.2f} ± {r['seed_sd']:.2f}" for r in cells) + " |")
    lines += ["", "The paired difference measures the SupCon increment with the original L1248 model fixed. E1–E3 were previously used for development. The historical comparison changes both source training and validation membership. No population p-values or confidence intervals are inferred; this two-arm supplement does not establish the full factorial interaction."]
    (destination / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_new(destination / "completed.json", {"status": "complete", "results_sha256": sha(result_path),
              "files_sha256": {p.name: sha(p) for p in destination.iterdir() if p.is_file()}})
    return destination
