"""Verify and describe historical dual-stream H ablations, excluding L124 runs."""
import hashlib
import json
import math
import statistics as st
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/ablation_review_h_l1248_20260917"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    if OUT.exists():
        raise FileExistsError("Use a new version instead of overwriting the review")
    sources = [
        ("outputs_supcon_tpp_legacy_reproduction/factorial_s32/factorial_results.json", {"H00", "H01"}),
        ("outputs_tpp_level8_ablation/l1248_h10_v1/level8_results.json", {"H10"}),
        ("outputs_tpp_level8_ablation/l1248_v1/level8_results.json", {"H11"}),
        ("outputs_tpp_residual_scale_ablation/h11_l1248_v1/scale_results.json", {"H11"}),
    ]
    rows, evidence, code_hashes = [], [], {}
    for relative, allowed in sources:
        index = read(ROOT / relative)
        assert index["status"] == "complete"
        selected = [r for r in index["rows"] if r["arm"] in allowed]
        for row in selected:
            attempt = Path(row["attempt_dir"])
            completion = read(attempt / "attempt_completed.json")
            assert completion["status"] == "success"
            assert sha(attempt / "attempt_completed.json") == row["completion_manifest_sha256"]
            for name, record in completion["artifacts"].items():
                path = attempt / name
                assert path.stat().st_size == record["size_bytes"]
                assert sha(path) == record["sha256"], str(path)
            config = read(attempt / "job_config.json")
            assert config["factorial"]["arm"] == row["arm"]
            assert config["target_env"] == row["target_env"] and config["seed_list"] == [row["seed"]]
            if config["factorial"]["tpp_on"]:
                assert config["model"]["tpp_levels"] == [1, 2, 4, 8]
            else:
                assert config["model"]["name"] == "pa_csi_dg_lite"
            for record in config["provenance"]["legacy_modules"].values():
                code_hashes[record["path"]] = record["sha256"]
            tpp = str(Path(config["provenance"]["tpp_snapshot_root"]) / "experiments/tpp_isolated/tpp_models.py")
            code_hashes[tpp] = config["provenance"]["tpp_models_sha256"]
            run = attempt / "payload/run"
            metrics = read(run / "metrics.json")
            assert metrics["epochs"] == 200 and metrics["total_optimizer_steps"] == 30000
            assert metrics["test_metrics"] == row["test_metrics"]
            assert sha(run / "metrics.json") == row["metrics_file_sha256"]
            assert sha(run / "best_model.pt") == row["checkpoint_sha256"]
            assert sha(run / "target_predictions.jsonl") == row["target_predictions_file_sha256"]
            predictions = [json.loads(line) for line in (run / "target_predictions.jsonl").read_text().splitlines()]
            assert len(predictions) == 3000 and [p["index"] for p in predictions] == list(range(3000))
            confusion = [[0] * 6 for _ in range(6)]
            for p in predictions:
                confusion[p["label"]][p["prediction"]] += 1
            f1 = []
            for k in range(6):
                denominator = sum(confusion[k]) + sum(r[k] for r in confusion)
                f1.append(2 * confusion[k][k] / denominator if denominator else 0.0)
            assert math.isclose(st.mean(f1), row["test_metrics"]["f1_macro"], abs_tol=1e-12, rel_tol=0)
            frozen = read(run / "checkpoint_frozen.json")
            assert frozen["strict_load"] and not frozen["target_data_opened"]
            assert frozen["checkpoint_sha256"] == row["checkpoint_sha256"]
            split = read(run / "split_manifest.json")
            sampler = read(run / "sampler_schedule.json")
            assert sampler["batches_seen"] == 30000 and sampler["samples_seen"] == 960000
            rows.append({"arm": row["arm"], "scale": row.get("scale", .25) if config["factorial"]["tpp_on"] else None,
                         "target": row["target_env"], "seed": row["seed"], "f1_percent": st.mean(f1) * 100,
                         "class_f1_percent": [v * 100 for v in f1], "parameters": frozen["parameter_count"],
                         "sampler": sampler["global_sample_schedule_sha256"],
                         "split": {p: split[p]["global_sample_id_sha256"] for p in ("train", "val", "test")},
                         "config_sha256": sha(attempt / "job_config.json"), "attempt": str(attempt)})
        evidence.append({"index": relative, "sha256": sha(ROOT / relative), "verified_runs": len(selected)})
        print(f"Verified {len(selected)} retained runs: {relative}", flush=True)
    for path, expected in code_hashes.items():
        assert sha(path) == expected, path
    assert len(rows) == 72
    core = [r for r in rows if r["scale"] in (None, .25)]
    lookup = {(r["target"], r["seed"], r["arm"]): r for r in core}
    assert len(core) == len(lookup) == 36
    for row in rows:
        reference = lookup[(row["target"], row["seed"], "H00")]
        assert row["split"] == reference["split"] and row["sampler"] == reference["sampler"]
    scores = []
    for arm in ("H00", "H01", "H10", "H11"):
        selected = [r for r in core if r["arm"] == arm]
        scores.append({"arm": arm, "mean_percent": st.mean(r["f1_percent"] for r in selected),
                       "parameters": selected[0]["parameters"],
                       "environments": {e: {"mean_percent": st.mean(r["f1_percent"] for r in selected if r["target"] == e),
                                            "seed_sd_pp": st.stdev(r["f1_percent"] for r in selected if r["target"] == e)}
                                        for e in ("E1", "E2", "E3")}})
    effects = []
    for a, b in [("H10", "H00"), ("H01", "H00"), ("H11", "H10"), ("H11", "H01"), ("H11", "H00")]:
        pairs = [{"target": r["target"], "seed": r["seed"],
                  "delta_pp": r["f1_percent"] - lookup[(r["target"], r["seed"], b)]["f1_percent"]}
                 for r in core if r["arm"] == a]
        effects.append({"comparison": f"{a}-{b}", "mean_pp": st.mean(p["delta_pp"] for p in pairs),
                        "positive_pairs": sum(p["delta_pp"] > 0 for p in pairs), "pairs": pairs,
                        "environment_mean_pp": {e: st.mean(p["delta_pp"] for p in pairs if p["target"] == e) for e in ("E1", "E2", "E3")}})
    interactions = []
    for r in core:
        if r["arm"] == "H00":
            v = {a: lookup[(r["target"], r["seed"], a)]["f1_percent"] for a in ("H00", "H01", "H10", "H11")}
            interactions.append({"target": r["target"], "seed": r["seed"], "delta_pp": v["H11"] - v["H10"] - v["H01"] + v["H00"]})
    scales = []
    for scale in (.125, .25, .5, .75, 1.0):
        selected = [r for r in rows if r["arm"] == "H11" and r["scale"] == scale]
        deltas = [r["f1_percent"] - lookup[(r["target"], r["seed"], "H11")]["f1_percent"] for r in selected]
        scales.append({"scale": scale, "mean_percent": st.mean(r["f1_percent"] for r in selected),
                       "delta_vs_025_pp": st.mean(deltas), "positive_pairs": sum(d > 0 for d in deltas)})
    report = {"recorded_at": datetime.now(timezone.utc).isoformat(), "excluded": "All L124 results, including R conclusions; no L124-to-L1248 comparison",
              "scope": "Historical dual-stream H with L1248 and matched non-TPP controls; exploratory H11 L1248 scale sensitivity",
              "scores": scores, "effects": effects, "interaction_mean_pp": st.mean(p["delta_pp"] for p in interactions),
              "interactions": interactions, "scales": scales, "evidence": evidence, "rows": rows,
              "verified_dependency_hashes": code_hashes, "population_inference": False}
    OUT.mkdir(parents=True)
    (OUT / "review.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k not in ("rows", "verified_dependency_hashes")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
