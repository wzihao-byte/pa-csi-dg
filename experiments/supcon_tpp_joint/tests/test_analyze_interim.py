from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Mapping

import pytest

from experiments.supcon_tpp_joint import (
    analyze_interim,
    check_protocol,
    generate_configs,
    protocol,
    train_joint,
)
from experiments.supcon_tpp_joint.runner_utils import fixed_label_metrics


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_predictions(
    path: Path,
    target: str,
    arm: str,
    *,
    continue_mode: bool,
    identity_drift: bool = False,
) -> Dict[str, Any]:
    labels = list(protocol.LABELS)
    predictions = list(labels)
    if continue_mode and arm in {"Y01", "Y10"}:
        predictions[0], predictions[1] = predictions[1], predictions[0]
    rows = []
    for index, (true, pred) in enumerate(zip(labels, predictions)):
        probability = [0.0] * len(protocol.LABELS)
        probability[pred] = 1.0
        rows.append(
            {
                "sample_id": f"{target}:sample-{index}",
                "group_id": (
                    f"{target}:drift" if identity_drift and index == 0 else f"{target}:B{index:02d}"
                ),
                "subject_id": None,
                "environment": target,
                "true": true,
                "pred": pred,
                "probability": probability,
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8"
    )
    metrics = fixed_label_metrics(labels, predictions, num_classes=len(protocol.LABELS))
    metrics["loss_ce"] = 0.1
    return metrics


def _small_label_protocol(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        protocol,
        "LABELS",
        {
            label: protocol.LabelSpec(label, f"class_{label}", 1, 1)
            for label in range(6)
        },
    )


def _synthetic_formal_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    continue_mode: bool,
    identity_drift: bool = False,
) -> Dict[str, Any]:
    _small_label_protocol(monkeypatch)
    monkeypatch.setattr(
        analyze_interim.check_protocol,
        "check_bundle",
        lambda _path: {"manifest": {"generation_mode": "formal_interim"}},
    )
    bundle = tmp_path / "formal_bundle"
    output = tmp_path / "formal_output"
    bundle.mkdir()
    output.mkdir()
    execution_hash = "e" * 64
    registry_jobs = []
    job_material: Dict[str, Dict[str, Any]] = {}
    selection_hashes: Dict[str, str] = {}

    for target in protocol.ENVIRONMENTS:
        selection_path = bundle / "selection_records" / f"selection_{target}.json"
        _write_json(selection_path, {"target_env": target})
        selection_hashes[target] = protocol.file_sha256(selection_path)
        selected_lambda = 0.025
        for seed in protocol.formal_seeds(target, "formal_interim"):
            for arm in protocol.ARMS:
                job_id = f"formal_interim__target_{target}__arm_{arm}__seed_{seed}"
                config_path = bundle / "formal_interim_configs" / f"{job_id}.json"
                config = {"experiment_name": job_id, "protocol": {"stage": "formal_interim"}}
                _write_json(config_path, config)
                lambda_supcon = selected_lambda if protocol.ARMS[arm].supcon_enabled else 0.0
                job = {
                    "job_id": job_id,
                    "stage": "formal_interim",
                    "target_env": target,
                    "source_envs": list(protocol.source_environments(target)),
                    "seed": seed,
                    "arm": arm,
                    "lambda_supcon": lambda_supcon,
                    "selected_lambda": selected_lambda,
                    "selection_record_file_sha256": selection_hashes[target],
                    "interim_decision_file_sha256": None,
                    "config_path": config_path.relative_to(bundle).as_posix(),
                    "config_content_sha256": protocol.canonical_sha256(config),
                    "config_file_sha256": protocol.file_sha256(config_path),
                }
                registry_jobs.append(job)
                job_material[job_id] = {"job": job, "config_path": config_path}

    registry = {
        "schema_version": protocol.REGISTRY_SCHEMA,
        "protocol_name": protocol.PROTOCOL_NAME,
        "protocol_version": protocol.PROTOCOL_VERSION,
        "protocol_lock_sha256": protocol.protocol_lock_sha256(),
        "stage": "formal_interim",
        "expected_job_count": 36,
        "jobs": registry_jobs,
    }
    registry_path = bundle / "formal_interim_job_registry.json"
    _write_json(registry_path, registry)
    registry_file_hash = protocol.file_sha256(registry_path)
    registry_content_hash = protocol.canonical_sha256(registry)

    result_rows = []
    run_rows = []
    for job_id, material in job_material.items():
        job = material["job"]
        target, seed, arm = job["target_env"], job["seed"], job["arm"]
        attempt_dir = (
            output
            / "runs"
            / target
            / f"seed_{seed}"
            / job_id
            / "attempt_001"
        ).resolve()
        attempt_dir.mkdir(parents=True)
        target_metrics = _write_predictions(
            attempt_dir / "target_predictions.jsonl",
            target,
            arm,
            continue_mode=continue_mode,
            identity_drift=(
                identity_drift and target == "E1" and seed == 101 and arm == "Y01"
            ),
        )
        _write_json(attempt_dir / "target_metrics.json", target_metrics)
        _write_json(attempt_dir / "metrics.json", {"target_metrics": target_metrics})
        _write_json(attempt_dir / "source_split_manifest.json", {"source": True})
        _write_json(attempt_dir / "target_evaluation_manifest.json", {"target": target})
        _write_json(attempt_dir / "split_manifest.json", {"combined": True})
        _write_json(attempt_dir / "phase_input_audit.json", {"unwrap_phase": False})
        _write_json(attempt_dir / "target_phase_input_audit.json", {"unwrap_phase": False})
        _write_json(attempt_dir / "history.json", [])
        _write_json(attempt_dir / "run_invocation.json", {"config": job_id})
        (attempt_dir / "best_model.pt").write_bytes(b"checkpoint")
        artifact_hashes = {
            name: protocol.file_sha256(attempt_dir / name)
            for name in sorted(analyze_interim.REQUIRED_ATTEMPT_ARTIFACTS)
        }
        manifest = {
            "schema_version": analyze_interim.ATTEMPT_MANIFEST_SCHEMA,
            "protocol_name": protocol.PROTOCOL_NAME,
            "protocol_version": protocol.PROTOCOL_VERSION,
            "protocol_lock_sha256": protocol.protocol_lock_sha256(),
            "stage": "formal_interim",
            "job_id": job_id,
            "target_env": target,
            "source_envs": job["source_envs"],
            "seed": seed,
            "arm": arm,
            "selected_lambda": job["selected_lambda"],
            "lambda_supcon": job["lambda_supcon"],
            "config_path": str(material["config_path"].resolve()),
            "config_content_sha256": job["config_content_sha256"],
            "config_file_sha256": job["config_file_sha256"],
            "selection_record_file_sha256": job["selection_record_file_sha256"],
            "interim_decision_file_sha256": None,
            "formal_registry_file_sha256": registry_file_hash,
            "formal_registry_content_sha256": registry_content_hash,
            "execution_lock_file_sha256": execution_hash,
            "artifact_file_sha256": artifact_hashes,
            "target_metrics": target_metrics,
        }
        manifest_path = attempt_dir / "attempt_manifest.json"
        _write_json(manifest_path, manifest)
        manifest_hash = protocol.file_sha256(manifest_path)

        result_rows.append(
            {
                "job_id": job_id,
                "stage": "formal_interim",
                "target_env": target,
                "seed": seed,
                "arm": arm,
                "selected_lambda": job["selected_lambda"],
                "lambda_supcon": job["lambda_supcon"],
                "config_content_sha256": job["config_content_sha256"],
                "selection_record_file_sha256": job["selection_record_file_sha256"],
                "interim_decision_file_sha256": None,
                "attempt_manifest_file_sha256": manifest_hash,
                "metrics_file_sha256": artifact_hashes["metrics.json"],
                "target_metrics_file_sha256": artifact_hashes["target_metrics.json"],
                "target_predictions_file_sha256": artifact_hashes["target_predictions.jsonl"],
                "best_model_file_sha256": artifact_hashes["best_model.pt"],
                "target_metrics": target_metrics,
                "best_source_val_f1_macro": 0.5,
                "best_epoch": 1,
            }
        )
        row = {field: "" for field in analyze_interim.RUN_REGISTRY_FIELDS}
        row.update(
            {
                "job_id": job_id,
                "stage": "formal_interim",
                "target_env": target,
                "source_envs": json.dumps(job["source_envs"], separators=(",", ":")),
                "seed": str(seed),
                "arm": arm,
                "lambda_supcon": format(job["lambda_supcon"], ".17g"),
                "selected_lambda": format(job["selected_lambda"], ".17g"),
                "selection_record_file_sha256": job["selection_record_file_sha256"],
                "config_path": str(material["config_path"].resolve()),
                "config_content_sha256": job["config_content_sha256"],
                "config_file_sha256": job["config_file_sha256"],
                "formal_registry_file_sha256": registry_file_hash,
                "formal_registry_content_sha256": registry_content_hash,
                "protocol_lock_file_sha256": "p" * 64,
                "bundle_manifest_file_sha256": "b" * 64,
                "execution_lock_file_sha256": execution_hash,
                "status": "completed",
                "attempt": "1",
                "output_dir": str(attempt_dir),
                "return_code": "0",
                "metrics_path": str(attempt_dir / "metrics.json"),
                "metrics_file_sha256": artifact_hashes["metrics.json"],
                "target_metrics_file_sha256": artifact_hashes["target_metrics.json"],
                "target_predictions_file_sha256": artifact_hashes["target_predictions.jsonl"],
                "best_model_file_sha256": artifact_hashes["best_model.pt"],
                "split_manifest_file_sha256": artifact_hashes["split_manifest.json"],
                "attempt_manifest_file_sha256": manifest_hash,
            }
        )
        run_rows.append(row)

    run_registry_path = output / "run_registry.csv"
    with run_registry_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=analyze_interim.RUN_REGISTRY_FIELDS)
        writer.writeheader()
        writer.writerows(run_rows)
    formal_results = {
        "schema_version": analyze_interim.FORMAL_RESULTS_SCHEMA,
        "protocol_name": protocol.PROTOCOL_NAME,
        "protocol_version": protocol.PROTOCOL_VERSION,
        "protocol_lock_sha256": protocol.protocol_lock_sha256(),
        "stage": "formal_interim",
        "formal_registry_file_sha256": registry_file_hash,
        "formal_registry_content_sha256": registry_content_hash,
        "execution_lock_file_sha256": execution_hash,
        "run_registry_file_sha256": protocol.file_sha256(run_registry_path),
        "completed_job_count": 36,
        "results": result_rows,
    }
    formal_results_path = output / "formal_results.json"
    _write_json(formal_results_path, formal_results)
    return {
        "bundle": bundle,
        "output": output,
        "registry": registry,
        "registry_path": registry_path,
        "formal_results_path": formal_results_path,
        "selection_hashes": selection_hashes,
    }


def _contrast_inputs(ds: float, dt: float, c3: float) -> tuple[dict, dict]:
    macro = {}
    sit = {}
    for target in protocol.ENVIRONMENTS:
        for seed in protocol.formal_seeds(target, "formal_interim"):
            macro[(target, seed, "Y11")] = 0.8
            macro[(target, seed, "Y01")] = 0.8 - ds
            macro[(target, seed, "Y10")] = 0.8 - dt
            macro[(target, seed, "Y00")] = 0.7
            sit[(target, seed, "Y11")] = 0.7
            sit[(target, seed, "Y00")] = 0.7 - c3
            sit[(target, seed, "Y01")] = 0.7
            sit[(target, seed, "Y10")] = 0.7
    return macro, sit


def test_effect_coded_model_uses_equal_target_mean_and_df6() -> None:
    observations = {}
    values = {"E1": (0.01, 0.02, 0.03), "E2": (0.04, 0.05, 0.06), "E3": (-0.02, -0.01, 0.0)}
    for target, target_values in values.items():
        for seed, value in zip(protocol.formal_seeds(target, "formal_interim"), target_values):
            observations[(target, seed)] = value
    model = analyze_interim.fit_target_effect_model(observations)
    assert model["residual_degrees_of_freedom"] == 6
    assert model["marginal_mean"] == pytest.approx(0.02)
    assert model["target_means"] == pytest.approx({"E1": 0.02, "E2": 0.05, "E3": -0.01})
    assert model["residual_mean_square"] == pytest.approx(0.0001)
    assert model["marginal_standard_error"] == pytest.approx(0.01 / 3.0)


def test_only_three_preregistered_futility_conditions_are_evaluated() -> None:
    macro, sit = _contrast_inputs(0.03, 0.03, 0.0)
    report = analyze_interim.evaluate_futility(macro, sit)
    assert [item["condition_id"] for item in report["futility_conditions"]] == [
        "sit_stand_harm_vs_control",
        "target_specific_harm",
        "minimum_practical_effect_futility",
    ]
    assert report["triggered_condition_ids"] == []

    macro, sit = _contrast_inputs(0.03, 0.03, -0.03)
    assert analyze_interim.evaluate_futility(macro, sit)["triggered_condition_ids"] == [
        "sit_stand_harm_vs_control"
    ]

    macro, sit = _contrast_inputs(0.0, 0.0, 0.0)
    assert analyze_interim.evaluate_futility(macro, sit)["triggered_condition_ids"] == [
        "minimum_practical_effect_futility"
    ]


def test_builds_atomic_continue_and_detects_prediction_tampering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    evidence = _synthetic_formal_evidence(tmp_path, monkeypatch, continue_mode=True)
    decision_path = tmp_path / "decision" / "interim_decision.json"
    payload = analyze_interim.analyze_and_write_decision(
        evidence["registry_path"], evidence["formal_results_path"], decision_path
    )
    assert payload["decision"] == "continue"
    assert payload["analysis"]["triggered_condition_ids"] == []
    assert len(payload["evidence"]["jobs"]) == 36
    assert analyze_interim.validate_decision_artifact(decision_path, require_continue=True) == payload
    # Idempotent re-entry validates rather than overwriting.
    assert (
        analyze_interim.analyze_and_write_decision(
            evidence["registry_path"], evidence["formal_results_path"], decision_path
        )
        == payload
    )

    prediction_path = next(evidence["output"].rglob("target_predictions.jsonl"))
    prediction_path.write_text(prediction_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(analyze_interim.InterimAnalysisError, match="artifact hash mismatch"):
        analyze_interim.validate_decision_artifact(decision_path)


def test_stop_decision_cannot_authorize_extension(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    evidence = _synthetic_formal_evidence(tmp_path, monkeypatch, continue_mode=False)
    decision_path = tmp_path / "decision" / "interim_decision.json"
    payload = analyze_interim.analyze_and_write_decision(
        evidence["registry_path"], evidence["formal_results_path"], decision_path
    )
    assert payload["decision"] == "stop"
    assert payload["analysis"]["triggered_condition_ids"] == [
        "minimum_practical_effect_futility"
    ]
    selections = {
        target: SimpleNamespace(file_sha256=evidence["selection_hashes"][target])
        for target in protocol.ENVIRONMENTS
    }
    with pytest.raises(ValueError, match="stop"):
        generate_configs.validate_interim_decision(
            decision_path,
            selections,
            interim_registry_content_sha256=protocol.canonical_sha256(evidence["registry"]),
        )


def test_ordered_target_identity_must_match_across_all_paired_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    evidence = _synthetic_formal_evidence(
        tmp_path, monkeypatch, continue_mode=True, identity_drift=True
    )
    with pytest.raises(analyze_interim.InterimAnalysisError, match="identity differs"):
        analyze_interim.build_decision_payload(
            evidence["registry_path"], evidence["formal_results_path"]
        )


def test_generator_and_checker_independently_recompute_continue_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    evidence = _synthetic_formal_evidence(tmp_path, monkeypatch, continue_mode=True)
    decision_path = tmp_path / "decision" / "interim_decision.json"
    analyze_interim.analyze_and_write_decision(
        evidence["registry_path"], evidence["formal_results_path"], decision_path
    )
    selections = {
        target: SimpleNamespace(file_sha256=evidence["selection_hashes"][target])
        for target in protocol.ENVIRONMENTS
    }
    artifact = generate_configs.validate_interim_decision(
        decision_path,
        selections,
        interim_registry_content_sha256=protocol.canonical_sha256(evidence["registry"]),
    )
    assert artifact.record["decision"] == "continue"

    extension_stub = tmp_path / "extension_stub"
    extension_stub.mkdir()
    shutil.copyfile(decision_path, extension_stub / "interim_decision.json")
    checked = check_protocol._validate_decision(
        extension_stub, selections, evidence["registry"]
    )
    assert checked["decision"] == "continue"

    tampered = json.loads(decision_path.read_text(encoding="utf-8"))
    tampered["analysis"]["triggered_condition_ids"] = ["invented_condition"]
    _write_json(decision_path, tampered)
    with pytest.raises(ValueError, match="independent"):
        generate_configs.validate_interim_decision(
            decision_path,
            selections,
            interim_registry_content_sha256=protocol.canonical_sha256(evidence["registry"]),
        )


def test_extension_runtime_revalidates_interim_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = tmp_path / "extension_bundle"
    config_path = bundle / "formal_extension_configs" / "job.json"
    selection_path = bundle / "selection_records" / "selection_E1.json"
    selection = {
        "selected_lambda": 0.025,
        "screen_registry": {"file_sha256": "r" * 64},
        "screen_results": {"file_sha256": "s" * 64},
    }
    _write_json(selection_path, selection)
    decision_path = bundle / "interim_decision.json"
    decision = {
        "protocol_name": protocol.PROTOCOL_NAME,
        "protocol_version": protocol.PROTOCOL_VERSION,
        "protocol_lock_sha256": protocol.protocol_lock_sha256(),
        "decision": "continue",
        "formal_interim_registry_content_sha256": "f" * 64,
    }
    _write_json(decision_path, decision)
    config = {
        "target_env": "E1",
        "protocol": {
            "stage": "formal_extension",
            "selected_lambda": 0.025,
            "selection_record": {
                "path": "selection_records/selection_E1.json",
                "file_sha256": protocol.file_sha256(selection_path),
                "screen_registry_file_sha256": "r" * 64,
                "screen_results_file_sha256": "s" * 64,
            },
            "interim_decision": {
                "path": "interim_decision.json",
                "file_sha256": protocol.file_sha256(decision_path),
                "formal_interim_registry_content_sha256": "f" * 64,
            },
        },
    }
    _write_json(config_path, config)
    monkeypatch.setattr(train_joint.select_lambda, "validate_selection_record", lambda *_args, **_kwargs: None)
    calls = []

    def validate(path: Path, **kwargs: Any) -> Mapping[str, Any]:
        calls.append((path, kwargs))
        return decision

    monkeypatch.setattr(train_joint.analyze_interim, "validate_decision_artifact", validate)
    train_joint.validate_bound_runtime_artifacts(config, config_path)
    assert calls == [
        (
            decision_path.resolve(),
            {
                "expected_registry_content_sha256": "f" * 64,
                "require_continue": True,
            },
        )
    ]
