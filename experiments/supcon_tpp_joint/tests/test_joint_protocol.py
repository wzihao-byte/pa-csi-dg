from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from experiments.supcon_tpp_joint import (
    check_protocol,
    generate_configs,
    protocol,
    screen_health,
    select_lambda,
)


@pytest.fixture
def minimal_base_config() -> dict:
    return {
        "experiment_name": "legacy",
        "device": "cpu",
        "data": {
            "format": "env_npy_triplets",
            "unwrap_phase": True,
            "phase_unwrap_axis": -1,
            "env_files": {
                environment: {
                    "amp": f"{environment}_amp.npy",
                    "phase": f"{environment}_phase.npy",
                    "label": f"{environment}_label.npy",
                }
                for environment in protocol.ENVIRONMENTS
            },
        },
        "model": {
            "name": "legacy",
            "fusion_dim": 256,
            "projection_hidden_dim": 256,
            "projection_dim": 128,
            "antenna_layout": {"mode": "tx", "num_tx": 3},
        },
        "training": {"num_workers": 0},
    }


def _screen_results_payload(registry_path: Path) -> dict:
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    rows = []
    for job in registry["jobs"]:
        target = job["target_env"]
        arm = job["arm"]
        value = job["lambda_supcon"]
        macro = 0.80
        c3 = 0.70
        if arm == "Y01":
            if target == "E2":
                macro = 0.805 if value == 0.01 else 0.83
            else:
                macro = 0.82 if value == 0.01 else 0.823
        elif arm == "Y11":
            if target == "E2":
                macro = 0.805 if value == 0.01 else 0.84
            else:
                macro = 0.83 if value == 0.01 else 0.834
            c3 = 0.71
        history = [
            {
                "epoch": epoch,
                "train": {
                    "loss_ce": 1.5 - 0.5 * (epoch - 1) / (protocol.EPOCHS - 1),
                    "gradient_norm_l2": {
                        "median": 1.0,
                        "p95": 1.2,
                        "maximum": 1.5,
                    },
                },
                "val": {"f1_macro": macro},
            }
            for epoch in range(1, protocol.EPOCHS + 1)
        ]
        metrics = screen_health.build_screen_metrics(
            history,
            best_source_val_sit_stand_f1=c3,
            positive_anchor_fraction=1.0,
        )
        rows.append(
            {
                "job_id": job["job_id"],
                "target_env": target,
                "arm": arm,
                "screen_seed": job["screen_seed"],
                "lambda_supcon": value,
                "config_content_sha256": job["config_content_sha256"],
                "metrics": metrics,
            }
        )
    return {
        "schema_version": protocol.SCREEN_RESULTS_SCHEMA,
        "protocol_name": protocol.PROTOCOL_NAME,
        "protocol_version": protocol.PROTOCOL_VERSION,
        "protocol_lock_sha256": protocol.protocol_lock_sha256(),
        "screen_registry_file_sha256": protocol.file_sha256(registry_path),
        "results": rows,
    }


def _selection_record_paths(
    tmp_path: Path,
    base_config: dict,
) -> tuple[Path, list[Path]]:
    screen_dir = tmp_path / "screen_bundle"
    generate_configs.write_generation_bundle(screen_dir, base_config=base_config)
    registry_path = screen_dir / "screen_job_registry.json"
    results_path = tmp_path / "screen_results.json"
    select_lambda.write_json(results_path, _screen_results_payload(registry_path))
    records = select_lambda.build_selection_records(registry_path, results_path)
    selection_dir = tmp_path / "selection_records_external"
    select_lambda.write_selection_records(selection_dir, records, clean=False)
    paths = [selection_dir / f"selection_{environment}.json" for environment in protocol.ENVIRONMENTS]
    return screen_dir, paths


def test_constants_and_linear_warmup_boundaries() -> None:
    protocol.validate_protocol_constants()
    assert tuple(protocol.ARMS) == ("Y00", "Y01", "Y10", "Y11")
    assert protocol.LABELS[2].name == "walking"
    assert protocol.LABELS[3].name == "sitting_down_standing_up"
    assert protocol.INNER_VALIDATION == "inferred_group_block_holdout"
    assert protocol.STRICT_GROUP_PROTOCOL is True
    assert protocol.SIDECAR_AVAILABLE is False
    assert protocol.SPLIT_SEED == 20260714
    assert protocol.LINEAR_WARMUP_STEPS == 6000
    assert protocol.linear_warmup_multiplier(0) == 0.0
    assert protocol.linear_warmup_multiplier(1) == pytest.approx(1 / 6000)
    assert protocol.linear_warmup_multiplier(5999) == pytest.approx(5999 / 6000)
    assert protocol.linear_warmup_multiplier(6000) == 1.0
    assert protocol.linear_warmup_multiplier(6001) == 1.0
    with pytest.raises(TypeError):
        protocol.linear_warmup_multiplier(1.5)  # type: ignore[arg-type]


def test_templates_use_strict_group_and_single_warmup_schema(minimal_base_config: dict) -> None:
    templates = generate_configs.build_arm_templates(minimal_base_config)
    assert check_protocol.check_templates(templates)["template_count"] == 4
    for config in templates.values():
        assert config["data"]["strict_group_protocol"] is True
        assert config["split"]["strict_group_protocol"] is True
        assert config["split"]["inner_validation"] == "inferred_group_block_holdout"
        assert config["split"]["validation_group_ids"] == {
            environment: list(groups) for environment, groups in protocol.VALIDATION_GROUP_IDS.items()
        }
        assert config["split"]["split_seed"] == 20260714
        assert "fallback_inner_validation" not in config["split"]
        assert "subject_fold" not in config["split"]
        assert config["losses"]["linear_warmup_steps"] == 6000
        assert "lambda_supcon_warmup_epochs" not in config["losses"]
        assert "lambda_supcon_warmup_steps" not in config["losses"]


def test_screen_jobs_physically_exclude_target_from_source_loader(minimal_base_config: dict) -> None:
    jobs = generate_configs.build_screen_job_registry(minimal_base_config)
    assert check_protocol.check_screen_job_registry(jobs)["screen_job_count"] == 18
    for job in jobs:
        config = job["config"]
        target = job["target_env"]
        sources = list(protocol.source_environments(target))
        assert config["data"]["strict_group_protocol"] is True
        assert config["data"]["source_only"] is True
        assert config["data"]["loaded_envs"] == sources
        assert set(config["data"]["env_files"]) == set(sources)
        assert target not in config["data"]["env_files"]
        assert config["target_data"]["target_env"] == target
        assert config["target_data"]["load_for_evaluation"] is False
        assert config["target_data"]["env_files"]["label"] == f"{target}_label.npy"


def test_protocol_lock_must_match_constants(tmp_path: Path) -> None:
    lock_path = tmp_path / "protocol_lock.json"
    select_lambda.write_json(lock_path, protocol.protocol_lock_payload())
    assert check_protocol.check_protocol_lock(lock_path)["content_sha256"] == protocol.protocol_lock_sha256()
    payload = json.loads(lock_path.read_text(encoding="utf-8"))
    payload["training"]["epochs"] = 199
    select_lambda.write_json(lock_path, payload)
    with pytest.raises(check_protocol.ProtocolValidationError, match="exactly match"):
        check_protocol.check_protocol_lock(lock_path)


def test_selection_applies_hard_gates_score_and_smallest_lambda_rule(
    tmp_path: Path, minimal_base_config: dict
) -> None:
    screen_dir = tmp_path / "screen"
    generate_configs.write_generation_bundle(screen_dir, base_config=minimal_base_config)
    registry_path = screen_dir / "screen_job_registry.json"
    results_path = tmp_path / "results.json"
    select_lambda.write_json(results_path, _screen_results_payload(registry_path))

    records = select_lambda.build_selection_records(registry_path, results_path)
    assert records["E1"]["selected_lambda"] == 0.01
    assert records["E2"]["selected_lambda"] == 0.025
    assert records["E3"]["selected_lambda"] == 0.01
    assert all(record["status"] == "selected" for record in records.values())


def test_screen_result_hash_field_binds_real_generated_config(
    tmp_path: Path, minimal_base_config: dict
) -> None:
    screen_dir = tmp_path / "screen"
    generate_configs.write_generation_bundle(screen_dir, base_config=minimal_base_config)
    registry_path = screen_dir / "screen_job_registry.json"
    payload = _screen_results_payload(registry_path)
    assert "config_content_sha256" in payload["results"][0]
    payload["results"][0]["config_content_sha256"] = "0" * 64
    results_path = tmp_path / "tampered_results.json"
    select_lambda.write_json(results_path, payload)
    with pytest.raises(select_lambda.SelectionValidationError, match="config_content_sha256"):
        select_lambda.build_selection_records(registry_path, results_path)


def test_selection_hard_gate_can_eliminate_low_lambda(tmp_path: Path, minimal_base_config: dict) -> None:
    screen_dir = tmp_path / "screen"
    generate_configs.write_generation_bundle(screen_dir, base_config=minimal_base_config)
    registry_path = screen_dir / "screen_job_registry.json"
    payload = _screen_results_payload(registry_path)
    for row in payload["results"]:
        if row["target_env"] == "E1" and row["arm"] == "Y11" and row["lambda_supcon"] == 0.01:
            row["metrics"]["best_source_val_sit_stand_f1"] = 0.60
    results_path = tmp_path / "results.json"
    select_lambda.write_json(results_path, payload)
    records = select_lambda.build_selection_records(registry_path, results_path)
    assert records["E1"]["selected_lambda"] == 0.025
    low = next(row for row in records["E1"]["candidate_evaluations"] if row["lambda_supcon"] == 0.01)
    assert "Y11_sit_stand_f1_drop_vs_Y10" in low["hard_gate_failures"]


def test_formal_interim_has_36_single_seed_jobs_from_records(
    tmp_path: Path, minimal_base_config: dict
) -> None:
    _, paths = _selection_record_paths(tmp_path, minimal_base_config)
    selections = select_lambda.load_selection_record_files(paths)
    jobs = generate_configs.build_formal_interim_job_registry(selections, minimal_base_config)
    summary = check_protocol.check_formal_job_registry(
        jobs, stage="formal_interim", selections=selections
    )
    assert summary["job_count"] == 36
    assert all(len(job["config"]["seed_list"]) == 1 for job in jobs)
    assert {job["seed"] for job in jobs if job["target_env"] == "E1"} == {101, 102, 103}
    assert all(job["config"]["protocol"]["selection_record"]["file_sha256"] for job in jobs)
    assert all(job["config"]["data"]["source_only"] is True for job in jobs)
    assert all(job["config"]["data"]["strict_group_protocol"] is True for job in jobs)
    assert all(job["config"]["target_data"]["load_for_evaluation"] is True for job in jobs)


def test_formal_generation_rejects_failed_or_tampered_selection_record(
    tmp_path: Path, minimal_base_config: dict
) -> None:
    _, paths = _selection_record_paths(tmp_path, minimal_base_config)
    record = json.loads(paths[0].read_text(encoding="utf-8"))
    record["selected_lambda"] = 0.025
    select_lambda.write_json(paths[0], record)
    with pytest.raises(select_lambda.SelectionValidationError):
        select_lambda.load_selection_record_files(paths)


def test_extension_rejects_continue_decision_without_36_run_evidence(
    tmp_path: Path, minimal_base_config: dict
) -> None:
    _, paths = _selection_record_paths(tmp_path, minimal_base_config)
    interim_dir = tmp_path / "formal_interim"
    generate_configs.write_generation_bundle(
        interim_dir,
        base_config=minimal_base_config,
        selection_record_paths=paths,
    )
    selections = select_lambda.load_selection_record_files(paths)
    interim_registry = json.loads(
        (interim_dir / "formal_interim_job_registry.json").read_text(encoding="utf-8")
    )
    decision_path = tmp_path / "interim_decision.json"
    decision = {
        "schema_version": protocol.INTERIM_DECISION_SCHEMA,
        "protocol_name": protocol.PROTOCOL_NAME,
        "protocol_version": protocol.PROTOCOL_VERSION,
        "protocol_lock_sha256": protocol.protocol_lock_sha256(),
        "decision": "continue",
        "completed_interim_job_count": 36,
        "formal_interim_registry_content_sha256": protocol.canonical_sha256(interim_registry),
        "selection_record_file_sha256": {
            environment: selections[environment].file_sha256 for environment in protocol.ENVIRONMENTS
        },
    }
    select_lambda.write_json(decision_path, decision)
    with pytest.raises(ValueError, match="evidence|schema drifted"):
        generate_configs.write_generation_bundle(
            tmp_path / "formal_extension",
            base_config=minimal_base_config,
            selection_record_paths=paths,
            interim_decision_path=decision_path,
        )


def test_stale_output_fails_closed_and_clean_is_explicit(tmp_path: Path, minimal_base_config: dict) -> None:
    output = tmp_path / "bundle"
    generate_configs.write_generation_bundle(output, base_config=minimal_base_config)
    with pytest.raises(generate_configs.StaleOutputError, match="--clean"):
        generate_configs.write_generation_bundle(output, base_config=minimal_base_config)
    summary = generate_configs.write_generation_bundle(output, base_config=minimal_base_config, clean=True)
    assert summary["screen_job_count"] == 18


def test_checker_rejects_stale_extra_file(tmp_path: Path, minimal_base_config: dict) -> None:
    output = tmp_path / "bundle"
    generate_configs.write_generation_bundle(output, base_config=minimal_base_config)
    (output / "stale.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(check_protocol.ProtocolValidationError, match="Stale/missing"):
        check_protocol.check_bundle(output)


def test_checker_rejects_semantically_tampered_summary_even_with_rehashed_manifest(
    tmp_path: Path, minimal_base_config: dict
) -> None:
    output = tmp_path / "bundle"
    generate_configs.write_generation_bundle(output, base_config=minimal_base_config)
    summary_path = output / "generation_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["screen_job_count"] = 17
    select_lambda.write_json(summary_path, summary)
    manifest_path = output / "bundle_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"]["generation_summary.json"] = protocol.file_sha256(summary_path)
    select_lambda.write_json(manifest_path, manifest)
    with pytest.raises(check_protocol.ProtocolValidationError, match="generation_summary"):
        check_protocol.check_bundle(output)


def test_checker_rejects_nested_registry_or_target_loader_drift(
    minimal_base_config: dict,
) -> None:
    jobs = generate_configs.build_screen_job_registry(minimal_base_config)
    jobs[0]["config"]["data"]["env_files"][jobs[0]["target_env"]] = {
        "amp": "leak",
        "phase": "leak",
        "label": "leak",
    }
    with pytest.raises(check_protocol.ProtocolValidationError, match="exactly the two source|leaked"):
        check_protocol.check_screen_job_registry(jobs)

    jobs = generate_configs.build_screen_job_registry(minimal_base_config)
    jobs[0]["source_envs"] = ["E1"]
    with pytest.raises(check_protocol.ProtocolValidationError, match="metadata mismatch"):
        check_protocol.check_screen_job_registry(jobs)


def test_non_whitelisted_template_drift_is_rejected(minimal_base_config: dict) -> None:
    templates = generate_configs.build_arm_templates(minimal_base_config)
    templates["Y11"]["training"]["lr"] = 0.01
    with pytest.raises(check_protocol.ProtocolValidationError, match="training.lr"):
        check_protocol.check_templates(templates)


def test_budget_counts_are_locked() -> None:
    budgets = check_protocol.check_budget_constants()
    assert budgets["screen_runs"] == 18
    assert budgets["formal_interim_jobs"] == 36
    assert budgets["formal_extension_jobs"] == 24
    assert budgets["formal_full_jobs"] == 60
    assert budgets["internal_stop_path_full_runs"] == 54
    assert budgets["internal_complete_path_full_runs"] == 78
