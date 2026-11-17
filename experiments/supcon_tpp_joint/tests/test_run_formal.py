from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from experiments.supcon_tpp_joint import generate_configs, protocol, run_formal, screen_health, select_lambda
from experiments.supcon_tpp_joint.runner_utils import fixed_label_metrics


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _base_config() -> dict:
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


def _healthy_screen_metrics(macro_f1: float, c3_f1: float) -> dict:
    history = [
        {
            "epoch": epoch,
            "train": {
                "loss_ce": 1.5 - 0.5 * (epoch - 1) / (protocol.EPOCHS - 1),
                "gradient_norm_l2": {"median": 1.0, "p95": 1.2, "maximum": 1.5},
            },
            "val": {"f1_macro": macro_f1},
        }
        for epoch in range(1, protocol.EPOCHS + 1)
    ]
    return screen_health.build_screen_metrics(
        history,
        best_source_val_sit_stand_f1=c3_f1,
        positive_anchor_fraction=1.0,
    )


@pytest.fixture
def formal_bundle(tmp_path: Path) -> Path:
    base = _base_config()
    screen_bundle = tmp_path / "screen"
    generate_configs.write_generation_bundle(screen_bundle, base_config=base)
    registry_path = screen_bundle / "screen_job_registry.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    macro = {"Y00": 0.80, "Y01": 0.82, "Y10": 0.81, "Y11": 0.83}
    c3 = {"Y00": 0.70, "Y01": 0.71, "Y10": 0.70, "Y11": 0.72}
    results = {
        "schema_version": protocol.SCREEN_RESULTS_SCHEMA,
        "protocol_name": protocol.PROTOCOL_NAME,
        "protocol_version": protocol.PROTOCOL_VERSION,
        "protocol_lock_sha256": protocol.protocol_lock_sha256(),
        "screen_registry_file_sha256": protocol.file_sha256(registry_path),
        "results": [
            {
                "job_id": row["job_id"],
                "target_env": row["target_env"],
                "arm": row["arm"],
                "screen_seed": row["screen_seed"],
                "lambda_supcon": row["lambda_supcon"],
                "config_content_sha256": row["config_content_sha256"],
                "metrics": _healthy_screen_metrics(macro[row["arm"]], c3[row["arm"]]),
            }
            for row in registry["jobs"]
        ],
    }
    results_path = screen_bundle / "screen_results.json"
    _write_json(results_path, results)
    records = select_lambda.build_selection_records(registry_path, results_path)
    selection_dir = tmp_path / "selections"
    select_lambda.write_selection_records(selection_dir, records, clean=False)
    formal = tmp_path / "formal"
    generate_configs.write_generation_bundle(
        formal,
        base_config=base,
        selection_record_paths=[
            selection_dir / f"selection_{target}.json" for target in protocol.ENVIRONMENTS
        ],
    )
    return formal


def _flag(command: list[str], flag: str) -> str:
    return command[command.index(flag) + 1]


def _csv_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_plan_and_dry_run_are_exactly_bound_and_deferred(formal_bundle: Path, tmp_path: Path) -> None:
    registry = formal_bundle / "formal_interim_job_registry.json"
    plan = run_formal.load_formal_plan(registry)
    assert plan.stage == "formal_interim"
    assert len(plan.jobs) == protocol.BUDGETS["formal_interim_jobs"] == 36
    assert [(job.target_env, job.seed, job.arm) for job in plan.jobs[:4]] == [
        ("E1", 101, arm) for arm in protocol.ARMS
    ]
    command = run_formal.build_train_command(plan.jobs[0], tmp_path / "attempt", device="cuda:0")
    assert "--epochs" not in command
    assert _flag(command, "--evaluate-target") == "true"
    assert _flag(command, "--seed") == "101"

    output = tmp_path / "formal_output"
    summary = run_formal.execute_formal(
        registry,
        output,
        device="cpu",
        dry_run=True,
        command_runner=lambda *args, **kwargs: pytest.fail("dry-run launched training"),
    )
    assert summary["completed_job_count"] == 0
    rows = _csv_rows(output / "run_registry.csv")
    assert len(rows) == 36
    assert {row["status"] for row in rows} == {"planned"}
    assert {row["attempt"] for row in rows} == {"1"}
    assert len({row["execution_lock_file_sha256"] for row in rows}) == 1
    execution_lock = json.loads((output / "execution_lock.json").read_text(encoding="utf-8"))
    assert execution_lock["training_code_sha256"]["experiments/supcon_tpp_joint/run_formal.py"]
    assert execution_lock["training_code_sha256"]["experiments/supcon_tpp_joint/check_protocol.py"]


def test_prediction_validation_recomputes_metrics_and_binds_identity(tmp_path: Path) -> None:
    path = tmp_path / "target_predictions.jsonl"
    rows = []
    index = 0
    for label, spec in protocol.LABELS.items():
        for _ in range(spec.support_per_environment):
            probability = [0.0] * len(protocol.LABELS)
            probability[label] = 1.0
            rows.append(
                {
                    "sample_id": f"E1:{index:04d}",
                    "group_id": f"E1:B{index // 300:02d}",
                    "subject_id": None,
                    "environment": "E1",
                    "true": label,
                    "pred": label,
                    "probability": probability,
                }
            )
            index += 1
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    evidence = run_formal._validate_prediction_file(path, "E1")
    assert evidence["count"] == 3000
    assert evidence["recomputed_metrics"]["f1_macro"] == 1.0
    assert len(evidence["ordered_sample_true_group_sha256"]) == 64

    rows[0]["pred"] = 1
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    with pytest.raises(run_formal.FormalExecutionError, match="prediction provenance"):
        run_formal._validate_prediction_file(path, "E1")


def test_seed_barrier_rejects_prediction_population_drift(
    formal_bundle: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = run_formal.load_formal_plan(formal_bundle / "formal_interim_job_registry.json")
    jobs = [job for job in plan.jobs if job.target_env == "E1" and job.seed == 101]
    rows = {job.job_id: {"status": "completed"} for job in jobs}

    def fake_completed(row: dict, job: run_formal.FormalJob, **kwargs: object):
        identity = "different" if job.arm == "Y11" else "same"
        hashes = {
            "calibration_sha256": "c",
            "source_split_manifest_sha256": "s",
            "target_evaluation_manifest_sha256": "t",
            "source_phase_audit_sha256": "p",
            "target_phase_audit_sha256": "q",
            "target_prediction_identity_sha256": identity,
        }
        metrics = {
            "resources": {
                "batch_schedule_audit": {
                    "schedule_sha256": "b",
                    "first_batch_sample_ids": ["x"],
                }
            },
            "parameter_report": {
                "trainable_parameter_count": 1,
                "initialization_hashes": {arm: "i" for arm in protocol.ARMS},
            },
        }
        return metrics, hashes

    monkeypatch.setattr(run_formal, "_completed_metrics", fake_completed)
    with pytest.raises(run_formal.FormalExecutionError, match="target_prediction_identity_sha256"):
        run_formal._validate_seed_block(
            "E1",
            101,
            jobs,
            rows,
            plan=plan,
            execution_lock_file_sha256="e" * 64,
        )


def test_recomputed_target_metrics_detect_tampering() -> None:
    labels = [0, 0, 1, 1, 2, 3, 4, 5]
    predictions = list(labels)
    recomputed = fixed_label_metrics(labels, predictions, num_classes=6)
    recorded = dict(recomputed)
    recorded["f1_macro"] = 0.5
    with pytest.raises(run_formal.FormalExecutionError, match="target predictions"):
        run_formal._assert_recomputed_target_metrics(recomputed, recorded, context="target")


def test_single_runner_os_lock_fails_closed(tmp_path: Path) -> None:
    lock = tmp_path / "formal" / ".runner.lock"
    with run_formal.exclusive_runner_lock(lock):
        with pytest.raises(run_formal.FormalExecutionError, match="already owns"):
            with run_formal.exclusive_runner_lock(lock):
                pytest.fail("a second runner acquired the same output root")
    # The persistent file is harmless: ownership was released by the OS lock.
    with run_formal.exclusive_runner_lock(lock):
        pass


@pytest.mark.parametrize("prior_status", ["running", "failed"])
def test_nonempty_attempt_is_preserved_and_resume_uses_new_attempt(
    formal_bundle: Path,
    tmp_path: Path,
    prior_status: str,
) -> None:
    plan = run_formal.load_formal_plan(formal_bundle / "formal_interim_job_registry.json")
    job = plan.jobs[0]
    execution_hash = "e" * 64
    row = run_formal._base_run_row(job, plan, execution_hash)
    old_output, _, _, _ = run_formal._prepare_attempt(
        row,
        job,
        tmp_path / "formal_output",
        device="cpu",
    )
    old_output.mkdir(parents=True)
    preserved = old_output / "failure_evidence.txt"
    preserved.write_text("do not delete", encoding="utf-8")
    row["status"] = prior_status

    new_output, _, _, _ = run_formal._prepare_attempt(
        row,
        job,
        tmp_path / "formal_output",
        device="cpu",
    )
    assert old_output.name == "attempt_001"
    assert new_output.name == "attempt_002"
    assert new_output != old_output
    assert preserved.read_text(encoding="utf-8") == "do not delete"


def test_prelaunch_revalidation_detects_training_code_drift(
    formal_bundle: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = run_formal.load_formal_plan(formal_bundle / "formal_interim_job_registry.json")
    job = plan.jobs[0]
    lock_path = tmp_path / "execution_lock.json"
    payload = run_formal.execution_lock_payload(
        plan,
        device="cpu",
        formal_output_root=lock_path.parent,
    )
    _write_json(lock_path, payload)
    lock_hash = protocol.file_sha256(lock_path)
    original = run_formal.training_code_sha256()
    changed = dict(original)
    changed["dg_dataset.py"] = "0" * 64
    monkeypatch.setattr(run_formal, "training_code_sha256", lambda: changed)
    with pytest.raises(run_formal.FormalExecutionError, match="fingerprint changed"):
        run_formal._revalidate_launch_binding(
            plan,
            job,
            execution_lock_path=lock_path,
            execution_lock_file_sha256=lock_hash,
            device="cpu",
        )


def test_tampered_attempt_manifest_fails_closed(tmp_path: Path) -> None:
    path = tmp_path / "attempt_manifest.json"
    expected = {"schema_version": run_formal.ATTEMPT_MANIFEST_SCHEMA, "job_id": "job"}
    run_formal._commit_or_validate_attempt_manifest(path, expected, create=True)
    tampered = dict(expected)
    tampered["job_id"] = "other"
    _write_json(path, tampered)
    with pytest.raises(run_formal.FormalExecutionError, match="drifted"):
        run_formal._commit_or_validate_attempt_manifest(path, expected, create=False)


def test_unsafe_bundle_output_is_rejected_before_lockfile_write(formal_bundle: Path) -> None:
    lock_path = formal_bundle / ".runner.lock"
    assert not lock_path.exists()
    with pytest.raises(run_formal.FormalExecutionError, match="Unsafe formal output"):
        run_formal.execute_formal(
            formal_bundle / "formal_interim_job_registry.json",
            formal_bundle,
            device="cpu",
            dry_run=True,
        )
    assert not lock_path.exists()
