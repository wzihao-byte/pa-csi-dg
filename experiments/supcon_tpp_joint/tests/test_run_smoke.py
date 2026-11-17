from __future__ import annotations

import csv
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, List, Mapping

import pytest

from experiments.supcon_tpp_joint import check_protocol, generate_configs, protocol, run_smoke


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


@pytest.fixture
def screen_bundle(tmp_path: Path) -> Path:
    bundle = tmp_path / "screen_bundle"
    generate_configs.write_generation_bundle(bundle, base_config=_base_config())
    return bundle


def _flag(command: List[str], name: str) -> str:
    return command[command.index(name) + 1]


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


class FakeSmokeTrainer:
    def __init__(self, artifact_root: Path, *, mutation: str | None = None) -> None:
        self.artifact_root = artifact_root
        self.mutation = mutation
        self.commands: List[List[str]] = []

    def __call__(self, command: List[str], *, cwd: Path, check: bool) -> SimpleNamespace:
        assert cwd == run_smoke.REPO_ROOT
        assert check is False
        self.commands.append(command)
        config_path = Path(_flag(command, "--config"))
        output_dir = Path(_flag(command, "--output-dir"))
        config = json.loads(config_path.read_text(encoding="utf-8"))
        arm = str(config["protocol"]["arm"])
        target_env = str(config["target_env"])
        seed = int(_flag(command, "--seed"))
        is_repeat = output_dir.name.endswith("__determinism_repeat")
        output_dir.mkdir(parents=True, exist_ok=False)

        calibration_path = self.artifact_root / f"calibration_{target_env}.json"
        calibration_payload: Mapping[str, Any] = {
            "outer_target": target_env,
            "source_envs": config["source_envs"],
        }
        if self.mutation in {"calibration_path", "calibration_hash"} and arm == "Y11" and not is_repeat:
            calibration_path = self.artifact_root / f"calibration_{target_env}_Y11.json"
            if self.mutation == "calibration_hash":
                calibration_payload = dict(calibration_payload) | {"drift": True}
        if not calibration_path.exists():
            _write_json(calibration_path, calibration_payload)
        calibration_hash = protocol.file_sha256(calibration_path)

        split_manifest = {
            "target_data_loaded": False,
            "loaded_environments": config["source_envs"],
            "train_index_sha256": "1" * 64,
            "val_index_sha256": "2" * 64,
        }
        if self.mutation == "split" and arm == "Y11" and not is_repeat:
            split_manifest["drift"] = True
        phase_audit = {
            "loaded_environments": config["source_envs"],
            "unwrap_phase": False,
            "records": {environment: {"exact_float32_match": True} for environment in config["source_envs"]},
        }
        if self.mutation == "phase" and arm == "Y11" and not is_repeat:
            phase_audit["drift"] = True

        initialization = "a" * 64
        if self.mutation == "initialization" and arm == "Y11" and not is_repeat:
            initialization = "b" * 64
        if self.mutation == "repeat_initialization" and is_repeat:
            initialization = "b" * 64
        initialization_hashes = {code: initialization for code in protocol.ARMS}
        parameter_audit = {
            "trainable_parameter_count": 12345,
            "initialization_hashes": initialization_hashes,
            "calibration_artifact": str(calibration_path.resolve()),
            "calibration_sha256": calibration_hash,
        }

        first_batch_ids = [f"sample-{index:02d}" for index in range(protocol.BATCH_SIZE)]
        if self.mutation == "repeat_batch_ids" and is_repeat:
            first_batch_ids[-1] = "sample-drift"
        schedules = [protocol.canonical_sha256({"seed": seed, "epoch": epoch}) for epoch in range(10)]
        if self.mutation == "schedule" and arm == "Y11" and not is_repeat:
            schedules[4] = "c" * 64
        if self.mutation == "repeat_schedule" and is_repeat:
            schedules[4] = "d" * 64
        batch_audit = {
            "schedule_sha256": schedules[0],
            "first_batch_sample_ids": first_batch_ids,
        }
        history = [
            {"epoch": epoch, "train": {"schedule_sha256": schedules[epoch - 1]}}
            for epoch in range(1, 11)
        ]
        resources = {
            "epochs": 10,
            "optimizer_steps": 10 * protocol.OPTIMIZER_STEPS_PER_EPOCH,
            "mean_seconds_per_epoch": 1.25,
            "peak_cuda_allocated_bytes": 100,
            "peak_cuda_reserved_bytes": 120,
            "batch_schedule_audit": batch_audit,
            "history": history,
        }
        stage0_gate = {
            "target_opened": False,
            "split_manifest_sha256": protocol.canonical_sha256(split_manifest),
            "phase_audit_sha256": protocol.canonical_sha256(phase_audit),
            "calibration_sha256": calibration_hash,
            "parameter_audit": parameter_audit,
        }
        metrics = {
            "arm": arm,
            "target_env": target_env,
            "seed": seed,
            "evaluate_target": False,
            "resources": resources,
            "parameter_report": parameter_audit,
            "calibration_artifact": str(calibration_path.resolve()),
            "calibration_sha256": calibration_hash,
            "screen_metrics": {"finite_training": True, "gradient_exploded": False},
        }
        _write_json(output_dir / "split_manifest.json", split_manifest)
        _write_json(output_dir / "phase_input_audit.json", phase_audit)
        _write_json(output_dir / "batch_schedule_audit.json", batch_audit)
        _write_json(output_dir / "parameter_audit.json", parameter_audit)
        _write_json(output_dir / "stage0_gate.json", stage0_gate)
        _write_json(output_dir / "history.json", history)
        _write_json(
            output_dir / "run_invocation.json",
            {
                "config_path": str(config_path.resolve()),
                "raw_config_sha256": protocol.canonical_sha256(config),
            },
        )
        _write_json(output_dir / "metrics.json", metrics)
        return SimpleNamespace(returncode=0)


def _csv_rows(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_preflight_is_complete_and_precedes_output_or_launch(
    screen_bundle: Path, tmp_path: Path
) -> None:
    registry = json.loads((screen_bundle / "screen_job_registry.json").read_text(encoding="utf-8"))
    jobs = registry["jobs"]
    broken = dict(jobs[0]) | {"config_file_sha256": "0" * 64}
    with pytest.raises(run_smoke.SmokeGateError, match="Config file hash mismatch"):
        run_smoke.validate_registry_config_hashes(screen_bundle, [broken])

    config_path = screen_bundle / str(jobs[0]["config_path"])
    config_path.write_text(config_path.read_text(encoding="utf-8") + " ", encoding="utf-8")
    output_dir = tmp_path / "must_not_exist"
    trainer = FakeSmokeTrainer(tmp_path / "artifacts")
    with pytest.raises(check_protocol.ProtocolValidationError, match="Bundle file hash mismatch"):
        run_smoke.execute_smoke(
            screen_bundle,
            "E1",
            device="cpu",
            output_dir=output_dir,
            dry_run=False,
            command_runner=trainer,
        )
    assert not output_dir.exists()
    assert trainer.commands == []


def test_success_runs_four_arms_and_one_minimal_determinism_repeat(
    screen_bundle: Path, tmp_path: Path
) -> None:
    output_dir = tmp_path / "smoke"
    trainer = FakeSmokeTrainer(tmp_path / "artifacts")
    report = run_smoke.execute_smoke(
        screen_bundle,
        "E1",
        device="cpu",
        output_dir=output_dir,
        dry_run=False,
        command_runner=trainer,
    )

    assert report["passed"] is True
    assert len(trainer.commands) == 5
    assert [_flag(command, "--epochs") for command in trainer.commands] == ["10"] * 5
    assert [_flag(command, "--evaluate-target") for command in trainer.commands] == ["false"] * 5
    first_config = Path(_flag(trainer.commands[0], "--config"))
    repeat_config = Path(_flag(trainer.commands[-1], "--config"))
    assert first_config == repeat_config
    assert _flag(trainer.commands[0], "--seed") == _flag(trainer.commands[-1], "--seed")
    assert report["deterministic_repeat"] == {
        "arm": "Y00",
        "seed": protocol.SCREEN_SEEDS["E1"],
        "target_env": "E1",
        "comparison_scope": [
            "first_batch_sample_ids",
            "all_epoch_batch_schedule_sha256",
            "initialization_hashes",
        ],
        "passed": True,
    }
    assert all(len(schedules) == 10 for schedules in report["epoch_batch_schedule_sha256"].values())
    rows = _csv_rows(output_dir / "smoke_registry.csv")
    assert len(rows) == 5
    assert [row["run_role"] for row in rows].count("primary") == 4
    assert [row["run_role"] for row in rows].count("determinism_repeat") == 1


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("split", "split manifests differ across arms"),
        ("phase", "phase input audits differ across arms"),
        ("calibration_path", "calibration artifact paths differ across arms"),
        ("calibration_hash", "calibration artifact hashes differ across arms"),
        ("initialization", "trainable initialization hashes differ across arms"),
        ("schedule", "epoch batch schedules differ across arms"),
    ],
)
def test_four_arm_gate_fails_closed_on_paired_artifact_drift(
    screen_bundle: Path, tmp_path: Path, mutation: str, message: str
) -> None:
    trainer = FakeSmokeTrainer(tmp_path / "artifacts", mutation=mutation)
    with pytest.raises(run_smoke.SmokeGateError, match=message):
        run_smoke.execute_smoke(
            screen_bundle,
            "E1",
            device="cpu",
            output_dir=tmp_path / "smoke",
            dry_run=False,
            command_runner=trainer,
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("repeat_batch_ids", "deterministic repeat batch IDs differ"),
        ("repeat_schedule", "deterministic repeat epoch batch schedules differ"),
        ("repeat_initialization", "deterministic repeat initialization hashes differ"),
    ],
)
def test_same_arm_same_seed_repeat_uses_only_preregistered_determinism_checks(
    screen_bundle: Path, tmp_path: Path, mutation: str, message: str
) -> None:
    trainer = FakeSmokeTrainer(tmp_path / "artifacts", mutation=mutation)
    with pytest.raises(run_smoke.SmokeGateError, match=message):
        run_smoke.execute_smoke(
            screen_bundle,
            "E1",
            device="cpu",
            output_dir=tmp_path / "smoke",
            dry_run=False,
            command_runner=trainer,
        )
