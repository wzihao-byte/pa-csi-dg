from __future__ import annotations

import csv
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, List

import pytest

from experiments.supcon_tpp_joint import generate_configs, protocol, screen_health, select_lambda
from experiments.supcon_tpp_joint import run_screen


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


class FakeTrainer:
    def __init__(self, artifact_root: Path, *, fail_call: int | None = None) -> None:
        self.artifact_root = artifact_root
        self.fail_call = fail_call
        self.commands: List[List[str]] = []

    def __call__(self, command: List[str], *, cwd: Path, check: bool) -> SimpleNamespace:
        assert cwd == run_screen.REPO_ROOT
        assert check is False
        self.commands.append(command)
        if self.fail_call is not None and len(self.commands) == self.fail_call:
            return SimpleNamespace(returncode=7)

        config_path = Path(_flag(command, "--config"))
        output_dir = Path(_flag(command, "--output-dir"))
        config = json.loads(config_path.read_text(encoding="utf-8"))
        target_env = config["target_env"]
        arm = config["protocol"]["arm"]
        seed = int(_flag(command, "--seed"))
        output_dir.mkdir(parents=True, exist_ok=False)

        calibration_path = self.artifact_root / f"calibration_{target_env}.json"
        if not calibration_path.exists():
            _write_json(calibration_path, {"outer_target": target_env, "source_envs": config["source_envs"]})
        calibration_hash = protocol.file_sha256(calibration_path)
        macro_by_arm = {"Y00": 0.80, "Y01": 0.82, "Y10": 0.81, "Y11": 0.83}
        c3_by_arm = {"Y00": 0.70, "Y01": 0.71, "Y10": 0.70, "Y11": 0.72}
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
                "val": {"f1_macro": macro_by_arm[arm]},
            }
            for epoch in range(1, protocol.EPOCHS + 1)
        ]
        screen_metrics = screen_health.build_screen_metrics(
            history,
            best_source_val_sit_stand_f1=c3_by_arm[arm],
            positive_anchor_fraction=1.0,
        )
        _write_json(
            output_dir / "run_invocation.json",
            {
                "config_path": str(config_path.resolve()),
                "raw_config_sha256": protocol.canonical_sha256(config),
            },
        )
        _write_json(
            output_dir / "metrics.json",
            {
                "target_env": target_env,
                "arm": arm,
                "seed": seed,
                "evaluate_target": False,
                "resources": {
                    "epochs": protocol.EPOCHS,
                    "optimizer_steps": protocol.TOTAL_OPTIMIZER_STEPS,
                    "history": history,
                },
                "calibration_artifact": str(calibration_path.resolve()),
                "calibration_sha256": calibration_hash,
                "screen_metrics": screen_metrics,
            },
        )
        return SimpleNamespace(returncode=0)


def _csv_rows(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_plan_is_fold_ordered_and_hash_bound(screen_bundle: Path) -> None:
    registry_path = screen_bundle / "screen_job_registry.json"
    plan = run_screen.load_screen_plan(registry_path)
    assert len(plan.jobs) == 18
    assert [job.target_env for job in plan.jobs] == ["E1"] * 6 + ["E2"] * 6 + ["E3"] * 6
    assert len({job.calibration_artifact_id for job in plan.jobs}) == 3
    assert plan.registry_file_sha256 == protocol.file_sha256(registry_path)
    assert all(job.config["training"]["epochs"] == 200 for job in plan.jobs)

    first_config = plan.jobs[0].config_path
    first_config.write_text(first_config.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(run_screen.ScreenExecutionError, match="Config file hash mismatch"):
        run_screen.load_screen_plan(registry_path)


def test_cli_requires_explicit_mode_and_command_has_no_epoch_override(screen_bundle: Path) -> None:
    with pytest.raises(SystemExit):
        run_screen.parse_args([])
    args = run_screen.parse_args(["--dry-run"])
    assert args.dry_run is True

    job = run_screen.load_screen_plan(screen_bundle / "screen_job_registry.json").jobs[0]
    command = run_screen.build_train_command(job, Path("run"), device="cuda:0")
    assert "--epochs" not in command
    assert _flag(command, "--evaluate-target") == "false"
    assert _flag(command, "--seed") == str(job.screen_seed)
    assert job.config["training"]["epochs"] == protocol.EPOCHS


def test_dry_run_execute_collect_select_and_resume(screen_bundle: Path, tmp_path: Path) -> None:
    registry_path = screen_bundle / "screen_job_registry.json"
    output_dir = tmp_path / "stage1"
    dry_summary = run_screen.execute_screen(
        registry_path,
        output_dir,
        device="cpu",
        dry_run=True,
        command_runner=lambda *args, **kwargs: pytest.fail("dry-run launched a process"),
    )
    assert dry_summary["mode"] == "dry_run"
    assert dry_summary["completed_job_count"] == 0
    dry_rows = _csv_rows(output_dir / "run_registry.csv")
    assert len(dry_rows) == 18
    assert {row["status"] for row in dry_rows} == {"planned"}
    assert {row["attempt"] for row in dry_rows} == {"1"}

    trainer = FakeTrainer(tmp_path / "artifacts")
    selection_dir = tmp_path / "selections"
    summary = run_screen.execute_screen(
        registry_path,
        output_dir,
        device="cpu",
        dry_run=False,
        select_lambdas=True,
        selection_dir=selection_dir,
        command_runner=trainer,
    )
    assert summary["completed_job_count"] == 18
    assert summary["selected_lambdas"] == {"E1": 0.01, "E2": 0.01, "E3": 0.01}
    assert len(trainer.commands) == 18
    command_targets = [
        json.loads(Path(_flag(command, "--config")).read_text(encoding="utf-8"))["target_env"]
        for command in trainer.commands
    ]
    assert command_targets == ["E1"] * 6 + ["E2"] * 6 + ["E3"] * 6
    assert all("--epochs" not in command for command in trainer.commands)

    rows = _csv_rows(output_dir / "run_registry.csv")
    assert {row["status"] for row in rows} == {"completed"}
    assert all(len(row["metrics_file_sha256"]) == 64 for row in rows)
    results_path = output_dir / "screen_results.json"
    results = json.loads(results_path.read_text(encoding="utf-8"))
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    select_lambda.validate_screen_results(
        results,
        registry,
        registry_file_sha256=protocol.file_sha256(registry_path),
    )
    assert set(results) == {
        "schema_version",
        "protocol_name",
        "protocol_version",
        "protocol_lock_sha256",
        "screen_registry_file_sha256",
        "results",
    }
    assert all(
        set(row)
        == {
            "job_id",
            "target_env",
            "arm",
            "screen_seed",
            "lambda_supcon",
            "config_content_sha256",
            "metrics",
        }
        for row in results["results"]
    )
    assert {path.name for path in selection_dir.iterdir()} == {
        "selection_E1.json",
        "selection_E2.json",
        "selection_E3.json",
        "selection_summary.json",
    }

    no_rerun = FakeTrainer(tmp_path / "unexpected")
    resumed = run_screen.execute_screen(
        registry_path,
        output_dir,
        device="cpu",
        dry_run=False,
        select_lambdas=True,
        selection_dir=selection_dir,
        command_runner=no_rerun,
    )
    assert resumed["completed_job_count"] == 18
    assert no_rerun.commands == []


def test_failure_stops_at_fold_barrier_and_resume_uses_new_attempt(
    screen_bundle: Path,
    tmp_path: Path,
) -> None:
    registry_path = screen_bundle / "screen_job_registry.json"
    output_dir = tmp_path / "stage1"
    failing = FakeTrainer(tmp_path / "artifacts", fail_call=3)
    with pytest.raises(run_screen.ScreenExecutionError, match="return code 7"):
        run_screen.execute_screen(
            registry_path,
            output_dir,
            device="cpu",
            dry_run=False,
            command_runner=failing,
        )
    assert len(failing.commands) == 3
    assert all(
        json.loads(Path(_flag(command, "--config")).read_text(encoding="utf-8"))["target_env"] == "E1"
        for command in failing.commands
    )
    failed_rows = _csv_rows(output_dir / "run_registry.csv")
    assert [row["status"] for row in failed_rows[:3]] == ["completed", "completed", "failed"]
    assert {row["status"] for row in failed_rows[6:]} == {"planned"}

    succeeding = FakeTrainer(tmp_path / "artifacts")
    summary = run_screen.execute_screen(
        registry_path,
        output_dir,
        device="cpu",
        dry_run=False,
        command_runner=succeeding,
    )
    assert summary["completed_job_count"] == 18
    assert len(succeeding.commands) == 16
    resumed_output = Path(_flag(succeeding.commands[0], "--output-dir"))
    assert resumed_output.name == "attempt_002"
    assert {row["status"] for row in _csv_rows(output_dir / "run_registry.csv")} == {"completed"}
