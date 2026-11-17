from __future__ import annotations

import hashlib
import json
import multiprocessing
import os
from pathlib import Path

import pytest

from experiments.supcon_tpp_joint.joint_models import CalibrationProvenance
from experiments.supcon_tpp_joint.train_joint import (
    CALIBRATION_BINDING_SCHEMA_VERSION,
    calibration_code_sha256,
    commit_calibration_pair,
    exclusive_calibration_lock,
    expected_calibration_binding,
    load_validated_calibration_pair,
)


def _hold_calibration_lock(lock_path: str, ready: object, release: object) -> None:
    with exclusive_calibration_lock(
        Path(lock_path),
        timeout_seconds=20.0,
        poll_seconds=0.01,
    ):
        ready.set()  # type: ignore[attr-defined]
        release.wait(20.0)  # type: ignore[attr-defined]


def _valid_provenance() -> CalibrationProvenance:
    sources = ("E2", "E3")
    return CalibrationProvenance(
        outer_target="E1",
        source_environments=sources,
        calibration_sample_ids=tuple(
            f"{source}:{index:04d}" for source in sources for index in range(256)
        ),
        smoke_sample_ids=tuple(
            f"{source}:smoke-{index:02d}" for source in sources for index in range(8)
        ),
        source_file_sha256={
            f"{source}.{kind}": hashlib.sha256(f"{source}.{kind}".encode()).hexdigest()
            for source in sources
            for kind in ("amp", "phase", "label")
        },
        source_index_sha256="d" * 64,
    )


def test_cross_process_lock_times_out_closed_without_deleting_live_lock(tmp_path: Path) -> None:
    context = multiprocessing.get_context("spawn")
    ready = context.Event()
    release = context.Event()
    lock_path = tmp_path / "calibration.json.lock"
    process = context.Process(
        target=_hold_calibration_lock,
        args=(str(lock_path), ready, release),
    )
    process.start()
    try:
        assert ready.wait(30.0), "child did not acquire the calibration lock"
        with pytest.raises(TimeoutError, match="calibration artifact lock"):
            with exclusive_calibration_lock(
                lock_path,
                timeout_seconds=0.15,
                poll_seconds=0.01,
            ):
                pass
        assert lock_path.is_file()
    finally:
        release.set()
        process.join(30.0)
        if process.is_alive():
            process.terminate()
            process.join(10.0)
    assert process.exitcode == 0
    with exclusive_calibration_lock(lock_path, timeout_seconds=1.0, poll_seconds=0.01):
        pass


def test_abnormal_owner_exit_releases_os_lock_without_lockfile_deletion(tmp_path: Path) -> None:
    context = multiprocessing.get_context("spawn")
    ready = context.Event()
    release = context.Event()
    lock_path = tmp_path / "calibration.json.lock"
    process = context.Process(
        target=_hold_calibration_lock,
        args=(str(lock_path), ready, release),
    )
    process.start()
    assert ready.wait(30.0), "child did not acquire the calibration lock"
    process.terminate()
    process.join(30.0)
    assert not process.is_alive()
    assert lock_path.is_file()
    with exclusive_calibration_lock(lock_path, timeout_seconds=2.0, poll_seconds=0.01):
        pass


def test_binding_captures_initialization_and_all_calibration_code_hashes() -> None:
    provenance = _valid_provenance()
    binding = expected_calibration_binding(
        {"joint_temporal": {"factor": "global_basis"}},
        provenance,
        initialization_sha256="a" * 64,
    )
    assert binding["binding_schema_version"] == CALIBRATION_BINDING_SCHEMA_VERSION
    assert binding["calibration_model_initialization_sha256"] == "a" * 64
    assert binding["calibration_initialization_seed"] == provenance.initialization_seed
    assert binding["calibration_code_sha256"] == calibration_code_sha256()
    assert set(binding["calibration_code_sha256"]) == {
        "dg_dataset.py",
        "dg_models.py",
        "experiments/supcon_tpp_joint/joint_models.py",
        "experiments/supcon_tpp_joint/protocol.py",
        "experiments/supcon_tpp_joint/runner_utils.py",
        "experiments/supcon_tpp_joint/train_joint.py",
        "experiments/tpp_isolated/tpp_models.py",
    }
    assert all(len(value) == 64 for value in binding["calibration_code_sha256"].values())


def test_pair_commit_uses_hash_bound_sidecar_and_cleans_unique_temporaries(tmp_path: Path) -> None:
    artifact_path = tmp_path / "calibration.json"
    binding_path = tmp_path / "calibration.json.binding.json"
    binding = commit_calibration_pair(
        artifact_path,
        binding_path,
        {"z": [1, 2], "a": "payload"},
        {"binding_schema_version": CALIBRATION_BINDING_SCHEMA_VERSION},
    )
    stored_binding = json.loads(binding_path.read_text(encoding="utf-8"))
    assert stored_binding == binding
    assert stored_binding["artifact_sha256"] == hashlib.sha256(artifact_path.read_bytes()).hexdigest()
    assert len(stored_binding["pair_commit_id"]) == 32
    assert not list(tmp_path.glob("*.tmp"))


def test_pair_commit_rolls_back_artifact_when_binding_publish_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact_path = tmp_path / "calibration.json"
    binding_path = tmp_path / "calibration.json.binding.json"
    original_replace = os.replace

    def fail_binding_replace(source: str | bytes | Path, destination: str | bytes | Path) -> None:
        if Path(destination) == binding_path:
            raise OSError("injected binding publish failure")
        original_replace(source, destination)

    monkeypatch.setattr(os, "replace", fail_binding_replace)
    with pytest.raises(OSError, match="injected"):
        commit_calibration_pair(
            artifact_path,
            binding_path,
            {"artifact": True},
            {"binding_schema_version": CALIBRATION_BINDING_SCHEMA_VERSION},
        )
    assert not artifact_path.exists()
    assert not binding_path.exists()
    assert not list(tmp_path.glob("*.tmp"))


def test_incomplete_pair_fails_closed_and_is_not_removed(tmp_path: Path) -> None:
    artifact_path = tmp_path / "calibration.json"
    binding_path = tmp_path / "calibration.json.binding.json"
    artifact_path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Incomplete calibration artifact pair"):
        load_validated_calibration_pair(
            artifact_path,
            binding_path,
            _valid_provenance(),
            {},
        )
    assert artifact_path.is_file()
    assert not binding_path.exists()
