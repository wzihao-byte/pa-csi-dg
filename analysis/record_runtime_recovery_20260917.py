"""Seal the September 17 host-interruption recovery provenance."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = Path(r"D:\pa-csi-dg-revision-runtime")


def read(path: Path):
    with path.open(encoding="utf-8-sig") as stream:
        return json.load(stream)


def sha256(path: Path) -> str:
    import hashlib
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def identity(pid: int) -> dict:
    process = psutil.Process(pid)
    return {"pid": pid, "process_create_time": process.create_time(), "command": process.cmdline()}


def main() -> None:
    recovery_output = RUNTIME / "coordinators/controls_to_inputs_v4"
    validation_path = recovery_output / "recovery_validation.json"
    validation = read(validation_path)
    recovery_status = read(recovery_output / "status.json")
    events_path = RUNTIME / "studies/r_inputs_v1/runs/E3_I_TWIN_AP_602/events.jsonl"
    events = [json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    audit = next(row for row in events if row.get("event") == "resume_state_audit" and row.get("time", "") >= "2026-09-17T13:30")
    resumed = next(row for row in events if row.get("event") == "resumed" and row.get("time", "") >= "2026-09-17T13:30")
    after = next(row for row in events if row.get("event") == "epoch_committed" and row.get("epoch") == 128)
    pids = {"inputs_coordinator": 37004, "inputs_runner": 43156, "h_queue": 6480,
            "final_evidence": 5532, "paper_assets": 2808, "fragment": 37532,
            "manuscript": 22240, "release": 45348}
    owners = {name: identity(pid) for name, pid in pids.items()}
    h_owner = ROOT / "outputs_h_grouped_l1248/v2/queue/owner.json"
    record = {
        "schema": "csi-revision-runtime-recovery/v3",
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "interruption": {
            "observed_at": "2026-09-17T13:32:14.012Z",
            "condition": "all prior Python owners absent; no nonempty runner error log",
            "last_committed_input": validation["active_run"],
            "completed_input_jobs": validation["completed_count"],
            "prior_inputs_coordinator_status_sha256": validation["prior_status_sha256"],
            "prior_input_lock": validation["stale_lock"],
            "archived_input_lock": recovery_status["archived_lock"],
            "archived_input_lock_sha256": recovery_status["archived_lock_sha256"],
            "prior_h_owner": str((ROOT / "outputs_h_grouped_l1248/v2/queue/owner.interrupted_20260917_1332Z.json").resolve()),
            "prior_h_owner_sha256": sha256(ROOT / "outputs_h_grouped_l1248/v2/queue/owner.interrupted_20260917_1332Z.json"),
            "h_training_started_before_interruption": False,
        },
        "exact_input_resume": {
            "recovery_validation_sha256": sha256(validation_path),
            "recovery_code_sha256": sha256(ROOT / "analysis/recover_multienv_inputs_after_interruption.py"),
            "frozen_input_bundle_sha256": validation["input_bundle_sha256"],
            "resume_state_audit": audit,
            "resume_event": resumed,
            "first_post_resume_commit": after,
            "model_optimizer_rng_exact": all(audit.get(key) is True for key in ("model_exact", "optimizer_exact", "rng_exact")),
            "target_data_loaded_before_resume": False,
        },
        "completed_source_analysis_reuse": {
            "completion_path": str((RUNTIME / "coordinators/formal_source_analysis_v3/completion.json").resolve()),
            "completion_sha256": sha256(RUNTIME / "coordinators/formal_source_analysis_v3/completion.json"),
            "completed_count": 117,
            "target_data_loaded": False,
            "identity_holder_code_sha256": sha256(ROOT / "analysis/hold_completed_source_analysis_identity.py"),
        },
        "h_recovery": {
            "current_owner_sha256": sha256(h_owner),
            "frozen_bundle_manifest_sha256": sha256(ROOT / "outputs_h_grouped_l1248/v2/bundle/bundle_manifest.json"),
            "training_started": False,
            "widar_dependency": False,
        },
        "current_owners": owners,
        "coordinator_versions": {
            "inputs": "controls_to_inputs_v4", "source": "formal_source_analysis_v3 (completed)",
            "evidence": "multienv_final_evidence_v4", "assets": "multienv_paper_assets_v4",
            "fragment": "multienv_fragment_v4", "manuscript": "multienv_manuscript_v4",
            "release": "multienv_release_v4",
        },
        "retained_pre_mutation_setup_failures": [
            str((RUNTIME / "logs/inputs_recovery_v4.stderr.log").resolve()),
            str((RUNTIME / "logs/source_completion_sentinel_v4.stderr.log").resolve()),
            str((RUNTIME / "logs/final_evidence_v4.stderr.log").resolve()),
        ],
        "validation": {"active_owner_count": len(owners), "widar_training_process_count": 0,
                       "input_source_training_complete": False, "h_training_started": False},
    }
    destination = ROOT / "docs/revision/evidence/runtime_recovery_20260917.json"
    if destination.exists():
        raise FileExistsError(destination)
    destination.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"path": str(destination), "sha256": sha256(destination)}, sort_keys=True))


if __name__ == "__main__":
    main()
