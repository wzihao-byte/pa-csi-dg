"""Generate the formal LaTeX result fragment after verified paper assets exist."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.revision_validation.study_io import now, read, sha256, write


def normalized(value: object) -> str:
    return str(value).replace("\\", "/").lower()


def process_identity(pid: int, required: tuple[str, ...]):
    process = psutil.Process(pid)
    command = process.cmdline()
    joined = normalized(" ".join(command))
    if not all(normalized(item) in joined for item in required):
        raise RuntimeError(f"PID {pid} is not the declared paper-assets coordinator")
    return process.create_time(), command


def live(pid: int, birth: float) -> bool:
    try:
        return psutil.Process(pid).create_time() == birth
    except psutil.NoSuchProcess:
        return False


def append_event(path: Path, event: dict):
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps({"at": now(), **event}, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets-pid", type=int, required=True)
    parser.add_argument("--assets-coordinator-output", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    coordinator_output = args.assets_coordinator_output.resolve()
    evidence = args.evidence.resolve()
    assets = args.assets.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Preserve existing formal-fragment coordinator attempt: {output}")
    birth, command = process_identity(
        args.assets_pid,
        ("analysis/continue_paper_assets_after_evidence.py", str(coordinator_output)),
    )
    output.mkdir(parents=True)
    frozen_builder = output / "code/paper/sensors_manuscript/build_formal_revision_fragment.py"
    frozen_builder.parent.mkdir(parents=True)
    shutil.copy2(ROOT / "paper/sensors_manuscript/build_formal_revision_fragment.py", frozen_builder)
    identity = {
        "pid": os.getpid(),
        "started_at": now(),
        "assets_pid": args.assets_pid,
        "assets_process_create_time": birth,
        "assets_command": command,
        "assets_coordinator_output": str(coordinator_output),
        "evidence": str(evidence),
        "assets": str(assets),
        "frozen_builder_sha256": sha256(frozen_builder),
        "mutable_upstream_json_observed_while_writer_live": False,
    }
    events = output / "events.jsonl"
    write(output / "config.json", identity)
    write(output / "status.json", {"status": "waiting_for_formal_paper_assets", **identity})
    append_event(events, {"event": "waiting_for_formal_paper_assets"})
    try:
        while live(args.assets_pid, birth):
            time.sleep(30)
        asset_status = read(coordinator_output / "status.json")
        if asset_status.get("status") != "completed":
            raise RuntimeError("Formal paper-assets coordinator did not complete")
        if asset_status.get("asset_manifest_sha256") != sha256(assets / "manifest.json"):
            raise RuntimeError("Paper-assets coordinator and asset manifest differ")
        builder_command = [
            sys.executable, "-X", "utf8", str(frozen_builder),
            "--evidence", str(evidence), "--assets", str(assets),
        ]
        write(output / "status.json", {"status": "building_formal_fragment", "command": builder_command, **identity})
        append_event(events, {"event": "formal_fragment_started", "command": builder_command})
        with (output / "builder.stdout.log").open("x", encoding="utf-8") as stdout, \
             (output / "builder.stderr.log").open("x", encoding="utf-8") as stderr:
            result = subprocess.run(
                builder_command,
                cwd=frozen_builder.parents[2],
                stdout=stdout,
                stderr=stderr,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
        if result.returncode:
            raise RuntimeError(f"Frozen formal fragment builder exited {result.returncode}")
        fragment_manifest = read(assets / "formal_fragment_manifest.json")
        for name, expected in fragment_manifest["files_sha256"].items():
            if sha256(assets / name) != expected:
                raise RuntimeError(f"Formal fragment artifact differs: {name}")
        completion = {
            "status": "completed",
            "completed_at": now(),
            "formal_fragment_manifest_sha256": sha256(assets / "formal_fragment_manifest.json"),
            "formal_results_tex_sha256": sha256(assets / "formal_results.tex"),
            "formal_narrative_values_sha256": sha256(assets / "formal_narrative_values.json"),
        }
        write(output / "status.json", completion)
        append_event(events, {"event": "completed", **completion})
    except BaseException:
        failure = {"status": "failed", "failed_at": now(), "error": traceback.format_exc(), **identity}
        write(output / "status.json", failure)
        append_event(events, {"event": "failed", "error": failure["error"]})
        raise


if __name__ == "__main__":
    main()
