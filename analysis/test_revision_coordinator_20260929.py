"""Exercise the revision coordinator with short, CPU-only synthetic child studies.

All fixtures and logs are retained under the dated QA directory on D:. The real
study queues and their live processes are never started, stopped, or modified.
"""
from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace

import psutil


ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "analysis/coordinate_revision_experiments_20260929.py"
QA_ROOT = Path("D:/pa-csi-dg-revision-runtime/qa/revision_coordinator_20260929")
CHILD = r'''
import argparse, hashlib, json, os, sys, time
from pathlib import Path
def read(p): return json.loads(p.read_text(encoding="utf-8"))
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def value_hash(v): return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(",",":"),ensure_ascii=False).encode("utf-8")).hexdigest()
def write(p,v):
    p.parent.mkdir(parents=True,exist_ok=True)
    q=p.with_name(p.name+".tmp")
    q.write_text(json.dumps(v),encoding="utf-8")
    q.replace(p)
p=argparse.ArgumentParser();p.add_argument("action");p.add_argument("--bundle",type=Path,required=True)
a=p.parse_args(); b=a.bundle; cfg=read(b/"protocol.json"); mode=cfg["fixture_mode"]
write(b/"invocation.json",{"pid":os.getpid(),"started_ns":time.monotonic_ns()})
write(b/"queue_status.json",{"status":"running"})
for j in cfg["jobs"]: write(b/"runs"/j["job_id"]/"status.json",{"status":"running","epoch":0,"epochs":1})
time.sleep(cfg.get("fixture_delay",.25))
if mode=="failure":
    write(b/"queue_failure.json",{"message":"intentional CPU-only fixture failure"});sys.exit(13)
if mode=="stopped": sys.exit(0)
for i,j in enumerate(cfg["jobs"]):
    run=b/"runs"/j["job_id"]
    write(run/"source_evidence.json",{"fixture":True})
    write(run/"status.json",{"status":"running" if mode=="incomplete_source" and i==0 else "completed","epoch":1,"epochs":1,
          "optimizer_updates":1,"config_value_sha256":value_hash(j),"artifacts_sha256":{"source_evidence.json":sha(run/"source_evidence.json")}})
    write(b/"evaluation"/j["job_id"]/"result.json",{"job_id":j["job_id"],"fixture":True})
    write(b/"evaluation"/j["job_id"]/"predictions.json",[{"sample_id":"fixture","true":0,"predicted":0}])
summary={"runs":[] if mode=="incomplete_evaluation" else [{"job_id":j["job_id"]} for j in cfg["jobs"]],
         "artifacts_sha256":{p.relative_to(b).as_posix():sha(p) for p in (b/"evaluation").glob("*/*.json")}}
write(b/"evaluation/summary.json",summary)
write(b/"target_gate.json",{"bundle_sha256":sha(b/"bundle_manifest.json"),
      "source_status_sha256":{j["job_id"]:sha(b/"runs"/j["job_id"]/"status.json") for j in cfg["jobs"]}})
write(b/"comparison/report.json",{"complete":True,"fixture":True})
state={"status":"completed","jobs":len(cfg["jobs"]),"bundle_sha256":sha(b/"bundle_manifest.json"),
       "evaluation_summary_sha256":sha(b/"evaluation/summary.json"),"comparison_report_sha256":sha(b/"comparison/report.json")}
write(b/"queue_status.json",state)
if mode=="missing_report": (b/"comparison/report.json").unlink()
if mode=="tampered_report": (b/"comparison/report.json").write_text('{"tampered":true}',encoding="utf-8")
if mode=="tampered_source_artifact": (b/"runs"/cfg["jobs"][0]["job_id"]/"source_evidence.json").write_text('{}',encoding="utf-8")
if mode=="tampered_prediction": (b/"evaluation"/cfg["jobs"][0]["job_id"]/"predictions.json").write_text('[]',encoding="utf-8")
write(b/"finished.json",{"finished_ns":time.monotonic_ns()})
'''


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def module():
    spec = importlib.util.spec_from_file_location("coordinator_under_test", CODE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    # Patch only this module's binding, not the real time module or child runtime.
    mod.time = SimpleNamespace(sleep=lambda seconds: time.sleep(min(seconds, .02)), time_ns=time.time_ns)
    return mod


def fixture(directory, modes=("success", "success"), delays=(.25, .25)):
    directory.mkdir(parents=True)
    bundles, entries = [], []
    for index, (mode, delay) in enumerate(zip(modes, delays)):
        bundle = directory / f"study_{index}"
        (bundle / "code").mkdir(parents=True)
        (bundle / "code/fixture_study.py").write_text(CHILD, encoding="utf-8")
        protocol = {"entry_module": "fixture_study", "fixture_mode": mode, "fixture_delay": delay,
                    "jobs": [{"job_id": f"fixture_{index}_{i}", "epochs": 1, "updates_per_epoch": 1} for i in range(2)]}
        write(bundle / "protocol.json", protocol)
        write(bundle / "bundle_manifest.json", {"files_sha256": {
            "protocol.json": sha(bundle / "protocol.json"),
            "code/fixture_study.py": sha(bundle / "code/fixture_study.py")}})
        entries.append({"bundle": str(bundle), "bundle_sha256": sha(bundle / "bundle_manifest.json"), "job_count": 2})
        bundles.append(bundle)
    coordinator = directory / "coordinator"
    coordinator.mkdir()
    plan = {"python": sys.executable, "studies": entries, "new_training_runs": 4,
            "coordinator_code_sha256": sha(CODE)}
    write(coordinator / "plan.json", plan)
    return coordinator / "plan.json", bundles


def invoke(mod, plan_path, expect_failure=False):
    capture = io.StringIO()
    error = None
    try:
        with contextlib.redirect_stdout(capture):
            mod.run(plan_path)
    except BaseException as exc:
        error = exc
    (plan_path.parent / "qa_captured_stdout.log").write_text(capture.getvalue(), encoding="utf-8")
    assert not (plan_path.parent / "coordinator.lock.json").exists(), "Coordinator lock leaked"
    if expect_failure:
        assert error is not None, "Coordinator accepted an incomplete or corrupted study"
        status_path = plan_path.parent / "status.json"
        assert not status_path.exists() or read(status_path)["status"] != "completed", "Coordinator incorrectly marked all studies completed"
        assert (plan_path.parent / "failure.json").exists(), "Failure was not recorded"
    elif error is not None:
        raise error
    return error


def completed_fixture(bundle):
    protocol = read(bundle / "protocol.json")
    for job in protocol["jobs"]:
        run = bundle / "runs" / job["job_id"]
        write(run / "source_evidence.json", {"fixture": True})
        value_hash = hashlib.sha256(json.dumps(job, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()
        write(run / "status.json", {"status": "completed", "epochs": 1, "epoch": 1, "optimizer_updates": 1,
              "config_value_sha256": value_hash, "artifacts_sha256": {"source_evidence.json": sha(run / "source_evidence.json")}})
        write(bundle / "evaluation" / job["job_id"] / "result.json", {"job_id": job["job_id"], "fixture": True})
        write(bundle / "evaluation" / job["job_id"] / "predictions.json", [{"sample_id": "fixture", "true": 0, "predicted": 0}])
    write(bundle / "evaluation/summary.json", {"runs": [{"job_id": job["job_id"]} for job in protocol["jobs"]],
          "artifacts_sha256": {path.relative_to(bundle).as_posix(): sha(path) for path in (bundle / "evaluation").glob("*/*.json")}})
    write(bundle / "target_gate.json", {"bundle_sha256": sha(bundle / "bundle_manifest.json"),
          "source_status_sha256": {job["job_id"]: sha(bundle / "runs" / job["job_id"] / "status.json") for job in protocol["jobs"]}})
    write(bundle / "comparison/report.json", {"complete": True, "fixture": True})
    write(bundle / "queue_status.json", {"status": "completed", "jobs": len(protocol["jobs"]),
          "bundle_sha256": sha(bundle / "bundle_manifest.json"),
          "evaluation_summary_sha256": sha(bundle / "evaluation/summary.json"),
          "comparison_report_sha256": sha(bundle / "comparison/report.json")})


def run_suite():
    directory = QA_ROOT / f"run_{time.time_ns()}"
    directory.mkdir(parents=True)
    mod = module()
    checks = []

    def case(name, function):
        try:
            function(directory / name)
            checks.append({"name": name, "passed": True})
        except BaseException:
            checks.append({"name": name, "passed": False, "error": traceback.format_exc()})

    def serial(path):
        plan, bundles = fixture(path)
        invoke(mod, plan)
        state = read(plan.parent / "status.json")
        assert state["status"] == "completed"
        assert all(row["completed"] == row["total"] == 2 for row in state["studies"])
        assert read(bundles[0] / "finished.json")["finished_ns"] < read(bundles[1] / "invocation.json")["started_ns"]

    def failure_blocks(path):
        plan, bundles = fixture(path, ("failure", "success"))
        invoke(mod, plan, expect_failure=True)
        assert not (bundles[1] / "invocation.json").exists()

    def zero_exit_incomplete(path):
        plan, bundles = fixture(path, ("stopped", "success"))
        invoke(mod, plan, expect_failure=True)
        assert not (bundles[1] / "invocation.json").exists()

    def skip_completed(path):
        plan, bundles = fixture(path)
        completed_fixture(bundles[0])
        before = sha(bundles[0] / "queue_status.json")
        invoke(mod, plan)
        assert not (bundles[0] / "invocation.json").exists()
        assert (bundles[1] / "invocation.json").exists()
        assert sha(bundles[0] / "queue_status.json") == before

    def reject_manifest(path):
        plan, bundles = fixture(path)
        original = read(bundles[0] / "bundle_manifest.json")
        write(bundles[0] / "bundle_manifest.json", {**original, "tampered": True})
        invoke(mod, plan, expect_failure=True)
        assert not any((b / "invocation.json").exists() for b in bundles)

    def reject_code(path):
        plan, bundles = fixture(path)
        (bundles[0] / "code/fixture_study.py").write_text(CHILD + "\n# tampered\n", encoding="utf-8")
        invoke(mod, plan, expect_failure=True)
        assert not any((b / "invocation.json").exists() for b in bundles)

    def live_wait(path):
        plan, bundles = fixture(path, delays=(.65, .25))
        command = [sys.executable, "-u", "-m", "fixture_study", "run", "--bundle", str(bundles[0])]
        with (path / "existing.stdout.log").open("wb") as out, (path / "existing.stderr.log").open("wb") as err:
            process = subprocess.Popen(command, cwd=bundles[0] / "code", stdin=subprocess.DEVNULL,
                                       stdout=out, stderr=err,
                                       creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        write(bundles[0] / "runner.lock.json", {"pid": process.pid, "create_time": psutil.Process(process.pid).create_time()})
        try:
            invoke(mod, plan)
            assert process.wait(timeout=5) == 0
            assert not (plan.parent / "0_child.json").exists(), "Already-running study was launched again"
            assert "Waiting for the already-running study" in (plan.parent / "qa_captured_stdout.log").read_text(encoding="utf-8")
            assert read(bundles[0] / "invocation.json")["pid"] == process.pid
            assert read(bundles[0] / "finished.json")["finished_ns"] < read(bundles[1] / "invocation.json")["started_ns"]
        finally:
            # This is only the exact fake child created immediately above.
            if process.poll() is None:
                process.wait(timeout=5)

    def invalid_completion(path, mode):
        plan, bundles = fixture(path, (mode, "success"))
        invoke(mod, plan, expect_failure=True)
        assert not (bundles[1] / "invocation.json").exists()

    for name, fn in (("serial_two_studies", serial), ("failure_blocks_next", failure_blocks),
                     ("zero_exit_without_completion_rejected", zero_exit_incomplete), ("skip_completed", skip_completed),
                     ("manifest_tamper_rejected", reject_manifest), ("code_tamper_rejected", reject_code),
                     ("wait_for_existing_live_study", live_wait)):
        case(name, fn)
    for mode in ("incomplete_source", "incomplete_evaluation", "missing_report", "tampered_report", "tampered_source_artifact", "tampered_prediction"):
        case(mode + "_rejected", lambda path, mode=mode: invalid_completion(path, mode))
    result = {"passed": all(check["passed"] for check in checks), "checks": checks,
              "coordinator_code_sha256": sha(CODE), "test_script_sha256": sha(__file__),
              "fixture_directory": str(directory), "actual_studies_touched": False,
              "gpu_used": False, "training_performed": False}
    write(directory / "result.json", result)
    write(QA_ROOT / "latest.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    raise SystemExit(0 if run_suite()["passed"] else 1)
