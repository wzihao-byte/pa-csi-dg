"""Windows CPU-only integration tests for scoped atomic-replace retries."""
from __future__ import annotations

import ctypes
from ctypes import wintypes
import hashlib
import json
import os
from pathlib import Path
import pickle
import random
import threading
import time
import traceback

from analysis.windows_atomic_retry_20260930 import install_replace_retry


ROOT = Path(__file__).resolve().parents[1]
QA = Path("D:/pa-csi-dg-revision-runtime/qa")
REAL_REPLACE = os.replace
PAYLOAD = bytes(range(256)) * 37 + b"\x00exact-checkpoint-tail\xff"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def failure(code, text="synthetic Windows file sharing failure"):
    error = PermissionError(13, text)
    error.winerror = code
    return error


def fixture(root, name):
    directory = root / name
    allowed, outside = directory / "allowed_bundle", directory / "outside"
    allowed.mkdir(parents=True)
    outside.mkdir()
    source, destination = allowed / "resume.pt.tmp", allowed / "resume.pt"
    source.write_bytes(PAYLOAD)
    destination.write_bytes(b"previous-complete-checkpoint")
    return allowed, outside, source, destination, directory / "events.jsonl"


def logs(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def acquire_blocking_handle(path):
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    create = kernel.CreateFileW
    create.argtypes = (wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, wintypes.LPVOID,
                       wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE)
    create.restype = wintypes.HANDLE
    # READ and WRITE sharing are allowed; deletion/rename sharing is deliberately
    # withheld to reproduce an actual Windows destination replacement denial.
    handle = create(str(path.resolve()), 0x80000000, 0x1 | 0x2, None, 3, 0x80, None)
    if handle == ctypes.c_void_p(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())
    kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
    kernel.CloseHandle.restype = wintypes.BOOL

    def close():
        if not kernel.CloseHandle(handle):
            raise ctypes.WinError(ctypes.get_last_error())

    return close


def run_suite():
    if os.name != "nt":
        raise RuntimeError("The required real sharing-denial integration check needs Windows")
    root = QA / "windows_atomic_retry_20260930_fixtures" / f"run_{time.time_ns()}"
    root.mkdir(parents=True)
    checks = []

    def check(name, fn):
        try:
            detail = fn(name)
            assert os.replace is REAL_REPLACE, "os.replace wrapper leaked from the test"
            checks.append({"name": name, "passed": True, "detail": detail})
        except BaseException:
            os.replace = REAL_REPLACE
            checks.append({"name": name, "passed": False, "error": traceback.format_exc()})

    def real_released(name):
        allowed, _, source, destination, event_log = fixture(root, name)
        close = acquire_blocking_handle(destination)
        try:
            try:
                REAL_REPLACE(source, destination)
            except OSError as error:
                denied = error.winerror
                assert denied in (5, 32, 33)
            else:
                raise AssertionError("The held Windows handle did not deny replacement")
            old_payload = destination.read_bytes()
            assert old_payload == b"previous-complete-checkpoint" and source.read_bytes() == PAYLOAD
            released = threading.Event()
            thread_errors = []

            def release(closer=close):
                try:
                    time.sleep(.22)
                    closer()
                    released.set()
                except BaseException as exc:
                    thread_errors.append(repr(exc))

            thread = threading.Thread(target=release, name="atomic-retry-fixture-release")
            thread.start()
            close = None
            with install_replace_retry([allowed], event_log, timeout_seconds=2) as handle:
                os.replace(source, destination)
                assert not handle.logging_errors
            thread.join(timeout=3)
            assert not thread.is_alive() and released.is_set() and not thread_errors
            assert destination.read_bytes() == PAYLOAD and not source.exists()
            records = logs(event_log)
            retries = [r for r in records if r["event"] == "atomic_replace_retry"]
            assert retries and any(r["event"] == "atomic_replace_succeeded" for r in records)
            assert all(.0 < r["sleep_seconds"] <= .5 for r in retries)
            return {"actual_windows_denial": denied, "retry_count": len(retries),
                    "payload_sha256": sha(destination), "destination_unchanged_until_atomic_replace": True}
        finally:
            if close is not None:
                close()

    def persistent_actual(name):
        allowed, _, source, destination, event_log = fixture(root, name)
        close = acquire_blocking_handle(destination)
        errors = []

        def captured(*args, **kwargs):
            try:
                return REAL_REPLACE(*args, **kwargs)
            except OSError as exc:
                errors.append(exc)
                raise

        os.replace = captured
        try:
            started = time.monotonic()
            with install_replace_retry([allowed], event_log, timeout_seconds=.18):
                try:
                    os.replace(source, destination)
                except OSError as exc:
                    assert exc is errors[0], "Persistent retry did not return the first OS exception"
                else:
                    raise AssertionError("Held destination incorrectly succeeded")
            elapsed = time.monotonic() - started
            assert .16 <= elapsed < 2 and len(errors) >= 2
            assert source.read_bytes() == PAYLOAD and destination.read_bytes() == b"previous-complete-checkpoint"
            assert logs(event_log)[-2]["event"] == "atomic_replace_failed"
            return {"elapsed_seconds": elapsed, "attempts": len(errors), "source_and_destination_preserved": True}
        finally:
            os.replace = REAL_REPLACE
            close()

    def winerror_retries(name, code):
        allowed, _, source, destination, event_log = fixture(root, name)
        calls = []
        first = failure(code)

        def original(*args, **kwargs):
            calls.append(1)
            if len(calls) == 1:
                raise first
            return REAL_REPLACE(*args, **kwargs)

        os.replace = original
        try:
            with install_replace_retry([allowed], event_log, timeout_seconds=.5):
                os.replace(source, destination)
            assert len(calls) == 2 and destination.read_bytes() == PAYLOAD
            return {"winerror": code, "attempts": len(calls)}
        finally:
            os.replace = REAL_REPLACE

    def not_retried(name, kind):
        allowed, outside, source, destination, event_log = fixture(root, name)
        original_error = failure(123 if kind == "unrelated_winerror" else 5)
        if kind == "outside_scope":
            source, destination = outside / "status.json.tmp", outside / "status.json"
            source.write_bytes(PAYLOAD)
            destination.write_bytes(b"outside-original")
        elif kind == "non_tmp_source":
            source = allowed / "incoming.bin"
            source.write_bytes(PAYLOAD)
        elif kind == "source_absent":
            source.unlink()
        calls = []

        def original(*args, **kwargs):
            calls.append(1)
            raise original_error

        os.replace = original
        try:
            with install_replace_retry([allowed], event_log, timeout_seconds=.15):
                try:
                    os.replace(source, destination)
                except OSError as exc:
                    assert exc is original_error
                else:
                    raise AssertionError("Original error was swallowed")
            assert len(calls) == 1
            assert not any(r["event"] == "atomic_replace_retry" for r in logs(event_log))
            return {"attempts": 1, "exception_identity_preserved": True}
        finally:
            os.replace = REAL_REPLACE

    def non_retryable_after_retry(name):
        allowed, _, source, destination, event_log = fixture(root, name)
        first, second = failure(5), FileNotFoundError(2, "new unrelated error")
        calls = []

        def original(*args, **kwargs):
            calls.append(1)
            raise first if len(calls) == 1 else second

        os.replace = original
        try:
            with install_replace_retry([allowed], event_log, timeout_seconds=.5):
                try:
                    os.replace(source, destination)
                except OSError as exc:
                    assert exc is second
                else:
                    raise AssertionError("Unexpected success")
            assert len(calls) == 2
            assert logs(event_log)[-2]["event"] == "atomic_replace_aborted"
            return {"attempts": 2, "new_non_retryable_exception_unchanged": True}
        finally:
            os.replace = REAL_REPLACE

    def rng_unchanged(name):
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        import numpy as np
        import torch
        allowed, _, source, destination, event_log = fixture(root, name)
        state_before = pickle.dumps((random.getstate(), np.random.get_state(), torch.random.get_rng_state().numpy().tobytes()))
        count = []

        def original(*args, **kwargs):
            count.append(1)
            if len(count) < 3:
                raise failure(32)
            return REAL_REPLACE(*args, **kwargs)

        os.replace = original
        try:
            with install_replace_retry([allowed], event_log, timeout_seconds=1):
                os.replace(source, destination)
        finally:
            os.replace = REAL_REPLACE
        state_after = pickle.dumps((random.getstate(), np.random.get_state(), torch.random.get_rng_state().numpy().tobytes()))
        assert state_before == state_after and len(count) == 3
        assert destination.read_bytes() == PAYLOAD
        return {"python_numpy_torch_cpu_rng_exactly_unchanged": True,
                "rng_state_sha256": hashlib.sha256(state_before).hexdigest(), "attempts": len(count)}

    def append_only_and_uninstall(name):
        allowed, _, source, destination, event_log = fixture(root, name)
        prefix = '{"existing_log_entry":true}\n'
        event_log.write_text(prefix, encoding="utf-8")
        handle = install_replace_retry([allowed], event_log)
        assert os.replace is handle.replacement
        os.replace(source, destination)
        handle.uninstall()
        handle.uninstall()
        assert event_log.read_text(encoding="utf-8").startswith(prefix)
        assert os.replace is REAL_REPLACE and destination.read_bytes() == PAYLOAD
        return {"existing_log_prefix_preserved": True, "manual_uninstall_idempotent": True}

    def invalid_settings(name):
        allowed, _, _, _, event_log = fixture(root, name)
        for timeout in (0, -1, float("nan"), float("inf")):
            try:
                install_replace_retry([allowed], event_log, timeout_seconds=timeout)
            except ValueError:
                pass
            else:
                raise AssertionError("Unbounded/invalid timeout accepted")
        for roots in ([], [Path(allowed.anchor)]):
            try:
                install_replace_retry(roots, event_log)
            except ValueError:
                pass
            else:
                raise AssertionError("An empty or drive-wide scope was accepted")
        return {"invalid_deadlines_and_unbounded_roots_rejected": True}

    check("actual_windows_locked_destination_released_then_exact_atomic_success", real_released)
    check("actual_windows_persistent_denial_is_bounded_and_preserves_original_exception", persistent_actual)
    for code in (5, 32, 33):
        check(f"retry_winerror_{code}_only_then_success", lambda name, code=code: winerror_retries(name, code))
    for kind in ("unrelated_winerror", "outside_scope", "non_tmp_source", "source_absent"):
        check(kind + "_not_retried", lambda name, kind=kind: not_retried(name, kind))
    check("non_retryable_error_after_retry_propagates_unchanged", non_retryable_after_retry)
    check("python_numpy_torch_cpu_rng_unchanged", rng_unchanged)
    check("append_only_log_and_idempotent_uninstall", append_only_and_uninstall)
    check("bounded_configuration_validation", invalid_settings)
    report = {"passed": all(row["passed"] for row in checks), "checks": checks,
              "host_os": os.name, "fixture_directory": str(root),
              "wrapper_sha256": sha(ROOT / "analysis/windows_atomic_retry_20260930.py"),
              "test_script_sha256": sha(__file__), "frozen_bundles_modified": False,
              "training_launched": False, "gpu_used": False,
              "actual_destination_lock_test": True, "replace_semantics": "original os.replace; no delete/write fallback"}
    text = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    (root / "result.json").write_text(text, encoding="utf-8")
    (QA / "windows_atomic_retry_20260930.json").write_text(text, encoding="utf-8")
    print(text)
    return report


if __name__ == "__main__":
    raise SystemExit(0 if run_suite()["passed"] else 1)
