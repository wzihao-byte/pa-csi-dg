"""Bounded Windows sharing-error retries for authorized atomic output writes.

This infrastructure shim never changes serialization, scientific code, RNG, or
the replace operation. It retries only a same-directory ``destination.tmp`` to
``destination`` replacement inside explicitly allowed roots, after Windows
reports access/sharing/lock errors and while the temporary file still exists.
All other calls and exceptions retain the original ``os.replace`` behavior.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import sys
import threading
import time


RETRY_WINERRORS = frozenset((5, 32, 33))


class ReplaceRetryHandle:
    """Own one installed wrapper; uninstall never removes somebody else's hook."""

    def __init__(self, roots, event_log, timeout_seconds):
        self.allowed_roots = tuple(roots)
        self.event_log = event_log
        self.timeout_seconds = timeout_seconds
        self.original_replace = os.replace
        self.logging_errors = []
        self._log_lock = threading.Lock()
        self._installed = False

        # Keep a stable function identity, rather than repeatedly constructing a
        # bound-method object when checking whether this wrapper is installed.
        def replacement(src, dst, *, src_dir_fd=None, dst_dir_fd=None):
            return self._replace(src, dst, src_dir_fd=src_dir_fd, dst_dir_fd=dst_dir_fd)

        self.replacement = replacement

    def _log(self, kind, *, required=False, **fields):
        record = {"time_utc": datetime.now(timezone.utc).isoformat(), "event": kind,
                  "pid": os.getpid(), "thread_id": threading.get_ident(), **fields}
        try:
            with self._log_lock:
                with self.event_log.open("a", encoding="utf-8", newline="\n") as stream:
                    stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                    stream.flush()
        except OSError as exc:
            # Audit failure must not turn a successful atomic replace into a
            # reported data-write failure, nor conceal the original OS error.
            self.logging_errors.append({"event": kind, "error": repr(exc)})
            if required:
                raise
            try:
                print(f"Atomic-replace retry audit log unavailable: {exc}", file=sys.stderr, flush=True)
            except (OSError, ValueError):
                pass

    def _scope(self, src, dst, src_dir_fd, dst_dir_fd):
        if os.name != "nt" or src_dir_fd is not None or dst_dir_fd is not None:
            return None
        try:
            source = Path(os.fsdecode(os.fspath(src))).resolve()
            destination = Path(os.fsdecode(os.fspath(dst))).resolve()
        except (OSError, TypeError, ValueError, RuntimeError):
            return None
        if source.parent != destination.parent or source.name.casefold() != (destination.name + ".tmp").casefold():
            return None
        if not any(source != root and source.is_relative_to(root) for root in self.allowed_roots):
            return None
        if not any(destination != root and destination.is_relative_to(root) for root in self.allowed_roots):
            return None
        return source, destination

    @staticmethod
    def _source_exists(source):
        try:
            return source.exists()
        except OSError:
            return False

    def _replace(self, src, dst, *, src_dir_fd=None, dst_dir_fd=None):
        # Call the original function with the caller's values. Do not replace
        # symlinks with resolved paths or otherwise alter os.replace semantics.
        kwargs = {}
        if src_dir_fd is not None:
            kwargs["src_dir_fd"] = src_dir_fd
        if dst_dir_fd is not None:
            kwargs["dst_dir_fd"] = dst_dir_fd
        try:
            return self.original_replace(src, dst, **kwargs)
        except OSError as first_error:
            scoped = self._scope(src, dst, src_dir_fd, dst_dir_fd)
            if scoped is None or getattr(first_error, "winerror", None) not in RETRY_WINERRORS:
                raise
            source, destination = scoped
            if not self._source_exists(source):
                raise
            started = time.monotonic()
            deadline = started + self.timeout_seconds
            delay, attempts, last_error = .05, 1, first_error
            details = {"source": str(source), "destination": str(destination),
                       "timeout_seconds": self.timeout_seconds}
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0 or not self._source_exists(source):
                    reason = "timeout" if remaining <= 0 else "source_missing"
                    self._log("atomic_replace_failed", **details, reason=reason, attempts=attempts,
                              elapsed_seconds=time.monotonic() - started,
                              winerror=getattr(last_error, "winerror", None))
                    raise first_error
                sleep_seconds = min(delay, remaining)
                self._log("atomic_replace_retry", **details, attempt=attempts + 1,
                          sleep_seconds=sleep_seconds, elapsed_seconds=time.monotonic() - started,
                          winerror=getattr(last_error, "winerror", None))
                time.sleep(sleep_seconds)
                # Sleeping until the budget boundary does not authorize an
                # additional attempt after that boundary.
                if time.monotonic() >= deadline or not self._source_exists(source):
                    continue
                attempts += 1
                try:
                    result = self.original_replace(src, dst, **kwargs)
                except OSError as retry_error:
                    if getattr(retry_error, "winerror", None) not in RETRY_WINERRORS:
                        self._log("atomic_replace_aborted", **details, reason="non_retryable_error",
                                  attempts=attempts, elapsed_seconds=time.monotonic() - started,
                                  winerror=getattr(retry_error, "winerror", None), error=repr(retry_error))
                        raise
                    last_error = retry_error
                    delay = min(delay * 2, .5)
                    continue
                self._log("atomic_replace_succeeded", **details, attempts=attempts,
                          elapsed_seconds=time.monotonic() - started)
                return result

    def install(self):
        if self._installed:
            return self
        if os.replace is not self.original_replace:
            raise RuntimeError("os.replace changed before this retry wrapper was installed")
        self._log("atomic_replace_retry_installed", required=True,
                  allowed_roots=[str(root) for root in self.allowed_roots],
                  timeout_seconds=self.timeout_seconds, retry_winerrors=sorted(RETRY_WINERRORS))
        os.replace = self.replacement
        self._installed = True
        return self

    def uninstall(self):
        if not self._installed:
            return
        if os.replace is not self.replacement:
            raise RuntimeError("Refusing to overwrite an os.replace hook installed after this one")
        os.replace = self.original_replace
        self._installed = False
        self._log("atomic_replace_retry_uninstalled")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.uninstall()
        return False


def install_replace_retry(allowed_roots, event_log, timeout_seconds=30):
    """Install a scoped retry wrapper and return its context/uninstall handle.

    The deadline is per failing replacement, measured using monotonic time.
    Persistent retryable failures raise the first original OS exception. A new
    non-retryable OS exception propagates unchanged immediately. The wrapper
    never deletes either file, rewrites payloads, or alters random state.
    """
    timeout = float(timeout_seconds)
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout_seconds must be positive and finite")
    if isinstance(allowed_roots, (str, bytes, os.PathLike)):
        allowed_roots = [allowed_roots]
    roots = tuple(dict.fromkeys(Path(os.fsdecode(os.fspath(root))).resolve() for root in allowed_roots))
    if not roots or any(not root.is_dir() or root == Path(root.anchor) for root in roots):
        raise ValueError("allowed_roots must be existing, explicit subdirectories")
    log_path = Path(event_log).resolve()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    return ReplaceRetryHandle(roots, log_path, timeout).install()
