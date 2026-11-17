"""Read mutable run JSON without blocking Windows atomic file replacement."""
from __future__ import annotations

import ctypes
import json
import os
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def shared_binary_read(path):
    if os.name != "nt":
        with Path(path).open("rb") as stream:
            yield stream
        return
    import msvcrt
    from ctypes import wintypes
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    create = kernel.CreateFileW
    create.argtypes = (wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, wintypes.LPVOID,
                       wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE)
    create.restype = wintypes.HANDLE
    handle = create(str(Path(path).resolve()), 0x80000000, 0x1 | 0x2 | 0x4, None, 3, 0x80, None)
    if handle == ctypes.c_void_p(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        descriptor = msvcrt.open_osfhandle(handle, os.O_RDONLY | os.O_BINARY)
    except BaseException:
        kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
        kernel.CloseHandle(handle)
        raise
    with os.fdopen(descriptor, "rb") as stream:
        yield stream


def read(path):
    with shared_binary_read(path) as stream:
        return json.loads(stream.read().decode("utf-8-sig"))


def optional(path):
    try:
        return read(path)
    except FileNotFoundError:
        return None
