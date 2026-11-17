"""Read-only Windows GPU gate: permit known desktop clients, block experiments.

This host's WDDM driver reports every desktop client as C+G, even through
--query-compute-apps. An empty PID list is therefore not an attainable idle
condition. Preserve all raw observations and fail closed on unknown clients,
compute-only clients, non-WDDM drivers, low free memory, or sustained load.
"""
from __future__ import annotations

import fnmatch
import json
import os
import subprocess
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone

import psutil


DESKTOP_PATHS = (
    "c:/windows/explorer.exe",
    "c:/windows/system32/dwm.exe",
    "c:/windows/systemapps/microsoft.windows.search_*/searchapp.exe",
    "c:/windows/systemapps/microsoft.windows.startmenuexperiencehost_*/startmenuexperiencehost.exe",
    "c:/windows/systemapps/microsoft.lockapp_*/lockapp.exe",
    "c:/windows/systemapps/microsoftwindows.client.cbs_*/textinputhost.exe",
    "c:/windows/systemapps/shellexperiencehost_*/shellexperiencehost.exe",
    "c:/program files (x86)/microsoft/edgewebview/application/*/msedgewebview2.exe",
    "c:/program files (x86)/steam/bin/cef/cef.win64/steamwebhelper.exe",
    "c:/program files/windowsapps/microsoft.yourphone_*/phoneexperiencehost.exe",
    "c:/program files/windowsapps/microsoft.549981c3f5f10_*/cortana.exe",
    "c:/program files/google/chrome/application/chrome.exe",
    "c:/program files/windowsapps/openai.codex_*/app/chatgpt.exe",
    "c:/users/howie/appdata/local/openai/codex/runtimes/cua_node/*/bin/node_modules/@oai/sky/bin/windows/swift/x64/codex-computer-use-swift.exe",
)


def desktop_client(row: dict) -> bool:
    if row["type"] not in ("G", "C+G"):
        return False
    path = row.get("executable") or row.get("nvidia_process_name") or ""
    path = path.replace("\\", "/").lower()
    if path:
        return any(fnmatch.fnmatchcase(path, pattern) for pattern in DESKTOP_PATHS)
    # The protected Desktop Window Manager hides its executable path from
    # both APIs on this host. Name still comes from the live OS process.
    return row.get("name", "").lower() == "dwm.exe" and row.get("executable_access_denied") is True


def snapshot() -> dict:
    result = subprocess.run(["nvidia-smi", "-q", "-x"], capture_output=True,
                            text=True, check=True,
                            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    xml = ET.fromstring(result.stdout)
    gpus, blockers = [], []
    for gpu in xml.findall("gpu"):
        driver = gpu.findtext("driver_model/current_dm")
        free = int(gpu.findtext("fb_memory_usage/free").split()[0])
        utilization = int(gpu.findtext("utilization/gpu_util").split()[0])
        rows = []
        for item in gpu.findall("processes/process_info"):
            row = {"pid": int(item.findtext("pid")), "type": item.findtext("type"),
                   "nvidia_process_name": item.findtext("process_name") or ""}
            try:
                process = psutil.Process(row["pid"])
                row.update(name=process.name(), process_create_time=process.create_time())
                try:
                    row["executable"] = process.exe()
                except psutil.AccessDenied:
                    row["executable_access_denied"] = True
            except psutil.NoSuchProcess:
                row["exited_during_probe"] = True
            except psutil.AccessDenied:
                row["identity_access_denied"] = True
            row["recognized_desktop_client"] = desktop_client(row)
            if not row["recognized_desktop_client"] and not row.get("exited_during_probe"):
                blockers.append({"reason": "unknown_or_compute_only_gpu_client", **row})
            rows.append(row)
        if driver != "WDDM":
            blockers.append({"reason": "driver_is_not_verified_WDDM", "driver_model": driver})
        if free < 8192:
            blockers.append({"reason": "less_than_8_GiB_free", "free_MiB": free})
        if utilization > 10:
            blockers.append({"reason": "gpu_utilization_above_10_percent", "gpu_util_percent": utilization})
        gpus.append({"uuid": gpu.findtext("uuid"), "driver_model": driver,
                     "free_MiB": free, "gpu_util_percent": utilization, "processes": rows})
    if len(gpus) != 1:
        blockers.append({"reason": "unexpected_gpu_count", "count": len(gpus)})
    return {"at": datetime.now(timezone.utc).isoformat(), "available": not blockers,
            "gpus": gpus, "blockers": blockers,
            "raw_nvidia_smi_xml": result.stdout}


def verify_available(samples: int = 3, interval_seconds: float = 2.0) -> dict:
    observations = []
    for index in range(samples):
        observations.append(snapshot())
        if index + 1 < samples:
            time.sleep(interval_seconds)
    return {"schema": "h-wddm-gpu-availability/v2", "available": all(x["available"] for x in observations),
            "sample_count": samples, "observations": observations,
            "policy": "Known desktop clients allowed; no unknown/compute-only GPU clients; >=8192 MiB free; <=10% GPU utilization in every sample"}


if __name__ == "__main__":
    report = verify_available()
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["available"] else 2)
