"""Three serial unauthenticated public requests, without changing acquisition state."""
import hashlib
import io
import json
import re
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote, urlsplit

import requests

BASE = "https://www.kaggle.com/api/v1/datasets/download/shuokanghuang/wimans"
DOCS = Path(__file__).resolve().parent


def probe(label, url, params):
    result = dict(label=label, requested_api=url, params=params,
                  at_utc=datetime.now(timezone.utc).isoformat())
    try:
        with requests.get(url, params=params, timeout=(8, 20), stream=True) as response:
            result.update(status=response.status_code, content_type=response.headers.get("Content-Type"),
                          server=response.headers.get("Server"),
                          final_host=urlsplit(response.url).hostname,
                          redirect_statuses=[r.status_code for r in response.history],
                          redirect_hosts=[urlsplit(r.headers.get("Location", "")).hostname for r in response.history])
            body = bytearray()
            for block in response.iter_content(65536):
                body.extend(block)
                if len(body) > 8 * 1024 * 1024:
                    raise ValueError("Bounded single-file probe exceeded 8 MiB")
            result["response_bytes"] = len(body)
            result["response_sha256"] = hashlib.sha256(body).hexdigest()
            if body.startswith(b"PK"):
                with zipfile.ZipFile(io.BytesIO(body)) as archive:
                    result["zip_members"] = archive.namelist()
                    result["mat_files"] = []
                    for name in archive.namelist():
                        payload = archive.read(name)
                        result["mat_files"].append(dict(name=name, bytes=len(payload),
                            matlab_v5_header=payload.startswith(b"MATLAB 5.0 MAT-file"),
                            sha256=hashlib.sha256(payload).hexdigest()))
            else:
                decoded = bytes(body[:12000]).decode("utf-8", errors="replace")
                title = re.search(r"<title[^>]*>(.*?)</title>", decoded, re.I | re.S)
                result["html_title"] = title.group(1).strip() if title else None
                result["body_starts_html"] = b"<html" in bytes(body[:2000]).lower() or b"<!doctype" in bytes(body[:2000]).lower()
    except Exception as error:
        result["error"] = type(error).__name__ + ": " + str(error)
    return result


if __name__ == "__main__":
    probes = [
        ("failed_file_original_query", BASE, dict(fileName="wifi_csi/mat/act_21_44.mat", datasetVersionNumber=1)),
        ("known_good_adjacent_original_query", BASE, dict(fileName="wifi_csi/mat/act_21_43.mat", datasetVersionNumber=1)),
        ("failed_file_path_endpoint", BASE + "/" + quote("wifi_csi/mat/act_21_44.mat", safe=""), dict(datasetVersionNumber=1)),
    ]
    report = dict(schema="bounded-wimans-download-diagnostic/v1", serial_requests=3,
                  authentication_used=False, acquisition_mutated=False, raw_payloads_saved=False,
                  probes=[probe(*item) for item in probes])
    (DOCS / "download_api_probe.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
