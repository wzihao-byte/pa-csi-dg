"""Inventory a predeclared Widar3.0 replication cohort from the public release.

Only archive metadata are read here; no model or performance result is consulted.
HTTP byte ranges are the release server's advertised ordinary download feature.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import sys
import urllib.parse
import urllib.request
import zipfile
from collections import Counter, OrderedDict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.revision_validation.study_io import now, read, sha256, value_hash, write

SHARE = "https://cloud.tsinghua.edu.cn/d/2760bb9557ca4d09a74d/"
COMMON = {str(i): i - 1 for i in range(1, 7)}
COHORT = {
    "20181130_user5_10_11.zip": {"room": 1, "users": {"5": COMMON, "10": COMMON, "11": COMMON}},
    "20181204.zip": {"room": 2, "users": {"1": COMMON}},
    "20181205.zip": {"room": 2, "users": {"2": {"1": 4, "2": 5}}},
    "20181208.zip": {"room": 2, "users": {"2": {str(i): i - 1 for i in range(1, 5)}}},
    "20181209.zip": {"room": 2, "users": {"6": COMMON}},
    "20181211.zip": {"room": 3, "users": {"7": COMMON, "8": COMMON, "9": COMMON}},
}


def public_download_url(path):
    page_url = SHARE + "files/?" + urllib.parse.urlencode({"p": path})
    with urllib.request.urlopen(page_url, timeout=40) as response:
        html = response.read().decode("utf-8")
    match = re.search(r"rawPath: '([^']+)'", html)
    if not match:
        raise RuntimeError("Public file view did not supply a download link")
    return json.loads('"' + match.group(1) + '"')


class PublicRangeReader(io.RawIOBase):
    def __init__(self, url, size, block_size=262144):
        self.url, self.size, self.position = url, size, 0
        self.block_size, self.cache, self.bytes_fetched = block_size, OrderedDict(), 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        position = offset if whence == 0 else self.position + offset if whence == 1 else self.size + offset
        if position < 0:
            raise ValueError("Negative seek")
        self.position = position
        return position

    def read(self, size=-1):
        end = self.size if size < 0 else min(self.size, self.position + size)
        chunks = []
        while self.position < end:
            number = self.position // self.block_size
            if number not in self.cache:
                start = number * self.block_size
                stop = min(start + self.block_size, self.size) - 1
                request = urllib.request.Request(self.url, headers={"Range": f"bytes={start}-{stop}"})
                with urllib.request.urlopen(request, timeout=60) as response:
                    if response.status != 206 or response.headers.get("Content-Range") != f"bytes {start}-{stop}/{self.size}":
                        raise RuntimeError("Server did not honor the expected public byte range")
                    block = response.read(stop - start + 2)
                if len(block) != stop - start + 1:
                    raise IOError("Truncated range response")
                self.bytes_fetched += len(block)
                self.cache[number] = block
                while len(self.cache) > 32:
                    self.cache.popitem(last=False)
            self.cache.move_to_end(number)
            offset = self.position % self.block_size
            block = self.cache[number][offset:offset + min(end - self.position, self.block_size - offset)]
            chunks.append(block)
            self.position += len(block)
        return b"".join(chunks)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError("Use a new inventory directory")
    output.mkdir(parents=True)
    source = ROOT / "docs/revision/sources/independent_candidates"
    listing = read(source / "widar_csi_listing.json")["dirent_list"]
    archive_sizes = {r["file_name"]: r["size"] for r in listing if not r["is_dir"]}
    selection = {"schema": "widar-replication-cohort/v1", "declared_at": now(), "archives": COHORT,
                 "class_names": ["Push&Pull", "Sweep", "Clap", "Slide", "Draw-O(Horizontal)", "Draw-Zigzag(Horizontal)"],
                 "receiver": "r1", "positions": [1, 2, 3, 4, 5], "orientations": [1, 2, 3, 4, 5],
                 "repetitions": [1, 2, 3, 4, 5],
                 "rationale": "Three subjects per room with six common gestures and disjoint subjects; fixed first receiver and first five recorded repetitions",
                 "selection_basis": "Official README metadata and public archive availability only; no model outputs",
                 "validation_subjects_if_room_is_source": {"1": "11", "2": "6", "3": "9"},
                 "invalid_files": "Record zero-byte/corrupt files individually, no outcome-based exclusions",
                 "external_inference": "Prospective cross-dataset retraining/replication; not direct transfer of MultiEnv activity labels",
                 "training_protocol_frozen": False, "target_performance_used": False,
                 "README_sha256": sha256(source / "Widar_README.pdf"),
                 "code_sha256": sha256(__file__), "source_share": SHARE}
    write(output / "selection_before_inventory.json", selection)
    selected, exclusions, inventories = [], [], []
    pattern = re.compile(r"user(\d+)-(\d+)-(\d+)-(\d+)-(\d+)-r(\d+)\.dat$", re.I)
    for archive, spec in COHORT.items():
        path = "/CSI/" + archive
        reader = PublicRangeReader(public_download_url(path), archive_sizes[archive])
        with zipfile.ZipFile(reader) as zipped:
            members = [{"name": item.filename, "size": item.file_size, "compressed_size": item.compress_size,
                        "crc32": item.CRC, "header_offset": item.header_offset} for item in zipped.infolist()]
        write(output / (archive + ".members.json"), members)
        inventories.append({"archive": archive, "archive_size": archive_sizes[archive], "member_count": len(members),
                            "member_metadata_sha256": sha256(output / (archive + ".members.json")),
                            "metadata_bytes_downloaded": reader.bytes_fetched})
        for item in members:
            match = pattern.search(item["name"].replace("\\", "/"))
            if not match:
                continue
            user, gesture, position, orientation, repetition, receiver = match.groups()
            if user not in spec["users"] or gesture not in spec["users"][user]:
                continue
            if receiver != "1" or any(int(v) not in range(1, 6) for v in (position, orientation, repetition)):
                continue
            row = {**item, "archive": archive, "room": spec["room"], "subject": user,
                   "label": spec["users"][user][gesture], "original_gesture_id": int(gesture),
                   "position": int(position), "orientation": int(orientation), "repetition": int(repetition),
                   "receiver": 1, "sample_id": archive.removesuffix(".zip") + ":" + item["name"]}
            if item["size"] == 0:
                exclusions.append({**row, "reason": "zero-byte member"})
            else:
                selected.append(row)
        print(f"Inventoried {archive}; selected cumulative {len(selected)}", flush=True)
    if len({r["sample_id"] for r in selected}) != len(selected):
        raise RuntimeError("Duplicate inventory identities")
    subjects = {room: {r["subject"] for r in selected if r["room"] == room} for room in (1, 2, 3)}
    if any(subjects[a] & subjects[b] for a, b in ((1, 2), (1, 3), (2, 3))):
        raise RuntimeError("Subjects cross selected rooms")
    write(output / "inventory.json", {"selection_sha256": sha256(output / "selection_before_inventory.json"),
          "complete_metadata_inventory": True, "signal_data_downloaded": False, "samples": selected,
          "exclusions": exclusions, "archives": inventories,
          "counts_by_room_subject_class": dict(sorted(Counter(f"room{r['room']}/user{r['subject']}/class{r['label']}" for r in selected).items())),
          "compressed_selected_bytes": sum(r["compressed_size"] for r in selected),
          "uncompressed_selected_bytes": sum(r["size"] for r in selected)})
    print(f"Selected {len(selected)} members; excluded {len(exclusions)} empty files; model evaluation has not begun")


if __name__ == "__main__":
    main()
