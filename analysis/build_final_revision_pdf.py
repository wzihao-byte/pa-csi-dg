"""Compile and render the final gated manuscript into an isolated QA directory."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageDraw


ROOT = Path(os.environ.get("PA_CSI_DG_ROOT", Path(__file__).resolve().parents[1])).resolve()


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_hash_table(directory: Path, manifest: dict) -> None:
    entries = manifest.get("files_sha256", manifest.get("artifacts_sha256"))
    if not isinstance(entries, dict):
        raise RuntimeError(f"No artifact hash table for {directory}")
    for name, expected in entries.items():
        if sha256(directory / name) != expected:
            raise RuntimeError(f"Artifact hash differs: {directory / name}")


def verify_inputs(evidence: Path, assets: Path) -> dict:
    formal = read(evidence / "verified_formal_results.json")
    if (formal.get("formal_run_count"), formal.get("formal_source_analysis_count")) != (216, 117):
        raise RuntimeError("Formal results are incomplete")
    if not formal.get("all_formal_studies_completed") or not formal.get("target_gates_verified"):
        raise RuntimeError("Formal study or target gate is incomplete")
    if formal.get("population_inference_performed"):
        raise RuntimeError("Unexpected population inference in formal evidence")
    verify_hash_table(evidence, read(evidence / "manifest.json"))
    asset_manifest = read(assets / "manifest.json")
    verify_hash_table(assets, asset_manifest)
    fragment = read(assets / "formal_fragment_manifest.json")
    for name, expected in fragment["files_sha256"].items():
        if sha256(assets / name) != expected:
            raise RuntimeError(f"Formal narrative artifact differs: {name}")
    if fragment["verified_formal_results_sha256"] != sha256(evidence / "verified_formal_results.json"):
        raise RuntimeError("Formal narrative and evidence hashes differ")
    if fragment["formal_asset_manifest_sha256"] != sha256(assets / "manifest.json"):
        raise RuntimeError("Formal narrative and asset hashes differ")
    return {"formal": formal, "asset_manifest": asset_manifest, "fragment": fragment}


def copy_source(destination: Path) -> None:
    source = ROOT / "paper/sensors_manuscript"
    ignored = {"manuscript.aux", "manuscript.log", "manuscript.out", "manuscript.pdf",
               "pa_csi_dg_sensors_manuscript.log", "revision_compile.txt"}
    for path in sorted(source.rglob("*")):
        if not path.is_file() or path.name in ignored or "__pycache__" in path.parts:
            continue
        target = destination / path.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)


def make_contact_sheet(pages: list[Path], output: Path) -> None:
    images = [Image.open(path).convert("RGB") for path in pages]
    thumb_width = 300
    thumbs = []
    for number, image in enumerate(images, start=1):
        height = round(image.height * thumb_width / image.width)
        thumb = image.resize((thumb_width, height), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (thumb_width + 20, height + 48), "white")
        canvas.paste(thumb, (10, 34))
        ImageDraw.Draw(canvas).text((10, 8), f"Page {number}", fill="black")
        thumbs.append(canvas)
    columns = 3
    rows = (len(thumbs) + columns - 1) // columns
    cell_width = max(image.width for image in thumbs)
    cell_height = max(image.height for image in thumbs)
    sheet = Image.new("RGB", (columns * cell_width, rows * cell_height), "#d9d9d9")
    for index, image in enumerate(thumbs):
        sheet.paste(image, ((index % columns) * cell_width, (index // columns) * cell_height))
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, optimize=True)
    for image in images + thumbs:
        image.close()


def compile_pdf(evidence: Path, assets: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError(f"Preserve existing final manuscript build: {output}")
    gate = verify_inputs(evidence, assets)
    source = output / "source"
    source.mkdir(parents=True)
    copy_source(source)
    pdflatex = shutil.which("pdflatex")
    pdftoppm = shutil.which("pdftoppm")
    pdfinfo = shutil.which("pdfinfo")
    if not all((pdflatex, pdftoppm, pdfinfo)):
        raise RuntimeError("pdflatex, pdftoppm and pdfinfo are required")
    command = [pdflatex, "-interaction=nonstopmode", "-halt-on-error", "-jobname=manuscript_final", "manuscript.tex"]
    logs = []
    for number in (1, 2):
        completed = subprocess.run(command, cwd=source, text=True, encoding="utf-8", errors="replace",
                                   capture_output=True, creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0)
        log_path = output / f"pdflatex_pass_{number}.stdout.log"
        log_path.write_text(completed.stdout + "\n" + completed.stderr, encoding="utf-8", newline="\n")
        logs.append(log_path)
        if completed.returncode:
            raise RuntimeError(f"pdflatex pass {number} failed; see {log_path}")
    pdf = source / "manuscript_final.pdf"
    latex_log = (source / "manuscript_final.log").read_text(encoding="utf-8", errors="replace")
    undefined = len(re.findall(r"(?:Citation .* undefined|Reference .* undefined|There were undefined references)", latex_log, re.I))
    overfull = len(re.findall(r"Overfull \\[hv]box", latex_log))
    underfull = len(re.findall(r"Underfull \\[hv]box", latex_log))
    if undefined or overfull:
        raise RuntimeError(f"Final LaTeX structural QA failed: undefined={undefined}, overfull={overfull}")
    info = subprocess.run([pdfinfo, str(pdf)], text=True, encoding="utf-8", errors="replace", capture_output=True,
                          creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0)
    if info.returncode:
        raise RuntimeError("pdfinfo failed")
    match = re.search(r"^Pages:\s+(\d+)", info.stdout, re.M)
    if not match:
        raise RuntimeError("Could not determine final PDF page count")
    pages = int(match.group(1))
    render = output / "rendered"
    render.mkdir()
    rendered = subprocess.run([pdftoppm, "-png", "-r", "150", str(pdf), str(render / "page")],
                              text=True, encoding="utf-8", errors="replace", capture_output=True,
                              creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0)
    if rendered.returncode:
        raise RuntimeError(f"pdftoppm failed: {rendered.stderr}")
    page_files = sorted(render.glob("page-*.png"), key=lambda path: int(path.stem.split("-")[-1]))
    if len(page_files) != pages:
        raise RuntimeError("Rendered page count differs from pdfinfo")
    contact = output / "contact_sheet.png"
    make_contact_sheet(page_files, contact)
    report = {
        "status": "awaiting_visual_qa",
        "built_at": now(),
        "source_manuscript_sha256": sha256(ROOT / "paper/sensors_manuscript/manuscript.tex"),
        "source_revision_body_sha256": sha256(ROOT / "paper/sensors_manuscript/revision_body.tex"),
        "verified_formal_results_sha256": sha256(evidence / "verified_formal_results.json"),
        "formal_asset_manifest_sha256": sha256(assets / "manifest.json"),
        "formal_fragment_manifest_sha256": sha256(assets / "formal_fragment_manifest.json"),
        "pdf": str(pdf),
        "pdf_sha256": sha256(pdf),
        "pages": pages,
        "latex_passes": 2,
        "undefined_references_or_citations": undefined,
        "overfull_boxes": overfull,
        "underfull_boxes": underfull,
        "contact_sheet": str(contact),
        "contact_sheet_sha256": sha256(contact),
        "rendered_pages": {str(index): {"path": str(path), "sha256": sha256(path)}
                           for index, path in enumerate(page_files, start=1)},
        "population_inference_performed": False,
        "formal_run_count": gate["formal"]["formal_run_count"],
        "formal_source_analysis_count": gate["formal"]["formal_source_analysis_count"],
    }
    write(output / "build_report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = compile_pdf(args.evidence.resolve(), args.assets.resolve(), args.output.resolve())
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
