"""Verify every terminal revision gate and emit one immutable completion record.

This verifier is intentionally downstream of the final release coordinator.  It
does not train, evaluate, rewrite prose, or create an approval.  A successful
run therefore means that all scientific, narrative, PDF, release, and document
sync approvals already exist and still match their referenced bytes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_STUDIES = {
    "r_core_v1": 36,
    "r_controls_v1": 81,
    "r_inputs_v1": 63,
    "r_widar_v2": 36,
}
REQUIRED_DOCUMENTS = (
    "docs/revision/REVISION_PLAN.md",
    "docs/revision/COMPLETION_AUDIT.md",
    "docs/revision/REVIEW_RESPONSE_MATRIX.md",
    "docs/revision/REVIEW_RESPONSE_DRAFT.md",
    "docs/revision/WIDAR_EXTERNAL_PROTOCOL.md",
    "docs/revision/INDEPENDENT_CONFIRMATION.md",
    "docs/revision/REPRODUCIBILITY_RELEASE_CHECKLIST.md",
    "experiments/revision_validation/EXPERIMENT_PLAN.md",
    "paper/sensors_manuscript/README.md",
    "paper/sensors_manuscript/manuscript.tex",
    "paper/sensors_manuscript/revision_body.tex",
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def verify_hash_table(directory: Path, table: dict[str, str]) -> None:
    require(isinstance(table, dict) and table, f"Missing hash table for {directory}")
    for relative, expected in table.items():
        path = directory / relative
        require(path.is_file(), f"Missing hashed artifact: {path}")
        require(sha256(path) == expected, f"Artifact hash differs: {path}")


def verify_formal_evidence(evidence: Path) -> tuple[dict, dict]:
    formal_path = evidence / "verified_formal_results.json"
    manifest_path = evidence / "manifest.json"
    require(formal_path.is_file() and manifest_path.is_file(), "Formal evidence is absent")
    formal = read(formal_path)
    require(formal.get("formal_run_count") == 216, "Formal run count is not 216")
    require(formal.get("formal_source_analysis_count") == 117,
            "Formal source-analysis count is not 117")
    require(formal.get("all_formal_studies_completed") is True,
            "Formal studies are not marked complete")
    require(formal.get("target_gates_verified") is True, "A formal target gate is unverified")
    require(formal.get("population_inference_performed") is False,
            "Formal evidence unexpectedly performs population inference")
    # The compiler uses short study keys rather than bundle directory names.
    counts = {
        "r_core_v1": formal.get("studies", {}).get("core", {}).get("job_count"),
        "r_controls_v1": formal.get("studies", {}).get("controls", {}).get("job_count"),
        "r_inputs_v1": formal.get("studies", {}).get("inputs", {}).get("job_count"),
        "r_widar_v2": formal.get("studies", {}).get("widar", {}).get("job_count"),
    }
    require(counts == EXPECTED_STUDIES, f"Formal study counts differ: {counts}")
    manifest = read(manifest_path)
    verify_hash_table(evidence, manifest.get("files_sha256"))
    return formal, manifest


def verify_source_analysis(runtime: Path) -> dict:
    completion_path = runtime / "coordinators/formal_source_analysis_v1/completion.json"
    require(completion_path.is_file(), "Formal source-analysis completion is absent")
    completion = read(completion_path)
    require(completion.get("completed_count") == 117, "Source-analysis completion count differs")
    require(completion.get("target_data_loaded") is False, "Source analysis opened target data")
    require(len(completion.get("completed", {})) == 117, "Source-analysis identities are incomplete")
    return completion


def verify_assets(assets: Path, evidence: Path) -> tuple[dict, dict]:
    manifest_path = assets / "manifest.json"
    fragment_path = assets / "formal_fragment_manifest.json"
    require(manifest_path.is_file() and fragment_path.is_file(), "Formal paper assets are absent")
    manifest = read(manifest_path)
    verify_hash_table(assets, manifest.get("files_sha256"))
    fragment = read(fragment_path)
    verify_hash_table(assets, fragment.get("files_sha256"))
    require(fragment.get("verified_formal_results_sha256") ==
            sha256(evidence / "verified_formal_results.json"),
            "Formal fragment is not bound to formal evidence")
    require(fragment.get("formal_asset_manifest_sha256") == sha256(manifest_path),
            "Formal fragment is not bound to the asset manifest")
    return manifest, fragment


def verify_final_manuscript(coordinator: Path, final_qa: Path, evidence: Path,
                            assets: Path) -> tuple[dict, dict, dict]:
    status_path = coordinator / "status.json"
    narrative_path = coordinator / "narrative_approval.json"
    require(status_path.is_file() and narrative_path.is_file(), "Final manuscript approvals are absent")
    status, narrative = read(status_path), read(narrative_path)
    require(status.get("status") == "completed", "Final manuscript coordinator is incomplete")
    require(final_qa.is_file(), "Final manuscript QA record is absent")
    require(status.get("final_qa_sha256") == sha256(final_qa),
            "Final manuscript status and QA hashes differ")
    require(narrative.get("status") == "approved", "Narrative QA is not approved")
    require(narrative.get("result_claims_checked_against_formal_evidence") is True,
            "Narrative result claims were not approved against formal evidence")
    require(not narrative.get("unresolved_numeric_claims"), "Narrative has unresolved numeric claims")
    expected_narrative_hashes = {
        "verified_formal_results_sha256": sha256(evidence / "verified_formal_results.json"),
        "formal_narrative_values_sha256": sha256(assets / "formal_narrative_values.json"),
        "manuscript_tex_sha256": sha256(ROOT / "paper/sensors_manuscript/manuscript.tex"),
        "revision_body_sha256": sha256(ROOT / "paper/sensors_manuscript/revision_body.tex"),
        "review_response_sha256": sha256(ROOT / "docs/revision/REVIEW_RESPONSE_DRAFT.md"),
    }
    for key, expected in expected_narrative_hashes.items():
        require(narrative.get(key) == expected, f"Narrative approval hash differs: {key}")
    qa = read(final_qa)
    require(qa.get("status") == "completed", "Final PDF QA is incomplete")
    require(qa.get("all_pages_visually_inspected") is True, "Final PDF pages were not all inspected")
    visual = qa.get("visual_qa", {})
    pages = qa.get("pages")
    require(isinstance(pages, int) and pages > 0, "Final PDF page count is invalid")
    require(visual.get("status") == "approved" and visual.get("all_pages_visually_inspected") is True,
            "Final visual approval is absent")
    require(visual.get("inspected_pages") == list(range(1, pages + 1)),
            "Final visual approval does not enumerate every page")
    require(not visual.get("layout_issues"), "Final visual approval contains layout issues")
    require(visual.get("pdf_sha256") == qa.get("pdf_sha256") and
            visual.get("contact_sheet_sha256") == qa.get("contact_sheet_sha256"),
            "Final visual approval is not bound to the rendered PDF and contact sheet")
    require(qa.get("undefined_references_or_citations") == 0 and qa.get("overfull_boxes") == 0,
            "Final PDF structural QA failed")
    workspace_pdf = ROOT / "paper/sensors_manuscript/manuscript.pdf"
    require(workspace_pdf.is_file() and qa.get("workspace_pdf_sha256") == sha256(workspace_pdf),
            "Workspace PDF does not match final QA")
    return status, narrative, qa


def verify_release(runtime: Path, coordinator: Path, release: Path) -> tuple[dict, dict, dict]:
    status_path = coordinator / "status.json"
    require(status_path.is_file(), "Release coordinator status is absent")
    status = read(status_path)
    require(status.get("status") == "completed", "Release coordinator is incomplete")
    require(Path(status.get("release", "")).resolve() == release.resolve(),
            "Release coordinator points to a different directory")
    archive = Path(status.get("archive", ""))
    require(release.is_dir() and archive.is_file(), "Release directory or ZIP is absent")
    manifest_path = release / "release_manifest.json"
    metadata_path = release / "release_metadata.json"
    require(manifest_path.is_file() and metadata_path.is_file(), "Release manifest or metadata is absent")
    require(status.get("release_manifest_sha256") == sha256(manifest_path),
            "Release coordinator and manifest hashes differ")
    require(status.get("archive_sha256") == sha256(archive), "Release ZIP hash differs")
    manifest, metadata = read(manifest_path), read(metadata_path)
    verify_hash_table(release, manifest.get("files_sha256"))
    require(manifest.get("file_count") == len(manifest["files_sha256"]),
            "Release file count differs from its manifest")
    require(metadata.get("formal_studies") == EXPECTED_STUDIES,
            "Release formal study counts differ")
    require(metadata.get("formal_run_count") == 216 and
            metadata.get("formal_source_analysis_count") == 117,
            "Release formal totals differ")
    checkpoints = read(release / "checkpoint_index.json")
    schedules = read(release / "sampler_schedule_index.json")
    require(len(checkpoints) == 288, "Release checkpoint index is not 216 formal + 72 historical")
    require(schedules, "Release schedule index is empty")
    formal_checkpoints = [row for row in checkpoints if row.get("study") in EXPECTED_STUDIES]
    require(len(formal_checkpoints) == 216, "Formal checkpoint index is incomplete")
    expected_keys = {(study, row["job_id"])
                     for study in EXPECTED_STUDIES
                     for row in read(runtime / "studies" / study / "protocol.json")["jobs"]}
    checkpoint_keys = {(row["study"], row["job_id"]) for row in formal_checkpoints}
    schedule_keys = {(row["study"], job_id) for row in schedules for job_id in row.get("jobs", [])}
    require(checkpoint_keys == expected_keys and len(checkpoint_keys) == 216,
            "Formal checkpoint identities differ from frozen protocols")
    require(schedule_keys == expected_keys and len(schedule_keys) == 216 and
            sum(len(row.get("jobs", [])) for row in schedules) == 216,
            "Formal schedule identities differ from frozen protocols")
    for row in formal_checkpoints:
        checkpoint = runtime / "studies" / row["study"] / "runs" / row["job_id"] / "best_model.pt"
        require(checkpoint.is_file(), f"Formal checkpoint is absent: {checkpoint}")
        require(row.get("checkpoint_bytes") == checkpoint.stat().st_size and
                row.get("checkpoint_sha256") == sha256(checkpoint),
                f"Formal checkpoint index differs: {row['study']}/{row['job_id']}")
    # Reopen all historical checkpoint files rather than treating the release
    # builder's index as independent evidence of its own verification.
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from analysis.build_revision_release_bundle import (
        historical_h_rows,
        historical_y_rows,
        verify_historical_h_checkpoints,
        verify_historical_y_checkpoints,
    )
    historical_actual = (
        verify_historical_h_checkpoints(historical_h_rows())
        + verify_historical_y_checkpoints(historical_y_rows())
    )
    require(len(historical_actual) == 72, "Direct historical checkpoint verification is incomplete")
    actual_by_key = {(row["protocol"], row["job_id"]): row for row in historical_actual}
    indexed_historical = [row for row in checkpoints if row.get("protocol") in {"H_historical", "Y_grouped"}]
    indexed_by_key = {(row["protocol"], row["job_id"]): row for row in indexed_historical}
    require(len(actual_by_key) == 72 and len(indexed_by_key) == 72 and
            set(actual_by_key) == set(indexed_by_key),
            "Historical checkpoint identities differ from the release index")
    for key, actual in actual_by_key.items():
        indexed = indexed_by_key[key]
        require(indexed.get("checkpoint_sha256") == actual["checkpoint_sha256"] and
                indexed.get("checkpoint_bytes") == actual["checkpoint_bytes"],
                f"Historical checkpoint index differs: {key[0]}/{key[1]}")
    return status, manifest, metadata


def verify_document_sync(approval: Path, evidence: Path, release: Path) -> dict:
    require(approval.is_file(), "Final document-sync approval is absent")
    value = read(approval)
    require(value.get("status") == "approved", "Final document sync is not approved")
    for flag in ("formal_states_synchronized", "author_actions_not_fabricated",
                 "pseudo_tx_checked_against_code", "historical_h_y_preserved",
                 "formal_tables_claims_checked_against_evidence"):
        require(value.get(flag) is True, f"Document-sync flag is not approved: {flag}")
    require(not value.get("unresolved_numeric_claims"), "Document sync has unresolved numeric claims")
    require(value.get("verified_formal_results_sha256") ==
            sha256(evidence / "verified_formal_results.json"),
            "Document sync is not bound to formal evidence")
    require(value.get("release_manifest_sha256") == sha256(release / "release_manifest.json"),
            "Document sync is not bound to the release")
    hashes = value.get("documents_sha256", {})
    require(set(hashes) == set(REQUIRED_DOCUMENTS), "Document-sync file set differs")
    for relative, expected in hashes.items():
        path = ROOT / relative
        require(path.is_file() and sha256(path) == expected, f"Synchronized document differs: {relative}")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, default=Path("D:/pa-csi-dg-revision-runtime"))
    parser.add_argument("--evidence", type=Path,
                        default=ROOT / "docs/revision/evidence/formal_revision_v1")
    parser.add_argument("--assets", type=Path,
                        default=ROOT / "paper/sensors_manuscript/revision_assets/formal_v1")
    parser.add_argument("--final-qa", type=Path,
                        default=ROOT / "docs/revision/evidence/manuscript_final_qa.json")
    parser.add_argument("--manuscript-coordinator", type=Path,
                        default=Path("D:/pa-csi-dg-revision-runtime/coordinators/final_manuscript_v3"))
    parser.add_argument("--release-coordinator", type=Path,
                        default=Path("D:/pa-csi-dg-revision-runtime/coordinators/revision_release_v6"))
    parser.add_argument("--release", type=Path,
                        default=Path("D:/pa-csi-dg-revision-runtime/releases/pa_csi_dg_revision_v6"))
    parser.add_argument("--document-approval", type=Path,
                        default=ROOT / "docs/revision/evidence/final_document_sync_approval.json")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "docs/revision/evidence/final_completion_audit.json")
    args = parser.parse_args()
    output = args.output.resolve()
    require(not output.exists(), f"Preserve existing completion audit: {output}")
    runtime, evidence, assets = args.runtime.resolve(), args.evidence.resolve(), args.assets.resolve()
    formal, evidence_manifest = verify_formal_evidence(evidence)
    source = verify_source_analysis(runtime)
    asset_manifest, fragment = verify_assets(assets, evidence)
    manuscript_status, narrative, final_qa = verify_final_manuscript(
        args.manuscript_coordinator.resolve(), args.final_qa.resolve(), evidence, assets)
    release_status, release_manifest, release_metadata = verify_release(
        runtime, args.release_coordinator.resolve(), args.release.resolve())
    documents = verify_document_sync(args.document_approval.resolve(), evidence, args.release.resolve())
    result = {
        "schema": "pa-csi-dg-final-completion-audit/v1",
        "status": "completed",
        "verified_at": now(),
        "formal_run_count": formal["formal_run_count"],
        "formal_source_analysis_count": source["completed_count"],
        "formal_studies": EXPECTED_STUDIES,
        "target_gates_verified": True,
        "population_inference_performed": False,
        "all_pages_visually_inspected": True,
        "release_file_count": release_manifest["file_count"],
        "hashes": {
            "verified_formal_results": sha256(evidence / "verified_formal_results.json"),
            "formal_evidence_manifest": sha256(evidence / "manifest.json"),
            "formal_asset_manifest": sha256(assets / "manifest.json"),
            "formal_fragment_manifest": sha256(assets / "formal_fragment_manifest.json"),
            "narrative_approval": sha256(args.manuscript_coordinator.resolve() / "narrative_approval.json"),
            "final_pdf_qa": sha256(args.final_qa.resolve()),
            "release_manifest": sha256(args.release.resolve() / "release_manifest.json"),
            "release_zip": release_status["archive_sha256"],
            "document_sync_approval": sha256(args.document_approval.resolve()),
        },
        "author_submission_actions_remain": True,
        "published_externally": release_status.get("published_externally", False),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                      encoding="utf-8", newline="\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
