"""Inventory local source files and verify the recorded relocation by SHA-256.

Unknown provenance remains blank. Holdout payloads are hashed, never parsed.
Run from the repository root with uv run --no-sync python -m program.acquire.inventory_sources.
"""

import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MANIFESTS = ROOT / "data/manifests"
FIELDS = "source_id source_name release_id landing_url download_url accessed_at downloaded_at original_filename local_path file_size sha256 citation license_or_terms_reference species genome_build assay endpoint value_unit log_transform upstream_processing schema_version acquisition_status previous_local_path provenance_note".split()
LANDING = {
    "GDSC": "https://cellmodelpassports.sanger.ac.uk/downloads",
    "DepMap": "https://depmap.org/portal/data_page/?tab=allData",
    "STRING": "https://version-12-0.string-db.org/cgi/download?species_text=Homo+sapiens",
    "DGIdb": "https://dgidb.org/downloads",
    "HGNC": "https://www.genenames.org/download/statistics-and-files/",
    "Cellosaurus": "https://www.cellosaurus.org/",
    "MSigDB": "https://www.gsea-msigdb.org/gsea/msigdb/human/collections.jsp",
    "CellModelPassports": "https://cellmodelpassports.sanger.ac.uk/downloads",
    "ChEMBL": "https://www.ebi.ac.uk/chembl/",
    "PRISM": "https://depmap.org/repurposing/",
    "GDSC1_external": "https://cellmodelpassports.sanger.ac.uk/downloads",
}


def file_hash(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    old_inventory = {}
    with (MANIFESTS / "local_inventory_2026-09-23.csv").open(encoding="utf-8-sig", newline="") as stream:
        old_inventory = {r["local_path"]: r for r in csv.DictReader(stream)}
    moves = json.loads((MANIFESTS / "relocation_plan_2026-09-23.json").read_text(encoding="utf-8"))
    # Paths removed after their identity was confirmed against newer originals
    # (e.g. legacy unverified copies replaced by verified release files).
    superseded = {r["new_path"]: r for r in json.loads((MANIFESTS / "superseded_relocations_2026-09-23.json").read_text(encoding="utf-8"))}
    previous = {r["new_path"]: r["old_path"] for r in moves if r["new_path"] not in superseded}
    acquisitions = {}
    for path in (MANIFESTS / "acquisition").glob("*.json"):
        item = json.loads(path.read_text(encoding="utf-8"))
        if item.get("acquisition_status") != "failed":
            acquisitions[item["local_path"]] = item
    rows, verified = [], []
    for folder in ["raw", "canonical", "holdout"]:
        for path in sorted((ROOT / "data" / folder).rglob("*")):
            if not path.is_file() or path.name.endswith(".part"):
                continue
            rel = path.relative_to(ROOT).as_posix()
            row = dict.fromkeys(FIELDS, "")
            row.update(local_path=rel, file_size=path.stat().st_size, sha256=file_hash(path),
                       original_filename=path.name, schema_version="1.0",
                       acquisition_status="local_present_provenance_incomplete")
            parts = path.relative_to(ROOT).parts
            if folder == "raw":
                row.update(source_id=parts[2], source_name=parts[2], release_id=parts[3])
            elif folder == "holdout":
                source = parts[3]
                row.update(source_id="GDSC1_external" if source == "GDSC1" else source,
                           source_name=source, release_id=parts[4],
                           provenance_note="Sealed external response; payload not analyzed.")
            else:
                row.update(source_id="local_derivative", source_name="local_derivative",
                           acquisition_status="derived_candidate_not_training_approved")
            row["landing_url"] = LANDING.get(row["source_id"], "")
            if row["release_id"] in {"unverified", "legacy_unverified"}:
                row["release_id"] = ""
                row["provenance_note"] = "Source release unknown; directory deliberately marked unverified."
            else:
                row["provenance_note"] += " Release label from local header/filename or source directory; see evidence records."
            if rel in acquisitions:
                item = acquisitions[rel]
                if row["sha256"] != item["sha256"]:
                    raise ValueError(f"Downloaded source hash changed: {rel}")
                for key in ["release_id", "download_url", "accessed_at", "downloaded_at", "acquisition_status"]:
                    row[key] = item.get(key, "")
                row["provenance_note"] = "Downloaded from recorded endpoint; response headers and container validation recorded."
            if rel in previous:
                old = previous[rel]
                expected = old_inventory[old]["sha256"]
                if row["sha256"] != expected:
                    raise ValueError(f"Relocation hash mismatch: {old} -> {rel}")
                row["previous_local_path"] = old
                verified.append({"old_path": old, "new_path": rel, "sha256": expected, "verified": True})
            if path.name == "GDSC2_fitted_dose_response_27Oct23.xlsx":
                row["previous_local_path"] = "<legacy_repo>/data/raw/drug_sensitivity/GDSC2_fitted_dose_response_27Oct23.xlsx"
                row["upstream_processing"] = "Byte-preserving copy from read-only legacy repository."
            if path.name == "GDSC_Raw_Data_Description.pdf":
                row["upstream_processing"] = "Extracted unchanged from GDSC2_public_raw_data_27Oct23.zip."
            if path.name == "GDSC_Fitted_Data_Description.pdf":
                row["download_url"] = "https://cog.sanger.ac.uk/cancerrxgene/GDSC_release8.5/GDSC_Fitted_Data_Description.pdf"
                row["acquisition_status"] = "downloaded_pdf_signature_checked"
            if rel == "data/raw/DepMap/26Q1/Model.csv":
                row["download_url"] = "https://storage.googleapis.com/depmap-external-downloads/downloads-by-canonical-id/public-26q1-5bbf.37/Model.csv"
                row["provenance_note"] = "26Q1 source path recovered from Windows Zone.Identifier; signed query stripped; not evidence for other matrices."
            if path.name == "release_notes_4606.json":
                row["download_url"] = "https://forum.depmap.org/t/announcing-the-26q1-release/4606.json"
            rows.append(row)
    if len(verified) != len(previous):
        raise ValueError("Some relocation entries are missing")
    with (MANIFESTS / "source_manifest.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    (MANIFESTS / "relocation_verified_2026-09-23.json").write_text(json.dumps(verified, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"files": len(rows), "bytes": sum(r["file_size"] for r in rows), "relocations_verified": len(verified)}))


if __name__ == "__main__":
    main()
