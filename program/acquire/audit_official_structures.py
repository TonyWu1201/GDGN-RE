"""Compare legacy candidate keys with GDSC's keyed official annotations.

Exact-key agreement is evidence, not automatic Core eligibility approval.
"""

import collections
import csv
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def read(relative_path):
    with (ROOT / relative_path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def main():
    official = collections.defaultdict(list)
    for row in read("data/raw/GDSC/release8.5/screened_compounds_rel_8.5_inchi.csv"):
        official[row["DRUG_ID"]].append(row)
    output = []
    for row in read("data/canonical/compound_master_raw.csv"):
        annotations = official[row["source_drug_id"]]
        keys = {r["INCHI_KEY"] for r in annotations if re.fullmatch(r"[A-Z]{14}-[A-Z]{10}-[A-Z]", r["INCHI_KEY"])}
        status = "official_key_missing"
        if keys:
            status = "official_key_exact_match" if keys == {row["inchikey"]} else "official_key_conflict_review_required"
        output.append({"source_drug_id": row["source_drug_id"], "source_name": row["source_name"],
                       "candidate_inchikey": row["inchikey"], "official_inchikey": "|".join(sorted(keys)),
                       "status": status, "candidate_pubchem_cid": row["pubchem_cid"]})
    with (ROOT / "data/canonical/compound_official_key_audit.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)
    summary = {"status_counts": dict(collections.Counter(row["status"] for row in output)),
               "candidate_count": len(output), "not_core_eligibility_approval": True,
               "unmatched_count": len(read("data/canonical/compound_unmatched.csv")),
               "evidence": "GDSC release 8.5 structure annotation, official CMP catalog snapshot 2026-09-23"}
    (ROOT / "data/manifests/structure_audit_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
