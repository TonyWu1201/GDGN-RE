"""数据目录与共享读取工具。"""

from __future__ import annotations

import csv
from functools import lru_cache
from pathlib import Path

# program/ 下两级的仓库根
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"

GDSC2_RESPONSE_CSV = (
    DATA_DIR / "canonical" / "GDSC" / "release8.5" / "GDSC2_fitted_dose_response_27Oct23.csv"
)


@lru_cache(maxsize=1)
def get_gdsc2_drugs() -> list[dict]:
    """提取 GDSC2 拟合表中的全部唯一药物（DRUG_ID, DRUG_NAME, PUTATIVE_TARGET）。

    DRUG_ID 是源药物编号（唯一），DRUG_NAME 可能是别名；此处按 (DRUG_ID, DRUG_NAME,
    PUTATIVE_TARGET) 去重并按 DRUG_ID 排序，不套用任何旧 common 列表。
    """
    seen: dict[str, dict] = {}
    with GDSC2_RESPONSE_CSV.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            drug_id = row["DRUG_ID"].strip()
            if not drug_id:
                continue
            if drug_id not in seen:
                seen[drug_id] = {
                    "DRUG_ID": drug_id,
                    "DRUG_NAME": row["DRUG_NAME"].strip(),
                    "PUTATIVE_TARGET": row["PUTATIVE_TARGET"].strip(),
                }
    return sorted(seen.values(), key=lambda d: int(d["DRUG_ID"]))