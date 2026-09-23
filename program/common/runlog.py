"""共享 IO/哈希/运行记录工具（rebuild-2026 阶段 C–E）。"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from program.common.paths import RUNS_DIR


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def sha256_frame(df: pd.DataFrame) -> str:
    """对 DataFrame 内容取确定性哈希（列序+行序+值）。"""
    payload = df.to_csv(index=False, lineterminator="\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def deterministic_json_bytes(obj: Any) -> bytes:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, indent=2).encode("utf-8")


def sha256_json(obj: Any) -> str:
    return hashlib.sha256(deterministic_json_bytes(obj)).hexdigest()


def git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        )
        return out.stdout.strip()
    except Exception:
        return "unknown"


def resolve_config(base: dict[str, Any]) -> dict[str, Any]:
    """生成 config_resolved：基础配置 + 环境版本 + git commit + 生成时间。"""
    import rdkit
    import pyarrow
    import numpy
    import scipy
    import sklearn

    cfg = dict(base)
    cfg["environment"] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "rdkit": rdkit.__version__,
        "pandas": pd.__version__,
        "pyarrow": pyarrow.__version__,
        "numpy": numpy.__version__,
        "scipy": scipy.__version__,
        "scikit-learn": sklearn.__version__,
    }
    cfg["git_commit"] = git_commit()
    cfg["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
    return cfg


def save_run_record(
    stage: str,
    config_resolved: dict[str, Any],
    input_hashes: dict[str, str],
    outputs: dict[str, str],
    status: str,
    started_at: float,
    error: str | None = None,
) -> Path:
    """每次阶段运行保存运行记录（§16.4 子集）；失败/中断照记，不覆盖旧记录。"""
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = RUNS_DIR / stage
    run_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "stage": stage,
        "status": status,
        "started_at": datetime.fromtimestamp(started_at, timezone.utc).isoformat(),
        "duration_s": round(time.time() - started_at, 1),
        "config_resolved": config_resolved,
        "input_sha256": input_hashes,
        "output_sha256": outputs,
        "error": error,
    }
    out = run_dir / f"run_{ts}_{status}.json"
    out.write_bytes(deterministic_json_bytes(record))
    return out


def read_config(path: Path) -> dict[str, Any]:
    import json

    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_config_resolved(stage: str, config_resolved: dict[str, Any]) -> Path:
    run_dir = RUNS_DIR / stage
    run_dir.mkdir(parents=True, exist_ok=True)
    out = run_dir / "config_resolved.json"
    out.write_bytes(deterministic_json_bytes(config_resolved))
    return out