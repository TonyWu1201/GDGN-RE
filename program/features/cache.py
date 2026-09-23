"""阶段 E 缓存哈希链（program/features/cache.py，§16.1）。

data/features/<data_hash>/<split_hash>/<preprocess_hash>/
- data_hash：来源清单 + canonical 关键文件内容哈希；
- split_hash：折文件内容哈希；
- preprocess_hash：config_resolved.json + RDKit 版本哈希。
配置改变 → 新哈希 → 新目录，不覆盖旧结果。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def hash_inputs(obj) -> str:
    """确定性 JSON 哈希（键序无关）。"""
    payload = json.dumps(obj, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def feature_dir_for(base: Path, data_hash: str, split_hash: str, preprocess_hash: str) -> Path:
    return Path(base) / data_hash / split_hash / preprocess_hash


def data_hash_from_files(files: list[Path]) -> str:
    h = hashlib.sha256()
    for p in sorted(files):
        h.update(str(p).encode("utf-8"))
        h.update(str(p.stat().st_size).encode("utf-8"))
        with open(p, "rb") as f:
            while chunk := f.read(1 << 20):
                h.update(chunk)
    return h.hexdigest()[:16]


def split_hash_from_fold_files(files: list[Path]) -> str:
    return data_hash_from_bytes(b"".join(f.read_bytes() for f in sorted(files)))


def data_hash_from_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()[:16]