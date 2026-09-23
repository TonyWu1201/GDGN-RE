"""Collision-safe experiment records; failures stay visible."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

from program.common import paths as P


def stable_hash(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temp, path)


def run_path(experiment: str, model: str, protocol: str, fold: str, seed: int, config: dict) -> Path:
    variant = "v_" + stable_hash(config)[:12]
    return P.DATA_DIR / "runs" / experiment / model / variant / f"{protocol}_{fold}" / str(seed)


def peak_memory_bytes() -> int | None:
    """Process peak working set, using only the Python standard library."""
    try:
        if platform.system() == "Windows":
            import ctypes
            from ctypes import wintypes
            class Counters(ctypes.Structure):
                _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                            ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                            ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]
            counters = Counters()
            counters.cb = ctypes.sizeof(counters)
            get_process = ctypes.windll.kernel32.GetCurrentProcess
            get_process.restype = wintypes.HANDLE
            get_memory = ctypes.windll.psapi.GetProcessMemoryInfo
            get_memory.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
            get_memory.restype = wintypes.BOOL
            handle = get_process()
            if get_memory(handle, ctypes.byref(counters), counters.cb):
                return int(counters.PeakWorkingSetSize)
            return None
        import resource
        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(value if platform.system() == "Darwin" else value * 1024)
    except Exception:
        return None


class RunRecord:
    def __init__(self, experiment: str, model: str, protocol: str, fold: str, seed: int,
                 config: dict, resume: bool = False):
        self.config = config
        self.key = stable_hash(config)
        self.path = run_path(experiment, model, protocol, fold, seed, config)
        self.start_time = time.monotonic()
        if self.path.exists():
            existing = self.path / "config_resolved.json"
            if not existing.exists() or json.loads(existing.read_text(encoding="utf-8"))["run_key"] != self.key:
                raise FileExistsError("run directory contains a different or incomplete configuration")
            if not resume:
                raise FileExistsError("run exists; pass --resume explicitly")
            if (self.path / "status.json").exists():
                status = json.loads((self.path / "status.json").read_text(encoding="utf-8"))["status"]
                if status in {"succeeded", "smoke"}:
                    raise FileExistsError("successful run is immutable")
        else:
            self.path.mkdir(parents=True)
            write_json(self.path / "config_resolved.json", {"run_key": self.key, **config})
        self.status("planned")

    def status(self, state: str, **extra) -> None:
        if state not in {"planned", "running", "succeeded", "failed", "incomplete", "smoke"}:
            raise ValueError(state)
        record = {
            "status": state, "run_key": self.key, "runtime_seconds": round(time.monotonic() - self.start_time, 3),
            "at_utc": datetime.now(timezone.utc).isoformat(),
            "peak_memory_bytes": peak_memory_bytes(),
            "hardware": {"platform": platform.platform(), "processor": platform.processor()}, **extra,
        }
        write_json(self.path / "status.json", record)
        with (self.path / "status_history.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    def fail(self, error: BaseException) -> None:
        (self.path / "error_log.txt").write_text(f"{type(error).__name__}: {error}\n", encoding="utf-8")
        description = str(error).lower()
        category = ("oom" if isinstance(error, MemoryError) or "out of memory" in description else
                    "non_finite" if "non-finite" in description or "nan" in description else
                    "non_convergence" if "converg" in description else "other")
        self.status("failed", error_type=type(error).__name__, error_category=category, error=str(error))
