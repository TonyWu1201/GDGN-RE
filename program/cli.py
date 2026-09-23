"""GDGN-RE 执行入口（rebuild-2026）。

五个子命令（计划 16.3）：
- audit-data：来源审计 + 阶段 C0–C4 + D1 瀑布与 DATA_CARD（已实现，串行驱动）
- build-splits：阶段 D2 + D3 + 泄漏审计（已实现）
- train / evaluate / explain：占位拒绝执行，防止误用（实验轮次实现）
"""

from __future__ import annotations

import argparse
import sys

SUBCOMMANDS = {
    "audit-data": "来源审计、哈希清单、实体映射与测量长表检查（C0–C4 + D1）",
    "build-splits": "封存确认区与开发划分（含泄漏审计）（D2 + D3）",
    "train": "按 configs 训练指定模型（未实现，实验轮次 A0–C）",
    "evaluate": "指标、封存区与外部数据评估（未实现）",
    "explain": "归因、干预与稳定性实验（未实现）",
}


def run_audit_data(args: argparse.Namespace) -> int:
    import subprocess

    steps = [
        ("C0", "program.acquire.inventory_sources"),
        ("C1", "program.harmonize.build_gene_master"),
        ("C2", "program.harmonize.build_cell_crosswalk"),
        ("C3", "program.harmonize.build_compound_master"),
        ("C4", "program.harmonize.build_response_measurements"),
        ("D1", "program.harmonize.build_cohorts"),
    ]
    for name, mod in steps:
        print(f"==> {name}: {mod}")
        r = subprocess.run([sys.executable, "-m", mod], check=False)
        if r.returncode != 0:
            print(f"[audit-data] {name} 失败（returncode={r.returncode}），停止后续阶段", file=sys.stderr)
            return r.returncode
    return 0


def run_build_splits(args: argparse.Namespace) -> int:
    import subprocess

    for name, mod in [("D2", "program.evaluation.build_sealed_regions"), ("D3", "program.evaluation.build_dev_splits")]:
        print(f"==> {name}: {mod}")
        r = subprocess.run([sys.executable, "-m", mod], check=False)
        if r.returncode != 0:
            print(f"[build-splits] {name} 失败（returncode={r.returncode}）", file=sys.stderr)
            return r.returncode
    return 0


def not_implemented(args: argparse.Namespace) -> int:
    print(
        f"[{args.command}] 尚未实现：rebuild-2026 实验轮次实现（见 guidance/重构计划/GDGN重构计划_rebuild-2026.md 第 14 节）",
        file=sys.stderr,
    )
    return 2


def main() -> int:
    parser = argparse.ArgumentParser(prog="gdgn-re", description="GDGN-RE (rebuild-2026) 执行入口")
    subparsers = parser.add_subparsers(dest="command", required=True)
    handlers = {
        "audit-data": run_audit_data,
        "build-splits": run_build_splits,
    }
    for name, help_text in SUBCOMMANDS.items():
        sub = subparsers.add_parser(name, help=help_text)
        sub.set_defaults(func=handlers.get(name, not_implemented))
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())