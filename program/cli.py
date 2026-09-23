"""GDGN-RE development CLI. A1 fitting is always an explicit user command."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def run_pipeline(steps: list[tuple[str, str]], label: str) -> int:
    for name, module in steps:
        print(f"==> {name}: {module}")
        result = subprocess.run([sys.executable, "-m", module], check=False)
        if result.returncode:
            print(f"[{label}] {name} failed ({result.returncode})", file=sys.stderr)
            return result.returncode
    return 0


def run_audit_data(_args: argparse.Namespace) -> int:
    return run_pipeline([
        ("C0", "program.acquire.inventory_sources"), ("C1", "program.harmonize.build_gene_master"),
        ("C2", "program.harmonize.build_cell_crosswalk"), ("C3", "program.harmonize.build_compound_master"),
        ("C4", "program.harmonize.build_response_measurements"), ("D1", "program.harmonize.build_cohorts"),
    ], "audit-data")


def run_build_splits(_args: argparse.Namespace) -> int:
    return run_pipeline([
        ("D2", "program.evaluation.build_sealed_regions"),
        ("D3", "program.evaluation.build_dev_splits"),
    ], "build-splits")


def run_train(args: argparse.Namespace) -> int:
    from program.experiments.pipeline import a1_plan, run_selected
    if args.dry_run:
        if args.stage != "A1":
            raise ValueError("--dry-run is the A1 manifest; use --stage A1")
        print(json.dumps(a1_plan(), ensure_ascii=False, indent=2))
        return 0
    if args.stage == "A0":
        names = ("global_mean", "ridge") if args.all else (args.model,)
        for name in names:
            path = run_selected("A0", model=name, protocol="lco", resume=args.resume)
            print(path)
        return 0
    if args.all:
        if args.diagnostic != "none" or args.cell_fraction != 1 or args.drug_fraction != 1:
            raise ValueError("--all is reserved for the preregistered A1 base matrix")
        failures = []
        for entry in a1_plan()["entries"]:
            try:
                path = run_selected("A1", model=entry["model"], protocol=entry["protocol"],
                                    seed=entry["seed"], resume=args.resume)
                print(path)
            except FileExistsError as exc:
                if args.resume and "successful run is immutable" in str(exc):
                    print(f"already complete: {entry}")
                else:
                    failures.append((entry, str(exc)))
            except Exception as exc:
                failures.append((entry, str(exc)))
        if failures:
            for entry, error in failures:
                print(f"FAILED {entry}: {error}", file=sys.stderr)
            return 1
        return 0
    if args.model is None:
        raise ValueError("--model is required for a single run")
    path = run_selected("A1", model=args.model, protocol=args.protocol, seed=args.seed,
                        resume=args.resume, diagnostic=args.diagnostic,
                        cell_fraction=args.cell_fraction, drug_fraction=args.drug_fraction)
    print(path)
    return 0


def run_evaluate(args: argparse.Namespace) -> int:
    from program.common import paths as P
    from program.evaluation.metrics import evaluate_file
    root = (P.DATA_DIR / "runs").resolve()
    path = Path(args.run_dir).resolve()
    if not path.is_relative_to(root) or len(path.parts) <= len(root.parts) or path.parts[len(root.parts)] not in {"A0", "A1"}:
        raise ValueError("M1–M3 evaluation accepts only A0/A1 development runs")
    if not (path / "config_resolved.json").is_file() or not (path / "predictions.parquet").is_file():
        raise ValueError("run lacks config or predictions")
    metrics = evaluate_file(path / "predictions.parquet")
    saved = json.loads((path / "metrics.json").read_text(encoding="utf-8"))
    if metrics != saved:
        raise ValueError("recomputed metrics differ from saved metrics")
    print(json.dumps({"run": str(path), "status": "recomputed_equal", "overall": metrics["overall"]},
                     ensure_ascii=False, indent=2))
    return 0


def run_report_a1(_args: argparse.Namespace) -> int:
    from program.experiments.report_a1 import build_report
    print(build_report())
    return 0


def not_implemented(args: argparse.Namespace) -> int:
    print(f"[{args.command}] not implemented in M1–M3", file=sys.stderr)
    return 2


def main() -> int:
    parser = argparse.ArgumentParser(prog="gdgn-re", description="GDGN-RE rebuild-2026")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("audit-data", help="rerun source and cohort preparation").set_defaults(func=run_audit_data)
    sub.add_parser("build-splits", help="rerun sealed/dev splits").set_defaults(func=run_build_splits)
    train = sub.add_parser("train", help="A0 smoke or explicit A1 development fit")
    train.add_argument("--stage", choices=("A0", "A1"), required=True)
    train.add_argument("--model", help="one registered baseline model")
    train.add_argument("--protocol", choices=("lco", "lpo"), default="lco")
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--all", action="store_true", help="A0 two smoke models or full A1 base matrix")
    train.add_argument("--dry-run", action="store_true", help="show A1 plan without fitting or reading labels")
    train.add_argument("--resume", action="store_true", help="retry failed/incomplete run; never overwrite success")
    train.add_argument("--diagnostic", choices=("none", "label_shuffle", "expression_ablation"), default="none")
    train.add_argument("--cell-fraction", type=float, default=1.0)
    train.add_argument("--drug-fraction", type=float, default=1.0)
    train.set_defaults(func=run_train)
    evaluate = sub.add_parser("evaluate", help="recompute metrics from saved development predictions")
    evaluate.add_argument("--run-dir", required=True)
    evaluate.set_defaults(func=run_evaluate)
    sub.add_parser("report-a1", help="build A1 registry, report and available learning curves").set_defaults(func=run_report_a1)
    sub.add_parser("explain", help="reserved for later modules").set_defaults(func=not_implemented)
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
