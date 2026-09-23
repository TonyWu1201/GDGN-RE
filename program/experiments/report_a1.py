"""Build honest A1 registry and report solely from persisted selected runs."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from program.common import paths as P
from program.experiments.pipeline import a1_plan


def _find_runs() -> list[dict]:
    root = P.DATA_DIR / "runs" / "A1"
    rows = []
    if not root.exists():
        return rows
    for status_file in sorted(root.rglob("status.json")):
        directory = status_file.parent
        config_file = directory / "config_resolved.json"
        if not config_file.exists():
            rows.append({"path": str(directory), "status": "incomplete", "reason": "config missing"})
            continue
        cfg = json.loads(config_file.read_text(encoding="utf-8"))
        status = json.loads(status_file.read_text(encoding="utf-8"))
        history_file = directory / "status_history.jsonl"
        history = [json.loads(line) for line in history_file.read_text(encoding="utf-8").splitlines()] if history_file.exists() else [status]
        trial_file = directory / "trial_history.json"
        recorded_trials = len(json.loads(trial_file.read_text(encoding="utf-8"))["trials"]) if trial_file.exists() else None
        row = {"path": str(directory), "status": status["status"], "model": cfg["model"],
               "protocol": cfg["protocol"], "seed": cfg["seed"], "diagnostic": cfg.get("diagnostic", "none"),
               "cell_fraction": cfg.get("cell_fraction", 1.0), "drug_fraction": cfg.get("drug_fraction", 1.0),
               "runtime_seconds": status.get("runtime_seconds"), "gpu_hours": status.get("gpu_hours"),
               "n_trials": status.get("n_trials", recorded_trials),
               "prior_failures": sum(x["status"] in {"failed", "incomplete"} for x in history),
               "n_train_cells": status.get("n_train_cells"),
               "n_train_drugs": status.get("n_train_drugs")}
        if status["status"] == "succeeded" and (directory / "metrics.json").exists():
            metrics = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))["overall"]
            row.update({"macro_spearman": metrics["macro_spearman_selection"], "rmse": metrics["rmse"],
                        "qualified_drugs": metrics["n_qualified_drugs"], "constant_drugs": metrics["constant_qualified_drugs"]})
        if status["status"] in {"failed", "incomplete"}:
            row["reason"] = status.get("error") or "incomplete"
        rows.append(row)
    return rows


def _learning_curves(rows: list[dict], figures: Path) -> list[str]:
    figures.mkdir(parents=True, exist_ok=True)
    paths = []
    completed = [r for r in rows if r["status"] == "succeeded" and r.get("macro_spearman") is not None]
    for axis in ("cell_fraction", "drug_fraction"):
        count_axis = "n_train_cells" if axis == "cell_fraction" else "n_train_drugs"
        chosen = [r for r in completed if r["diagnostic"] == "none" and
                  (r["drug_fraction"] == 1 if axis == "cell_fraction" else r["cell_fraction"] == 1)]
        if len(set(r[axis] for r in chosen)) < 2:
            continue
        fig, ax = plt.subplots(figsize=(7, 4))
        for (protocol, model), group in pd.DataFrame(chosen).groupby(["protocol", "model"]):
            if group[axis].nunique() < 2:
                continue
            points = group.groupby(count_axis)["macro_spearman"].mean().sort_index()
            ax.plot(points.index, points.values, marker="o", label=f"{protocol}/{model}")
        ax.set(xlabel=count_axis, ylabel="same-drug macro Spearman", title=f"A1 learning curve: {axis}")
        ax.legend(loc="best", fontsize=8)
        fig.tight_layout()
        output = figures / f"a1_{axis}.pdf"
        fig.savefig(output)
        plt.close(fig)
        paths.append(str(output))
    return paths


def _drug_distribution(rows: list[dict], figures: Path) -> str | None:
    groups = {}
    for row in rows:
        if row["status"] != "succeeded" or row.get("diagnostic") != "none" or row.get("cell_fraction") != 1 or row.get("drug_fraction") != 1:
            continue
        metrics_path = Path(row["path"]) / "metrics.json"
        if not metrics_path.exists():
            continue
        per_drug = json.loads(metrics_path.read_text(encoding="utf-8"))["overall"]["per_drug"]
        values = [item["spearman_selection"] for item in per_drug if item["qualified"]]
        if values:
            groups.setdefault((row["protocol"], row["model"]), []).append(values)
    if not groups:
        return None
    labels, distributions = [], []
    for key, seed_values in sorted(groups.items()):
        labels.append("/".join(key))
        distributions.append([v for seed in seed_values for v in seed])
    figures.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.65), 5))
    ax.boxplot(distributions, tick_labels=labels, showfliers=False)
    ax.set(ylabel="per-drug Spearman selection score", title="A1 per-drug distribution")
    ax.tick_params(axis="x", labelrotation=60)
    fig.tight_layout()
    path = figures / "a1_per_drug_distribution.pdf"
    fig.savefig(path)
    plt.close(fig)
    return str(path)


def build_report() -> Path:
    plan = a1_plan()
    runs = _find_runs()
    registry = P.GUIDANCE_DIR / "EXPERIMENT_REGISTRY.md"
    report = P.GUIDANCE_DIR / "BASELINE_REPORT.md"
    figures = P.GUIDANCE_DIR / "figures"
    curve_paths = _learning_curves(runs, figures)
    drug_plot = _drug_distribution(runs, figures)
    by_key = {(r.get("protocol"), r.get("model"), r.get("seed")) for r in runs
              if r.get("diagnostic") == "none" and r.get("cell_fraction") == 1 and r.get("drug_fraction") == 1
              and r["status"] == "succeeded"}
    missing = [entry for entry in plan["entries"]
               if (entry["protocol"], entry["model"], entry["seed"]) not in by_key]
    registry_lines = ["# A1 实验台账", "", "本文件由已保存的 A1 运行记录自动生成；失败、中断与负结果保留。", "",
                      f"预定已选配置拟合 {plan['selected_fit_count']} 次；成功 {len(by_key)} 次；待完成 {len(missing)} 次。", "",
                      "| 协议 | 模型 | 种子 | 诊断 | 细胞比例 | 药物比例 | 状态 | 历史失败 | trials | 耗时(s) | 路径/原因 |",
                      "| --- | --- | ---: | --- | ---: | ---: | --- | ---: | ---: | ---: | --- |"]
    for r in runs:
        registry_lines.append(f"| {r.get('protocol','')} | {r.get('model','')} | {r.get('seed','')} | "
                              f"{r.get('diagnostic','')} | {r.get('cell_fraction','')} | {r.get('drug_fraction','')} | "
                              f"{r['status']} | {r.get('prior_failures',0)} | {r.get('n_trials','')} | {r.get('runtime_seconds','')} | "
                              f"{r.get('reason') or r['path']} |")
    if not runs:
        registry_lines.append("| — | — | — | — | — | — | 未执行 | — | — | — | — |")
    registry.write_text("\n".join(registry_lines) + "\n", encoding="utf-8")

    lines = ["# A1 强基线报告", "", "证据等级：仅已成功的真实 A1 运行属于本地可核对开发证据；未完成项目不作科学判断。", "",
             f"计划 {plan['selected_fit_count']} 次已选配置拟合；成功 {len(by_key)} 次；缺 {len(missing)} 次。", "",
             "## 逐种子结果", "", "| 协议 | 模型 | 种子 | 同药物 macro Spearman | RMSE | 有效药物 | 恒定药物 |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    base = [r for r in runs if r["status"] == "succeeded" and r.get("diagnostic") == "none"
            and r.get("cell_fraction") == 1 and r.get("drug_fraction") == 1]
    for r in base:
        lines.append(f"| {r['protocol']} | {r['model']} | {r['seed']} | {r.get('macro_spearman')} | "
                     f"{r.get('rmse')} | {r.get('qualified_drugs')} | {r.get('constant_drugs')} |")
    if not base:
        lines.append("| — | — | — | 未运行 | — | — | — |")
    strata_rows = []
    for r in base:
        metric_path = Path(r["path"]) / "metrics.json"
        for dimension, groups in json.loads(metric_path.read_text(encoding="utf-8"))["strata"].items():
            for group_name, values in groups.items():
                strata_rows.append({"protocol": r["protocol"], "model": r["model"], "seed": r["seed"],
                                    "dimension": dimension, "group": group_name, **values})
    if strata_rows:
        strata_file = P.GUIDANCE_DIR / "A1_STRATA.csv"
        pd.DataFrame(strata_rows).to_csv(strata_file, index=False)
        lines += ["", f"预定义组织、外推和化学组分层的完整表：`{strata_file}`。"]
    lines += ["", "## 种子汇总", "", "种子均值和离散程度仅描述训练稳定性，不作为独立生物重复。", "",
              "| 协议 | 模型 | 成功种子数 | macro 均值 | macro 标准差 | RMSE 均值 |",
              "| --- | --- | ---: | ---: | ---: | ---: |"]
    if base:
        grouped = pd.DataFrame(base).groupby(["protocol", "model"])
        for (protocol, model), group in grouped:
            scores = group["macro_spearman"].dropna()
            lines.append(f"| {protocol} | {model} | {len(group)} | "
                         f"{scores.mean() if len(scores) else None} | {scores.std(ddof=1) if len(scores) > 1 else None} | "
                         f"{group['rmse'].mean()} |")
    else:
        lines.append("| — | — | 0 | — | — | — |")
    lines += ["", "## 诊断与学习曲线", ""]
    diagnostics = [r for r in runs if r.get("diagnostic") != "none" or
                   r.get("cell_fraction") != 1 or r.get("drug_fraction") != 1]
    lines.append(f"诊断运行记录 {len(diagnostics)} 条；学习曲线图 {len(curve_paths)} 张。")
    lines += ["", "| 协议 | 模型 | 种子 | 诊断 | 细胞比例 | 药物比例 | macro | RMSE |",
              "| --- | --- | ---: | --- | ---: | ---: | ---: | ---: |"]
    for r in diagnostics:
        lines.append(f"| {r.get('protocol')} | {r.get('model')} | {r.get('seed')} | {r.get('diagnostic')} | "
                     f"{r.get('cell_fraction')} | {r.get('drug_fraction')} | {r.get('macro_spearman')} | {r.get('rmse')} |")
    if not diagnostics:
        lines.append("| — | — | — | 未运行 | — | — | — | — |")
    for path in curve_paths:
        lines.append(f"- `{path}`")
    if drug_plot:
        lines.append(f"- `{drug_plot}`")
    lines += ["", "标签打乱、表达消融和两类学习曲线均需完成并核对后，才能判断基础差异信号。",
              "smoke 不进入本报告；LPO 仅为工程桥接，不能替代 LCO 主结论。", "",
              "## 未完成与失败", ""]
    for entry in missing:
        lines.append(f"- 待完成：{entry['protocol']}/{entry['model']}/seed={entry['seed']}")
    for r in runs:
        if r["status"] in {"failed", "incomplete"}:
            lines.append(f"- {r['status']}：{r['path']}；{r.get('reason','')}")
    if not missing and not any(r["status"] in {"failed", "incomplete"} for r in runs):
        lines.append("- 无。")
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report
