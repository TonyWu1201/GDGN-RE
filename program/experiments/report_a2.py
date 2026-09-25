"""Build an honest A2 candidate report from persisted development runs."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from program.common import paths as P
from program.evaluation.metrics import assert_metrics_close, evaluate_file
from program.evaluation.statistics import align_pair
from program.experiments.a2_pipeline import _find_a1, a2_plan


def _runs() -> list[dict]:
    root = P.DATA_DIR / "runs" / "A2"
    items = []
    for status_file in sorted(root.rglob("status.json")):
        directory = status_file.parent
        try:
            status = json.loads(status_file.read_text(encoding="utf-8"))
            config = json.loads((directory / "config_resolved.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if config.get("stage") != "A2":
            continue
        item = {"directory": directory, "status": status["status"], "config": config,
                "runtime_seconds": status.get("runtime_seconds"), "error": status.get("error", "")}
        if status["status"] == "succeeded":
            try:
                prediction = pd.read_parquet(directory / "predictions.parquet")
                actual = evaluate_file(directory / "predictions.parquet")
                assert_metrics_close(actual, json.loads((directory / "metrics.json").read_text(encoding="utf-8")))
                item["predictions"], item["metrics"] = prediction, actual["overall"]
                item["parameter_count"] = json.loads((directory / "selected_epoch.json").read_text(encoding="utf-8"))["parameter_count"]
            except (OSError, ValueError) as exc:
                item["status"], item["error"] = "invalid_saved_run", str(exc)
        items.append(item)
    return items


def _paired_difference(candidate: pd.DataFrame, baseline: pd.DataFrame) -> tuple[float, float]:
    entities = candidate[["sample_id", "cell_id", "compound_id"]].merge(
        baseline[["sample_id", "cell_id", "compound_id"]], on="sample_id", validate="one_to_one",
        suffixes=("_candidate", "_baseline"))
    if (len(entities) != len(candidate) or len(entities) != len(baseline) or
            not np.array_equal(entities["cell_id_candidate"], entities["cell_id_baseline"]) or
            not np.array_equal(entities["compound_id_candidate"], entities["compound_id_baseline"])):
        raise ValueError("paired predictions have different entity IDs")
    paired = align_pair(candidate, baseline)
    if paired.empty:
        raise ValueError("empty paired comparison")
    from program.evaluation.metrics import evaluate_predictions
    a = paired.rename(columns={"y_pred_a": "y_pred"}).drop(columns="y_pred_b")
    b = paired.rename(columns={"y_pred_b": "y_pred"}).drop(columns="y_pred_a")
    ma, mb = evaluate_predictions(a), evaluate_predictions(b)
    if ma["macro_spearman_selection"] is None or mb["macro_spearman_selection"] is None:
        raise ValueError("paired macro score unavailable")
    return ma["macro_spearman_selection"] - mb["macro_spearman_selection"], ma["rmse"] / mb["rmse"]


def build_report() -> Path:
    plan = a2_plan()
    runs = _runs()
    out = P.GUIDANCE_DIR / "M4" / "A2_CANDIDATE_REPORT.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# A2 候选报告", "",
             "证据等级：仅已保存且可重算的 A2 正式开发运行属于本地可核对开发证据；smoke 与代码测试不进入科学结果。",
             "", f"计划新配置：{', '.join(plan['selected_models'])}；计划已选配置拟合 {plan['selected_fit_count']}/{plan['limit']}。", "",
             "## 运行台账", "", "| 协议 | 模型 | 种子 | 状态 | macro | RMSE | 参数量 | 秒 | 说明 |",
             "| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |"]
    by_key: dict[tuple[str, str, int], list[dict]] = {}
    for item in runs:
        cfg = item["config"]
        by_key.setdefault((cfg["protocol"], cfg["model"], cfg["seed"]), []).append(item)
        metric = item.get("metrics", {})
        note = item["error"].replace("|", "/").replace("\n", " ")[:120]
        lines.append(f"| {cfg['protocol']} | {cfg['model']} | {cfg['seed']} | {item['status']} | "
                     f"{metric.get('macro_spearman_selection', '—')} | {metric.get('rmse', '—')} | {item.get('parameter_count', '—')} | "
                     f"{item['runtime_seconds']} | {note} |")
    for entry in plan["entries"]:
        key = (entry["protocol"], entry["model"], entry["seed"])
        if key not in by_key:
            lines.append(f"| {entry['protocol']} | {entry['model']} | {entry['seed']} | 未运行 | — | — | — | — | — |")
    if not runs:
        lines += ["", "A2 正式运行尚未开始；下列门槛目前不能判定。"]
    successful_groups: dict[tuple[str, str, int], list[dict]] = {}
    for item in runs:
        if item["status"] == "succeeded":
            cfg = item["config"]
            successful_groups.setdefault((cfg["protocol"], cfg["model"], cfg["seed"]), []).append(item)
    unique_success = {key: items[0] for key, items in successful_groups.items() if len(items) == 1}
    ambiguous = [key for key, items in successful_groups.items() if len(items) > 1]
    if ambiguous:
        lines += ["", f"同模型/协议/种子有多个成功变体：{ambiguous}。保留全部记录，配对比较暂不选取其中任何一条。"]
    lines += ["", "## 与 A1 配对比较", "",
              "同一 `sample_id` 与标签严格对齐。LCO 主任务比较 A1 每药物 ElasticNet，并列出 A1 MLP 或 A2 重训 R0 参考；LPO 单列，不替代 LCO 结论。",
              "", "| 协议 | 模型 | 种子 | 对简单对照 Δmacro | RMSE 比 | 对 R0 Δmacro |",
              "| --- | --- | ---: | ---: | ---: | ---: |"]
    comparisons: dict[tuple[str, str], list[dict]] = {}
    refitted_r0 = {(protocol, seed): item["predictions"]
                   for (protocol, name, seed), item in unique_success.items() if name == "r0"}
    for item in unique_success.values():
        cfg = item["config"]
        protocol, name, seed = cfg["protocol"], cfg["model"], cfg["seed"]
        simple_name = "per_drug_elasticnet" if protocol == "lco" else "ridge"
        try:
            simple = pd.read_parquet(_find_a1(simple_name, protocol, 42) / "predictions.parquet")
            r0 = refitted_r0.get((protocol, seed))
            if r0 is None:
                r0 = pd.read_parquet(_find_a1("mlp", protocol, seed) / "predictions.parquet")
            gain, ratio = _paired_difference(item["predictions"], simple)
            r0_gain, _ = _paired_difference(item["predictions"], r0)
        except (OSError, ValueError) as exc:
            lines.append(f"| {protocol} | {name} | {seed} | 无法比较：{str(exc).replace('|', '/')} | — | — |")
            continue
        lines.append(f"| {protocol} | {name} | {seed} | {gain:.6f} | {ratio:.6f} | {r0_gain:.6f} |")
        comparisons.setdefault((protocol, name), []).append({"seed": seed, "gain": gain, "rmse_ratio": ratio})
    lines += ["", "## 筛选状态", "",
              "§15.1 开发筛选：LCO 相对对应简单对照 macro 提升 ≥0.02，三种子至少两个同向，原尺度 RMSE 恶化 ≤2%。种子不作为独立生物重复；最终保留还需多折与统计确认。",
              ""]
    for name in plan["selected_models"]:
        if name == "p_linear":
            lines.append("- p_linear：确定性浅层通路对照；不应用三种子神经结构门槛。")
            continue
        values = comparisons.get(("lco", name), [])
        if len(values) < 3:
            lines.append(f"- {name}：LCO 可比较种子 {len(values)}/3，待完成；不得写作通过。")
            continue
        gain = float(np.mean([item["gain"] for item in values]))
        direction = sum(item["gain"] > 0 for item in values)
        ratio = float(np.mean([item["rmse_ratio"] for item in values]))
        passed = gain >= 0.02 and direction >= 2 and ratio <= 1.02
        lines.append(f"- {name}：平均 Δmacro={gain:.6f}；正向种子 {direction}/3；平均 RMSE 比={ratio:.6f}；"
                     f"开发筛选{'通过' if passed else '未通过'}。")
    lines += ["", "## 机制对照", "", "仅对同协议、同种子、同测试样本的成功运行计算；缺失对照不自动视为零效应。", "",
              "| 协议 | 候选 | 对照 | 种子 | Δmacro |",
              "| --- | --- | --- | ---: | ---: |"]
    successes = unique_success
    control_map = {"r1_k16": ["additive"], "r1_k32": ["additive"],
                   "r2_drug_equal": ["r1_k16"], "r2_rank": ["r1_k16"],
                   "p": ["p_linear", "p_projection", "p_random"]}
    for (protocol, name, seed), item in sorted(successes.items()):
        for control in control_map.get(name, []):
            reference = successes.get((protocol, control, 42 if control == "p_linear" else seed))
            if reference is None:
                lines.append(f"| {protocol} | {name} | {control} | {seed} | 对照未运行 |")
                continue
            try:
                delta, _ = _paired_difference(item["predictions"], reference["predictions"])
                lines.append(f"| {protocol} | {name} | {control} | {seed} | {delta:.6f} |")
            except ValueError as exc:
                lines.append(f"| {protocol} | {name} | {control} | {seed} | 无法比较：{str(exc).replace('|', '/')} |")
    lines += ["", "交互机制还需优于特征加性与参数量接近的 R0；P 通路先验还需同维浅层、同容量无先验及匹配随机成员关系对照。任何缺项均不作机制有效的主张。",
              "", "失败、中断和负结果均在上表保留；本报告不读取封存区响应标签。", ""]
    out.write_text("\n".join(lines), encoding="utf-8")
    return out
