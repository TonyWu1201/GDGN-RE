# GDSC 数据字典核读笔记（C0 · 2026-09-23）

> 来源（本地冻结文件，逐条附页码）：
> - [Fitted p.X]：`data/raw/GDSC/release8.5/GDSC_Fitted_Data_Description.pdf`（Version 1.0.0, 21 September 2017），共 2 页
> - [Raw p.X]：`data/raw/GDSC/release8.5/GDSC_Raw_Data_Description.pdf`，共 4 页
> - 证据等级：`local_verified`（以下每条均为本地 PDF 原文摘录）；未覆盖的字段明确标 `unknown`，不猜测。
> - 注意：Fitted 字典为 2017 年版本，列名与 release8.5 实际表头存在演化（如 IC50_RESULTS_ID → NLME_RESULT_ID），差异逐条登记如下。

## 1. 拟合表（GDSC2_fitted_dose_response）字段语义

| 字段（8.5 实际表头） | 字典出处 | 官方定义（摘录/转述） | 对管线的结论 |
| --- | --- | --- | --- |
| `LN_IC50` | [Fitted p.2] | "Natural log of the fitted IC50 … To convert to micromolar take the exponent of this value, i.e. exp(IC50_nat_log)" | LN_IC50 = ln(IC50[µM])。**换算 µM 用 exp()；不得再取 log**（计划 §5.3）。pIC50 = 6 − LN_IC50/ln(10)（前置条件：IC50 单位为 µM，已证实） |
| `MIN_CONC` / `MAX_CONC` | [Fitted p.2]（2017 列名 `MIN_CONC_MICROMOLAR` / `MAX_CONC_MICROMOLAR`） | "Maximum/Minimum micromolar screening concentration of the drug" | 单位 **µM 已证实**；8.5 表头省略了 `_MICROMOLAR` 后缀，语义按字典继承 |
| `AUC` | [Fitted p.2] | "Area Under the Curve for the fitted model. Presented as a fraction of the total area between the highest and lowest screening concentration." | 归一化到 [最低,最高] 浓度区间的**分数**；方向（较高=更耐药）字典**未明文**——仅作辅助列，不作主标签，不同来源不直接可比（cohort.json 已注明） |
| `RMSE` | [Fitted p.2] | "how well the modelled curve fits the data points. **Curves with RMSE > 0.3 are excluded prior to release as part of quality control.**" | **官方 QC 阈值 0.3 已证实**：正式 release 中应无 RMSE>0.3 的行；长表 quality_flag 以此为准逐行核验（发现超标行单独列出） |
| `Z_SCORE` | [Fitted p.2] | "Z = (x − μ)/σ，其中 μ、σ 为该药物在全部处理细胞系上 LN_IC50 的均值/标准差" | **按药物标准化**，跨药物可比性依赖各药分布假设；随行携带但标 `not_training_label`（§5.3） |
| `DATASET` | [Fitted p.1]（2017 列名 `DATASET_VERSION`） | "Each dataset is processed (curve fitted and ANOVA analysis) as a whole." | DATASET 标识一个整体处理（拟合+ANOVA）的数据集；8.5 表中实际取值在 C4 枚举核对。与 `SCREENING_SITE`（screened_compounds）的关系：字典未定义映射，C4 以数据实测交叉表回答 |
| `NLME_RESULT_ID` / `NLME_CURVE_ID` | [Fitted p.1]（2017 对应列 `IC50_RESULTS_ID` = "Identifier for the fitted dose response"） | 2017 字典仅定义"拟合剂量响应标识符"；NLME 前缀对应 Raw p.1 所述 Vis et al. 2016 非线性混合效应模型 | 语义部分证实（拟合结果标识符）；RESULT 与 CURVE 两级的确切区分**文档未定义**→ long 表 `measurement_id` 以 (NLME_RESULT_ID, NLME_CURVE_ID, 源行号) 组合保证唯一，语义限制如实记录 |
| `CELL_LINE_NAME` / `COSMIC_ID` | [Fitted p.1] / [Raw p.2] | 主名；COSMIC 数据库标识符 | 主键映射以 `SANGER_MODEL_ID`（8.5 新增，2017 字典无此列）为准，COSMIC/名称仅佐证 |
| `DRUG_ID` | [Fitted p.1] | "Unique identifier for a drug. Used for internal lab tracking" | 药物身份键；同名药物不同 DRUG_ID 视为不同记录，合并仅在结构同 InChIKey 时按 §4.2 分组规则处理 |
| `PUTATIVE_TARGET` / `PATHWAY_NAME` | [Fitted p.1] | 推定靶点 / 通路（响应文件内嵌注释列） | 仅注释用；正式靶点字段以 screened_compounds + 审定表为准 |

## 2. 原始数据（GDSC2_public_raw_data）关键字段（供 C4 重复质控按需取证）

| 字段 | 字典出处 | 定义 | 对管线结论 |
| --- | --- | --- | --- |
| `RESEARCH_PROJECT` | [Raw p.2] | 数据集项目名（例 GDSC_SA） | 可作 screen 级别证据（不同筛选项目=不同测量组） |
| `BARCODE` / `SCAN_ID` | [Raw p.2] | 板条码 / 板读数扫描 ID；"A plate might be scanned more than once but only one SCAN_ID will pass internal QC" | BARCODE↔SCAN_ID 一一对应（已证实）；可用于恢复重复测量的板级来源 |
| `CELL_ID` / `MASTER_CELL_ID` | [Raw p.2] | 每次扩培唯一 CELL_ID；一系一 MASTER_CELL_ID、可多 CELL_ID | **同细胞系多次扩培=重复测量的物理来源**；重复识别按此区分技术/生物学重复 |
| `DRUGSET_ID` | [Raw p.1-2] | 一块板的药物集合与布局 | 同 (cell, drug) 多次出现若属不同 DRUGSET=独立实验组，**不得均值合并**（§5.2） |
| `ASSAY` | [Raw p.2] | 终点测定类型（例 Glo = Promega CellTiter-Glo） | fitted 表不含此列 → 长表 `assay=unknown`，除非按需解包 ZIP 取证 |
| `DURATION` | [Raw p.2] | "Duration of the assay **in days** from cell line drug treatment to end point measurement" | 单位=天（72h≈3 天的常见值字典未明文规定为常数）；fitted 表不含 → `treatment_duration=unknown`，不同时长保持不同测量组 |
| `TAG` | [Raw p.3] | `Lx-Dx-S`：库药物 x、剂量档 x（D1=最大浓度）、单药；`-C` 联合；`NC/PC/R` 对照 | L 序号区分库，可恢复重复的库归属；51 个内部库药物即 TAG 证据（DECISION_LOG 2026-09-23） |
| `CONC` | [Raw p.2] | "Micromolar concentration of the drug" | 再次证实浓度单位 µM |
| 实验设计 | [Raw p.1] | 1536/384 孔板；每板单细胞系；组合处理数据不公开 | 单药处理为主；组合数据不在本管线范围 |

## 3. C0 结论与长表字段映射决定

1. `endpoint=LN_IC50`、`response_unit=ln(uM)`、`log_base=e`——**已证实**，禁止再取 log。
2. `min/max_concentration` 单位 µM——**已证实**（两处独立证实）。
3. `quality_flag` 采用官方 RMSE>0.3 阈值核验；非有限值（NaN/inf）单列排除类。
4. `extrapolation_flag`：exp(LN_IC50) 与 [MIN_CONC, MAX_CONC] 比较，同单位 µM，边界（恰等于端点）不记外推。
5. `assay` / `treatment_duration`：fitted 表无此两列 → 记 `unknown` 并列分析限制；如需恢复按 §2 从原始 ZIP 解包取证（按需，产物另登记哈希）。
6. `DATASET` 实际取值与 `SCREENING_SITE` 交叉关系：在 C4 用全表实测回答（本地可核对项，无需求文档）。
7. 2017 字典与 8.5 表头的列名演化（IC50_RESULTS_ID→NLME_*；新增 SANGER_MODEL_ID）如实记录，不影响上述语义结论。

## 4. 依赖与配置核对结果（2026-09-23）

- 依赖（uv 管理，`uv run --no-sync` 实测）：rdkit **2026.03.6**、pandas **3.0.6**、pyarrow **25.0.1**、numpy/scipy/scikit-learn/torch/openpyxl 在位；本轮 `uv add pypdf==6.19.0`（已登记 DECISION_LOG）。
- 配置核对：`uv run --no-sync python -m program.acquire.inventory_sources` → `{"files": 194, "bytes": 11007768564, "relocations_verified": 143}`，与 source_manifest.csv（194 文件 / 11,007,768,564 字节）一致，**0 失配**。
- git 忽略：`.gitignore` 已含 `data/processed/`（本次新增，见 DECISION_LOG 2026-09-23 阶段 C0 前置决策）；splits 目录不入忽略名单（按计划入 git）。