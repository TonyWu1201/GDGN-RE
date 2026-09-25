# M4（A2）代码实施与本机初步测试报告

日期：2026-09-25
证据等级：本机可核对的工程验证；正式 A2 开发实验未运行，以下不构成模型增益或科学结论。

## 1. 实施范围

- 实现 A2 运行入口、配置展开、4 配置/24 次拟合预算检查、A1 对照预检、失败可见的运行记录和报告生成入口。默认正式矩阵仅 `additive` 与 `r1_k16`，LCO/LPO 各三种子，共 12 条计划拟合；`data/runs/A2/` 本机尚无正式运行。
- `program/models/a2.py` 提供 R0 重训备选、特征加性、R1-K16/K32、R2 药物等权与单项排序损失、P 通路加性，以及 P 的匹配随机分组、无先验投影和同通路浅层线性对照。R1 保存 `mu/b_cell/b_drug/interaction`；P 保存有符号通路贡献，均检查分支之和与总预测一致。
- `program/features/a2_pathways.py` 在 inner train 和 outer train 的唯一训练细胞上分别拟合 Hallmark 成员基因与通路分数状态；LPO 独立构建状态。表达干预后可用保存状态同步重算通路分数；随机成员对照通过二分图换边保持集合大小及基因参与次数。
- `program/experiments/a2_pipeline.py` 复用开发 split、训练折表达预处理、RunRecord 与指标。正式拟合前核对 A1 MLP/R0 与简单基线的 cohort、样本角色、实体 ID、标签、表达/指纹源及保存指标。A1 R0 不可复用时可将 `r0` 放在正式矩阵首位重训并占用预算。
- 同步 tree 拟合后单线程预测的位级稳定性修复；`evaluate` 改为按运行配置读取有效药物门槛，并以极小浮点容差重算，指标结构、计数、ID 和非有限值仍严格检查。
- `program/experiments/report_a2.py` 从保存的正式运行生成 `A2_CANDIDATE_REPORT.md`，列出缺失、失败、参数量、配对指标与机制对照；没有正式运行时仅标“未运行”。根目录 `README.md` 提供异机命令；配置选择及新增实现值登记在 `guidance/DECISION_LOG.md`。

## 2. A1 衔接检查

`train --stage A2 --dry-run` 本机通过。LCO 与 LPO 各核对三个 A1 MLP 种子和一个对应简单对照（LCO 每药物 ElasticNet，LPO Ridge），其逐样本预测可重算，样本 ID、实体 ID、标签与活动 split 对齐。A1 神经网络、数据装配和表达预处理实现哈希与本机一致，故当前可复用 A1 MLP 为 R0。

两协议 split 文件的保存字节哈希与本机哈希不同；此前已定位为 CRLF/LF 换行差异。本次入口保留两种字节哈希，同时以训练/测试样本集合及逐样本标签校验内容等价；没有静默忽略差异。此次仅核对 A2 所需的 8 条 A1 对照运行，没有宣称本机重算全部 56 条 A1 运行。

固定输入维度下参数量：A1 R0 为 416,001，A2 特征加性为 407,811，R1-K16 为 409,859。R1-K16 与 R0 相差约 1.5%；正式机制判断仍需配对指标优于加性和 R0。

## 3. 本机测试记录

| 检查 | 结果 | 说明 |
| --- | --- | --- |
| `uv lock --check --no-cache` | 通过，解析 60 个包 | 直接 `uv lock --check` 曾因本机 uv 默认缓存目录创建失败；改用无缓存参数后通过，未修改依赖。 |
| `uv run --no-cache --no-sync python -m pytest tests -q -p no:cacheprovider` | **68 passed** | 包含合成加性/交互恢复、R1 中心化及分支求和、P 贡献/随机成员守恒/训练折状态/干预重算、R2 排序方向与噪声过滤、预算与历史配置检查、重复成功版本拒绝自动比较、tree 位级稳定性、smoke 重算门槛等。 |
| `uv run --no-cache --no-sync python -m program.cli train --stage A2 --dry-run` | 通过 | 生成默认 12/24 清单，并完成双协议 A1 对照预检；未拟合。 |
| `uv run --no-cache --no-sync python -m program.cli report-a2` | 通过 | 候选报告列出 12 条“未运行”，无 smoke 指标。 |
| `git diff --check` | 通过 | 未发现空白错误。 |

补充随机成员关系重叠率审计后，再运行 `tests/test_a2_models.py`，13 项通过。

真实输入短训练均采用 `--smoke`，每条仅一个 trial、最多 2 epoch，并逐条用 `evaluate --run-dir` 从保存预测重算成功：

| 模型 | 协议 | 状态 |
| --- | --- | --- |
| additive | LCO | smoke，重算通过 |
| r1_k16 | LCO | smoke，重算通过 |
| p | LCO、LPO | 两条 smoke，重算通过 |
| p_random | LCO | smoke，重算通过 |
| p_projection | LCO | smoke，重算通过 |
| p_linear | LCO | smoke，重算通过 |

首次 additive smoke 的独立评估因入口默认 `min_cells=10`、而 smoke 记录为 2 而报指标不一致；已改为读取该次运行的 `config_resolved.json`，回归测试覆盖，原运行保存结果未改写。所有 smoke 结果位于 `data/runs/A2_SMOKE/`，不得作为模型选择或科学结果。

## 4. 异机正式运行条件与限制

1. 目标机器同步本次代码，以及被 `.gitignore` 排除的 A1 逐样本预测、数据、split、特征与本地指导文件；先运行 `uv lock --check --no-cache` 和 A2 dry-run。缺少或不一致时先解决来源，不绕过预检。
2. 默认矩阵用 `uv run --no-cache --no-sync python -m program.cli train --stage A2 --all` 显式启动。中断后加 `--resume`；成功运行不可覆盖。每条运行后用 `evaluate --run-dir` 重算，全部完成再运行 `report-a2`。
3. R2 排序的训练噪声阈值和温度目前为 `null`，因此未纳入默认正式矩阵。若启用，须先仅依据训练数据定值、写入 `models.json` 并登记决策。P 的完整通路先验主张还需同通路线性、无先验投影和匹配随机成员关系对照；所有正式拟合均占 4 配置/24 次预算。
4. A2 当前只有工程就绪证据。LCO +0.02、三种子至少两个同向、RMSE 恶化 ≤2% 及后续多折确认，须等待异机正式运行后判断。A2 候选报告目前标记“未运行”。
