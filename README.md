# GDGN-RE

## M1–M3 开发实验

项目使用 `uv`。活动队列和开发折由已完成的数据预处理阶段提供；`data/holdout/` 标签不属于以下入口的输入。

### 已验收的 A0 smoke

```powershell
uv run --no-cache --no-sync python -m program.cli train --stage A0 --all
```

这会在 LCO quick fold 的固定不超过 2,000 条真实样本上分别运行全局均值和 Ridge，结果标记为 `smoke`。运行目录位于 `data/runs/A0/`，不进入科学结果。命令会拒绝覆盖成功运行；失败重试需加 `--resume`。可用以下命令重算已保存的预测指标：

```powershell
uv run --no-cache --no-sync python -m program.cli evaluate --run-dir "<上一步输出的运行目录>"
```

### A1 正式运行（由项目负责人启动）

先检查不读取响应标签、也不拟合模型的矩阵：

```powershell
uv run --no-cache --no-sync python -m program.cli train --stage A1 --dry-run
```

当前矩阵为 LCO quick 和 LPO 各一折，12 次朴素/单侧诊断、20 次学习模型已选配置拟合；确定性模型每折一次，树和 MLP 三种子。每家族搜索不超过 12 trials。确认后启动：

```powershell
uv run --no-cache --no-sync python -m program.cli train --stage A1 --all
```

中断后用同命令加 `--resume`，已成功运行保持不可变。单独运行示例：

```powershell
uv run --no-cache --no-sync python -m program.cli train --stage A1 --protocol lco --model ridge --seed 42
```

基础矩阵完成后，可显式运行诊断；下面仅是代码入口示例，诊断不会由 `--all` 自动启动：

```powershell
uv run --no-cache --no-sync python -m program.cli train --stage A1 --protocol lco --model ridge --diagnostic label_shuffle
uv run --no-cache --no-sync python -m program.cli train --stage A1 --protocol lco --model ridge --diagnostic expression_ablation
uv run --no-cache --no-sync python -m program.cli train --stage A1 --protocol lco --model ridge --cell-fraction 0.5
uv run --no-cache --no-sync python -m program.cli train --stage A1 --protocol lco --model ridge --drug-fraction 0.5
```

按相同命令改变细胞/药物比例形成曲线；固定原 outer test。运行后生成台账、基线报告和可用图表：

药物数曲线在较小训练比例时包含未见训练药物的测试样本，因此只用于分析训练药物多样性；不能当成纯粹的已知药物 LCO 结果。每个点记录实际训练细胞数、药物数和样本清单。

```powershell
uv run --no-cache --no-sync python -m program.cli report-a1
```

报告会列出所有失败和缺失项；诊断不足时不会判定基础信号成立。LPO 仅作工程桥接，LCO 是主任务。当前实现不包含封存区或外部验证入口。

## M4 / A2 最小机制（正式拟合在另一台电脑）

A1 的三个 MLP 种子作为 R0 锚点；A2 dry-run 会核对其数据、样本 ID、标签和保存指标。默认正式矩阵只包含特征加性与 R1-K16，各在 LCO/LPO 运行三个种子，共 12 次已选配置拟合。所有正式 A2 配置由 `configs/models.json` 的 `a2_execution.selected_models` 冻结，最多 4 个新配置、24 次拟合；增加 R2 或 P 分支前先在 `guidance/DECISION_LOG.md` 登记取舍。R2 排序的噪声阈值和温度目前为空，必须由训练数据确定并登记后才能选择。

若异机 dry-run 判定 A1 MLP 无法复用，先查明数据或实现差异；确需重训时把 `r0` 放在 `selected_models` 首位并删去相应可选配置，仍守住 4 配置/24 次上限。其余 A2 模型必须等同协议、同种子的重训 R0 成功后运行。

本机先检查代码与输入，不运行完整 A2：

```powershell
uv run --no-cache --no-sync python -m pytest tests -q -p no:cacheprovider
uv run --no-cache --no-sync python -m program.cli train --stage A2 --dry-run
uv run --no-cache --no-sync python -m program.cli train --stage A2 --model additive --protocol lco --seed 42 --smoke
```

`--smoke` 只用小样本和短训练，运行状态为 `smoke`，不进入候选报告。另一台电脑同步代码及带哈希的 A1 预测、数据和特征资产后，先执行上述 dry-run，再显式启动冻结矩阵：

```powershell
uv lock --check --no-cache
uv run --no-cache --no-sync python -m program.cli train --stage A2 --all
```

中断后同一命令加 `--resume`；成功运行不可覆盖。单条运行可用 `--stage A2 --model r1_k16 --protocol lco --seed 42`。对每条输出目录用 `evaluate --run-dir "<输出目录>"` 重算，再运行：

```powershell
uv run --no-cache --no-sync python -m program.cli report-a2
```

候选报告写入 `guidance/M4/A2_CANDIDATE_REPORT.md`，仅汇总正式 A2 运行。`guidance/`、`data/features/`、`data/runs/` 被 `.gitignore` 排除，跨机需单独同步相关资产并校验哈希；开发入口不读取 `data/holdout/` 响应标签。
