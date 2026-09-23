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
