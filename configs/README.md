# configs 字段说明

配置全部为 JSON，读取使用标准库 `json`。JSON 不支持注释，字段含义与规则写在本文件，预注册数值以 `guidance/重构计划/GDGN重构计划_rebuild-2026.md` 对应章节为准。

| 文件 | 内容 | 计划依据 |
| --- | --- | --- |
| data_sources.json | P0–P3 来源登记、版本冻结策略、网络核验备注 | 第 3 节 |
| cohort.json | 三个队列、初始质控门槛、重复聚合、主标签定义 | 第 5、6 节 |
| split_policy.json | 封存确认区、六协议、三阶段流程、测试输入规则 | 第 8 节 |
| baselines.json | 11 类对照、调参预算、公平性、基础信号验收 | 第 10、11 节 |
| models.json | 默认结构与训练配方、R0/R1/R2/P/G1 阶梯、候选进入条件 | 第 11、12 节 |
| experiment_budget.json | A0–C 分轮预算、选择门槛、停止规则 | 第 14、15 节 |
| interpretation.json | 三层次解释、最小解释实验、防循环验证 | 第 13 节 |

## 约定

- 每个配置文件带 `_meta` 对象：`schema_version`、`project_tag`（`rebuild-2026`）、`description`。
- 所有数值均为预注册起点：执行前可根据重复测量噪声、覆盖与 pilot 调整一次并冻结（计划 6.3、15.1），之后不再按结果回调。
- 修改任何预注册值之前，先在 `guidance/DECISION_LOG.md` 登记改动与理由；与计划文本冲突时以计划为准。
- 运行时产生的已解析配置随每次运行保存为 `config_resolved.json`（计划 16.4），配置改变不得覆盖同一结果目录。
