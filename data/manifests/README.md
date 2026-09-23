# 数据来源清单

当前进展以 guidance/DATA_COLLECTION_PROGRESS_2026-09-23.md 和 sources.json 为准。

- source_manifest.csv：当前文件位置、大小、SHA-256 和来源；空值表示未知。它包含隔离的外部响应文件的文件级信息，不包含响应标签。
- sources.json：来源状态与待办。当前 collection_complete=true、ready_for_training=true；当前训练入口须采用 DATA_CARD 所列修复后 cohort_hash。
- acquisition/：下载记录，包括失败。
- relocation_plan_2026-09-23.json、relocation_verified_2026-09-23.json：迁移前后路径与哈希核验。
- local_inventory_2026-09-23.csv、local_audit_2026-09-23.json：迁移前历史快照，旧路径不再用于读取当前数据。

raw 已按来源/版本组织；unverified 表示来源版本仍待核验。canonical 中带 candidate/raw/audit/conflict 的表不能当作已批准训练主表。原始 GDSC2 在 raw/GDSC/release8.5，历史派生 CSV 在 canonical/GDSC/release8.5。GDSC1/PRISM 响应在 holdout/external，不得进入开发训练、调参或早停。
