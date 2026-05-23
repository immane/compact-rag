# compact-rag 任务索引

> 基于设计文档 v1.2 解耦生成 | 日期: 2026-05-24
>
> **执行方案**: [多 Agent 实施执行方案](../processes/implementation.md) — 14 个 Agent × 7 个 Phase 的完整执行计划

## 任务列表

| 编号 | 任务 | 依赖 | 优先级 | 预计工时 |
|------|------|------|--------|----------|
| 01 | [配置管理](./01-config-management.md) | 无 | P0 | 4h |
| 02 | [公共基础设施](./02-common-infrastructure.md) | 01 | P0 | 3h |
| 03 | [关系型数据库层](./03-relational-database.md) | 01, 02 | P0 | 12h |
| 04 | [文档摄入管道](./04-document-ingestion.md) | 03, 06 | P0 | 10h |
| 05 | [表格提取子系统](./05-table-extraction.md) | 02 | P0 | 8h |
| 06 | [向量化服务](./06-embedding-service.md) | 01 | P0 | 4h |
| 07 | [向量存储层 ChromaDB](./07-vector-store.md) | 06 | P0 | 6h |
| 08 | [混合检索层](./08-hybrid-retrieval.md) | 07 | P0 | 10h |
| 09 | [生成层 LLM 抽象](./09-llm-generation.md) | 01, 02 | P0 | 6h |
| 10 | [Tool Calling 子系统](./10-tool-calling.md) | 02 | P1 | 8h |
| 11 | [RAG 管线编排](./11-rag-pipeline.md) | 08, 09, 10, 03 | P0 | 8h |
| 12 | [API 层](./12-api-layer.md) | 11, 03, 12 | P0 | 12h |
| 13 | [文件存储子系统](./13-file-storage.md) | 01, 02 | P0 | 10h |
| 14 | [Streamlit 管理后台](./14-streamlit-admin.md) | 12, 13 | P1 | 16h |
| 15 | [数据库设计](./15-database-design.md) | 03 | P0 | — |
| 16 | [错误处理与异常体系](./16-error-handling.md) | 02 | P0 | 4h |
| 17 | [测试策略](./17-testing-strategy.md) | 全部 | P0 | 持续 |
| 18 | [部署方案](./18-deployment.md) | 全部 | P2 | 4h |
| 19 | [性能优化策略](./19-performance-optimization.md) | 06, 07, 08, 03 | P1 | 持续 |

## 依赖图

```
01-config ──┬── 02-common ──┬── 03-database ──┬── 04-ingestion
            │               │                 ├── 11-pipeline
            │               │                 └── 12-api
            │               ├── 05-table-extraction
            │               ├── 09-llm ──────── 11-pipeline
            │               ├── 10-tool ─────── 11-pipeline
            │               ├── 13-file-storage
            │               └── 16-errors
            │
            ├── 06-embedding ── 07-vector-store ── 08-retrieval ──┬── 11-pipeline
            │                                                     └── 12-api
            ├── 09-llm
            └── 13-file-storage

14-admin ──── requires: 12-api, 13-file-storage
15-database ─ requires: 03-database
17-testing ─  requires: all
18-deploy ─── requires: all
19-perf ───── requires: 06, 07, 08, 03
```

## 实施分期

| Phase | 任务 | 目标 |
|-------|------|------|
| **Phase 1** | 01, 02, 03, 13, 12(部分) | 可运行的空服务 |
| **Phase 2** | 04, 05, 06, 07 | 文档摄入能力 |
| **Phase 3** | 08, 09 | 检索 + 生成 |
| **Phase 4** | 11, 12(完整) | RAG 管线 + 对话 |
| **Phase 5** | 10, 16 | Tool Calling + 鲁棒性 |
| **Phase 6** | 14 | Streamlit 管理后台 |
| **Phase 7** | 17, 18, 19 | 测试 + 部署 + 优化 |
