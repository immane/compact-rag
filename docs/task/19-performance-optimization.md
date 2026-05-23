# 任务 19: 性能优化策略

> **依赖**: 06-向量化服务, 07-向量存储层, 08-混合检索层, 03-关系数据库 | **优先级**: P1 | **预计工时**: 持续

## 目标

在各模块中实施性能优化措施，确保系统在低资源环境下（CPU-only）高效运行。

## 优化策略总结

### 1. Embedding 优化

| 优化项 | 方法 | 效果 |
|--------|------|------|
| 批量编码 | `batch_size=64` | 2-3x 吞吐提升 |
| ONNX Runtime | `backend="onnx"` 或 `use_onnx=True` | 2-3x 推理加速 |
| int8 量化 | `(emb * 127).astype(np.int8)` | 4x 内存节省 |
| max_seq_length | 192 取代 512（对短文本查询） | 1.5-2x 加速 |
| Matryoshka 截断 | `truncate_dim=128`（需支持模型） | 2-3x 检索加速 |
| 模型缓存 | 单例模式，进程级缓存 | 避免重复加载（~500ms 节省） |

**实现要点**：

```python
class EmbeddingService:
    _instance = None  # 单例缓存

    def __new__(cls, settings: EmbeddingSettings):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    async def encode(self, texts: list[str]) -> np.ndarray:
        # 批量编码
        embeddings = await asyncio.to_thread(
            self.model.encode, texts,
            batch_size=self.settings.batch_size,
            normalize_embeddings=self.settings.normalize,
            show_progress_bar=False,
        )
        return embeddings
```

### 2. ChromaDB 优化

| 优化项 | 方法 |
|--------|------|
| 持久化路径 | 使用 SSD 存储 `persist_directory` |
| 批量写入 | `collection.add(documents=[...], embeddings=[...], ids=[...])` 一次调用 |
| 元数据索引 | 仅在过滤字段上建索引（doc_id, collection_name） |
| 定期清理 | 删除过期或不再使用的 collection |
| 距离函数 | 使用 `cosine` 而非 `l2`（与 normalize 配合） |

### 3. 关系数据库优化

| 优化项 | 方法 |
|--------|------|
| 连接池 | `pool_size=5`, `max_overflow=10` |
| 索引 | `documents.collection_id`, `documents.file_hash`, `documents.status`, `messages.conversation_id`, `document_chunks.document_id` |
| 批量插入 | `session.add_all([...])` 替代逐条 add |
| 分页查询 | `.limit(page_size).offset((page-1)*page_size)` |
| 避免 N+1 | 使用 `selectinload` / `joinedload` 预加载关联数据 |

### 4. 检索优化

| 优化项 | 方法 | 效果 |
|--------|------|------|
| Dense top_k 限制 | `dense_top_k=100`（召回池）再精排 | 平衡召回与延迟 |
| BM25 预分词 | 索引时存储分词结果 | 检索时免分词 |
| RRF k 值调优 | `k=60`（经典值，鲁棒性最优） | — |
| Cross-Encoder 候选数 | 仅对融合后的 Top-50 做重排 | 减少 Cross-Encoder 调用 |
| BM25 索引定期重建 | 每次新文档摄入后增量更新 | 保持索引新鲜度 |

**性能基准**（8 万条文档）：

| 配置 | 检索延迟 | 内存 | Recall@10 |
|------|---------|------|----------|
| BM25 only | 15ms | 120MB | 0.72 |
| Dense only (MiniLM+ONNX) | 8ms | 180MB | 0.81 |
| **Hybrid (RRF)** | 20ms | 220MB | **0.87** |
| **Hybrid + Cross-Encoder** | 35ms | 320MB | **0.91** |

### 5. API 优化

| 优化项 | 方法 | 实现 |
|--------|------|------|
| 流式响应 | SSE 分块传输 | `StreamingResponse` + `AsyncGenerator` |
| 请求限流 | 令牌桶算法 | `slowapi` + `@limiter.limit("100/minute")` |
| 响应缓存 | 相同 query 短期缓存 | 可选：`functools.lru_cache` 或 Redis |
| 并发处理 | `asyncio.gather` 并行执行 | Dense + Sparse 检索并行 |
| 连接复用 | httpx 连接池 | `httpx.AsyncClient` 单例 |

### 6. 文件存储优化

| 优化项 | 方法 |
|--------|------|
| CDN 加速 | 七牛云 Kodo 原生 CDN；阿里云 OSS + CDN 回源 |
| 大文件分片 | 断点续传（OSS: `resumable_upload`，S3: `multipart_threshold`） |
| 临时文件 TTL | `TempFileCleaner` 定时清理 `temp/` 目录 |
| 预签名 URL | 避免直接暴露存储地址，防止盗链 |
| 就近上传 | 选择离用户最近的云存储区域 |

### 7. 启动优化

```python
# 延迟加载 — 不使用的模块不加载
class LazyEmbeddingService:
    """仅在第一次 encode 调用时加载模型"""
    def __init__(self, settings):
        self.settings = settings
        self._model = None

    async def _ensure_model(self):
        if self._model is None:
            self._model = SentenceTransformer(self.settings.model_name)

# CLI lazy imports — 不在 import 时加载重量级模块
# main.py
@app.command()
def serve(...):
    from compact_rag.api.router import app  # 延迟导入
    uvicorn.run(app, ...)
```

## 验收标准

- [ ] Embedding 批量编码正确工作
- [ ] ONNX 模式下推理速度提升
- [ ] 数据库索引在 Alembic 迁移中创建
- [ ] Hybrid 检索端到端延迟 < 50ms（万级文档）
- [ ] API 限流功能生效
- [ ] SSE 流式响应延迟可接受
- [ ] 大文件上传支持分片（> 100MB）
- [ ] TempFileCleaner 定期清理临时文件
