# 任务 08: 混合检索层

> **依赖**: 07-向量存储层 | **优先级**: P0 | **预计工时**: 10h

## 目标

实现混合检索系统：Dense 向量检索 + BM25 稀疏检索 + RRF 融合 + Cross-Encoder 重排序，并提供统一的 HybridRetriever 编排器。

## 产出文件

```
src/compact_rag/retrieval/
├── __init__.py
├── dense.py               # ChromaDB 向量检索
├── sparse.py              # BM25 关键词检索
├── fusion.py              # RRF / RSF 融合
├── reranker.py            # Cross-Encoder 重排序
├── query_transformer.py   # 查询改写（HyDE / 多查询扩展）
└── retriever.py           # HybridRetriever 编排器
```

## 详细需求

### 1. `dense.py` — Dense 检索

```python
class DenseRetriever:
    """
    封装 VectorStore.search() 为标准检索器接口
    负责将 query 转为向量并执行相似度搜索
    """
    def __init__(self, vector_store: VectorStore): ...

    async def search(self, query: str, top_k: int = 100,
                     collection: str = None) -> list[SearchResult]:
        """执行向量检索，返回 SearchResult 列表"""
```

### 2. `sparse.py` — BM25 检索

```python
class BM25Retriever:
    """
    基于 rank_bm25 的稀疏检索器
    内置中文分词（jieba）+ 英文分词支持
    """

    def __init__(self):
        self.bm25 = None
        self.documents: list[str] = []      # 纯文本列表
        self.doc_ids: list[str] = []        # 对应 chroma_id
        self._is_indexed = False

    def index(self, documents: list[str], doc_ids: list[str]):
        """
        构建 BM25 索引
        - 对每个文档执行 _tokenize()
        - 创建 BM25Okapi(tokenized_corpus, k1=1.5, b=0.75)
        """

    def search(self, query: str, top_k: int = 100) -> list[tuple[str, float]]:
        """返回 (doc_id, bm25_score) 列表"""

    def rebuild_index(self, documents: list[str], doc_ids: list[str]):
        """热重建索引（增量更新用）"""

    def _tokenize(self, text: str) -> list[str]:
        """中文用 jieba.lcut，英文用 split()"""
```

### 3. `fusion.py` — 融合算法

```python
def rrf_fusion(
    dense_results: list[SearchResult],
    sparse_results: list[SearchResult],
    k: int = 60,
    top_k: int = 50,
) -> list[SearchResult]:
    """
    Reciprocal Rank Fusion
    score(d) = Σ 1 / (k + rank_i(d))
    按融合分数降序排列，返回 top_k 个结果
    """

def rsf_fusion(
    dense_results: list[SearchResult],
    sparse_results: list[SearchResult],
    alpha: float = 0.5,
    top_k: int = 50,
) -> list[SearchResult]:
    """
    Relative Score Fusion
    score(d) = alpha * norm_dense_score + (1-alpha) * norm_sparse_score
    """
```

### 4. `reranker.py` — 重排序

```python
class RerankerService:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """加载 CrossEncoder 模型"""

    async def rerank(self, query: str,
                     candidates: list[SearchResult]) -> list[SearchResult]:
        """
        对候选结果用 Cross-Encoder 精细打分
        1. 构建 (query, candidate.content) 对
        2. model.predict(pairs)
        3. 按新分数降序重排搜索结果
        """

    @property
    def is_available(self) -> bool:
        """模型是否成功加载"""
```

| 模型 | CPU 延迟 | 推荐场景 |
|------|---------|---------|
| MiniLM-L-6-v2 | ~10ms | 轻量，默认 |
| MiniLM-L-12-v2 | ~20ms | 均衡 |
| bge-reranker-base | ~50ms | 中文优化 |

### 5. `query_transformer.py` — 查询改写（可选）

```python
class QueryTransformer:
    """
    查询改写策略（按需启用）
    - HyDE: 先让 LLM 生成假设答案，用假设答案向量检索
    - MultiQuery: 生成多个查询变体，合并检索结果
    """
    async def hyde_transform(self, query: str, llm_client) -> str: ...
    async def multi_query_expand(self, query: str, llm_client) -> list[str]: ...
```

### 6. `retriever.py` — 混合检索编排器

```python
class HybridRetriever:
    def __init__(self, vector_store, bm25_retriever, reranker,
                 settings: RetrievalSettings, query_transformer=None):
        ...

    async def retrieve(self, query: str, top_k: int = 10,
                       collection: str = None) -> list[SearchResult]:
        """
        完整检索流程:
        1. [可选] QueryTransformer 改写
        2. Dense 检索: vector_store.search(query, top_k=settings.dense_top_k)
        3. Sparse 检索: bm25.search(query, top_k=settings.sparse_top_k)
        4. Fusion: rrf_fusion(dense, sparse, top_k=settings.fusion_top_k)
        5. [可选] Rerank: reranker.rerank(query, fused)
        6. 返回 top_k
        """
```

### 7. 性能基准参考

| 配置 | 检索延迟 | 内存 | Recall@10 |
|------|---------|------|----------|
| BM25 only | 15ms | 120MB | 0.72 |
| Dense only (MiniLM+ONNX) | 8ms | 180MB | 0.81 |
| **Hybrid (RRF)** | 20ms | 220MB | **0.87** |
| **Hybrid + Cross-Encoder** | 35ms | 320MB | **0.91** |

*注：8 万条文档条件下的参考数据*

## 验收标准

- [ ] Dense 检索返回与 VectorStore 一致的结果
- [ ] BM25 索引构建和搜索正常，中文分词有效
- [ ] RRF 融合后结果排序合理（出现在两个列表中靠前的文档排名更高）
- [ ] Cross-Encoder 重排序后相关性提升
- [ ] `HybridRetriever.retrieve()` 端到端工作
- [ ] BM25 索引重建后旧数据被正确替换
- [ ] 降级模式：Reranker 不可用时跳过重排序
