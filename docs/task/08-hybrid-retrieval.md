# 任务 07: 向量存储层 ChromaDB

> **依赖**: 06-向量化服务 | **优先级**: P0 | **预计工时**: 6h

## 目标

封装 ChromaDB 的 CRUD 操作，提供文档块的批量添加、向量检索、按文档删除、集合列表等接口。

## 产出文件

```
src/compact_rag/storage/
├── __init__.py
├── schema.py              # Pydantic 数据模型 (SearchResult, DocumentChunk)
└── vector_store.py        # ChromaDB CRUD 封装
```

## 详细需求

### 1. `schema.py` — 共享数据模型

```python
class SearchResult(BaseModel):
    id: str                  # chroma_id
    content: str             # chunk 文本
    score: float             # 相似度/距离分数
    metadata: dict           # ChromaDB metadata

class DocumentChunk(BaseModel):   # 向量库 + SQL 之间的桥梁
    content: str
    page_number: int | None
    chunk_index: int
    is_table: bool
    token_count: int
    content_hash: str
    metadata: dict
```

### 2. `vector_store.py` — VectorStore

```python
class VectorStore:
    def __init__(self, settings: ChromaDBSettings, embedding_service: EmbeddingService):
        """
        - 创建 PersistentClient(path=settings.persist_directory)
        - 获取或创建 Collection
        - 设置默认 embedding function 为 None（自行管理向量）
        """

    async def add_documents(self, chunks: list[DocumentChunk],
                            embeddings: np.ndarray) -> list[str]:
        """批量添加文档块到向量存储，返回 chroma_id 列表"""

    async def search(self, query: str, top_k: int = 10,
                     where: dict = None) -> list[SearchResult]:
        """
        向量相似度搜索
        1. embedding_service.encode_query(query)
        2. collection.query(query_embeddings=[vec], n_results=top_k,
                            include=["documents", "metadatas", "distances"])
        3. 转换为 SearchResult 列表
        """

    async def delete_by_document(self, doc_id: str):
        """按文档 ID 删除所有相关块 (where={"doc_id": doc_id})"""

    async def delete_by_ids(self, chroma_ids: list[str]):
        """按 chroma_id 列表删除"""

    async def list_collections(self) -> list[str]:
        """列出所有 ChromaDB Collection"""

    async def count(self, where: dict = None) -> int:
        """统计文档块数量"""

    def _ensure_collection(self):
        """确保 Collection 存在，不存在则创建"""
```

### 3. Metadata 结构

每个 chunk 在 ChromaDB 中的 metadata：

```json
{
    "doc_id": "uuid-of-document",
    "chroma_id": "auto-generated-chroma-id",
    "chunk_index": 0,
    "page_number": 3,
    "filename": "report.pdf",
    "collection_name": "finance-2024",
    "is_table": false,
    "token_count": 245
}
```

### 4. 与关系数据库同步

- 每次 `add_documents` 后返回 `chroma_id` 列表，调用方负责同步写入 `document_chunks` 表
- 删除文档时：先 `delete_by_document` 删除向量，再删除 SQL 记录
- `chroma_id` 是 ChromaDB 自动生成的 UUID，也是两套存储之间的关联键

### 5. ChromaDB 配置

```python
class ChromaDBSettings:
    persist_directory: str = "./data/chromadb"
    collection_name: str = "default"
```

## 验收标准

- [ ] `add_documents` 正确写入向量，返回 chroma_id 列表
- [ ] `search` 返回按相似度降序排列的结果
- [ ] `delete_by_document` 正确删除指定文档的所有块
- [ ] `count` 返回正确的文档块数量
- [ ] 不同 Collection 之间数据隔离
- [ ] ChromaDB 数据持久化，重启后数据不丢失
- [ ] 空查询返回空列表，不抛异常
