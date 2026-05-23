# 任务 04: 文档摄入管道

> **依赖**: 03-关系数据库, 06-向量化服务 | **优先级**: P0 | **预计工时**: 10h

## 目标

实现多格式文档的完整摄入管道：文件加载 → 表格提取 → 分块 → 向量化 → 双写（ChromaDB + SQL）。

## 产出文件

```
src/compact_rag/ingestion/
├── __init__.py
├── loader.py              # 多格式文档加载器
├── chunker.py             # 分块策略实现
└── pipeline.py            # 摄入流程编排器
```

## 详细需求

### 1. `loader.py` — 文件加载器

使用抽象基类 + 工厂模式：

```python
class BaseLoader(ABC):
    @abstractmethod
    async def load(self, file_path: str) -> list[LoadedPage]:
        """加载并解析文件，返回页面列表"""
```

实现以下 Loader：

| Loader | 扩展名 | 依赖 | 输出 |
|--------|--------|------|------|
| `PDFLoader` | `.pdf` | pypdf | 按页提取纯文本 |
| `DOCXLoader` | `.docx` | python-docx | 按段落提取文本 |
| `TextLoader` | `.txt` | 零依赖 | 直接读取 |
| `MarkdownLoader` | `.md` | 零依赖 | 直接读取，保留结构 |
| `HTMLLoader` | `.html` | bs4 + markdownify | HTML → Markdown |

每个 `LoadedPage` 包含：
- `page_number`: int
- `content`: str
- `tables`: list[ExtractedTable]
- `metadata`: dict

`LoaderFactory.get_loader(file_path)` 根据扩展名返回对应 Loader。

**元数据提取**（所有 Loader 共同产出）：
- `filename`, `file_type`, `file_size`
- `page_count`（PDF/DOCX）
- `hash`：SHA256 去重用
- `table_count`：检测到的表格数量

### 2. `chunker.py` — 分块策略

三种策略，默认 `recursive`：

**策略 1: 递归字符分割 (RecursiveCharacterTextSplitter)**
- 分隔符优先级：`["\n\n", "\n", "。", ".", "，", ",", " ", ""]`
- 默认 `chunk_size=500`, `chunk_overlap=50`
- 适用于大多数文档

**策略 2: 语义分割 (SemanticChunker)**
- 基于 embedding 相似度阈值检测断点
- 相邻句子余弦相似度低于阈值时分段

**策略 3: 表格感知分割 (TableAwareChunker)**
- 检测 Markdown 表格边界（`|---|` 行）
- 保持表格整体完整性
- 表格前后各保留一行纯文本作为上下文
- 超大表格（>50行）拆分为：表头行 + 每30行一组数据行

输出 `DocumentChunk`：
```python
class DocumentChunk:
    content: str
    page_number: int | None
    chunk_index: int
    is_table: bool
    token_count: int
    content_hash: str
    metadata: dict
```

### 3. `pipeline.py` — 摄入流程编排

```python
class IngestionPipeline:
    def __init__(self, settings, loader_factory, chunker, embedding_service,
                 vector_store, doc_repo, chunk_repo, ingestion_repo,
                 storage_backend):
        ...

    async def ingest_file(self, file_path: str, collection_name: str,
                          force: bool = False) -> IngestionResult:
        """摄入单个文件，返回 IngestionResult"""

    async def ingest_directory(self, dir_path: str, collection_name: str) -> list[IngestionResult]:
        """批量摄入目录下所有支持的文件"""

    async def ingest_url(self, url: str, collection_name: str) -> IngestionResult:
        """从 URL 下载并摄入"""
```

完整流程（以 `ingest_file` 为例）：
1. 检查文件是否存在 + 格式支持
2. 计算 SHA256 哈希 → 如已存在且 `force=False`，跳过（返回 `skipped`）
3. 创建 `IngestionJob` (status=running)
4. 上传到 StorageBackend 临时区
5. 选择 Loader → 加载文档
6. 提取表格 → Markdown 转换（调用 05-table-extraction）
7. 统一文本流 → Chunking
8. Embedding 批量编码
9. 双写：
   - ChromaDB: `collection.add(embeddings, documents, metadatas, ids)`
   - SQL: INSERT `documents` + `document_chunks`（批量 `session.add_all`）
10. 原始文件持久化到 StorageBackend
11. 临时文件清理（TTL）
12. 更新 `IngestionJob` 为 completed/failed

**增量更新**：通过 SHA256 哈希实现，已存在且未修改的跳过。`force=True` 时强制重新摄入（先删除旧数据）。

### 4. IngestionResult

```python
class IngestionResult:
    doc_id: str
    filename: str
    status: Literal["completed", "skipped", "failed"]
    chunk_count: int
    table_count: int
    error_message: str | None
    duration_ms: float
```

## 验收标准

- [ ] 支持 `.pdf`, `.docx`, `.txt`, `.md`, `.html` 五种格式加载
- [ ] 增量摄入：相同文件不重复处理
- [ ] `force=True` 正确重新摄入
- [ ] 分块大小和重叠符合配置
- [ ] 表格感知分块不破坏表格结构
- [ ] 摄入完成后 ChromaDB 和 SQL 数据一致
- [ ] 摄入失败时 status=failed，error_message 有值
- [ ] 批量摄入 `ingest_directory` 支持嵌套子目录
- [ ] IngestionJob 正确跟踪进度（processed_files/total_files）
