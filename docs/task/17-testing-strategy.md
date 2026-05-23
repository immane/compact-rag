# 任务 17: 测试策略

> **依赖**: 全部模块 | **优先级**: P0 | **预计工时**: 持续

## 目标

建立完整的测试体系，覆盖单元测试、集成测试、API 测试和端到端测试，目标代码覆盖率 > 85%。

## 产出文件

```
tests/
├── conftest.py                    # 全局 pytest fixtures
├── fixtures/                      # 测试数据
│   ├── sample.pdf
│   ├── sample.docx
│   ├── sample.txt
│   └── sample_table.html
├── test_config/
│   └── test_settings.py
├── test_common/
│   ├── test_logger.py
│   └── test_exceptions.py
├── test_storage/
│   ├── test_db_models.py
│   ├── test_repositories.py
│   ├── test_vector_store.py
│   └── test_file_storage.py
├── test_ingestion/
│   ├── test_loader.py
│   ├── test_chunker.py
│   ├── test_table_extractor.py
│   └── test_pipeline.py
├── test_embedding/
│   └── test_service.py
├── test_retrieval/
│   ├── test_dense.py
│   ├── test_sparse.py
│   ├── test_fusion.py
│   ├── test_reranker.py
│   └── test_retriever.py
├── test_generation/
│   ├── test_llm.py
│   └── test_prompt.py
├── test_tool/
│   ├── test_schema.py
│   └── test_engine.py
├── test_rag/
│   └── test_pipeline.py
├── test_api/
│   ├── test_chat.py
│   ├── test_documents.py
│   ├── test_collections.py
│   ├── test_conversations.py
│   └── test_system.py
└── test_admin/
    ├── test_client.py
    └── test_pages.py
```

## 测试层级

| 层级 | 范围 | 框架 | 目标覆盖率 |
|------|------|------|-----------|
| 单元测试 | 单个函数/类 | pytest | 85%+ |
| 集成测试 | 模块间交互 | pytest + pytest-asyncio | 70%+ |
| API 测试 | HTTP 接口 | httpx + TestClient | 90%+ |
| 端到端测试 | 完整问答流程 | pytest | 关键路径 |

## 各模块测试重点

| 模块 | 测试重点 |
|------|---------|
| **Loader** | 各格式正确解析；元数据提取正确；损坏文件容错；空文件处理 |
| **Chunker** | 分块大小一致性；重叠正确性；表格完整性保留；边界条件（空文本、超长文本） |
| **Table Extractor** | Camelot/pdfplumber 后备逻辑；Markdown 输出正确性；质量评估函数 |
| **Embedding** | 向量维度正确；批量/单条一致性；ONNX 模式可用 |
| **VectorStore** | 写入读取一致；按元数据过滤；删除正确性；集合隔离 |
| **FileStorage** | LocalFile/MinIO CRUD 正确性；预签名 URL；工厂函数切换后端；TTL 清理 |
| **BM25** | 中文分词正确；排序合理性；空查询边缘情况 |
| **Fusion (RRF)** | 融合后排序合理性；参数 k 敏感度 |
| **Reranker** | 重排后精度提升；与融合结果兼容 |
| **LLM Client** | 各 provider 实例化；消息格式兼容；流式输出正确；超时处理 |
| **Tool Engine** | 参数解析；工具路由；错误恢复；重试逻辑 |
| **RAG Pipeline** | 端到端一致性；引用标注正确；对话历史持久化 |
| **API** | 请求校验；流式 SSE 正确；错误码规范；并发安全性 |
| **Admin** | API 客户端方法覆盖；页面渲染无异常；交互操作正确性 |

## 核心 Fixtures (`conftest.py`)

```python
@pytest.fixture
async def test_settings():
    """创建测试用 Settings（SQLite 内存 + 临时目录）"""

@pytest.fixture
async def test_db(test_settings):
    """创建临时 SQLite 数据库，测试后自动清理"""

@pytest.fixture
async def test_async_session(test_db):
    """提供异步 SQLAlchemy session"""

@pytest.fixture
async def test_chromadb():
    """创建临时 ChromaDB 实例（内存模式或临时目录）"""

@pytest.fixture
def test_documents():
    """标准测试文档集（PDF/TXT/MD 各若干）"""

@pytest.fixture
def mock_llm_client():
    """模拟 LLM 客户端，返回固定 ChatResponse"""

@pytest.fixture
def mock_embedding_service():
    """模拟 EmbeddingService，返回固定维度随机向量"""

@pytest.fixture
async def test_rag_pipeline(test_settings, test_chromadb, mock_llm_client):
    """组装完整的测试版 RAGPipeline"""

@pytest.fixture
async def test_client(test_rag_pipeline):
    """FastAPI TestClient with test dependencies overridden"""
```

## 运行测试

```bash
# 运行全部测试
pytest

# 按模块运行
pytest tests/test_retrieval/

# 带覆盖率报告
pytest --cov=src/compact_rag --cov-report=term-missing

# 按标记运行
pytest -m "integration"
pytest -m "slow"          # 慢速测试（LLM/Embedding 实际调用）

# pytest.ini 配置
# [pytest]
# markers =
#     slow: 慢速测试（需要实际 LLM/Embedding 服务）
#     integration: 集成测试
#     unit: 单元测试
```

## 验收标准

- [ ] 所有模块有对应测试文件
- [ ] 单元测试覆盖率 > 85%
- [ ] API 端点覆盖率 > 90%
- [ ] CI/CD pipeline 中 `pytest` 通过
- [ ] 端到端测试覆盖核心问答流程
- [ ] Mock 正确隔离外部依赖（LLM API、ChromaDB 等）
