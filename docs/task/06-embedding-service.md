# 任务 06: 向量化服务

> **依赖**: 01-配置管理 | **优先级**: P0 | **预计工时**: 4h

## 目标

封装 `sentence-transformers` 为统一的 EmbeddingService，支持批量编码、查询编码、ONNX 加速，并暴露维度信息。

## 产出文件

```
src/compact_rag/embedding/
├── __init__.py
└── service.py             # sentence-transformers 封装
```

## 详细需求

### 1. EmbeddingService

```python
class EmbeddingService:
    """sentence-transformers 封装服务"""

    def __init__(self, settings: EmbeddingSettings):
        """
        初始化：
        - 加载模型（单例模式，进程级缓存）
        - 如果 use_onnx=True，转换为 ONNX Runtime
        - 设置 max_seq_length
        """

    async def encode(self, texts: list[str]) -> np.ndarray:
        """批量编码文本为向量 (n, dimension)"""

    async def encode_query(self, query: str) -> np.ndarray:
        """编码单条查询 (dimension,)"""

    @property
    def dimension(self) -> int:
        """返回嵌入向量维度"""
```

### 2. 关键行为

- **模型加载**：使用 `SentenceTransformer(settings.model_name, device=settings.device)`
- **Batch 编码**：`model.encode(texts, batch_size=settings.batch_size, normalize_embeddings=settings.normalize, show_progress_bar=False)`
- **ONNX 加速**：`model.to_onnx()` 或 `SentenceTransformer(..., backend="onnx")`
- **max_seq_length**：`model.max_seq_length = settings.max_seq_length`
- **单例模式**：整个进程只加载一次模型，通过模块级缓存

### 3. 推荐的模型配置

| 场景 | 模型 | 维度 | 内存 | 推荐理由 |
|------|------|------|------|----------|
| 极低内存 | all-MiniLM-L6-v2 | 384 | ~90MB | 最小最快 |
| 均衡（默认） | BGE-small-zh-v1.5 | 384 | ~95MB | 中英文兼顾 |
| 高精度 | BGE-base-en-v1.5 | 768 | ~440MB | 精度最高 |

### 4. 加速方案（按需）

| 方案 | 配置 | 加速比 | 适用条件 |
|------|------|--------|----------|
| ONNX Runtime | `use_onnx=True` | 2-3x | 任何 CPU |
| OpenVINO | 安装 `optimum-intel` | 3x | Intel CPU |
| int8 量化 | 模型转换后保存 | 4x 内存节省 | 内存紧张 |
| Matryoshka 截断 | `truncate_dim=128` | 2-4x | 需支持模型 |

## 验收标准

- [ ] `encode(["hello", "world"])` 返回 shape `(2, dimension)` 的 numpy 数组
- [ ] `encode_query("test")` 返回 shape `(dimension,)` 的数组
- [ ] `dimension` 属性与模型声明一致
- [ ] ONNX 模式下编码结果与原始模型一致（误差 < 1e-4）
- [ ] 模型仅在首次实例化时加载（单例缓存验证）
- [ ] 空文本列表 `encode([])` 返回空数组 `(0, dimension)`
