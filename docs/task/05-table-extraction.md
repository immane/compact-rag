# 任务 05: 表格提取子系统

> **依赖**: 02-公共基础设施 | **优先级**: P0 | **预计工时**: 8h

## 目标

从 PDF 和 HTML 文档中提取表格并转换为 Markdown 格式，采用分层后备策略保证最高成功率。

## 产出文件

```
src/compact_rag/ingestion/
└── table_extractor.py      # 表格提取 + Markdown 转换
```

## 详细需求

### 1. 提取策略

采用分层后备（Fallback）模式：

**PDF 文件处理流**：
```
PDF 文件 → 是否为扫描件？
  ├── 是 → PaddleOCR (按需启用，需 GPU)
  └── 否 → Camelot (Lattice 模式优先)
              ├── 成功 → 输出 Markdown
              └── 失败 → pdfplumber (后备方案)
```

**HTML/Word 文件处理流**：
- `markdownify` 直接处理 HTML 内嵌表格
- `Pandoc` 作为 Word 格式转换后备

### 2. 优先级定义

| 优先级 | 方案 | 适用场景 | 依赖 |
|--------|------|----------|------|
| P0 | Camelot + pdfplumber | 80%+ 数字 PDF 表格 | camelot-py, pdfplumber |
| P1 | markdownify | HTML 内嵌表格 | markdownify |
| P2 | Pandoc | Word/HTML 格式转换 | pandoc (系统级) |
| P3 | PaddleOCR | 扫描版 PDF（按需） | paddleocr (需 GPU) |

### 3. 核心实现

```python
class TableExtractor:
    """PDF 表格提取器"""

    async def extract_from_pdf(self, file_path: str) -> list[ExtractedTable]:
        """
        PDF 表格提取主入口
        1. 检测是否为扫描件（通过 pdfplumber 判断文本覆盖度）
        2. 优先使用 Camelot Lattice 模式
        3. 失败时回退到 pdfplumber
        4. 如为扫描件且启用 OCR → 调用 PaddleOCR
        """

    async def extract_from_html(self, html_content: str) -> list[ExtractedTable]:
        """HTML 内表格提取，使用 markdownify"""

    async def extract_from_docx(self, file_path: str) -> list[ExtractedTable]:
        """Word 文档表格提取"""

class ExtractedTable:
    page_number: int
    rows: int
    columns: int
    markdown: str          # Markdown 格式表格
    quality_score: float    # 0-1 质量评估分数
    method: str             # camelot / pdfplumber / markdownify / paddleocr
```

### 4. 质量评估函数

```python
def evaluate_table_quality(markdown_table: str) -> dict:
    """
    检查：
    - 行数 ≥ 2（至少表头 + 1行数据）
    - 分隔行存在有效（|---|...|）
    - 每行列数一致
    - 返回 {score: float, issues: list[str]}
    """
```

### 5. Markdown 输出格式示例

```
| 年份 | 营收 (亿元) | 净利润 (亿元) |
|------|------------|--------------|
| 2021 | 120.5      | 15.2         |
| 2022 | 145.8      | 18.7         |
| 2023 | 168.3      | 22.1         |
```

## 验收标准

- [ ] Camelot 可正确提取标准 PDF 数字表格
- [ ] Camelot 失败时自动回退到 pdfplumber
- [ ] 表格质量评分合理（无效表格 < 0.5，有效表格 > 0.8）
- [ ] HTML 表格通过 markdownify 正确转为 Markdown
- [ ] 各提取方案输出的表格列数一致
- [ ] 损坏的 PDF 文件不会导致崩溃，返回空列表 + 日志警告
