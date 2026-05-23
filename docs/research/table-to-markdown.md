# 如何低成本把表格转成 Markdown

> 调研日期：2026-05-23
> 适用场景：RAG 系统中的表格数据提取与转换

---

## 一、问题背景

在构建 RAG（Retrieval-Augmented Generation）系统时，文档中的**表格数据**是一个非常重要的信息源。然而，绝大多数 RAG 系统主要处理纯文本，对表格的支持较弱。如果将表格内容直接以原始格式（PDF、HTML、Excel）喂给 LLM，会导致：

1. **信息丢失** —— 表格的结构化关系无法被 LLM 有效理解
2. **Token 浪费** —— 非文本格式占用大量 token
3. **检索效果差** —— Embedding 模型对表格的语义理解不足

将表格转换为 **Markdown 格式**是一种低成本的折中方案：

- Markdown 表格语法简单、可读性强
- 大多数 LLM 能很好地理解 Markdown 表格
- 占用 token 较少
- 方便 Embedding 模型处理

本文聚焦于**开源、低成本**的 Python 方案，重点解决从 **PDF**、**HTML**、**Excel**、**图片**等来源提取表格并转为 Markdown 的问题。

---

## 二、开源方案对比

### 方案 1：pdfplumber

| 项目 | 说明 |
|------|------|
| **GitHub** | [jsvine/pdfplumber](https://github.com/jsvine/pdfplumber) |
| **Stars** | ~5k+ |
| **License** | MIT |
| **安装** | `pip install pdfplumber` |
| **依赖** | pdfminer.six（纯 Python） |

**核心能力：**
- 精确提取 PDF 中的字符、矩形、线条等元素
- 基于线条和文字对齐进行表格检测
- 支持 `extract_table()` 和 `extract_tables()` 方法
- 提供可视化调试工具

**优缺点：**

| 优点 | 缺点 |
|------|------|
| 纯 Python，安装简单 | 对扫描版 PDF 无效（无 OCR） |
| 表格提取精度高（有线表格） | 对复杂合并单元格支持一般 |
| 可精细调整参数 | 处理速度较慢 |
| 可视化调试方便 | 不直接输出 Markdown（需自行转换） |

**适用场景：** 数字原生 PDF（非扫描），表格有明确边框线。

---

### 方案 2：Camelot

| 项目 | 说明 |
|------|------|
| **GitHub** | [camelot-dev/camelot](https://github.com/camelot-dev/camelot) |
| **Stars** | ~3.7k |
| **License** | MIT |
| **安装** | `pip install camelot-py` |
| **依赖** | Ghostscript、OpenCV、pdfminer |

**核心能力：**
- 专为 PDF 表格提取设计
- 两种模式：Lattice（有线表格）、Stream（无线表格）
- 直接输出 pandas DataFrame
- 内置精度评估指标（accuracy、whitespace）

**优缺点：**

| 优点 | 缺点 |
|------|------|
| 表格提取准确率较高 | 需安装 Ghostscript（额外依赖） |
| 支持 Lattice/Stream 两种模式 | 对扫描版 PDF 无效 |
| 提供 accuracy 指标，便于自动过滤 | 处理复杂版式有时不如 pdfplumber 灵活 |
| 支持导出 Markdown、CSV、Excel、JSON | 社区维护频率有所下降 |
| 内置与 Tabula/pdfplumber 的对比基准 | |

**适用场景：** 数字原生 PDF，尤其是有清晰边框的表格。

---

### 方案 3：tabula-py

| 项目 | 说明 |
|------|------|
| **GitHub** | [chezou/tabula-py](https://github.com/chezou/tabula-py) |
| **Stars** | ~2.2k |
| **License** | MIT |
| **安装** | `pip install tabula-py` |
| **依赖** | Java 8+（依赖 tabula-java） |

**核心能力：**
- 封装 Tabula Java 引擎
- 直接从 PDF 提取表格到 DataFrame
- 支持批量转换和远程 PDF

**优缺点：**

| 优点 | 缺点 |
|------|------|
| 提取质量稳定 | **必须安装 Java**，部署成本高 |
| 支持批量处理 | 处理速度较慢 |
| 社区成熟 | 对无线表格处理一般 |
| 可直接输出 CSV/JSON | 不如 Camelot 的 Lattice 模式准确 |

**适用场景：** 有 Java 环境的团队，处理简单有线表格。

---

### 方案 4：Pandoc

| 项目 | 说明 |
|------|------|
| **官网** | [pandoc.org](https://pandoc.org/) |
| **License** | GPL-2.0 |
| **安装** | `brew install pandoc` / `pip install pandoc` |
| **依赖** | Haskell（但以独立二进制分发） |

**核心能力：**
- 通用文档格式转换器
- 支持 HTML、LaTeX、Docx、PDF 等转 Markdown
- 对 HTML 表格转换效果优秀

**优缺点：**

| 优点 | 缺点 |
|------|------|
| 格式转换能力极强 | PDF 转 Markdown 需经过中间格式 |
| 对 HTML 表格转 Markdown 非常成熟 | 对 PDF 表格提取不是主业 |
| 命令行使用方便 | 复杂表格布局可能丢失信息 |
| 支持自定义 writer | |

**适用场景：** HTML 转 Markdown、中间格式转换、docx 转 markdown。

---

### 方案 5：markdownify（HTML 转 Markdown）

| 项目 | 说明 |
|------|------|
| **GitHub** | [matthewwithanm/python-markdownify](https://github.com/matthewwithanm/python-markdownify) |
| **Stars** | ~700+ |
| **License** | MIT |
| **安装** | `pip install markdownify` |

**核心能力：**
- 将 HTML 转换为 Markdown
- 支持 `<table>` 标签转换
- 高度可定制（strip/convert 标签过滤）

**优缺点：**

| 优点 | 缺点 |
|------|------|
| 轻量、安装简单 | 本身不支持 PDF |
| 对标准 HTML 表格转换效果好 | 对复杂表格处理能力有限 |
| 可自定义转换规则 | 需要先将其他格式转为 HTML |
| 活跃维护 | |

**适用场景：** HTML 内容的 Markdown 化。

---

### 方案 6：OCR 方案（用于扫描版 PDF）

| 工具 | 说明 | License |
|------|------|---------|
| **PaddleOCR** | 百度开源 OCR，支持表格识别 | Apache-2.0 |
| **Tesseract** | Google 开源 OCR | Apache-2.0 |
| **PaddleOCR table** | 专门识别表格结构 | Apache-2.0 |
| **Surya OCR** | 新兴 OCR，支持表格 | MIT |

**核心思路（扫描版 PDF）：**
1. 先用 OCR 识别图像中的文字
2. 再用表格结构识别模型恢复表格
3. 最后转为 Markdown

**PaddleOCR 表格识别示例：**
```python
from paddleocr import PPStructure

engine = PPStructure(show_log=True)
result = engine("scanned_table.png")
# result 中包含表格的 HTML 结构
```

**优缺点：**

| 优点 | 缺点 |
|------|------|
| 可处理扫描版/图片中的表格 | **需要 GPU**（CPU 也可但极慢） |
| PaddleOCR 中文支持好 | 部署包较大 |
| 能识别复杂表格结构 | 免费但需要一定计算资源 |

---

## 三、方案总结对比

### 综合对比矩阵

| 方案 | 输入源 | 输出格式 | 精度 | 速度 | 安装复杂度 | 成本 | 适合场景 |
|------|--------|----------|------|------|-----------|------|---------|
| **pdfplumber** | 数字 PDF | List of lists | ⭐⭐⭐⭐ | ⭐⭐⭐ | 低（纯Python） | 免费 | 有线表格、数字PDF |
| **Camelot** | 数字 PDF | DataFrame | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | 中（需Ghostscript） | 免费 | PDF 专用表格提取 |
| **tabula-py** | 数字 PDF | DataFrame | ⭐⭐⭐⭐ | ⭐⭐ | 高（需Java） | 免费 | 有 Java 环境时 |
| **Pandoc** | HTML/Docx | Markdown | ⭐⭐⭐⭐⭐(HTML) | ⭐⭐⭐⭐ | 低 | 免费 | 格式转换 |
| **markdownify** | HTML | Markdown | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 极低 | 免费 | HTML→Markdown |
| **PaddleOCR** | 图片/扫描PDF | HTML/MD | ⭐⭐⭐⭐ | ⭐⭐ | 高（含DL模型） | 免费(需算力) | 扫描文档 |
| **Unstructured** | 多格式 | Markdown | ⭐⭐⭐ | ⭐⭐⭐ | 中 | 免费 | RAG 一站式 |

### 精度 vs 成本分析

```
精度
  ↑
  │   Camelot
  │   pdfplumber   *  PaddleOCR (GPU)
  │        
  │   tabula-py
  │   Pandoc (HTML)
  │   markdownify
  │                    Tesseract OCR
  │
  └──────────────────────────────→ 成本（时间/算力/部署）
     低                 高
```

> 对于 RAG 系统，**最推荐的方案是 Camelot + pdfplumber 的组合：**
> 1. 优先使用 Camelot（Lattice 模式）提取有线表格，精度高
> 2. Camelot 失败时 fallback 到 pdfplumber
> 3. 最终统一输出为 Markdown 格式

---

## 四、代码示例

### 4.1 使用 pdfplumber 提取表格并转为 Markdown

```python
import pdfplumber

def pdf_table_to_markdown(pdf_path, page_number=0):
    """
    从 PDF 中提取表格并转为 Markdown 格式
    """
    with pdfplumber.open(pdf_path) as pdf:
        page = pdf.pages[page_number]
        tables = page.extract_tables()

        if not tables:
            return ""

        markdown_output = ""
        for table_idx, table in enumerate(tables):
            if not table:
                continue

            # 构建 Markdown 表格
            rows = []
            for row_idx, row in enumerate(table):
                # 处理 None 和空值
                cleaned_row = [cell if cell else "" for cell in row]
                rows.append(cleaned_row)

            if not rows:
                continue

            # 表头
            header = rows[0]
            markdown_output += "| " + " | ".join(header) + " |\n"
            # 分隔行
            markdown_output += "| " + " | ".join(["---"] * len(header)) + " |\n"
            # 数据行
            for data_row in rows[1:]:
                markdown_output += "| " + " | ".join(data_row) + " |\n"

            markdown_output += "\n"

        return markdown_output

# 使用
md = pdf_table_to_markdown("document.pdf")
print(md)
```

### 4.2 使用 Camelot 提取并转为 Markdown（推荐）

```python
import camelot
import pandas as pd

def camelot_to_markdown(pdf_path, flavor="lattice"):
    """
    使用 Camelot 提取表格并输出 Markdown
    flavor: "lattice"（有线表格）或 "stream"（无线表格）
    """
    tables = camelot.read_pdf(pdf_path, flavor=flavor, pages="all")

    if tables.n == 0:
        print("未检测到表格")
        return ""

    output = ""
    for i, table in enumerate(tables):
        df: pd.DataFrame = table.df

        # 转为 Markdown
        # 表头
        header = list(df.columns)
        output += f"### 表格 {i+1}\n\n"
        output += "| " + " | ".join(header) + " |\n"
        output += "| " + " | ".join(["---"] * len(header)) + " |\n"

        # 数据行
        for _, row in df.iterrows():
            output += "| " + " | ".join(str(cell) for cell in row) + " |\n"

        output += "\n"

    return output

# 使用
md = camelot_to_markdown("document.pdf", flavor="lattice")
print(md)

# Camelot 直接导出 Markdown（v1.0+ 支持）
tables = camelot.read_pdf("document.pdf")
tables[0].to_markdown("output.md")
```

### 4.3 使用 Pandoc（HTML/Word 转 Markdown）

```python
import subprocess

def pandoc_convert_to_markdown(input_path, output_path="output.md"):
    """
    使用 Pandoc 将文档转为 Markdown
    支持: .html, .docx, .epub, .tex 等
    """
    cmd = [
        "pandoc",
        input_path,
        "-f", "html",          # 输入格式，根据实际文件调整
        "-t", "markdown",
        "--wrap", "preserve",
        "-o", output_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"错误: {result.stderr}")
        return ""
    return output_path

# 使用
pandoc_convert_to_markdown("document.html")
```

### 4.4 使用 markdownify（轻量 HTML 转 Markdown）

```python
from markdownify import markdownify as md

html_table = """
<table>
  <tr>
    <th>姓名</th>
    <th>年龄</th>
    <th>城市</th>
  </tr>
  <tr>
    <td>张三</td>
    <td>28</td>
    <td>北京</td>
  </tr>
  <tr>
    <td>李四</td>
    <td>32</td>
    <td>上海</td>
  </tr>
</table>
"""

markdown = md(html_table, heading_style="ATX")
print(markdown)
```

### 4.5 一站式方案：Unstructured Library

```python
from unstructured.partition.pdf import partition_pdf
from unstructured.partition.html import partition_html
from unstructured.staging.base import convert_to_markdown

# 从 PDF 提取元素（包括表格）
elements = partition_pdf("document.pdf", strategy="auto")

# 转为 Markdown
markdown_output = ""
for element in elements:
    if element.category == "Table":
        # 表格元素会以 HTML 格式包含
        markdown_output += element.metadata.text_as_html + "\n\n"
    else:
        markdown_output += str(element) + "\n\n"

print(markdown_output)
```

### 4.6 扫描版 PDF 表格提取（PaddleOCR）

```python
# 安装: pip install paddlepaddle paddleocr
from paddleocr import PPStructure
import json

def scanned_pdf_table_to_markdown(image_path):
    """
    从扫描版 PDF 或图片中提取表格
    注意：扫描 PDF 需要先转为图片（每页一张）
    """
    engine = PPStructure(show_log=False)
    result = engine(image_path)

    markdown_tables = []
    for item in result:
        if item['type'] == 'table':
            html = item['res']['html']
            # 可以再用 markdownify 将 HTML 转为 Markdown
            from markdownify import markdownify as md
            md_table = md(html)
            markdown_tables.append(md_table)

    return "\n\n".join(markdown_tables)

# 使用
md = scanned_pdf_table_to_markdown("scanned_page.png")
print(md)
```

---

## 五、在 RAG 系统中的集成建议

### 5.1 推荐架构

```
                      ┌──────────────────────┐
                      │    文档输入           │
                      │ (PDF/HTML/Excel/图片) │
                      └──────────┬───────────┘
                                 │
                                 ▼
                      ┌──────────────────────┐
                      │  文档类型检测          │
                      └──────────┬───────────┘
                                 │
              ┌──────────────────┼──────────────────┐
              ▼                  ▼                  ▼
      ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
      │  数字 PDF    │  │  HTML/Word   │  │ 扫描PDF/图片 │
      └──────┬───────┘  └──────┬───────┘  └──────┬───────┘
             │                 │                  │
             ▼                 ▼                  ▼
      ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
      │ Camelot/     │  │  Pandoc/     │  │  PaddleOCR   │
      │ pdfplumber   │  │  markdownify │  │  (PPStructure)│
      └──────┬───────┘  └──────┬───────┘  └──────┬───────┘
             │                 │                  │
             └─────────────────┼──────────────────┘
                               │
                               ▼
                     ┌──────────────────┐
                     │  统一 Markdown    │
                     │  表格输出          │
                     └────────┬─────────┘
                              │
                              ▼
                     ┌──────────────────┐
                     │  Chunk 处理      │
                     │  保留表格结构     │
                     └────────┬─────────┘
                              │
                              ▼
                     ┌──────────────────┐
                     │  Embedding →     │
                     │  Vector DB       │
                     └──────────────────┘
```

### 5.2 分块策略建议

表格在 RAG 中的分块处理有两种方式：

**方式 A：表格整体作为一个 Chunk**
- 适用于小型表格（< 50 行）
- 保留完整上下文
- Chunk 较大时需要注意 token 限制

**方式 B：表格分块 + 上下文保留**
```python
def table_chunking(markdown_table, max_chunk_size=1000):
    """将大表格按行分组，每组合并为一个 Chunk"""
    lines = markdown_table.strip().split("\n")
    if len(lines) < 3:
        return [markdown_table]

    # 前两行是表头和分隔行
    header = lines[0] + "\n" + lines[1] + "\n"
    data_lines = lines[2:]

    chunks = []
    current_chunk = header
    for line in data_lines:
        if len(current_chunk) + len(line) + 1 > max_chunk_size:
            chunks.append(current_chunk.strip())
            current_chunk = header + line + "\n"
        else:
            current_chunk += line + "\n"

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks
```

### 5.3 Prompt 设计建议

在 RAG 系统的 Prompt 中，建议包含以下表格处理指引：

```text
当文档中包含 Markdown 格式的表格时：
- 保留表格的原始结构
- 使用 Markdown 表格语法（| 和 ---）
- 表头放在第一行
- 表格前后各有一个空行
- 如果表格过大，可以分块但保留表头信息
```

### 5.4 质量评估指标

在 RAG Pipeline 中加入表格提取质量的自动评估：

```python
def evaluate_table_quality(markdown_table):
    """评估表格质量"""
    lines = markdown_table.strip().split("\n")
    if len(lines) < 3:
        return {"valid": False, "reason": "行数不足"}

    # 检查分隔行
    separator = lines[1]
    if not all(c in "|- " for c in separator):
        return {"valid": False, "reason": "缺少有效的分隔行"}

    # 检查列数一致性
    col_counts = [len(line.split("|")) for line in lines]
    if len(set(col_counts)) > 1:
        return {
            "valid": False,
            "reason": "列数不一致",
            "detail": f"列数分布: {col_counts}"
        }

    return {"valid": True, "rows": len(lines) - 2, "cols": col_counts[0] - 2}
```

### 5.5 成本估算

| 方案 | 单页处理时间 | 内存消耗 | GPU 需求 | 部署难度 | 适合规模 |
|------|-------------|---------|---------|---------|---------|
| pdfplumber | 0.5~2s | 低 | 无 | 低 | 大规模 |
| Camelot | 1~3s | 中 | 无 | 中 | 中大规模 |
| tabula-py | 2~5s | 中 | 无 | 高 | 中小规模 |
| Pandoc (HTML) | <0.1s | 低 | 无 | 低 | 大规模 |
| markdownify | <0.01s | 极低 | 无 | 极低 | 任意规模 |
| PaddleOCR | 5~20s | 高 | 推荐有 | 高 | 小规模 |

### 5.6 最终推荐

对于 `compact-rag` 项目的**低成本表格转 Markdown** 方案，优先级如下：

| 优先级 | 方案 | 理由 |
|--------|------|------|
| **P0** | **Camelot + pdfplumber 组合** | 覆盖 80%+ 的数字 PDF 表格场景，免费、开源、精度高 |
| **P1** | **markdownify** | 处理 HTML 表格，极轻量 |
| **P2** | **Pandoc** | 处理 Word/HTML 格式转换 |
| **P3** | **PaddleOCR** | 仅在遇到扫描版 PDF 时启用，按需使用 |

---

## 六、参考资料

1. pdfplumber 官方文档: https://github.com/jsvine/pdfplumber
2. Camelot 官方文档: https://camelot-py.readthedocs.io/
3. tabula-py 官方文档: https://github.com/chezou/tabula-py
4. Pandoc 官网: https://pandoc.org/
5. markdownify: https://github.com/matthewwithanm/python-markdownify
6. PaddleOCR: https://github.com/PaddlePaddle/PaddleOCR
7. Camelot vs 其他工具对比: https://github.com/camelot-dev/camelot/wiki/Comparison-with-other-PDF-Table-Extraction-libraries-and-tools
8. Unstructured.io: https://unstructured.io/
9. RAG 最佳实践: https://docs.llamaindex.ai/en/stable/
