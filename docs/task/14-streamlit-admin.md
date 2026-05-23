# 任务 14: Streamlit 管理后台

> **依赖**: 12-API 层, 13-文件存储子系统 | **优先级**: P1 | **预计工时**: 16h

## 目标

实现基于 Streamlit 的可视化管理后台，通过内部 HTTP 客户端复用 REST API，提供仪表盘、集合管理、文档管理、摄入监控、对话浏览、RAG 问答台、API 密钥管理、文件存储浏览共 8 个页面。

## 产出文件

```
src/compact_rag/admin/
├── __init__.py
├── app.py                 # Streamlit 主入口 + 页面路由
├── client.py              # 内部 HTTP 客户端，封装所有 API 调用
├── config.py              # Admin 配置读取（API base URL, 密码等）
├── pages/
│   ├── __init__.py
│   ├── dashboard.py       # 系统概览仪表盘
│   ├── collections.py     # 集合管理 CRUD
│   ├── documents.py       # 文档上传/列表/详情/删除
│   ├── ingestion.py       # 摄入任务监控 + 进度
│   ├── conversations.py   # 对话历史浏览 + 导出
│   ├── playground.py      # RAG 问答调试台
│   ├── api_keys.py        # API Key 管理
│   └── storage.py         # 文件存储浏览 + 清理
└── components/
    ├── __init__.py
    ├── stats.py           # 统计卡片组件
    ├── status.py          # 状态徽章组件
    └── charts.py          # 图表可视化组件（plotly）
```

## 详细需求

### 1. 设计原则

| 原则 | 说明 |
|------|------|
| **零前端依赖** | 100% Python + Streamlit |
| **复用 API (dogfooding)** | 所有操作通过 `AdminAPIClient` 调用 `/v1/*` |
| **只读优先** | 管理操作需显式确认，避免误删 |
| **分页支持** | 列表页面支持分页和搜索 |
| **配置驱动** | API Base URL 通过配置传入 |
| **可选部署** | Streamlit 独立进程，不影响主 API |

### 2. `client.py` — 内部 HTTP 客户端

```python
class AdminAPIClient:
    """
    封装所有 API 调用，供 Streamlit 页面使用
    使用 httpx.Client (同步) 因为 Streamlit 是同步框架
    """
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip("/")
        self.client = httpx.Client(timeout=30.0)

    # 系统
    def health(self) -> dict: ...
    def info(self) -> dict: ...

    # 集合
    def list_collections(self, page=1, page_size=20) -> dict: ...
    def create_collection(self, **kwargs) -> dict: ...
    def delete_collection(self, name: str) -> dict: ...

    # 文档
    def list_documents(self, collection=None, status=None,
                       page=1, page_size=20) -> dict: ...
    def upload_document(self, file_data: bytes, filename: str,
                        collection: str) -> dict: ...
    def get_document(self, doc_id: str) -> dict: ...
    def delete_document(self, doc_id: str) -> dict: ...

    # 摄入
    def list_ingestion_jobs(self, status=None, collection=None) -> dict: ...
    def get_ingestion_job(self, job_id: str) -> dict: ...

    # 对话
    def list_conversations(self, page=1, page_size=20) -> dict: ...
    def get_conversation(self, conv_id: str) -> dict: ...
    def delete_conversation(self, conv_id: str) -> dict: ...

    # 问答
    def chat(self, messages: list[dict], collection=None,
             stream=False, **kwargs) -> dict: ...
    def chat_stream(self, messages: list[dict], **kwargs) -> httpx.Response: ...

    # API Keys
    def list_api_keys(self) -> dict: ...
    def create_api_key(self, name: str, permissions: list[str]) -> dict: ...
    def toggle_api_key(self, key_id: str, active: bool) -> dict: ...
    def delete_api_key(self, key_id: str) -> dict: ...

    # 存储
    def list_storage_files(self, **filters) -> dict: ...
    def get_file_url(self, storage_key: str) -> str: ...
    def delete_storage_file(self, storage_key: str) -> dict: ...
```

### 3. `app.py` — 主入口

```python
def main():
    st.set_page_config(
        page_title="Compact-RAG Admin", page_icon="🔍",
        layout="wide", initial_sidebar_state="expanded")

    with st.sidebar:
        st.title("🔍 Compact-RAG")
        st.caption("企业 RAG 系统管理后台")

        # API 连接状态
        api_status = check_api_health()
        if api_status:
            st.success("🟢 API 已连接")
        else:
            st.error("🔴 API 不可达"); st.stop()

        # API Base URL 配置
        api_base = st.text_input("API Base URL", value=get_api_base_url())

        # 页面导航
        page = st.radio("导航", [
            "🏠 仪表盘", "📁 集合管理", "📄 文档管理",
            "⚙️ 摄入监控", "💬 对话浏览", "🧪 RAG 问答台",
            "🔑 API 密钥", "📦 文件存储",
        ], label_visibility="collapsed")

    # 路由分发
    pages = {
        "🏠 仪表盘": render_dashboard,
        "📁 集合管理": render_collections,
        "📄 文档管理": render_documents,
        "⚙️ 摄入监控": render_ingestion,
        "💬 对话浏览": render_conversations,
        "🧪 RAG 问答台": render_playground,
        "🔑 API 密钥": render_api_keys,
        "📦 文件存储": render_storage,
    }
    pages[page](client=get_or_create_client(api_base))
```

### 4. 各页面功能摘要

| 页面 | 关键功能 |
|------|---------|
| **仪表盘** | 4 列统计卡片（文档/集合/对话/存储）；服务健康状态面板；最近摄入任务列表；系统配置快照 |
| **集合管理** | CRUD 表格 + 分页；创建表单（名称、描述、embedding 模型、chunk 参数）；删除二次确认 |
| **文档管理** | 文件上传（拖拽 + 选择集合）；文档列表（集合/状态过滤）；文档详情（元数据 + chunks 预览）；删除 |
| **摄入监控** | 任务列表（状态色标）；实时进度条（processed/total）；错误详情展开；临时文件管理 |
| **对话浏览** | 对话列表（时间/集合/模型过滤）；对话消息详情（含引用来源）；JSON/CSV 导出 |
| **RAG 问答台** | 集合选择器；检索参数实时调整；多轮聊天界面；流式输出开关；引用来源展示 |
| **API 密钥** | 密钥 CRUD 表格；创建表单（名称 + 权限）；激活/停用开关 |
| **文件存储** | 文件列表（类型/后端/集合过滤）；存储用量统计；文件预览下载；临时文件清理 |

### 5. RAG 问答台 (Playground) 详细设计

```
┌──────────────────────────────────────────────┐
│  [侧边栏]                    │  [聊天区域]     │
│                              │                │
│  检索参数:                   │  User: 你好     │
│  集合: [select dropdown]    │  Assistant: ... │
│  Top-K: [1-20 slider=5]    │    📎 引用来源   │
│  ☑ Cross-Encoder 重排序     │                │
│  ☑ 混合检索                 │  ────────────── │
│                              │                │
│  LLM 参数:                  │  输入框: ___    │
│  Temperature: [0.0-1.0]    │                │
│  ☐ 流式输出                 │                │
│                              │                │
│  [🔄 重置对话]              │                │
└──────────────────────────────────────────────┘
```

### 6. 组件 (`components/`)

```python
# stats.py
def render_stat_card(title: str, value, delta: str = None, icon: str = None):
    """渲染统计卡片"""

# status.py
def render_status_badge(status: str):
    """渲染状态徽章 (success/warning/error/info)"""

# charts.py
def render_bar_chart(data, x, y, title: str):
    """渲染柱状图 (使用 plotly)"""
```

### 7. 启动方式

```bash
# 安装依赖
pip install -e ".[admin]"

# 启动（先启动 API，再启动 Admin）
streamlit run src/compact_rag/admin/app.py \
    --server.port 8501 \
    --server.address 127.0.0.1

# 或通过 CLI
compact-rag admin       # 内部启动 Streamlit 子进程
```

### 8. 安全措施

| 措施 | 说明 |
|------|------|
| 默认仅本地访问 | `--server.address 127.0.0.1` |
| 可选认证密码 | `ADMIN_PASSWORD` 环境变量 |
| API 操作审计 | 操作记录在服务端日志 |
| 网络隔离 | 生产环境放在内网或 VPN 后 |

## 验收标准

- [ ] 所有 8 个页面可正常导航和渲染
- [ ] AdminAPIClient 与本地 API 服务通信正常
- [ ] 仪表盘统计数据正确
- [ ] RAG 问答台可实现完整的检索→回答流程
- [ ] 文档上传端到端工作（上传 → 解析 → 可检索）
- [ ] 对话导出功能正常（JSON/CSV 下载）
- [ ] 删除操作均有二次确认
- [ ] API 不可达时显示错误提示而非崩溃
- [ ] `compact-rag admin` CLI 命令可一键启动
