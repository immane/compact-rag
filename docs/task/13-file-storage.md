# 任务 13: 文件存储子系统

> **依赖**: 01-配置管理, 02-公共基础设施 | **优先级**: P0 | **预计工时**: 10h

## 目标

实现统一的文件存储抽象层，通过 `StorageBackend` 抽象接口 + 策略模式支持 Local / MinIO / OSS / Kodo / S3 多种后端，配置驱动一键切换。

## 产出文件

```
src/compact_rag/storage/
├── __init__.py
└── file_storage.py        # StorageBackend 抽象 + 各后端实现 + 工厂函数
```

## 详细需求

### 1. `StorageBackend` 抽象接口

```python
class StorageBackend(ABC):
    """统一的文件存储后端抽象接口 —— 策略模式核心"""

    @abstractmethod
    async def upload_file(self, local_path: str, remote_key: str) -> str:
        """上传文件，返回访问 URL"""

    @abstractmethod
    async def upload_bytes(self, data: bytes, remote_key: str,
                           content_type: str = "") -> str:
        """上传字节数据到存储后端，返回文件访问 URL"""

    @abstractmethod
    async def download_file(self, remote_key: str, local_path: str) -> str:
        """下载文件到本地路径，返回本地路径"""

    @abstractmethod
    async def download_bytes(self, remote_key: str) -> bytes:
        """读取文件为字节数据"""

    @abstractmethod
    async def delete(self, remote_key: str) -> bool:
        """删除文件，返回是否成功"""

    @abstractmethod
    async def list(self, prefix: str = "") -> list[str]:
        """列出指定前缀下的所有文件键值"""

    @abstractmethod
    async def get_url(self, remote_key: str, expires: int = 3600) -> str:
        """获取文件访问 URL（支持预签名 / CDN 加速）"""

    @abstractmethod
    async def exists(self, remote_key: str) -> bool:
        """检查文件是否存在"""
```

### 2. 后端实现

| 后端 | 实现类 | SDK | 推荐场景 |
|------|--------|-----|---------|
| 本地 | `LocalFileBackend` | 零依赖 | 开发测试、单机部署 |
| MinIO | `MinIOBackend` | `minio` | 开发测试（Docker），私有化部署 |
| 阿里云 OSS | `OSSBackend` | `oss2` | 中国大陆生产 |
| 七牛云 Kodo | `KodoBackend` | `qiniu` | 中国大陆生产（CDN 优先） |
| AWS S3 | `S3Backend` | `boto3` | 海外生产 |

**LocalFileBackend**:
- 路径策略: `{root_dir}/{remote_key}`
- `get_url()` 返回 `{base_url}/{remote_key}`

**MinIOBackend**:
- 自动创建 bucket
- `get_url()` 返回预签名 URL（支持过期时间）

### 3. 工厂函数

```python
@lru_cache()
def get_storage_backend(settings) -> StorageBackend:
    """根据配置获取存储后端实例（单例缓存）"""
    if settings.backend == "local":
        return LocalFileBackend(root_dir=settings.local.root_dir,
                                base_url=settings.local.base_url)
    elif settings.backend == "minio":
        return MinIOBackend(endpoint=settings.minio.endpoint, ...)
    elif settings.backend == "oss":
        return OSSBackend(endpoint=settings.oss.endpoint, ...)
    elif settings.backend == "kodo":
        return KodoBackend(access_key=...)
    elif settings.backend == "s3":
        return S3Backend(region=...)
    raise ValueError(f"Unknown storage backend: {settings.backend}")
```

### 4. 文件路径策略

```python
def build_storage_key(collection_id: str, filename: str,
                      category: str = "docs") -> str:
    """
    构建持久化存储路径:
    {category}/{collection_id}/{year}/{month}/{day}/{hash}{ext}
    例: docs/finance-2024/2026/05/24/a1b2c3d4e5f6.pdf
    """
    now = datetime.utcnow()
    date_path = f"{now.year}/{now.month:02d}/{now.day:02d}"
    file_hash = hashlib.sha256(filename.encode()).hexdigest()[:16]
    ext = Path(filename).suffix
    return f"{category}/{collection_id}/{date_path}/{file_hash}{ext}"
```

### 5. 临时文件清理

```python
class TempFileCleaner:
    """临时文件 TTL 自动清理器"""

    def __init__(self, backend: StorageBackend, ttl_hours: int = 1):
        self.backend = backend
        self.ttl = timedelta(hours=ttl_hours)

    async def clean_expired(self) -> int:
        """
        清理过期临时文件（temp/ 前缀）
        按路径中的日期时间戳判断是否过期
        返回清理的文件数量
        """
```

### 6. 配置示例 (`config/storage.yaml`)

```yaml
storage:
  backend: minio          # local | minio | oss | kodo | s3
  local:
    root_dir: ./data/storage
    base_url: http://localhost:8000/files
  minio:
    endpoint: localhost:9000
    access_key: ${MINIO_ACCESS_KEY}
    secret_key: ${MINIO_SECRET_KEY}
    bucket: compact-rag
    secure: false
  # oss, kodo, s3 配置略
```

### 7. 后端选择决策流

```
开发/测试 → MinIO (Docker) 或 LocalFile
中国大陆生产 → 有 CDN 需求? → 是 → 七牛云 Kodo
                          → 否 → 阿里云 OSS
海外生产 → AWS S3
私有化部署 → MinIO (K8s/Docker)
```

## 验收标准

- [ ] LocalFileBackend 完整 CRUD 操作正常
- [ ] MinIOBackend 连接 MinIO Docker 正常，预签名 URL 有效
- [ ] 工厂函数根据配置返回正确后端实例
- [ ] `build_storage_key` 路径格式正确，防碰撞
- [ ] `TempFileCleaner` 正确清理过期临时文件
- [ ] 后端不可用时抛出 `StorageBackendError`
- [ ] 文件不存在时 `download` 抛出 `FileNotFoundError`
