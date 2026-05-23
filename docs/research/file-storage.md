# 企业级文件存储方案调研 — 本地存储与云存储

> **调研日期**: 2026-05-23  
> **调研目的**: 为 compact-rag RAG 系统选择最优的文件存储后端  
> **适用场景**: 文档摄入临时存储、分块后持久化存储、CDN 加速分发

---

## 1. 背景与需求分析

### 1.1 RAG 系统中文件存储的典型需求

| 需求维度 | 说明 |
|---------|------|
| **文档摄入** | 上传 PDF/Word/Markdown/图片等原始文件，经解析后用于向量化 |
| **分块持久化** | 解析后的文本块、元数据需要存储，可能关联原始文件路径 |
| **图片/附件引用** | 对话中引用图片时需提供可访问的 URL |
| **临时文件** | 上传过程中产生的临时文件需定期清理 |
| **模型微调数据** | 收集的用户反馈数据需归档存储 |
| **多租户隔离** | 企业场景下不同租户的数据隔离 |

### 1.2 文件生命周期管理

```
上传 -> [临时存储] -> 解析完成 -> [持久化存储] -> 归档/删除
                        |
                 向量数据库
```

- **临时阶段**: 上传到解析完成，通常存活 几分钟 ~ 几小时
- **持久化阶段**: 解析完成后的原始文件、分析结果，存活 天 ~ 月
- **归档阶段**: 不再频繁访问的数据，存活 月 ~ 年甚至永久

### 1.3 选择存储方案的考量维度

| 维度 | 说明 |
|------|------|
| **成本** | 存储单价 + 流量费 + API 调用费 + 运维人力 |
| **速度** | 上传/下载延迟，内网 vs 外网，是否支持 CDN |
| **可靠性** | 数据冗余（多副本/纠删码），SLA 保障 |
| **合规** | 数据驻留要求（国内场景需国内云），加密支持 |
| **可扩展性** | 能否支持 PB 级存储，是否需要扩容 |
| **生态兼容** | 是否兼容 S3 API，是否有成熟的 Python SDK |
| **易用性** | 部署复杂度，运维成本 |

---

## 2. 本地存储方案

### 2.1 MinIO -- S3 兼容自托管对象存储

MinIO 是高性能、S3 兼容的对象存储系统，完全开源，适合私有化部署。

#### 安装部署（Docker）

```bash
docker run -p 9000:9000 -p 9001:9001 \
  -e MINIO_ROOT_USER=admin \
  -e MINIO_ROOT_PASSWORD=password \
  quay.io/minio/minio server /data --console-address ":9001"
```

#### Python SDK 安装

```bash
pip install minio
```

#### MinIO Python CRUD 示例

```python
from minio import Minio
from minio.error import S3Error
import io
from datetime import timedelta

# ============ 初始化客户端 ============
client = Minio(
    endpoint="localhost:9000",
    access_key="admin",
    secret_key="password",
    secure=False,
)

bucket_name = "rag-files"

# 确保 bucket 存在
found = client.bucket_exists(bucket_name)
if not found:
    client.make_bucket(bucket_name)

# ============ 上传文件 ============
result = client.fput_object(
    bucket_name=bucket_name,
    object_name="docs/2026/report.pdf",
    file_path="/tmp/report.pdf",
    content_type="application/pdf",
)
print(f"Uploaded: {result.object_name}")

# ============ 上传字节流 ============
data = b"Hello, RAG System!"
result = client.put_object(
    bucket_name=bucket_name,
    object_name="docs/hello.txt",
    data=io.BytesIO(data),
    length=len(data),
    content_type="text/plain",
)

# ============ 下载文件 ============
client.fget_object(
    bucket_name=bucket_name,
    object_name="docs/2026/report.pdf",
    file_path="/tmp/downloaded_report.pdf",
)

# ============ 下载为字节流 ============
response = client.get_object(
    bucket_name=bucket_name,
    object_name="docs/hello.txt",
)
content = response.read()
response.close()
response.release_conn()

# ============ 列出对象 ============
objects = client.list_objects(
    bucket_name=bucket_name,
    prefix="docs/",
    recursive=True,
)
for obj in objects:
    print(f"  {obj.object_name}  ({obj.size} bytes)")

# ============ 删除对象 ============
client.remove_object(
    bucket_name=bucket_name,
    object_name="docs/hello.txt",
)

# ============ 生成预签名 URL ============
url = client.presigned_get_object(
    bucket_name=bucket_name,
    object_name="docs/2026/report.pdf",
    expires=timedelta(hours=1),
)
print(f"临时访问 URL: {url}")

# ============ 生成预签名上传 URL ============
upload_url = client.presigned_put_object(
    bucket_name=bucket_name,
    object_name="uploads/new_file.pdf",
    expires=timedelta(hours=2),
)
print(f"预签名上传 URL: {upload_url}")
```

#### 优势与局限

| 优势 | 局限 |
|------|------|
| 完全免费，S3 API 兼容 | 需自行运维服务器 |
| 高性能，支持纠删码 | 小规模部署成本可能高于云存储 |
| 支持 Docker/K8s 部署 | 外网访问需配置反向代理 |
| 内置 Web 管理界面 | 无 CDN 加速能力（需自行配置） |
| 支持桶策略、生命周期管理 | 数据备份需自行管理 |

---

### 2.2 本地文件系统

最简单的方案，直接利用操作系统的文件系统进行读写。

```python
import os
import shutil
from pathlib import Path

STORAGE_ROOT = Path("/data/rag-storage")

class LocalFileStorage:
    """本地文件系统存储实现"""

    def upload(self, local_path: str, storage_key: str) -> str:
        dest = STORAGE_ROOT / storage_key
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(local_path, dest)
        return str(dest)

    def download(self, storage_key: str, local_path: str) -> str:
        src = STORAGE_ROOT / storage_key
        shutil.copy2(src, local_path)
        return local_path

    def read_bytes(self, storage_key: str) -> bytes:
        with open(STORAGE_ROOT / storage_key, "rb") as f:
            return f.read()

    def delete(self, storage_key: str) -> bool:
        path = STORAGE_ROOT / storage_key
        if path.is_file():
            path.unlink()
            return True
        elif path.is_dir():
            shutil.rmtree(path)
            return True
        return False

    def list_files(self, prefix: str = "") -> list[str]:
        base = STORAGE_ROOT / prefix
        if not base.exists():
            return []
        return [
            str(p.relative_to(STORAGE_ROOT))
            for p in base.rglob("*")
            if p.is_file()
        ]
```

#### 路径策略建议

```
/data/rag-storage/
  +-- temp/                  # 临时文件（TTL 清理）
  |   +-- {session_id}/
  +-- docs/                  # 持久化文档
  |   +-- {collection_id}/
  |       +-- {year}/{month}/{day}/
  |           +-- {hash}.pdf
  +-- images/                # 图片
  |   +-- {hash}.png
  +-- exports/               # 导出/归档
```

#### 优势与局限

| 优势 | 局限 |
|------|------|
| 零依赖，实现简单 | 无内置冗余/备份 |
| 延迟极低（本地磁盘） | 扩展困难（单机瓶颈） |
| 无需网络开销 | 无访问控制/认证 |
| 适合开发测试 | 生产环境不推荐 |

---

### 2.3 NAS 集成（NFS/SMB 挂载）

对于已有 NAS 设备的企业，可将 NAS 挂载到应用服务器上使用。

```bash
# 挂载 NFS
mount -t nfs nas-server:/volume1/rag-storage /mnt/rag-storage

# 挂载 SMB/CIFS
mount -t cifs //nas-server/rag-storage /mnt/rag-storage \
  -o username=admin,password=***
```

| 考量点 | 说明 |
|--------|------|
| **延迟** | 相比本地磁盘高，取决于网络 |
| **可靠性** | 取决于 NAS 设备的 RAID 配置 |
| **并发** | NFS 有锁机制，高并发可能瓶颈 |
| **推荐场景** | 中小团队，已有 NAS 设备 |

---

### 2.4 SeaweedFS / Ceph -- 分布式存储

#### SeaweedFS

轻量级分布式文件系统，适合存储大量小文件。

```
特点:
- 专为小文件优化（图片、文档）
- 支持 S3 API 兼容
- 支持 FUSE 挂载
- 部署简单，资源占用低
- Python 可通过 S3 API 或 HTTP API 访问
```

#### Ceph

企业级分布式存储系统，功能全面。

```
特点:
- 支持对象存储(RGW)、块存储(RBD)、文件系统(CephFS)
- S3/Swift API 兼容
- 自我修复、无单点故障
- 部署较复杂，适合大规模集群
- Python 可通过 boto3 (S3) 或 librados 访问
```

#### 对比

| 特性 | SeaweedFS | Ceph | MinIO |
|------|-----------|------|-------|
| 部署复杂度 | 低 | 高 | 低 |
| 小文件性能 | 优 | 一般 | 良 |
| 大文件性能 | 良 | 优 | 优 |
| S3 兼容 | 是 | 是 | 是 |
| 资源占用 | 低 | 高 | 中 |
| 生产推荐度 | 特定场景 | 大规模 | 通用 |

---

### 2.5 本地方案对比表

| 方案 | 部署难度 | 成本 | 可靠性 | 性能 | S3 API | Python SDK | 推荐场景 |
|------|---------|------|--------|------|--------|-----------|---------|
| **本地文件系统** | 低 | 免费 | 低 | 极高 | 否 | 无需 | 开发测试 |
| **MinIO** | 中 | 免费(自运维) | 高 | 高 | 是 | minio-py | 生产首选(私有化) |
| **NAS 挂载** | 中 | 硬件成本 | 中 | 中 | 否 | 无需 | 中小团队 |
| **SeaweedFS** | 中高 | 免费 | 高 | 高 | 是 | boto3 兼容 | 海量小文件 |
| **Ceph** | 高 | 免费 | 很高 | 高 | 是 | boto3 兼容 | 大规模集群 |

---

## 3. 国内云存储方案

### 3.1 七牛云 Kodo

七牛云是国内领先的对象存储服务商之一，以 CDN 加速见长。

#### SDK 安装

```bash
pip install qiniu
```

#### 认证方式

```python
from qiniu import Auth

access_key = "your-access-key"
secret_key = "your-secret-key"
q = Auth(access_key, secret_key)
```

#### 核心 API 示例

```python
from qiniu import Auth, put_file, put_data, BucketManager
from qiniu import build_batch_delete

# ============ 初始化 ============
access_key = "your-access-key"
secret_key = "your-secret-key"
bucket_name = "rag-bucket"
q = Auth(access_key, secret_key)

# ============ 上传文件 ============
token = q.upload_token(bucket_name, key=None, expires=3600)

# 方式一：上传本地文件
ret, info = put_file(token, "docs/report.pdf", "/tmp/report.pdf")
print(f"Upload Hash: {ret.get('hash')}, Key: {ret.get('key')}")

# 方式二：上传字节流
ret, info = put_data(token, "docs/hello.txt", b"Hello RAG!")
print(f"Upload result: {ret}")

# ============ 下载文件（公开空间）============
base_url = "http://your-domain.bkt.clouddn.com"
public_url = f"{base_url}/docs/report.pdf"

# ============ 生成私有空间下载链接 ============
private_url = q.private_download_url(
    url=f"http://your-domain.bkt.clouddn.com/docs/report.pdf",
    expires=3600,
)
print(f"私有下载地址: {private_url}")

# ============ 管理操作 ============
bucket = BucketManager(q)

# 列出文件
ret, eof, info = bucket.list(bucket_name, prefix="docs/", limit=100)
for item in ret.get("items", []):
    print(f"  {item['key']}  ({item['fsize']} bytes)")

# 删除文件
ret, info = bucket.delete(bucket_name, "docs/hello.txt")

# 批量删除
keys = ["docs/a.txt", "docs/b.txt"]
ops = build_batch_delete(bucket_name, keys)
ret, info = bucket.batch(ops)

# 获取文件信息
ret, info = bucket.stat(bucket_name, "docs/report.pdf")
if ret:
    print(f"File size: {ret['fsize']}, MIME: {ret['mimeType']}")
```

#### 生成外链（公开资源）

```python
domain = "your-bucket.s3.cn-east-1.qiniucs.com"
file_url = f"https://{domain}/docs/report.pdf"
```

#### 价格信息（中国大陆，华东-浙江2）

| 计费项 | 价格 |
|--------|------|
| **标准存储** | 0.115 元/GB/月 |
| **低频存储** | 0.075 元/GB/月 |
| **外网流出流量** (0-100TB) | 0.26 元/GB |
| **CDN 回源流量** | 0.15 元/GB |
| **PUT/DELETE 请求** | 0.01 元/万次 |
| **GET 请求** | 0.01 元/万次 |

**免费额度**: 10GB 存储 + 10GB CDN 回源流量 + 10万次写请求 + 100万次读请求 / 月

#### CDN 加速配置

```
七牛云在对象存储层面深度集成 CDN:
1. Bucket 默认提供 CDN 加速域名
2. 支持绑定自定义 CDN 域名
3. CDN 回源流量 0.15 元/GB
4. 适合 RAG 图片/文档分发场景
```

---

### 3.2 阿里云 OSS

阿里云对象存储（OSS），国内市场份额领先，功能最全面。

#### SDK 安装

```bash
pip install oss2
```

#### 认证方式

```python
import oss2

# 方式一：永久密钥认证
auth = oss2.Auth("your-access-key-id", "your-access-key-secret")

# 方式二：STS 临时授权（推荐生产使用）
auth = oss2.StsAuth(
    access_key_id="tmp-access-key-id",
    access_key_secret="tmp-access-key-secret",
    security_token="tmp-security-token",
)
```

#### 核心 API 示例

```python
import oss2
from oss2 import Bucket, ObjectIterator

# ============ 初始化 ============
auth = oss2.Auth("your-access-key-id", "your-access-key-secret")
endpoint = "oss-cn-hangzhou.aliyuncs.com"
bucket_name = "rag-bucket"
bucket = Bucket(auth, endpoint, bucket_name)

# ============ 上传文件 ============
bucket.put_object_from_file("docs/report.pdf", "/tmp/report.pdf")

# 上传字节流
result = bucket.put_object("docs/hello.txt", b"Hello RAG!")
print(f"ETag: {result.etag}")

# 大文件断点续传
oss2.resumable_upload(
    bucket, "docs/large-file.zip", "/tmp/large-file.zip",
    multipart_threshold=10*1024*1024,
    num_threads=4,
)

# ============ 下载文件 ============
bucket.get_object_to_file("docs/report.pdf", "/tmp/downloaded.pdf")

# 流式读取
result = bucket.get_object("docs/hello.txt")
content = result.read()

# ============ 列出文件 ============
for obj in ObjectIterator(bucket, prefix="docs/"):
    print(f"  {obj.key}  ({obj.size} bytes, {obj.last_modified})")

# ============ 删除文件 ============
bucket.delete_object("docs/hello.txt")

# 批量删除
result = bucket.batch_delete_objects(["docs/a.txt", "docs/b.txt"])
print(f"Deleted: {result.deleted_keys}")

# ============ 生成预签名 URL ============
# 下载预签名 URL (有效期 1 小时)
download_url = bucket.sign_url(
    method="GET",
    key="docs/report.pdf",
    expires=3600,
)
print(f"下载链接: {download_url}")

# 上传预签名 URL
upload_url = bucket.sign_url(
    method="PUT",
    key="uploads/new_file.pdf",
    expires=3600,
)
print(f"上传链接: {upload_url}")
```

#### STS 临时授权（生产推荐）

```python
from aliyunsdkcore.client import AcsClient
from aliyunsdksts.request.v20150401 import AssumeRoleRequest

client = AcsClient("your-access-key-id", "your-access-key-secret", "cn-hangzhou")
request = AssumeRoleRequest.AssumeRoleRequest()
request.set_RoleArn("acs:ram::123456789:role/rag-oss-role")
request.set_RoleSessionName("rag-session")
response = client.do_action_with_exception(request)

auth = oss2.StsAuth(
    access_key_id=response["Credentials"]["AccessKeyId"],
    access_key_secret=response["Credentials"]["AccessKeySecret"],
    security_token=response["Credentials"]["SecurityToken"],
)
```

#### 价格信息（中国大陆）

| 计费项 | 价格 |
|--------|------|
| **标准存储** (本地冗余) | 0.12 元/GB/月 |
| **低频存储** | 0.08 元/GB/月 |
| **归档存储** | 0.033 元/GB/月 |
| **外网流出流量** (忙时) | 0.50 元/GB |
| **外网流出流量** (闲时) | 0.25 元/GB |
| **CDN 回源流量** | 0.15 元/GB |
| **PUT 请求** (>500万次/月) | 0.01 元/万次 |
| **GET 请求** (>2000万次/月) | 0.01 元/万次 |

**免费额度**: 5GB 存储 + 5GB 下行流量 + 100 万次 GET 请求 / 月

---

### 3.3 腾讯云 COS

腾讯云对象存储（COS），深度整合微信/小程序生态。

#### SDK 安装

```bash
pip install -U cos-python-sdk-v5
```

#### 认证方式

```python
from qcloud_cos import CosConfig, CosS3Client
import os

config = CosConfig(
    Region="ap-beijing",
    SecretId=os.environ.get("COS_SECRET_ID"),
    SecretKey=os.environ.get("COS_SECRET_KEY"),
    Token=None,
    Scheme="https",
)
client = CosS3Client(config)
```

#### 核心 API 示例

```python
from qcloud_cos import CosConfig, CosS3Client
import os

# ============ 初始化 ============
config = CosConfig(
    Region="ap-beijing",
    SecretId=os.environ.get("COS_SECRET_ID"),
    SecretKey=os.environ.get("COS_SECRET_KEY"),
    Scheme="https",
)
client = CosS3Client(config)
bucket = "rag-bucket-1250000000"  # BucketName-AppID

# ============ 上传文件 ============
with open("/tmp/report.pdf", "rb") as fp:
    response = client.put_object(
        Bucket=bucket,
        Body=fp,
        Key="docs/report.pdf",
    )
print(response["ETag"])

# 高级上传（自动分片，推荐）
response = client.upload_file(
    Bucket=bucket,
    LocalFilePath="/tmp/large-file.zip",
    Key="docs/large-file.zip",
    PartSize=10,
    MAXThread=10,
)

# 上传字节流
response = client.put_object(
    Bucket=bucket,
    Body=b"Hello RAG!",
    Key="docs/hello.txt",
)

# ============ 下载文件 ============
response = client.get_object(
    Bucket=bucket,
    Key="docs/report.pdf",
)
response["Body"].get_stream_to_file("/tmp/downloaded.pdf")

# 流式读取
response = client.get_object(Bucket=bucket, Key="docs/hello.txt")
content = response["Body"].get_raw_stream().read()

# ============ 列出文件 ============
response = client.list_objects(
    Bucket=bucket,
    Prefix="docs/",
)
if "Contents" in response:
    for obj in response["Contents"]:
        print(f"  {obj['Key']}  ({obj['Size']} bytes)")

# ============ 删除文件 ============
client.delete_object(Bucket=bucket, Key="docs/hello.txt")

# 批量删除
client.delete_objects(
    Bucket=bucket,
    Delete={
        "Object": [{"Key": "docs/a.txt"}, {"Key": "docs/b.txt"}],
        "Quiet": "true",
    },
)

# ============ 生成预签名 URL ============
url = client.generate_presigned_url(
    Bucket=bucket,
    Method="GET",
    Key="docs/report.pdf",
    Expired=3600,
)
print(f"预签名 URL: {url}")
```

#### 价格信息（中国大陆）

| 计费项 | 价格 |
|--------|------|
| **标准存储** | 0.118 元/GB/月 |
| **低频存储** | 0.08 元/GB/月 |
| **归档存储** | 0.033 元/GB/月 |
| **外网流出流量** (0-10TB) | 0.50 元/GB |
| **CDN 回源流量** | 0.15 元/GB |
| **PUT 请求** | 0.01 元/万次 |
| **GET 请求** | 0.01 元/万次 |

**免费额度**: 50GB 存储 + 10GB 外网下行流量 + 200 万次 GET 请求 / 月

---

### 3.4 华为云 OBS / 百度云 BOS

#### 华为云 OBS

```bash
pip install esdk-obs-python
```

```python
from obs import ObsClient

obs_client = ObsClient(
    access_key_id="your-access-key",
    secret_access_key="your-secret-key",
    server="https://obs.cn-north-4.myhuaweicloud.com",
)

obs_client.putFile("rag-bucket", "docs/report.pdf", "/tmp/report.pdf")
obs_client.getObject("rag-bucket", "docs/report.pdf", "/tmp/downloaded.pdf")
obs_client.deleteObject("rag-bucket", "docs/report.pdf")
```

#### 百度云 BOS

```bash
pip install baidubce
```

```python
from baidubce.services.bos.bos_client import BosClient
from baidubce.bce_client_configuration import BceClientConfiguration

config = BceClientConfiguration(
    credentials=("your-access-key", "your-secret-key"),
    endpoint="https://bj.bcebos.com",
)
client = BosClient(config)

client.put_object_from_file("rag-bucket", "docs/report.pdf", "/tmp/report.pdf")
client.get_object_to_file("rag-bucket", "docs/report.pdf", "/tmp/downloaded.pdf")
```

### 3.5 国内云存储方案对比表

| 特性 | 七牛云 Kodo | 阿里云 OSS | 腾讯云 COS | 华为云 OBS | 百度云 BOS |
|------|------------|-----------|-----------|-----------|-----------|
| **标准存储** (元/GB/月) | 0.115 | 0.12 | 0.118 | 0.099 | 0.119 |
| **外网流量** (元/GB) | 0.26 | 0.50 | 0.50 | 0.50 | 0.49 |
| **CDN 回源流量** (元/GB) | 0.15 | 0.15 | 0.15 | 0.15 | 0.15 |
| **PUT 请求** (元/万次) | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 |
| **GET 请求** (元/万次) | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 |
| **免费额度** | 10GB | 5GB | 50GB | 5GB | 5GB |
| **CDN 加速** | 深度集成 | 支持 | 支持 | 支持 | 支持 |
| **STS 临时授权** | 支持 | 支持 | 支持 | 支持 | 支持 |
| **Python SDK** | qiniu | oss2 | cos-python-sdk-v5 | esdk-obs-python | baidubce |
| **SDK 成熟度** | 高 | 极高 | 高 | 中 | 中 |
| **S3 兼容** | 支持 | 不支持 | 不支持 | 支持 | 支持 |
| **可用区** | 华东/华北/华南/北美等 | 全球多区域 | 全球多区域 | 全球多区域 | 国内+香港 |

> **注**: 价格来自各官网 2026 年 5 月数据，实际以官网为准。七牛云的外网流量费最低，且 CDN 集成最成熟。

---

## 4. 国际云存储方案

### 4.1 AWS S3 (boto3)

全球最成熟的对象存储服务，S3 API 成为行业标准。

#### SDK 安装

```bash
pip install boto3
```

#### 基本使用

```python
import boto3
from botocore.config import Config
from boto3.s3.transfer import TransferConfig

# ============ 初始化 ============
s3_client = boto3.client(
    "s3",
    region_name="us-east-1",
    aws_access_key_id="your-access-key",
    aws_secret_access_key="your-secret-key",
    config=Config(
        retries={"max_attempts": 3, "mode": "adaptive"},
        connect_timeout=10,
    ),
)

bucket_name = "rag-bucket"

# ============ 上传文件 ============
# 上传字节流
s3_client.put_object(
    Bucket=bucket_name,
    Key="docs/hello.txt",
    Body=b"Hello RAG!",
    ContentType="text/plain",
)

# 上传文件
s3_client.upload_file(
    Filename="/tmp/report.pdf",
    Bucket=bucket_name,
    Key="docs/report.pdf",
)

# 大文件上传（自动分片）
s3_client.upload_file(
    Filename="/tmp/large-file.zip",
    Bucket=bucket_name,
    Key="docs/large-file.zip",
    ExtraArgs={"ContentType": "application/zip"},
    Config=TransferConfig(
        multipart_threshold=10 * 1024 * 1024,
        max_concurrency=10,
    ),
)

# ============ 下载文件 ============
s3_client.download_file(
    Bucket=bucket_name,
    Key="docs/report.pdf",
    Filename="/tmp/downloaded.pdf",
)

# 流式读取
response = s3_client.get_object(Bucket=bucket_name, Key="docs/hello.txt")
content = response["Body"].read()

# ============ 列出文件 ============
response = s3_client.list_objects_v2(
    Bucket=bucket_name,
    Prefix="docs/",
)
if "Contents" in response:
    for obj in response["Contents"]:
        print(f"  {obj['Key']}  ({obj['Size']} bytes)")

# ============ 删除文件 ============
s3_client.delete_object(Bucket=bucket_name, Key="docs/hello.txt")

# ============ 生成预签名 URL ============
download_url = s3_client.generate_presigned_url(
    ClientMethod="get_object",
    Params={"Bucket": bucket_name, "Key": "docs/report.pdf"},
    ExpiresIn=3600,
)
print(f"下载链接: {download_url}")

upload_url = s3_client.generate_presigned_url(
    ClientMethod="put_object",
    Params={
        "Bucket": bucket_name,
        "Key": "uploads/new_file.pdf",
        "ContentType": "application/pdf",
    },
    ExpiresIn=3600,
)
print(f"上传链接: {upload_url}")
```

### 4.2 Google Cloud Storage (GCS)

```bash
pip install google-cloud-storage
```

```python
from google.cloud import storage

client = storage.Client.from_service_account_json("service-account.json")
bucket = client.bucket("rag-bucket")

blob = bucket.blob("docs/report.pdf")
blob.upload_from_filename("/tmp/report.pdf")
blob.download_to_filename("/tmp/downloaded.pdf")

url = blob.generate_signed_url(
    version="v4",
    expiration=3600,
    method="GET",
)
```

### 4.3 Azure Blob Storage

```bash
pip install azure-storage-blob
```

```python
from azure.storage.blob import BlobServiceClient
from datetime import datetime, timedelta

conn_str = "DefaultEndpointsProtocol=https;AccountName=...;AccountKey=...;"
service_client = BlobServiceClient.from_connection_string(conn_str)
container_client = service_client.get_container_client("rag-container")

with open("/tmp/report.pdf", "rb") as data:
    container_client.upload_blob(name="docs/report.pdf", data=data)

with open("/tmp/downloaded.pdf", "wb") as file:
    blob_client = container_client.get_blob_client("docs/report.pdf")
    file.write(blob_client.download_blob().readall())

blob_client = container_client.get_blob_client("docs/report.pdf")
sas_url = blob_client.generate_signed_url(
    permission="r",
    expiry=datetime.utcnow() + timedelta(hours=1),
)
```

### 4.4 国际 vs 国内方案功能对比

| 对比维度 | AWS S3 | GCS | Azure Blob | 阿里云 OSS | 七牛云 Kodo |
|---------|--------|-----|------------|-----------|------------|
| **全球节点** | 最多 | 多 | 多 | 多 | 较少 |
| **国内访问速度** | 慢(绕路) | 慢 | 慢 | 快 | 快 |
| **国内合规** | 需北京区域 | 受限 | 世纪互联 | 完全合规 | 完全合规 |
| **标准存储价格** | $0.023/GB | $0.020/GB | $0.018/GB | Y0.12/GB | Y0.115/GB |
| **S3 API 兼容** | 原生 | 兼容 | 兼容 | 不兼容 | 兼容 |
| **Python SDK** | boto3 | google-cloud-storage | azure-storage-blob | oss2 | qiniu |
| **RAG 推荐度** | 海外首选 | 海外备选 | 海外备选 | 国内首选 | 国内优选 |

> **重要提示**: 若目标用户在中国大陆，强烈建议使用国内云存储。海外云存储在国内访问延迟高，且可能存在合规风险。

---

## 5. 统一抽象层设计

### 5.1 StorageBackend 接口设计

```python
from abc import ABC, abstractmethod
from typing import List


class StorageBackend(ABC):
    """统一的存储后端抽象接口"""

    @abstractmethod
    def upload_file(self, local_path: str, remote_key: str) -> str:
        """上传文件到存储后端。返回文件的访问 URL。"""
        ...

    @abstractmethod
    def upload_bytes(self, data: bytes, remote_key: str,
                     content_type: str = "") -> str:
        """上传字节数据到存储后端。返回文件的访问 URL。"""
        ...

    @abstractmethod
    def download_file(self, remote_key: str, local_path: str) -> str:
        """从存储后端下载文件到本地。返回本地路径。"""
        ...

    @abstractmethod
    def download_bytes(self, remote_key: str) -> bytes:
        """从存储后端读取文件内容为字节。"""
        ...

    @abstractmethod
    def delete(self, remote_key: str) -> bool:
        """删除存储后端上的文件。"""
        ...

    @abstractmethod
    def list(self, prefix: str = "") -> List[str]:
        """列出指定前缀下的所有文件键值。"""
        ...

    @abstractmethod
    def get_url(self, remote_key: str, expires: int = 3600) -> str:
        """获取文件的可访问 URL（支持预签名）。"""
        ...

    @abstractmethod
    def exists(self, remote_key: str) -> bool:
        """检查文件是否存在。"""
        ...
```

### 5.2 各后端实现示例

#### LocalFileBackend

```python
class LocalFileBackend(StorageBackend):
    """本地文件系统实现"""

    def __init__(self, root_dir: str = "./storage", base_url: str = ""):
        import shutil
        self.shutil = shutil
        self.root = Path(root_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        self.base_url = base_url

    def upload_file(self, local_path: str, remote_key: str) -> str:
        dest = self.root / remote_key
        dest.parent.mkdir(parents=True, exist_ok=True)
        self.shutil.copy2(local_path, dest)
        return f"{self.base_url}/{remote_key}"

    def upload_bytes(self, data: bytes, remote_key: str,
                     content_type: str = "") -> str:
        dest = self.root / remote_key
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        return f"{self.base_url}/{remote_key}"

    def download_file(self, remote_key: str, local_path: str) -> str:
        src = self.root / remote_key
        self.shutil.copy2(src, local_path)
        return local_path

    def download_bytes(self, remote_key: str) -> bytes:
        return (self.root / remote_key).read_bytes()

    def delete(self, remote_key: str) -> bool:
        path = self.root / remote_key
        if path.exists():
            path.unlink()
            return True
        return False

    def list(self, prefix: str = "") -> List[str]:
        base = self.root / prefix
        if not base.exists():
            return []
        return [str(p.relative_to(self.root))
                for p in base.rglob("*") if p.is_file()]

    def get_url(self, remote_key: str, expires: int = 3600) -> str:
        return f"{self.base_url}/{remote_key}"

    def exists(self, remote_key: str) -> bool:
        return (self.root / remote_key).exists()
```

#### MinIOBackend

```python
class MinIOBackend(StorageBackend):
    """MinIO S3 兼容对象存储实现"""

    def __init__(self, endpoint: str, access_key: str, secret_key: str,
                 bucket: str, secure: bool = True, region: str = "us-east-1"):
        from minio import Minio
        self.client = Minio(
            endpoint=endpoint, access_key=access_key,
            secret_key=secret_key, secure=secure, region=region,
        )
        self.bucket = bucket
        self._ensure_bucket()

    def _ensure_bucket(self):
        if not self.client.bucket_exists(self.bucket):
            self.client.make_bucket(self.bucket)

    def upload_file(self, local_path: str, remote_key: str) -> str:
        self.client.fput_object(self.bucket, remote_key, local_path)
        return self.get_url(remote_key)

    def upload_bytes(self, data: bytes, remote_key: str,
                     content_type: str = "") -> str:
        import io
        self.client.put_object(
            self.bucket, remote_key, io.BytesIO(data),
            length=len(data),
            content_type=content_type or "application/octet-stream",
        )
        return self.get_url(remote_key)

    def download_file(self, remote_key: str, local_path: str) -> str:
        self.client.fget_object(self.bucket, remote_key, local_path)
        return local_path

    def download_bytes(self, remote_key: str) -> bytes:
        response = self.client.get_object(self.bucket, remote_key)
        data = response.read()
        response.close()
        response.release_conn()
        return data

    def delete(self, remote_key: str) -> bool:
        self.client.remove_object(self.bucket, remote_key)
        return True

    def list(self, prefix: str = "") -> List[str]:
        objects = self.client.list_objects(
            self.bucket, prefix=prefix, recursive=True)
        return [obj.object_name for obj in objects]

    def get_url(self, remote_key: str, expires: int = 3600) -> str:
        from datetime import timedelta
        return self.client.presigned_get_object(
            self.bucket, remote_key,
            expires=timedelta(seconds=expires),
        )

    def exists(self, remote_key: str) -> bool:
        try:
            self.client.stat_object(self.bucket, remote_key)
            return True
        except Exception:
            return False
```

#### OSSBackend

```python
class OSSBackend(StorageBackend):
    """阿里云 OSS 实现"""

    def __init__(self, access_key_id: str, access_key_secret: str,
                 endpoint: str, bucket: str):
        import oss2
        self.auth = oss2.Auth(access_key_id, access_key_secret)
        self.bucket_obj = oss2.Bucket(self.auth, endpoint, bucket)
        self.bucket_name = bucket

    def upload_file(self, local_path: str, remote_key: str) -> str:
        self.bucket_obj.put_object_from_file(remote_key, local_path)
        return self.get_url(remote_key)

    def upload_bytes(self, data: bytes, remote_key: str,
                     content_type: str = "") -> str:
        self.bucket_obj.put_object(remote_key, data)
        return self.get_url(remote_key)

    def download_file(self, remote_key: str, local_path: str) -> str:
        self.bucket_obj.get_object_to_file(remote_key, local_path)
        return local_path

    def download_bytes(self, remote_key: str) -> bytes:
        result = self.bucket_obj.get_object(remote_key)
        return result.read()

    def delete(self, remote_key: str) -> bool:
        self.bucket_obj.delete_object(remote_key)
        return True

    def list(self, prefix: str = "") -> List[str]:
        import oss2
        return [obj.key for obj in
                oss2.ObjectIterator(self.bucket_obj, prefix=prefix)]

    def get_url(self, remote_key: str, expires: int = 3600) -> str:
        return self.bucket_obj.sign_url("GET", remote_key, expires)

    def exists(self, remote_key: str) -> bool:
        return self.bucket_obj.object_exists(remote_key)
```

### 5.3 配置驱动切换

```yaml
# config/storage.yaml
storage:
  backend: minio          # local | minio | oss | cos | kodo | s3

  local:
    root_dir: ./data/storage
    base_url: http://localhost:8000/files

  minio:
    endpoint: localhost:9000
    access_key: ${MINIO_ACCESS_KEY}
    secret_key: ${MINIO_SECRET_KEY}
    bucket: compact-rag
    secure: false

  oss:
    access_key_id: ${OSS_ACCESS_KEY_ID}
    access_key_secret: ${OSS_ACCESS_KEY_SECRET}
    endpoint: oss-cn-hangzhou.aliyuncs.com
    bucket: compact-rag

  kodo:
    access_key: ${QINIU_ACCESS_KEY}
    secret_key: ${QINIU_SECRET_KEY}
    bucket: compact-rag
    domain: https://cdn.yourdomain.com

  s3:
    region: us-east-1
    access_key_id: ${AWS_ACCESS_KEY_ID}
    secret_access_key: ${AWS_SECRET_ACCESS_KEY}
    bucket: compact-rag
```

### 5.4 工厂函数

```python
from functools import lru_cache


@lru_cache()
def get_storage_backend() -> StorageBackend:
    """根据配置获取存储后端实例"""
    import os
    backend_type = os.environ.get("STORAGE_BACKEND", "local")

    if backend_type == "local":
        return LocalFileBackend(
            root_dir=os.environ.get("STORAGE_LOCAL_ROOT", "./data/storage"),
            base_url=os.environ.get("STORAGE_LOCAL_BASE_URL", ""),
        )
    elif backend_type == "minio":
        return MinIOBackend(
            endpoint=os.environ["MINIO_ENDPOINT"],
            access_key=os.environ["MINIO_ACCESS_KEY"],
            secret_key=os.environ["MINIO_SECRET_KEY"],
            bucket=os.environ["MINIO_BUCKET"],
            secure=os.environ.get("MINIO_SECURE", "true").lower() == "true",
        )
    elif backend_type == "oss":
        return OSSBackend(
            access_key_id=os.environ["OSS_ACCESS_KEY_ID"],
            access_key_secret=os.environ["OSS_ACCESS_KEY_SECRET"],
            endpoint=os.environ["OSS_ENDPOINT"],
            bucket=os.environ["OSS_BUCKET"],
        )
    else:
        raise ValueError(f"Unknown storage backend: {backend_type}")
```

---

## 6. 文件存储与 RAG 系统的集成

### 6.1 文档摄入流程中的文件存储位置

```
用户上传文件
    |
    v
+-------------------------------------------+
|  临时存储 (Temp Storage)                   |
|  - 原始上传文件                            |
|  - 路径: temp/{session_id}/{filename}      |
|  - TTL: 1小时                              |
|  - 后端: MinIO 或本地文件系统              |
+-------------------------------------------+
    |
    v (解析完成)
+-------------------------------------------+
|  持久化存储 (Persistent Storage)           |
|  - 解析后原始文档                          |
|  - 路径: docs/{collection_id}/{hash}{ext} |
|  - 存储: 长期保留                         |
|  - 后端: MinIO 或云存储                   |
+-------------------------------------------+
    |
    v (不再频繁访问)
+-------------------------------------------+
|  归档存储 (Archive Storage)               |
|  - 历史数据                                |
|  - 后端: 云存储低频/归档存储              |
+-------------------------------------------+
```

### 6.2 存储路径策略

```python
import hashlib
from datetime import datetime
from pathlib import Path


def build_storage_key(
    collection_id: str,
    filename: str,
    content: bytes = None,
) -> str:
    """构建持久化存储路径"""
    now = datetime.utcnow()
    date_path = f"{now.year}/{now.month:02d}/{now.day:02d}"

    if content:
        file_hash = hashlib.sha256(content).hexdigest()[:16]
    else:
        file_hash = hashlib.sha256(filename.encode()).hexdigest()[:16]

    ext = Path(filename).suffix
    return f"docs/{collection_id}/{date_path}/{file_hash}{ext}"


def build_temp_key(session_id: str, filename: str) -> str:
    """构建临时文件路径"""
    return (f"temp/{session_id}/"
            f"{int(datetime.utcnow().timestamp())}_{filename}")
```

### 6.3 临时文件清理（TTL 策略）

```python
import asyncio
from datetime import datetime, timedelta


class TempFileCleaner:
    """临时文件清理器"""

    def __init__(self, backend: StorageBackend, ttl_hours: int = 1):
        self.backend = backend
        self.ttl = timedelta(hours=ttl_hours)

    async def clean_expired(self):
        """清理过期临时文件"""
        for key in self.backend.list(prefix="temp/"):
            parts = key.split("/")
            if len(parts) >= 3:
                try:
                    timestamp = int(parts[2].split("_")[0])
                    file_time = datetime.fromtimestamp(timestamp)
                    if datetime.now() - file_time > self.ttl:
                        self.backend.delete(key)
                        print(f"Cleaned temp file: {key}")
                except (ValueError, IndexError):
                    continue

    async def start(self, interval_minutes: int = 30):
        """启动定时清理循环"""
        while True:
            await self.clean_expired()
            await asyncio.sleep(interval_minutes * 60)
```

### 6.4 图片/附件在问答中的引用方式

```python
def get_file_url_for_response(
    remote_key: str,
    expires: int = 3600,
) -> str:
    """生成问答响应中引用的文件 URL"""
    backend = get_storage_backend()
    return backend.get_url(remote_key, expires=expires)
```

---

## 7. 推荐方案

### 7.1 环境推荐

#### 开发测试环境

```
推荐: MinIO (Docker) 或 本地文件系统

原因:
  - 零成本
  - 部署简单（一行 Docker 命令）
  - MinIO 与生产环境的 S3 API 完全兼容
  - 无需网络，延迟极低

启动命令:
  docker run -p 9000:9000 -p 9001:9001 \
    -e MINIO_ROOT_USER=minioadmin \
    -e MINIO_ROOT_PASSWORD=minioadmin \
    quay.io/minio/minio server /data --console-address ":9001"
```

#### 生产环境（国内场景）

```
首选: 阿里云 OSS  或  七牛云 Kodo

阿里云 OSS 优势:
  - 功能最全面，文档最完善
  - SDK 成熟度高，生态丰富
  - 支持多种数据冗余策略
  - 全球节点最多

七牛云 Kodo 优势:
  - 外网流量费最低 (0.26 元/GB vs 0.50 元/GB)
  - CDN 集成最深度
  - 免费额度更高 (10GB)
  - S3 API 兼容

推荐组合:
  - 文档存储: 七牛云 Kodo (低流量成本)
  - 图片/富媒体: 七牛云 Kodo + CDN (加速分发)
  - 备份归档: 阿里云 OSS 归档存储 (低成本)
```

#### 生产环境（海外场景）

```
首选: AWS S3

原因:
  - 最成熟，全球节点最多
  - boto3 SDK 文档丰富
  - S3 API 是行业标准
  - RAG 系统配套生态丰富 (Bedrock, OpenSearch 等)
```

### 7.2 选择决策流程图

```
需要文件存储
    |
    v
+-------------------+
| 部署在哪个区域？  |
+-------------------+
    |
    +-- 中国大陆 -------------------------------------+
    |                                                  |
    |       +------------------------------+           |
    |       | 有私有化部署需求？           |           |
    |       +------------------------------+           |
    |          |               |                       |
    |          是              否                       |
    |          v               v                       |
    |   +----------+    +----------+                   |
    |   |  MinIO   |    | 云存储？  |                   |
    |   | (Docker/ |    +----------+                   |
    |   |   K8s)   |       |     |                     |
    |   +----------+       是    否                     |
    |                       v     v                     |
    |               +----------+ +----------+           |
    |               | 有 CDN   | | 阿里云   |           |
    |               | 加速需求？| |  OSS     |           |
    |               +----------+ +----------+           |
    |                  |     |                          |
    |                  是    否                          |
    |                  v     v                          |
    |            +--------+ +--------+                  |
    |            |七牛云  | | 阿里云 |                  |
    |            | Kodo   | |  OSS   |                  |
    |            +--------+ +--------+                  |
    |
    +-- 海外 -------> AWS S3
    |
    +-- 混合云 -----> MinIO (私有) + 云存储 (备份)
```

### 7.3 成本估算示例

假设 RAG 系统月处理 100GB 文档，存储 1TB 数据，月外网流量 500GB：

| 方案 | 存储费 | 流量费 | API 费 | 月总成本 |
|------|--------|--------|--------|---------|
| **MinIO（自托管）** | 服务器折旧 | 带宽费 | 0 | ~Y500-1000 |
| **七牛云 Kodo** | Y115 | Y130 | ~Y10 | **~Y255** |
| **阿里云 OSS** | Y120 | Y250 | ~Y10 | **~Y380** |
| **腾讯云 COS** | Y118 | Y250 | ~Y10 | **~Y378** |
| **AWS S3** | $23 | $50 | ~$1 | **~$74** |

> 七牛云在外网流量费上有明显优势，适合流量敏感型场景。

---

## 8. 参考资料与链接

### Python SDK 源码/GitHub 地址

| SDK | 安装命令 | GitHub URL |
|-----|---------|-----------|
| **MinIO** | `pip install minio` | https://github.com/minio/minio-py |
| **阿里云 OSS** | `pip install oss2` | https://github.com/aliyun/aliyun-oss-python-sdk |
| **腾讯云 COS** | `pip install cos-python-sdk-v5` | https://github.com/tencentyun/cos-python-sdk-v5 |
| **七牛云 Kodo** | `pip install qiniu` | https://github.com/qiniu/python-sdk |
| **AWS S3 (boto3)** | `pip install boto3` | https://github.com/boto/boto3 |
| **GCS** | `pip install google-cloud-storage` | https://github.com/googleapis/python-storage |
| **Azure Blob** | `pip install azure-storage-blob` | https://github.com/Azure/azure-sdk-for-python |
| **华为云 OBS** | `pip install esdk-obs-python` | https://github.com/huaweicloud/obs-sdk-python |
| **百度云 BOS** | `pip install baidubce` | https://github.com/baidubce/bce-sdk-python |

### 价格计算器链接

| 服务商 | 价格计算器 |
|--------|-----------|
| **阿里云 OSS** | https://www.aliyun.com/price/detail/oss |
| **腾讯云 COS** | https://buy.cloud.tencent.com/price/cos |
| **七牛云 Kodo** | https://www.qiniu.com/prices/kodo |
| **华为云 OBS** | https://www.huaweicloud.com/pricing/calculator.html |
| **AWS S3** | https://calculator.aws.amazon.com/ |
| **GCS** | https://cloud.google.com/products/calculator |

### MinIO 官方文档

- MinIO 官方文档: https://min.io/docs/
- MinIO Python SDK 文档: https://min.io/docs/minio/linux/developers/python/
- MinIO Python SDK 示例: https://github.com/minio/minio-py/tree/master/examples

### 各大云存储官方文档

| 服务商 | Python SDK 文档地址 |
|--------|-------------------|
| **阿里云 OSS** | https://help.aliyun.com/zh/oss/developer-reference/python-installation |
| **腾讯云 COS** | https://cloud.tencent.com/document/product/436/12269 |
| **七牛云 Kodo** | https://developer.qiniu.com/kodo/sdk/python |
| **华为云 OBS** | https://support.huaweicloud.com/sdk-python-devg-obs/ |
| **AWS S3** | https://boto3.amazonaws.com/v1/documentation/api/latest/guide/s3.html |
| **GCS** | https://cloud.google.com/storage/docs/reference/libraries |
| **Azure Blob** | https://learn.microsoft.com/azure/storage/blobs/storage-quickstart-blobs-python |

---

## 附录

### A. pip 安装命令汇总

```bash
# 本地方案
pip install minio                    # MinIO SDK

# 国内云方案
pip install oss2                     # 阿里云 OSS
pip install cos-python-sdk-v5        # 腾讯云 COS
pip install qiniu                    # 七牛云 Kodo
pip install esdk-obs-python          # 华为云 OBS
pip install baidubce                 # 百度云 BOS

# 国际云方案
pip install boto3                    # AWS S3
pip install google-cloud-storage     # GCS
pip install azure-storage-blob       # Azure Blob
```

### B. 快速启动检查清单

- [ ] 确定部署区域（国内/海外/混合）
- [ ] 确定是否有私有化部署要求
- [ ] 是否需要 CDN 加速
- [ ] 预算范围（月存储量 x 流量估算）
- [ ] 安装对应 SDK: pip install minio oss2 cos-python-sdk-v5 qiniu boto3
- [ ] 配置环境变量（AccessKey/SecretKey）
- [ ] 实现 StorageBackend 接口统一调用
- [ ] 设置临时文件 TTL 清理策略
- [ ] 预签名 URL 用于前端直传/下载

---

> **文档维护者**: compact-rag 团队
> **最后更新**: 2026-05-23
> **下一阶段**: 在 compact-rag 中实现 StorageBackend 抽象层，并默认集成 MinIO 支持
