# MinerU VLM Web API

基于 MinerU 2.0 VLM 模型的 PDF 解析 Web API 服务，支持多种 VLM 后端（transformers、sglang-engine、sglang-client）。

## 🚀 特性

- **多种 VLM 后端支持**：transformers、sglang-engine、sglang-client
- **高性能解析**：基于 MinerU 2.0 的最新 VLM 模型
- **灵活的 API 接口**：支持文件上传和路径输入
- **批量处理**：支持一次处理多个文件
- **丰富的输出格式**：Markdown、JSON、图像等
- **容器化部署**：基于 SGLang 的 Docker 镜像
- **开发/生产环境分离**：针对不同使用场景优化

## 📋 系统要求

### 硬件要求
- **GPU**: NVIDIA RTX 4090 或更高性能的 GPU（**必需**）
- **显存**: 至少 24GB（推荐 32GB+）
- **内存**: 至少 32GB RAM
- **存储**: 至少 50GB 可用空间

### 软件要求
- **Docker** 和 **Docker Compose**
- **NVIDIA Container Toolkit**（**必需** - 用于 Docker 访问 GPU）
- **CUDA 12.4+**
- **NVIDIA 驱动** 535.x 或更高版本

## 🛠️ 环境准备

详细的环境准备步骤请参考 [DEPLOYMENT.md](./DEPLOYMENT.md)

## 🚀 快速开始

### 生产环境部署

生产环境使用预构建的镜像，模型已包含在镜像中，开箱即用。

```bash
# 1. 克隆项目
git clone https://github.com/opendatalab/MinerU.git
cd MinerU/projects/web_api_vlm

# 2. 启动生产服务
docker-compose -f docker-compose.prod.yml up -d

# 3. 查看服务状态
docker-compose -f docker-compose.prod.yml ps

# 4. 查看日志
docker-compose -f docker-compose.prod.yml logs -f mineru-vlm-api
```

### 开发环境部署

开发环境支持代码热更新，模型动态下载，适合代码开发和调试。

```bash
# 1. 克隆项目
git clone https://github.com/opendatalab/MinerU.git
cd MinerU/projects/web_api_vlm

# 2. 启动开发服务
docker-compose -f docker-compose.dev.yml up --build

# 3. 或后台运行
docker-compose -f docker-compose.dev.yml up --build -d

# 4. 查看日志
docker-compose -f docker-compose.dev.yml logs -f mineru-vlm-api

# 5. 进入容器调试
docker-compose -f docker-compose.dev.yml exec mineru-vlm-api bash
```

## 📁 项目结构

```
projects/web_api_vlm/
├── README.md                    # 项目说明文档
├── DEPLOYMENT.md               # 详细部署指南
├── app.py                      # 主应用程序
├── entrypoint.sh              # 容器启动脚本
├── requirements.txt           # Python 依赖
├── mineru.json               # 配置文件
├── test_api.py               # API 测试脚本
├── Dockerfile.prod           # 生产环境 Dockerfile
├── Dockerfile.dev            # 开发环境 Dockerfile
├── docker-compose.prod.yml   # 生产环境 Docker Compose
├── docker-compose.dev.yml    # 开发环境 Docker Compose
├── .dockerignore             # Docker 忽略文件
├── output/                   # 输出文件目录
├── logs/                     # 日志文件目录
├── temp/                     # 临时文件目录
└── models_cache/             # 模型缓存目录
```

## 🔧 环境差异说明

### 生产环境 (docker-compose.prod.yml)

**特点：**
- 使用预构建镜像 `rstarall/mineru-vlm-api:latest`
- 模型已打包在镜像中，无需额外下载
- 优化的资源配置和健康检查
- 自动重启策略
- 只挂载数据和配置目录

**适用场景：**
- 生产部署
- 稳定服务运行
- 性能要求高的场景

**启动命令：**
```bash
# 基础启动
docker-compose -f docker-compose.prod.yml up -d

# 包含 SGLang 服务器
docker-compose -f docker-compose.prod.yml --profile sglang up -d

# 停止服务
docker-compose -f docker-compose.prod.yml down
```

### 开发环境 (docker-compose.dev.yml)

**特点：**
- 从源码构建镜像
- 模型动态下载（首次启动）
- 代码文件通过 volume 挂载，支持热更新
- 包含开发工具（vim、htop、tree）
- 交互模式支持
- DEBUG 级别日志

**适用场景：**
- 代码开发
- 功能调试
- 配置测试

**启动命令：**
```bash
# 基础启动
docker-compose -f docker-compose.dev.yml up --build

# 后台运行
docker-compose -f docker-compose.dev.yml up --build -d

# 重启服务（代码修改后）
docker-compose -f docker-compose.dev.yml restart mineru-vlm-api

# 进入容器
docker-compose -f docker-compose.dev.yml exec mineru-vlm-api bash

# 停止服务
docker-compose -f docker-compose.dev.yml down
```

## 🔨 构建生产镜像

如果需要自定义构建生产镜像：

```bash
# 1. 构建镜像
docker-compose -f docker-compose.prod.yml build

# 2. 标记镜像
docker tag web_api_vlm_mineru-vlm-api:latest rstarall/mineru-vlm-api:latest

# 3. 推送镜像（可选）
docker push rstarall/mineru-vlm-api:latest
```

## 📖 API 使用说明

### 基础信息

- **API 地址**: `http://localhost:8000`
- **API 文档**: `http://localhost:8000/docs`
- **健康检查**: `http://localhost:8000/health`

### 主要接口

#### VLM 文档解析 `/vlm_parse`

**POST** `/vlm_parse`

支持的参数：
- `file`: 上传的文件（与 file_path 二选一）
- `file_path`: 文件路径（与 file 二选一）
- `backend`: VLM 后端类型
  - `vlm-transformers`: 通用性好，适合开发测试
  - `vlm-sglang-engine`: 性能优化，适合生产环境
  - `vlm-sglang-client`: 连接外部 SGLang 服务器
- `server_url`: SGLang 服务器地址（仅 sglang-client 需要）
- `start_page`: 开始页码（默认 0）
- `end_page`: 结束页码（默认处理全部）
- `return_images`: 是否返回图像 base64 编码
- `return_middle_json`: 是否返回中间 JSON 数据
- `return_model_output`: 是否返回模型原始输出
- `save_files`: 是否保存文件到磁盘

**示例请求（cURL）：**
```bash
# 上传文件解析
curl -X POST "http://localhost:8000/vlm_parse" \
  -F "file=@document.pdf" \
  -F "backend=vlm-sglang-engine" \
  -F "return_images=true" \
  -F "save_files=true"
```

**示例请求（Python）：**
```python
import requests

# 上传文件
with open("document.pdf", "rb") as f:
    response = requests.post(
        "http://localhost:8000/vlm_parse",
        files={"file": f},
        data={
            "backend": "vlm-sglang-engine",
            "return_images": True,
            "save_files": True
        }
    )
    result = response.json()
```

## 🔧 配置说明

### mineru.json 配置文件

配置文件通过 volume 挂载，支持热更新（无需重新构建镜像）：

```json
{
    "bucket_info": {
        "bucket-name-1": ["ak", "sk", "endpoint"]
    },
    "latex-delimiter-config": {
        "display": {
            "left": "$$",
            "right": "$$"
        },
        "inline": {
            "left": "$",
            "right": "$"
        }
    },
    "llm-aided-config": {
        "title_aided": {
            "api_key": "your_api_key",
            "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "model": "qwen2.5-32b-instruct",
            "enable": false
        }
    },
    "models-dir": {
        "pipeline": "",
        "vlm": ""
    },
    "config_version": "1.3.0"
}
```

### 环境变量

| 变量 | 开发环境 | 生产环境 | 说明 |
|------|---------|---------|------|
| `MINERU_MODEL_SOURCE` | auto | local | 模型源配置 |
| `LOG_LEVEL` | debug | info | 日志级别 |
| `MINERU_DOWNLOAD_SOURCE` | modelscope | modelscope | 模型下载源 |
| `OMP_NUM_THREADS` | - | 16 | OpenMP 线程数 |
| `PYTORCH_CUDA_ALLOC_CONF` | - | max_split_size_mb:512 | CUDA 内存配置 |

## 🔍 常见问题

### 1. 开发环境代码修改不生效

```bash
# 重启容器应用代码更改
docker-compose -f docker-compose.dev.yml restart mineru-vlm-api
```

### 2. 模型下载失败

```bash
# 检查网络连接和模型源配置
docker-compose -f docker-compose.dev.yml logs mineru-vlm-api
```

### 3. GPU 不可用

```bash
# 检查 NVIDIA Container Toolkit
docker run --rm --gpus all nvidia/cuda:12.4-base-ubuntu22.04 nvidia-smi
```

### 4. 镜像构建太慢

开发环境不包含模型，构建更快：
```bash
# 开发环境构建
docker-compose -f docker-compose.dev.yml build
```

## 📞 支持

- 详细部署指南：[DEPLOYMENT.md](./DEPLOYMENT.md)
- API 测试脚本：`python test_api.py`
- 项目仓库：[GitHub](https://github.com/opendatalab/MinerU)

## 📄 许可证

本项目遵循 [MIT License](../../LICENSE.md)。 