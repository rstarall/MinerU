# MinerU VLM Web API

基于 MinerU 2.0 VLM 模型的 PDF 解析 Web API 服务，支持多种 VLM 后端（transformers、sglang-engine、sglang-client）。

## 🚀 特性

- **多种 VLM 后端支持**：transformers、sglang-engine、sglang-client
- **高性能解析**：基于 MinerU 2.0 的最新 VLM 模型
- **灵活的 API 接口**：支持文件上传和路径输入
- **批量处理**：支持一次处理多个文件
- **丰富的输出格式**：Markdown、JSON、图像等
- **容器化部署**：基于 SGLang 的 Docker 镜像

## 📋 系统要求

### 硬件要求
- **GPU**: NVIDIA RTX 4090 或更高性能的 GPU
- **显存**: 至少 24GB（推荐 32GB+）
- **内存**: 至少 32GB RAM
- **存储**: 至少 50GB 可用空间

### 软件要求
- Docker 和 Docker Compose
- NVIDIA Container Toolkit
- CUDA 12.4+

## 🛠️ 安装部署

### 方式一：Docker 构建（推荐）

1. **克隆项目**
```bash
git clone https://github.com/opendatalab/MinerU.git
cd MinerU/projects/web_api_vlm
```

2. **构建 Docker 镜像**
```bash
# 国内用户（使用代理）
docker build --build-arg http_proxy=http://127.0.0.1:7890 \
             --build-arg https_proxy=http://127.0.0.1:7890 \
             -t mineru-vlm-api:latest .

# 国外用户
docker build -t mineru-vlm-api:latest .
```

3. **启动服务**
```bash
docker run --rm -it \
  --gpus all \
  --shm-size 32g \
  --ipc=host \
  -p 8000:8000 \
  -v $(pwd)/output:/app/output \
  mineru-vlm-api:latest
```

### 方式二：Docker Compose（便于管理）

1. **创建 docker-compose.yml**
```yaml
version: '3.8'

services:
  mineru-vlm-api:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./output:/app/output
      - ./logs:/app/logs
    environment:
      - MINERU_MODEL_SOURCE=local
      - LOG_LEVEL=info
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 60s
```

2. **启动服务**
```bash
docker-compose up -d
```

### 方式三：本地安装

1. **安装依赖**
```bash
pip install -r requirements.txt
```

2. **下载模型**
```bash
mineru-models-download -s modelscope -m all
```

3. **启动服务**
```bash
python app.py
```

## 📖 API 使用说明

### 基础信息

- **API 地址**: `http://localhost:8000`
- **API 文档**: `http://localhost:8000/docs`
- **健康检查**: `http://localhost:8000/health`

### 主要接口

#### 1. VLM 文档解析 `/vlm_parse`

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

# 使用文件路径解析
curl -X POST "http://localhost:8000/vlm_parse" \
  -F "file_path=/path/to/document.pdf" \
  -F "backend=vlm-transformers"
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
print(result["md_content"])
```

#### 2. 批量处理 `/batch_vlm_parse`

**POST** `/batch_vlm_parse`

支持一次上传多个文件进行批量处理（最多 10 个文件）。

**示例请求：**
```bash
curl -X POST "http://localhost:8000/batch_vlm_parse" \
  -F "files=@doc1.pdf" \
  -F "files=@doc2.pdf" \
  -F "backend=vlm-sglang-engine" \
  -F "save_files=true"
```

### 响应格式

**成功响应：**
```json
{
  "md_content": "# 文档标题\n\n文档内容...",
  "file_name": "document",
  "backend": "vlm-sglang-engine",
  "pages_processed": "0-end",
  "images": {
    "image1.jpg": "data:image/jpeg;base64,..."
  },
  "middle_json": {...},
  "saved_path": "/app/output/document"
}
```

**错误响应：**
```json
{
  "error": "错误描述信息"
}
```

## 🔧 配置说明

### 环境变量

- `MINERU_MODEL_SOURCE`: 模型来源（`local`、`huggingface`、`modelscope`）
- `PORT`: API 服务端口（默认 8000）
- `LOG_LEVEL`: 日志级别（`debug`、`info`、`warning`、`error`）

### 配置文件 `mineru.json`

```json
{
    "models-dir": {
        "pipeline": "/opt/models/pipeline",
        "vlm": "/opt/models/vlm"
    },
    "config_version": "1.3.0"
}
```

## 📊 性能优化建议

### 1. 硬件配置
- 使用 NVIDIA RTX 4090 或更高性能的 GPU
- 确保足够的显存（24GB+）和内存（32GB+）
- 使用 SSD 存储以提高 I/O 性能

### 2. Docker 优化
```bash
# 分配更多共享内存
--shm-size 32g

# 使用 host 网络模式（如果需要）
--network host

# 设置 IPC 模式
--ipc=host
```

### 3. 后端选择
- **开发/测试**: 使用 `vlm-transformers`
- **生产环境**: 使用 `vlm-sglang-engine`
- **分布式部署**: 使用 `vlm-sglang-client`

## 🔍 监控和日志

### 健康检查
```bash
curl http://localhost:8000/health
```

### 查看日志
```bash
# Docker 日志
docker logs <container_id>

# Docker Compose 日志
docker-compose logs -f mineru-vlm-api
```

### 性能监控
```bash
# GPU 使用情况
nvidia-smi

# 容器资源使用
docker stats
```

## 🚨 故障排除

### 常见问题

1. **GPU 内存不足**
   - 减少并发请求数量
   - 使用更小的 batch size
   - 确保没有其他程序占用 GPU

2. **模型下载失败**
   - 检查网络连接
   - 尝试使用不同的模型源（ModelScope/HuggingFace）
   - 手动下载模型文件

3. **API 响应缓慢**
   - 检查 GPU 利用率
   - 优化输入文件大小
   - 考虑使用 sglang-engine 后端

### 日志级别设置
```bash
# 启动时设置详细日志
docker run -e LOG_LEVEL=debug mineru-vlm-api:latest
```

## 🤝 贡献指南

1. Fork 项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

## 📄 许可证

本项目基于 Apache 2.0 许可证 - 查看 [LICENSE](../../LICENSE.md) 文件了解详情。

## 🙏 致谢

- [MinerU](https://github.com/opendatalab/MinerU) - 核心 PDF 解析库
- [SGLang](https://github.com/sgl-project/sglang) - 高性能 LLM 推理框架
- [FastAPI](https://fastapi.tiangolo.com/) - 现代 Web API 框架 