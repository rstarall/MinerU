# MinerU VLM Web API 项目结构

```
projects/web_api_vlm/
├── app.py                    # FastAPI 主应用程序
├── requirements.txt          # Python 依赖包列表
├── mineru.json              # MinerU 配置文件
├── Dockerfile               # Docker 构建文件
├── docker-compose.yml       # Docker Compose 配置
├── entrypoint.sh           # 容器启动脚本
├── test_api.py             # API 测试脚本
├── .env.example            # 环境变量示例
├── .dockerignore           # Docker 忽略文件
├── .gitignore              # Git 忽略文件
├── README.md               # 项目说明文档
├── CHANGELOG.md            # 更新日志
└── PROJECT_STRUCTURE.md    # 项目结构说明（本文件）
```

## 📁 文件说明

### 核心文件

- **`app.py`**: FastAPI 主应用程序，包含所有 API 端点和业务逻辑
- **`requirements.txt`**: Python 依赖包列表，包含 MinerU VLM 支持
- **`mineru.json`**: MinerU 配置文件，定义模型路径和配置
- **`Dockerfile`**: 基于 SGLang 镜像的 Docker 构建文件
- **`entrypoint.sh`**: 容器启动脚本，包含环境检查和服务启动

### 部署文件

- **`docker-compose.yml`**: Docker Compose 配置，支持主服务和可选的 SGLang 服务器
- **`.env.example`**: 环境变量配置示例
- **`.dockerignore`**: Docker 构建时忽略的文件列表

### 测试和工具

- **`test_api.py`**: 完整的 API 测试脚本，支持多种测试场景
- **`.gitignore`**: Git 版本控制忽略文件

### 文档

- **`README.md`**: 详细的项目说明、安装部署和使用指南
- **`CHANGELOG.md`**: 版本更新日志
- **`PROJECT_STRUCTURE.md`**: 项目结构说明（本文件）

## 🏗️ 架构设计

### API 层次结构

```
FastAPI Application (app.py)
├── / (根路径)
├── /health (健康检查)
├── /vlm_parse (VLM 文档解析)
└── /batch_vlm_parse (批量解析)
```

### VLM 后端支持

1. **vlm-transformers**: 基于 transformers 库的通用后端
2. **vlm-sglang-engine**: SGLang 引擎模式，高性能推理
3. **vlm-sglang-client**: SGLang 客户端模式，连接外部服务器

### 数据流

```
HTTP Request → FastAPI → VLM Backend → MinerU Core → Response
     ↓              ↓           ↓            ↓
File Upload → Validation → Model Inference → JSON/MD Output
```

## 🔧 配置管理

### 环境变量

- `MINERU_MODEL_SOURCE`: 模型来源配置
- `PORT`: API 服务端口
- `LOG_LEVEL`: 日志级别
- `CUDA_VISIBLE_DEVICES`: GPU 设备选择

### 配置文件

- `mineru.json`: MinerU 核心配置
- `docker-compose.yml`: 容器编排配置
- `.env`: 环境变量配置

## 🧪 测试覆盖

### 测试类型

1. **健康检查测试**: 验证服务状态
2. **根路径测试**: 验证基础接口
3. **VLM 解析测试**: 验证核心功能
4. **批量处理测试**: 验证批量能力

### 测试用法

```bash
# 基础测试
python test_api.py --url http://localhost:8000

# 带文件测试
python test_api.py --file document.pdf --backend vlm-sglang-engine

# 批量测试
python test_api.py --files doc1.pdf doc2.pdf
```

## 📊 性能考虑

### 资源需求

- **GPU**: NVIDIA RTX 4090+ (24GB+ 显存)
- **内存**: 32GB+ RAM
- **存储**: 50GB+ 可用空间
- **网络**: 用于模型下载和 API 访问

### 优化建议

1. **硬件优化**: 使用高性能 GPU 和 SSD
2. **容器优化**: 合理分配共享内存和 IPC
3. **后端选择**: 根据场景选择合适的 VLM 后端
4. **并发控制**: 限制同时处理的文件数量

## 🚀 部署模式

### 1. 单容器模式

```bash
docker run --gpus all --shm-size 32g -p 8000:8000 mineru-vlm-api:latest
```

### 2. Docker Compose 模式

```bash
docker-compose up -d
```

### 3. 分布式模式

```bash
# 启动 SGLang 服务器
docker-compose --profile sglang up sglang-server

# 启动 API 服务（连接到 SGLang 服务器）
docker-compose up mineru-vlm-api
```

## 🔍 监控和维护

### 健康检查

- HTTP 健康检查端点: `/health`
- Docker 健康检查: 内置容器健康检查
- 服务状态监控: 通过日志和指标

### 日志管理

- 应用日志: 通过 `loguru` 记录
- 容器日志: Docker 日志驱动
- 错误追踪: 详细的异常信息和堆栈跟踪

### 性能监控

- GPU 使用率: `nvidia-smi`
- 内存使用: `docker stats`
- API 响应时间: 内置性能测量

这个项目结构设计遵循了现代 Web API 开发的最佳实践，提供了完整的部署、测试和维护解决方案。 