# MinerU VLM Web API 快速开始

这是一个 5 分钟快速部署指南，帮助您快速启动 MinerU VLM Web API 服务。

## 🚀 快速部署（5分钟）

### 前提条件

确保您的系统满足以下要求：

- ✅ Ubuntu 20.04+ 或其他 Linux 发行版
- ✅ NVIDIA GPU（RTX 4090 或更好，24GB+ 显存）
- ✅ 32GB+ 内存
- ✅ 50GB+ 可用存储空间
- ✅ 已安装 Docker 和 NVIDIA Container Toolkit

### 一键部署

```bash
# 1. 克隆项目（30秒）
git clone https://github.com/opendatalab/MinerU.git
cd MinerU/projects/web_api_vlm

# 2. 快速启动（3-4分钟，包含模型下载）
docker-compose up -d

# 3. 验证服务（30秒）
curl http://localhost:8000/health
```

就这么简单！🎉

## 📋 验证部署

### 检查服务状态

```bash
# 查看容器状态
docker-compose ps

# 查看启动日志
docker-compose logs -f mineru-vlm-api
```

### 访问 API 文档

打开浏览器访问：http://localhost:8000/docs

### 测试 API 功能

```bash
# 下载测试 PDF
wget -O test.pdf "https://arxiv.org/pdf/2301.00001.pdf"

# 测试 VLM 解析
curl -X POST "http://localhost:8000/vlm_parse" \
  -F "file=@test.pdf" \
  -F "backend=vlm-transformers" \
  -F "return_images=false"

# 或使用测试脚本
python test_api.py --url http://localhost:8000 --file test.pdf
```

## 🎯 使用示例

### 1. 基础文档解析

```python
import requests

# 上传并解析 PDF
with open("document.pdf", "rb") as f:
    response = requests.post(
        "http://localhost:8000/vlm_parse",
        files={"file": f},
        data={"backend": "vlm-transformers"}
    )

result = response.json()
print("解析结果：")
print(result["md_content"])
```

### 2. 高性能解析（SGLang 后端）

```bash
curl -X POST "http://localhost:8000/vlm_parse" \
  -F "file=@document.pdf" \
  -F "backend=vlm-sglang-engine" \
  -F "return_middle_json=true"
```

### 3. 批量处理

```bash
curl -X POST "http://localhost:8000/batch_vlm_parse" \
  -F "files=@doc1.pdf" \
  -F "files=@doc2.pdf" \
  -F "files=@doc3.pdf" \
  -F "backend=vlm-sglang-engine"
```

## 🔧 常用配置

### 切换后端类型

编辑 `docker-compose.yml` 中的环境变量：

```yaml
environment:
  - MINERU_MODEL_SOURCE=local
  - DEFAULT_BACKEND=vlm-sglang-engine  # 默认后端
```

### 调整资源限制

```yaml
deploy:
  resources:
    limits:
      memory: 64G
      cpus: '32'
    reservations:
      memory: 32G
      devices:
        - driver: nvidia
          count: 1
          capabilities: [gpu]
```

### 启用 SGLang 服务器模式

```bash
# 启动独立的 SGLang 服务器
docker-compose --profile sglang up -d sglang-server

# 使用 sglang-client 连接
curl -X POST "http://localhost:8000/vlm_parse" \
  -F "file=@document.pdf" \
  -F "backend=vlm-sglang-client" \
  -F "server_url=http://localhost:30000"
```

## 📊 性能对比

| 后端类型 | 解析速度 | 显存占用 | 适用场景 |
|---------|---------|---------|---------|
| vlm-transformers | 较慢 | ~16GB | 开发测试 |
| vlm-sglang-engine | 快 | ~20GB | 生产单机 |
| vlm-sglang-client | 最快 | ~24GB | 分布式 |

## 🚨 常见问题

### Q: 容器启动失败？

```bash
# 检查 GPU 访问
docker run --rm --gpus all nvidia/cuda:12.4-base nvidia-smi

# 检查内存
free -h

# 查看详细错误
docker-compose logs mineru-vlm-api
```

### Q: 模型下载慢？

```bash
# 使用国内镜像源
export MINERU_MODEL_SOURCE=modelscope
docker-compose down && docker-compose up -d
```

### Q: API 响应慢？

```bash
# 切换到高性能后端
curl -X POST "http://localhost:8000/vlm_parse" \
  -F "file=@document.pdf" \
  -F "backend=vlm-sglang-engine"
```

### Q: 显存不足？

```bash
# 减少并发，设置环境变量
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256
docker-compose restart mineru-vlm-api
```

## 🎓 进阶用法

### 自定义配置

1. 复制配置模板：
   ```bash
   cp .env.example .env
   ```

2. 编辑配置文件：
   ```bash
   # .env
   MINERU_MODEL_SOURCE=local
   LOG_LEVEL=debug
   CUDA_VISIBLE_DEVICES=0
   ```

3. 重启服务：
   ```bash
   docker-compose down && docker-compose up -d
   ```

### 生产环境部署

参见详细文档：
- [完整部署指南](DEPLOYMENT.md)
- [项目结构说明](PROJECT_STRUCTURE.md)
- [API 详细文档](README.md)

## 🤝 获取帮助

- 📖 [完整文档](README.md)
- 🐛 [报告问题](https://github.com/opendatalab/MinerU/issues)
- 💬 社区讨论：Discord / 微信群
- 📧 邮件支持：通过 GitHub Issues

## 🎉 下一步

恭喜！您已经成功部署了 MinerU VLM Web API。现在您可以：

1. 🔗 集成到您的应用程序中
2. 📊 监控服务性能和日志
3. 🔧 根据需要调整配置
4. 📈 扩展到生产环境

开始构建您的智能文档处理应用吧！🚀 