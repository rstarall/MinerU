# 更新日志

## [1.0.0] - 2025-01-20

### 新增
- 🎉 首次发布 MinerU VLM Web API
- 🚀 支持多种 VLM 后端：vlm-transformers、vlm-sglang-engine、vlm-sglang-client
- 📝 完整的 FastAPI 接口文档
- 🐳 基于 SGLang 的 Docker 容器化部署
- 📊 批量处理多个文件
- 🔍 健康检查和监控接口
- 🧪 完整的 API 测试脚本

### 特性
- 支持 PDF 和图像文件解析
- 多种输出格式：Markdown、JSON、图像 base64
- 页面范围选择解析
- 可配置的模型源（本地、HuggingFace、ModelScope）
- 完整的错误处理和日志记录

### 技术栈
- FastAPI 0.104+
- MinerU 2.0 VLM 模型
- SGLang 高性能推理框架
- Docker 容器化部署
- NVIDIA GPU 加速支持

### 系统要求
- NVIDIA RTX 4090+ GPU
- 24GB+ 显存
- 32GB+ 内存
- CUDA 12.4+
- Docker + NVIDIA Container Toolkit 