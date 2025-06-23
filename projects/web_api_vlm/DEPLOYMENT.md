# MinerU VLM Web API 部署指南

本文档详细说明了如何在不同环境下部署 MinerU VLM Web API。

## 📋 部署前准备

### 系统要求

| 组件 | 最低要求 | 推荐配置 |
|------|---------|---------|
| GPU | NVIDIA RTX 4090 (24GB 显存) | NVIDIA RTX A6000 (48GB 显存) |
| 内存 | 32GB RAM | 64GB+ RAM |
| 存储 | 50GB 可用空间 (SSD) | 100GB+ SSD |
| CPU | 16 核心 | 32+ 核心 |
| 操作系统 | Ubuntu 20.04+ | Ubuntu 22.04 LTS |

### 🔧 必需软件依赖

#### 1. NVIDIA 驱动程序

```bash
# 检查是否已安装
nvidia-smi

# Ubuntu 安装 NVIDIA 驱动
sudo apt update
sudo apt install nvidia-driver-535
sudo reboot

# CentOS/RHEL 安装
sudo dnf install nvidia-driver
sudo reboot

# 验证安装
nvidia-smi
```

#### 2. Docker 和 Docker Compose

```bash
# 安装 Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# 安装 Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# 将用户添加到 docker 组
sudo usermod -aG docker $USER
newgrp docker

# 验证安装
docker --version
docker-compose --version
```

#### 3. NVIDIA Container Toolkit（**必需**）

**⚠️ 重要：这是在 Docker 容器中使用 NVIDIA GPU 的必需组件！**

##### Ubuntu/Debian 安装

```bash
# 添加 NVIDIA GPG 密钥和仓库
distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
   && curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
   && curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
      sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
      sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# 更新包列表并安装
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# 配置 Docker 运行时
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

##### CentOS/RHEL 安装

```bash
# 添加 NVIDIA 仓库
distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
   && curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/nvidia-container-toolkit.repo | \
      sudo tee /etc/yum.repos.d/nvidia-container-toolkit.repo

# 安装 nvidia-container-toolkit
sudo yum install -y nvidia-container-toolkit

# 配置 Docker 运行时
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

##### 验证安装

```bash
# 测试 GPU 访问
docker run --rm --gpus all nvidia/cuda:12.4-base-ubuntu22.04 nvidia-smi

# 如果看到 GPU 信息输出，说明安装成功
```

## 🚀 部署方式

### 生产环境部署

#### 方式一：使用预构建镜像（推荐）

```bash
# 1. 克隆项目
git clone https://github.com/opendatalab/MinerU.git
cd MinerU/projects/web_api_vlm

# 2. 创建必要目录
mkdir -p output logs temp models_cache

# 3. 启动生产服务
docker-compose -f docker-compose.prod.yml up -d

# 4. 查看服务状态
docker-compose -f docker-compose.prod.yml ps

# 5. 查看启动日志
docker-compose -f docker-compose.prod.yml logs -f mineru-vlm-api

# 6. 验证服务
curl http://localhost:8000/health
```

#### 方式二：从源码构建

```bash
# 1. 修改 docker-compose.prod.yml，启用 build 配置
# 取消注释 build 部分，注释 image 行

# 2. 构建并启动
docker-compose -f docker-compose.prod.yml up --build -d

# 3. 构建单独的镜像
docker-compose -f docker-compose.prod.yml build

# 4. 标记镜像
docker tag web_api_vlm_mineru-vlm-api rstarall/mineru-vlm-api:latest
```

#### 方式三：包含 SGLang 服务器的分布式部署

```bash
# 1. 启动完整的分布式服务
docker-compose -f docker-compose.prod.yml --profile sglang up -d

# 2. 验证 SGLang 服务器
curl http://localhost:30000/health

# 3. 验证主 API 服务
curl http://localhost:8000/health
```

### 开发环境部署

#### 基础开发环境

```bash
# 1. 克隆项目
git clone https://github.com/opendatalab/MinerU.git
cd MinerU/projects/web_api_vlm

# 2. 创建必要目录
mkdir -p output logs temp models_cache

# 3. 启动开发服务
docker-compose -f docker-compose.dev.yml up --build

# 4. 或后台运行
docker-compose -f docker-compose.dev.yml up --build -d

# 5. 查看日志
docker-compose -f docker-compose.dev.yml logs -f mineru-vlm-api
```

#### 开发环境高级操作

```bash
# 重启服务（代码修改后）
docker-compose -f docker-compose.dev.yml restart mineru-vlm-api

# 进入容器进行调试
docker-compose -f docker-compose.dev.yml exec mineru-vlm-api bash

# 查看容器内文件
docker-compose -f docker-compose.dev.yml exec mineru-vlm-api ls -la /app

# 查看实时日志
docker-compose -f docker-compose.dev.yml logs -f --tail=100 mineru-vlm-api

# 停止服务
docker-compose -f docker-compose.dev.yml down
```

### CPU 模式部署（无 GPU 环境）

对于没有 NVIDIA GPU 的环境，可以修改配置文件：

```bash
# 创建 CPU 模式的 docker-compose 文件
cp docker-compose.dev.yml docker-compose.cpu.yml

# 编辑文件，删除或注释 GPU 相关配置：
# deploy:
#   resources:
#     reservations:
#       devices:
#         - driver: nvidia
#           count: all
#           capabilities: [gpu]

# 启动 CPU 模式
CUDA_VISIBLE_DEVICES="" docker-compose -f docker-compose.cpu.yml up -d
```

## 🔧 配置优化

### 生产环境配置优化

#### 1. 环境变量配置

创建 `.env` 文件：

```bash
# GPU 配置
CUDA_VISIBLE_DEVICES=0,1
NVIDIA_VISIBLE_DEVICES=all

# 模型源配置
MINERU_MODEL_SOURCE=local
MINERU_DOWNLOAD_SOURCE=modelscope

# 服务配置
PORT=8000
HOST=0.0.0.0
LOG_LEVEL=info

# 性能优化
OMP_NUM_THREADS=16
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# 网络配置（中国用户）
PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
PIP_TRUSTED_HOST=pypi.tuna.tsinghua.edu.cn
```

#### 2. Docker 资源限制

```yaml
# 在 docker-compose.prod.yml 中调整资源配置
deploy:
  resources:
    limits:
      memory: 64G
      cpus: '32'
    reservations:
      memory: 32G
      cpus: '16'
      devices:
        - driver: nvidia
          count: all
          capabilities: [gpu]
```

#### 3. 存储配置

```bash
# 创建专用存储目录
sudo mkdir -p /data/mineru/{output,logs,models,temp}
sudo chown -R $USER:$USER /data/mineru

# 更新 docker-compose.prod.yml 中的挂载路径
volumes:
  - /data/mineru/output:/app/output
  - /data/mineru/logs:/app/logs
  - /data/mineru/models:/root/.cache
  - /data/mineru/temp:/app/temp
```

### 开发环境配置优化

#### 1. 代码热更新配置

开发环境已配置代码热更新：

```yaml
volumes:
  # 代码文件挂载，支持热更新
  - ./app.py:/app/app.py:ro
  - ./entrypoint.sh:/app/entrypoint.sh:ro
  - ./mineru.json:/root/mineru.json:ro
  - ./test_api.py:/app/test_api.py:ro
```

#### 2. 调试配置

```yaml
environment:
  - LOG_LEVEL=debug
  - PYTHONPATH=/app
  - FLASK_ENV=development

# 交互模式
stdin_open: true
tty: true
```

## 🔍 验证部署

### 健康检查

```bash
# 基础健康检查
curl http://localhost:8000/health

# 详细系统信息
curl http://localhost:8000/system/info

# API 文档
# 浏览器访问：http://localhost:8000/docs
```

### 功能测试

```bash
# 运行 API 测试脚本
python test_api.py

# 或在容器内运行
docker-compose -f docker-compose.dev.yml exec mineru-vlm-api python test_api.py
```

### 性能测试

```bash
# 监控 GPU 使用
watch -n 1 nvidia-smi

# 监控容器资源使用
docker stats

# 监控服务日志
docker-compose -f docker-compose.prod.yml logs -f mineru-vlm-api
```

## 🔧 监控和维护

### 日志管理

```bash
# 查看所有日志
docker-compose -f docker-compose.prod.yml logs

# 查看特定服务日志
docker-compose -f docker-compose.prod.yml logs mineru-vlm-api

# 实时查看日志
docker-compose -f docker-compose.prod.yml logs -f --tail=100

# 日志轮转配置（添加到 docker-compose.yml）
logging:
  driver: "json-file"
  options:
    max-size: "100m"
    max-file: "5"
```

### 备份和恢复

```bash
# 备份配置和数据
tar -czf mineru-backup-$(date +%Y%m%d).tar.gz \
  mineru.json output/ logs/ models_cache/

# 恢复数据
tar -xzf mineru-backup-20240101.tar.gz
```

### 更新部署

```bash
# 生产环境更新
docker-compose -f docker-compose.prod.yml pull
docker-compose -f docker-compose.prod.yml up -d

# 开发环境更新
docker-compose -f docker-compose.dev.yml build --no-cache
docker-compose -f docker-compose.dev.yml up -d
```

## 🚨 故障排除

### 常见问题及解决方案

#### 1. GPU 相关问题

```bash
# 检查 NVIDIA 驱动
nvidia-smi

# 检查 Docker GPU 支持
docker run --rm --gpus all nvidia/cuda:12.4-base-ubuntu22.04 nvidia-smi

# 检查容器内 GPU 访问
docker-compose -f docker-compose.dev.yml exec mineru-vlm-api nvidia-smi
```

#### 2. 模型下载问题

```bash
# 检查网络连接
docker-compose -f docker-compose.dev.yml exec mineru-vlm-api curl -I https://www.modelscope.cn

# 手动下载模型
docker-compose -f docker-compose.dev.yml exec mineru-vlm-api \
  python -c "import os; os.system('mineru-models-download -s modelscope -m all')"

# 查看模型下载日志
docker-compose -f docker-compose.dev.yml logs | grep -i download
```

#### 3. 内存和存储问题

```bash
# 检查磁盘空间
df -h

# 检查 Docker 镜像占用
docker system df

# 清理不用的镜像和容器
docker system prune -f

# 检查容器内存使用
docker stats
```

#### 4. 网络连接问题

```bash
# 检查端口占用
netstat -tulpn | grep :8000

# 检查防火墙设置
sudo ufw status

# 测试内部网络连接
docker-compose -f docker-compose.dev.yml exec mineru-vlm-api curl localhost:8000/health
```

### 调试模式

#### 开发环境调试

```bash
# 启用详细日志
docker-compose -f docker-compose.dev.yml up --build

# 进入容器调试
docker-compose -f docker-compose.dev.yml exec mineru-vlm-api bash

# 手动启动服务进行调试
docker-compose -f docker-compose.dev.yml exec mineru-vlm-api python app.py
```

#### 生产环境调试

```bash
# 临时切换到调试模式
docker-compose -f docker-compose.prod.yml exec mineru-vlm-api-prod bash

# 查看环境变量
docker-compose -f docker-compose.prod.yml exec mineru-vlm-api-prod env

# 检查配置文件
docker-compose -f docker-compose.prod.yml exec mineru-vlm-api-prod cat /root/mineru.json
```

## 📈 性能优化建议

### 硬件优化

1. **GPU 优化**
   - 使用最新的 NVIDIA GPU（RTX 4090, A100, H100）
   - 确保足够的显存（24GB+）
   - 考虑多 GPU 并行处理

2. **内存优化**
   - 至少 32GB RAM，推荐 64GB+
   - 使用高速内存（DDR4-3200+）

3. **存储优化**
   - 使用 NVMe SSD
   - 将模型缓存放在高速存储上
   - 考虑使用网络存储（NFS, S3）

### 软件优化

1. **Docker 优化**
   ```bash
   # 增加共享内存
   --shm-size 32g
   
   # 优化 IPC 设置
   --ipc=host
   
   # 网络优化
   --network host  # 仅在需要时使用
   ```

2. **环境变量优化**
   ```bash
   # CUDA 优化
   export CUDA_VISIBLE_DEVICES=0,1
   export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
   
   # 并行处理优化
   export OMP_NUM_THREADS=16
   export MKL_NUM_THREADS=16
   ```

3. **模型优化**
   - 使用量化模型减少显存占用
   - 考虑模型蒸馏
   - 使用模型并行处理

## 🔒 安全配置

### 生产环境安全

1. **网络安全**
   ```bash
   # 使用防火墙限制访问
   sudo ufw allow from 192.168.1.0/24 to any port 8000
   
   # 配置反向代理（Nginx）
   # 启用 HTTPS
   # 配置速率限制
   ```

2. **容器安全**
   ```yaml
   # 使用非 root 用户
   user: "1000:1000"
   
   # 只读文件系统
   read_only: true
   
   # 安全选项
   security_opt:
     - no-new-privileges:true
   ```

3. **数据安全**
   ```bash
   # 加密敏感配置
   # 使用 Docker secrets
   # 定期备份数据
   ```

## 📞 技术支持

### 获取帮助

- **项目文档**: [GitHub](https://github.com/opendatalab/MinerU)
- **问题报告**: [GitHub Issues](https://github.com/opendatalab/MinerU/issues)
- **讨论交流**: [GitHub Discussions](https://github.com/opendatalab/MinerU/discussions)

### 诊断信息收集

出现问题时，请收集以下信息：

```bash
# 系统信息
uname -a
nvidia-smi
docker --version
docker-compose --version

# 服务状态
docker-compose -f docker-compose.prod.yml ps
docker-compose -f docker-compose.prod.yml logs --tail=100

# 资源使用
docker stats
df -h
free -h
```

## 📝 更新日志

本文档会随着项目更新而持续更新，请定期查看最新版本。 