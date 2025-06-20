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

### 软件依赖

```bash
# 安装 Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# 安装 Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# 安装 NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update && sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

## 🚀 部署方式

### 方式一：快速单容器部署

适用于快速测试和开发环境。

```bash
# 1. 克隆项目
git clone https://github.com/opendatalab/MinerU.git
cd MinerU/projects/web_api_vlm

# 2. 构建镜像
docker build -t mineru-vlm-api:latest .

# 3. 启动服务
docker run --rm -it \
  --name mineru-vlm-api \
  --gpus all \
  --shm-size 32g \
  --ipc=host \
  -p 8000:8000 \
  -v $(pwd)/output:/app/output \
  -e MINERU_MODEL_SOURCE=local \
  mineru-vlm-api:latest
```

### 方式二：Docker Compose 部署（推荐）

适用于生产环境和持久化服务。

```bash
# 1. 准备配置文件
cp .env.example .env
# 编辑 .env 文件根据需要调整配置

# 2. 启动服务
docker-compose up -d

# 3. 查看日志
docker-compose logs -f mineru-vlm-api

# 4. 停止服务
docker-compose down
```

### 方式三：分布式部署

适用于高性能和高可用性需求。

```bash
# 1. 启动 SGLang 服务器
docker-compose --profile sglang up -d sglang-server

# 2. 等待 SGLang 服务器启动完成
docker-compose logs -f sglang-server

# 3. 启动 API 服务
docker-compose up -d mineru-vlm-api

# 4. 验证服务状态
curl http://localhost:8000/health
curl http://localhost:30000/health
```

## 🔧 配置优化

### 环境变量配置

创建 `.env` 文件：

```bash
# 模型源配置
MINERU_MODEL_SOURCE=local

# 服务配置
PORT=8000
LOG_LEVEL=info

# GPU 配置
CUDA_VISIBLE_DEVICES=0

# 性能优化
OMP_NUM_THREADS=16
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
```

### Docker 资源限制

```yaml
# docker-compose.yml 资源配置示例
services:
  mineru-vlm-api:
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
              count: 1
              capabilities: [gpu]
```

### 存储配置

```bash
# 创建持久化存储目录
mkdir -p /data/mineru/{output,logs,models,temp}

# 更新 docker-compose.yml 中的挂载路径
volumes:
  - /data/mineru/output:/app/output
  - /data/mineru/logs:/app/logs
  - /data/mineru/models:/opt/models
  - /data/mineru/temp:/app/temp
```

## 🔍 验证部署

### 健康检查

```bash
# 检查服务状态
curl http://localhost:8000/health

# 检查 API 信息
curl http://localhost:8000/

# 检查 SGLang 服务器（如果启用）
curl http://localhost:30000/health
```

### 功能测试

```bash
# 运行测试脚本
python test_api.py --url http://localhost:8000

# 测试文件上传（需要准备测试 PDF 文件）
curl -X POST "http://localhost:8000/vlm_parse" \
  -F "file=@test.pdf" \
  -F "backend=vlm-transformers"
```

## 📊 性能调优

### GPU 优化

```bash
# 查看 GPU 状态
nvidia-smi

# 设置 GPU 性能模式
sudo nvidia-smi -pm 1

# 设置 GPU 频率
sudo nvidia-smi -ac 1215,1410
```

### 内存优化

```bash
# 检查内存使用
free -h
docker stats

# 设置 swap（如果需要）
sudo fallocate -l 32G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

### 网络优化

```bash
# 如果需要外网访问，配置防火墙
sudo ufw allow 8000/tcp
sudo ufw allow 30000/tcp  # SGLang 服务器端口

# 配置反向代理（Nginx 示例）
sudo apt-get install nginx

# /etc/nginx/sites-available/mineru
server {
    listen 80;
    server_name your-domain.com;
    
    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        client_max_body_size 100M;
    }
}
```

## 🛡️ 安全配置

### 网络安全

```bash
# 限制容器网络访问
docker-compose.yml:
  networks:
    mineru-network:
      driver: bridge
      ipam:
        config:
          - subnet: 172.20.0.0/16
```

### 文件权限

```bash
# 设置适当的文件权限
sudo chown -R 1000:1000 /data/mineru
sudo chmod -R 755 /data/mineru
```

### API 安全

在生产环境中，建议添加认证和限流：

```python
# 在 app.py 中添加认证中间件
from fastapi.security import HTTPBearer
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address

# 添加限流
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# 添加 CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境中应限制具体域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

## 📈 监控和日志

### 日志配置

```bash
# 配置日志轮转
# /etc/logrotate.d/mineru
/data/mineru/logs/*.log {
    daily
    missingok
    rotate 7
    compress
    delaycompress
    notifempty
    create 644 root root
}
```

### 监控配置

```yaml
# docker-compose.yml 添加监控服务
  prometheus:
    image: prom/prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml

  grafana:
    image: grafana/grafana
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
```

## 🔄 备份和恢复

### 数据备份

```bash
# 备份脚本
#!/bin/bash
BACKUP_DIR="/backup/mineru/$(date +%Y%m%d)"
mkdir -p $BACKUP_DIR

# 备份配置文件
cp /data/mineru/mineru.json $BACKUP_DIR/
cp .env $BACKUP_DIR/
cp docker-compose.yml $BACKUP_DIR/

# 备份输出数据
tar -czf $BACKUP_DIR/output.tar.gz /data/mineru/output/

# 备份日志
tar -czf $BACKUP_DIR/logs.tar.gz /data/mineru/logs/
```

### 恢复流程

```bash
# 恢复配置
cp $BACKUP_DIR/*.json /data/mineru/
cp $BACKUP_DIR/.env ./
cp $BACKUP_DIR/docker-compose.yml ./

# 恢复数据
tar -xzf $BACKUP_DIR/output.tar.gz -C /
tar -xzf $BACKUP_DIR/logs.tar.gz -C /

# 重启服务
docker-compose down
docker-compose up -d
```

## 🚨 故障排除

### 常见问题

1. **容器启动失败**
   ```bash
   # 检查 Docker 日志
   docker logs mineru-vlm-api
   
   # 检查 GPU 访问
   docker run --rm --gpus all nvidia/cuda:11.8-base nvidia-smi
   ```

2. **模型下载失败**
   ```bash
   # 手动下载模型
   docker exec -it mineru-vlm-api bash
   mineru-models-download -s modelscope -m all
   ```

3. **内存不足**
   ```bash
   # 增加 swap
   sudo swapon --show
   sudo fallocate -l 32G /swapfile
   sudo swapon /swapfile
   ```

4. **GPU 内存不足**
   ```bash
   # 减少并发请求
   # 在环境变量中设置
   export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256
   ```

### 支持渠道

- GitHub Issues: https://github.com/opendatalab/MinerU/issues
- 官方文档: https://github.com/opendatalab/MinerU
- 社区讨论: Discord / 微信群

## 📝 部署检查清单

- [ ] 硬件要求确认
- [ ] 软件依赖安装
- [ ] 项目代码下载
- [ ] 环境变量配置
- [ ] 镜像构建成功
- [ ] 容器启动正常
- [ ] 健康检查通过
- [ ] API 功能测试
- [ ] 性能基准测试
- [ ] 监控配置完成
- [ ] 备份策略制定
- [ ] 安全配置审查

完成以上检查清单后，您的 MinerU VLM Web API 服务就可以投入使用了。 