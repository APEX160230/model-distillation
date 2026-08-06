# 部署指南

## 环境要求

| 项目 | 最低 | 推荐 |
|------|------|------|
| CPU | 2 核 | 4 核 |
| 内存 | 4 GB | 8 GB |
| 磁盘 | 10 GB | 20 GB |
| GPU | 不需要 | 不需要 |

## 架构

```
用户浏览器
    │
    ▼
FastAPI (端口 8000)
    ├── HybridRetriever
    │   ├── QueryRouter (查询路由)
    │   ├── ConceptMapper (概念映射)
    │   ├── BM25 (关键词检索)
    │   ├── VectorRetriever (ChromaDB 向量检索)
    │   └── GraphRAG (知识图谱多跳查询)
    │
    └── Generator → Ollama (端口 11434)
                    └── qwen25-15b-tcm (Q4 量化, ~1.2GB)
```

## 方式一：直接部署（推荐 2核4G 服务器）

### 1. 安装系统依赖

```bash
# Python 3.12+
python3 --version

# 安装 Ollama
curl -fsSL https://ollama.com/install.sh | sh
```

### 2. 加载模型

```bash
# 导入微调后的 GGUF 模型
ollama create qwen25-15b-tcm -f Modelfile

# 验证
ollama list
# 应显示: qwen25-15b-tcm

# 测试
ollama run qwen25-15b-tcm "你好"
```

Modelfile 示例:
```
FROM ./models/qwen25-15b-tcm-q4_k_m.gguf
PARAMETER temperature 0.7
PARAMETER num_ctx 2048
```

### 3. 安装 Python 依赖

```bash
python3 -m venv .venv
source .venv/bin/activate  # Linux
# .venv\Scripts\activate   # Windows

pip install -e ".[rag,serve]"
```

### 4. 初始化向量库

```bash
# 首次运行需要构建 ChromaDB 向量库
python -m src.data.extract
python -m src.data.clean
python -m src.rag.embed
```

### 5. 启动服务

```bash
# 开发模式
python -m src.serve.main

# 生产模式（推荐用 gunicorn 管理 uvicorn worker）
pip install gunicorn
gunicorn src.serve.main:app -w 1 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:8000

# 后台运行
nohup python -m src.serve.main > server.log 2>&1 &
```

### 6. Nginx 反向代理（可选）

```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;

        # SSE 支持
        proxy_buffering off;
        proxy_cache off;
        proxy_read_timeout 120s;
    }
}
```

## 方式二：Docker 部署

```bash
# 构建并启动
docker-compose up -d

# 查看日志
docker-compose logs -f app

# 停止
docker-compose down
```

首次启动后需要拉取模型:
```bash
# 进入 Ollama 容器拉取模型
docker exec -it tcm-ollama ollama pull qwen2.5:1.5b
# 或导入微调模型
docker cp models/qwen25-15b-tcm-q4_k_m.gguf tcm-ollama:/tmp/
docker exec -it tcm-ollama ollama create qwen25-15b-tcm -f /tmp/Modelfile
```

## 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `PORT` | `8000` | 服务端口 |
| `MODEL` | `qwen25-15b-tcm` | Ollama 模型名 |
| `USE_HYBRID` | `true` | 启用混合检索（false 回退纯向量） |
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama 服务地址 |
| `HF_ENDPOINT` | `https://hf-mirror.com` | HuggingFace 镜像 |
| `LOG_LEVEL` | `INFO` | 日志级别 |

## 内存预算（2核4G）

| 组件 | 内存占用 |
|------|----------|
| Ollama (Q4 模型) | ~1.2 GB |
| ChromaDB + Embedding | ~0.8 GB |
| FastAPI + NetworkX | ~0.2 GB |
| 系统 + 其他 | ~0.8 GB |
| **总计** | **~3.0 GB** |
| 余量 | ~1.0 GB |

## 性能指标

| 指标 | 数值 |
|------|------|
| 检索延迟 | < 100 ms |
| 生成延迟 (p50) | 2-5 s |
| 生成延迟 (p95) | 8-15 s |
| 并发 | 1（锁死） |
| recall@5 | 100% |
| LLM judge 均分 | 3.56 / 5 |

## 故障排查

### Ollama 连接失败
```bash
# 检查 Ollama 是否运行
curl http://localhost:11434/api/tags

# 检查模型是否加载
ollama list
```

### ChromaDB 初始化失败
```bash
# 删除并重建向量库
rm -rf data/chroma/
python -m src.rag.embed
```

### 内存不足
```bash
# 检查内存使用
free -h

# 添加 Swap
sudo fallocate -l 4G /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```
