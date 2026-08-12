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
    │  https (nginx :443, SSE 需关闭缓冲)
    ▼
tcm-backend (systemd, FastAPI :8002)
    ├── QueryRouter     查询路由
    ├── ConceptMapper   概念映射
    ├── BM25            关键词检索
    ├── VectorRetriever 向量检索（ChromaDB）
    ├── GraphRAG        知识图谱多跳查询
    └── Generator ──► Ollama (127.0.0.1:11434)
                    └── tcm-model (Q4 量化, ~1.2 GB)
```

## 方式一：裸机部署（推荐 2 核 4G 服务器）

### 1. 安装系统依赖

```bash
# Python 3.11+
python3 --version

# 安装 Ollama
curl -fsSL https://ollama.com/install.sh | sh
```

### 2. 导入微调模型

将微调后的 GGUF 文件（如 `qwen25-15b-tcm-q4km.gguf`）放到工作目录，与 `Modelfile` 同目录：

```bash
ollama create tcm-model -f Modelfile

# 验证
ollama list
# 应显示: tcm-model

# 测试
ollama run tcm-model "桂枝汤主治什么？"
```

> 提示：系统提示词（口语化讲解风格）已内置在代码 `src/rag/generate.py` 的 `SYSTEM_PROMPT` 中，无需在 Modelfile 重复配置。

### 3. 安装 Python 依赖

```bash
python3 -m venv .venv
source .venv/bin/activate  # Linux
# .venv\Scripts\activate   # Windows

pip install -e ".[rag,serve]"
```

### 4. 初始化向量库

```bash
# 构建 ChromaDB 向量库（幂等：条文数据未变更则跳过）
# 首次运行会自动下载 bge-small-zh-v1.5 模型
python scripts/build_chroma.py

# 服务器无外网或下载失败时，可指定本地模型路径
python scripts/build_chroma.py --model /path/to/bge-small-zh-v1.5
```

### 5. 配置 systemd 服务

```ini
# /etc/systemd/system/tcm-backend.service
[Unit]
Description=TCM Assistant Backend
After=network.target ollama.service

[Service]
User=ubuntu
WorkingDirectory=/home/ubuntu/tcm/backend
ExecStart=/home/ubuntu/tcm/backend/.venv/bin/python -m src.serve.main
Restart=always
RestartSec=5
Environment=PORT=8002
Environment=MODEL=tcm-model

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now tcm-backend
sudo systemctl status tcm-backend
```

### 6. Nginx 反向代理（HTTPS）

```nginx
# /etc/nginx/sites-available/tcm
server {
    listen 80;
    server_name tcm.example.com;
    return 301 https://$host$request_uri;
}

server {
    listen 443 ssl;
    server_name tcm.example.com;

    ssl_certificate     /etc/letsencrypt/live/tcm.example.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/tcm.example.com/privkey.pem;

    location / {
        proxy_pass http://127.0.0.1:8002;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;

        # SSE 流式必需
        proxy_buffering off;
        proxy_cache off;
        proxy_read_timeout 120s;
        proxy_http_version 1.1;
    }
}
```

## 方式二：Docker 部署

```bash
# 构建并启动
docker-compose up -d

# 查看日志
docker-compose logs -f app
```

首次启动后需要导入模型：

```bash
# 进入 Ollama 容器
docker exec -it tcm-ollama sh
# 导入微调模型
ollama create tcm-model -f /Modelfile
```

## 方式三：CI/CD 自动部署（推荐）

推送 main 分支后，GitHub Actions 自动执行：

1. **测试**（74 个：单元 + 集成）→ 2. **脚本校验 + 前端冒烟** → 3. **部署**

部署脚本 `scripts/deploy.sh` 流程：

```
[1/6] 上传源码 src/
[2/6] 上传数据资产 + 重建脚本
[3/6] 上传 Modelfile
[4/6] 条文变更时重建向量库（md5 幂等，未变更跳过）
[5/6] 重启 tcm-backend 服务
[6/6] 健康检查（内网 + 公网 https）
```

需要配置的 GitHub Secrets：

| Secret | 说明 |
|--------|------|
| `DEPLOY_HOST` | 服务器地址（如 zzy1n.cc） |
| `DEPLOY_USER` | SSH 用户（默认 ubuntu） |
| `DEPLOY_KEY` | 部署专用 SSH 私钥 |

## 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `PORT` | `8000` | 服务端口 |
| `MODEL` | `tcm-model` | Ollama 模型名 |
| `USE_HYBRID` | `true` | 启用混合检索（false 回退纯向量） |
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama 服务地址 |
| `HF_ENDPOINT` | `https://hf-mirror.com` | HuggingFace 镜像 |
| `LOG_LEVEL` | `INFO` | 日志级别 |

## 内存预算（2 核 4G）

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
| recall@5 | 100%（评测集）/ 90.5%（真实场景） |
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
python scripts/build_chroma.py
```

### 国内服务器无法下载 bge 模型（hf-mirror 308 问题）

huggingface_hub 对 hf-mirror 的 `resolve/` 路径 HEAD 请求会失败（返回 308 且无法跟随），导致 `LocalEntryNotFoundError`。解决方案：

```bash
# 1. 用 requests GET 直连下载模型文件到本地（跟随 308 重定向）
python ~/.workbuddy/skills/hf-mirror-download-fix/scripts/download_model_direct.py \
    --repo BAAI/bge-small-zh-v1.5 --out /home/ubuntu/tcm/models/bge-small-zh-v1.5

# 2. 构建时指定本地路径
python scripts/build_chroma.py --model /home/ubuntu/tcm/models/bge-small-zh-v1.5
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
