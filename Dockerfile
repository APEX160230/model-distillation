# ── 构建阶段 ──
FROM python:3.12-slim AS builder

WORKDIR /build

# 系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ && \
    rm -rf /var/lib/apt/lists/*

# 先装依赖（利用 Docker 层缓存）
COPY pyproject.toml .
RUN pip install --no-cache-dir --prefix=/install \
    "fastapi>=0.111" "uvicorn[standard]>=0.30" \
    "chromadb>=0.5" "sentence-transformers>=2.7" \
    "networkx>=3.0" "jieba>=0.42" "rank_bm25>=0.2" \
    "requests>=2.31" "pandas>=2.0"

# ── 运行阶段 ──
FROM python:3.12-slim

WORKDIR /app

# 从构建阶段复制已安装的包
COPY --from=builder /install /usr/local

# 复制源码
COPY src/ ./src/
COPY data/processed/classics/ ./data/processed/classics/
COPY data/processed/sft_train_p1.jsonl ./data/processed/

# 环境变量
ENV PYTHONUNBUFFERED=1
ENV HF_ENDPOINT=https://hf-mirror.com
ENV PORT=8000
ENV MODEL=qwen25-15b-tcm
ENV USE_HYBRID=true

# ChromaDB 持久化目录
VOLUME ["/app/data/chroma"]

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/api/health')" || exit 1

CMD ["python", "-m", "src.serve.main"]
