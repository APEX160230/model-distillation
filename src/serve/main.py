"""服务启动入口

用法:
    python -m src.serve.main

    # 或指定端口
    PORT=8080 python -m src.serve.main
"""
import os

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import uvicorn
from src.rag.pipeline import RAGPipeline
from src.serve.api import create_app


def main():
    pipeline = RAGPipeline(
        chroma_path="data/chroma",
        model="qwen25-15b-tcm",
        top_k=5,
    )
    app = create_app(pipeline)

    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port, workers=1)


if __name__ == "__main__":
    main()
