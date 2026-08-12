"""服务启动入口

用法:
    # 默认端口 8000
    python -m src.serve.main

    # 指定端口
    PORT=8080 python -m src.serve.main

    # 禁用混合检索（回退纯向量，用于对比）
    USE_HYBRID=false python -m src.serve.main
"""
import logging
import os
import sys

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

# 日志配置
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("tcm_serve")


def main():
    from src.rag.pipeline import RAGPipeline
    from src.serve.api import create_app
    import uvicorn

    use_hybrid = os.environ.get("USE_HYBRID", "true").lower() != "false"
    model = os.environ.get("MODEL", "tcm-model")
    port = int(os.environ.get("PORT", "8000"))

    logger.info(f"启动 RAG 管道: model={model}, hybrid={use_hybrid}")

    pipeline = RAGPipeline(
        chroma_path="data/chroma",
        model=model,
        top_k=5,
        use_hybrid=use_hybrid,
    )

    # 打印知识图谱统计
    if pipeline.hybrid_retriever:
        stats = pipeline.hybrid_retriever.graph_stats
        logger.info(f"知识图谱: {stats}")

    app = create_app(pipeline)
    logger.info(f"服务启动: http://0.0.0.0:{port}")

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        workers=1,
        log_level="info",
    )


if __name__ == "__main__":
    main()
