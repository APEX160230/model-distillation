"""FastAPI Web 服务

提供 SSE 流式对话、健康检查、图谱统计接口。
并发限制为 1（2 核 4G 服务器约束）。
"""
from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel


class ChatRequest(BaseModel):
    question: str


def create_app(pipeline) -> FastAPI:
    """创建 FastAPI 应用

    Args:
        pipeline: RAGPipeline 实例
    """
    app = FastAPI(title="中医经典知识检索与理解助手", version="0.2.0")

    # 预加载知识图谱
    from src.data.kg_build import build_graph, graph_stats
    kg_graph = build_graph()
    kg_stats = graph_stats(kg_graph)

    @app.get("/", response_class=HTMLResponse)
    async def index():
        """返回前端页面"""
        html_path = Path(__file__).parent / "frontend.html"
        return HTMLResponse(html_path.read_text(encoding="utf-8"))

    @app.get("/api/health")
    async def health():
        """健康检查"""
        return {
            "status": "ok",
            "vector_store_count": pipeline.retriever.count(),
            "model": "qwen2.5:1.5b",
        }

    @app.get("/api/graph/stats")
    async def graph_stats_endpoint():
        """图谱统计"""
        return kg_stats

    @app.post("/api/chat")
    async def chat(req: ChatRequest):
        """流式对话（SSE）"""
        if not req.question.strip():
            raise HTTPException(status_code=400, detail="问题不能为空")

        def generate():
            # 检索
            docs, stream = pipeline.stream_query(req.question)

            # 先发送检索结果
            retrieved = [
                {
                    "clause_id": d.clause_id,
                    "chapter": d.chapter,
                    "text": d.text,
                    "score": round(d.score, 4),
                }
                for d in docs
            ]
            yield f"data: {json.dumps({'type': 'retrieved', 'docs': retrieved}, ensure_ascii=False)}\n\n"

            # 流式发送生成内容
            for chunk in stream:
                yield f"data: {json.dumps({'type': 'chunk', 'content': chunk}, ensure_ascii=False)}\n\n"

            # 结束标记
            yield f"data: {json.dumps({'type': 'done'})}\n\n"

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
        )

    return app
