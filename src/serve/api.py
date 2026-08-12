"""FastAPI Web 服务

提供 SSE 流式对话、健康检查、图谱统计接口。
并发限制为 1（2 核 4G 服务器约束）。

P2.1: 暴露路由类型、概念映射、GraphRAG 统计到前端。
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from pydantic import BaseModel, Field

logger = logging.getLogger("tcm_serve")


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=500, description="用户问题")


class ChatResponse(BaseModel):
    answer: str
    route_type: str
    retrieved_clauses: list[int]
    latency: float


def _serialize_context_extras(extras: dict[str, Any] | None) -> dict[str, Any] | None:
    """将 context_extras 序列化为 JSON 安全格式"""
    if not extras:
        return None
    safe = {}
    for key, val in extras.items():
        try:
            # 测试是否能 JSON 序列化
            json.dumps(val, ensure_ascii=False)
            safe[key] = val
        except (TypeError, ValueError):
            safe[key] = str(val)
    return safe


def create_app(pipeline) -> FastAPI:
    """创建 FastAPI 应用

    Args:
        pipeline: RAGPipeline 实例
    """
    app = FastAPI(
        title="中医经典知识检索与理解助手",
        description="基于 LoRA 微调 + GraphRAG 的伤寒论问答系统",
        version="1.0.0",
    )

    # CORS — 允许前端独立部署
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    # 并发锁 — 2核4G 约束，同时只处理一个推理请求
    inference_lock = asyncio.Lock()

    # ── 健康检查 ──
    @app.get("/api/health")
    async def health():
        """健康检查"""
        try:
            # 尝试获取检索器状态
            if pipeline.hybrid_retriever:
                stats = pipeline.hybrid_retriever.graph_stats
                return {
                    "status": "ok",
                    "model": pipeline._generator._model,
                    "retriever": "hybrid",
                    "graph_stats": stats,
                }
            else:
                count = pipeline.retriever.count()
                return {
                    "status": "ok",
                    "model": pipeline._generator._model,
                    "retriever": "vector",
                    "vector_store_count": count,
                }
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return JSONResponse(
                status_code=503,
                content={"status": "degraded", "error": str(e)},
            )

    # ── 图谱统计 ──
    @app.get("/api/graph/stats")
    async def graph_stats():
        """知识图谱统计"""
        if pipeline.hybrid_retriever:
            return pipeline.hybrid_retriever.graph_stats
        raise HTTPException(status_code=404, detail="GraphRAG 未启用")

    # ── 示例问题 ──
    @app.get("/api/examples")
    async def examples():
        """返回示例问题列表"""
        return {
            "examples": [
                {"category": "辨证建议", "question": "我这两天头痛，怕冷，不出汗"},
                {"category": "辨证建议", "question": "发烧了，有出汗，怕风"},
                {"category": "辨证建议", "question": "嘴里发苦，嗓子干，有点头晕"},
                {"category": "辨证建议", "question": "拉肚子，肚子疼，手脚发凉"},
                {"category": "辨证建议", "question": "这几天拉肚子，肚子胀，没胃口"},
                {"category": "知识问答", "question": "桂枝汤的组成是什么？"},
                {"category": "知识问答", "question": "什么是太阳病？"},
                {"category": "知识问答", "question": "桂枝汤和麻黄汤有什么区别？"},
                {"category": "知识问答", "question": "什么是太阳中风证？"},
            ]
        }

    # ── 流式对话 ──
    @app.post("/api/chat")
    async def chat(req: ChatRequest, request: Request):
        """流式对话（SSE）

        SSE 事件类型：
        - retrieved: 检索到的经典原文 + 路由类型 + 上下文
        - chunk: 生成的文本片段
        - done: 生成完成 + 延迟
        - error: 错误信息
        """
        question = req.question.strip()
        if not question:
            raise HTTPException(status_code=400, detail="问题不能为空")

        client_ip = request.client.host if request.client else "unknown"
        logger.info(f"[{client_ip}] Question: {question[:80]}")

        # 获取并发锁
        if inference_lock.locked():
            logger.warning(f"[{client_ip}] Server busy, rejecting request")
            raise HTTPException(status_code=503, detail="服务器正忙，请稍后再试")

        async def generate():
            async with inference_lock:
                start_time = time.time()
                try:
                    # 检索 + 流式生成
                    docs, stream, route_type, context_extras = pipeline.stream_query(question)

                    # 发送检索结果
                    retrieved = [
                        {
                            "clause_id": d.clause_id,
                            "chapter": d.chapter,
                            "text": d.text,
                            "score": round(d.score, 4) if hasattr(d, "score") else 0.0,
                        }
                        for d in docs
                    ]

                    meta = {
                        "type": "retrieved",
                        "docs": retrieved,
                        "route_type": route_type,
                        "context_extras": _serialize_context_extras(context_extras),
                    }
                    yield f"data: {json.dumps(meta, ensure_ascii=False)}\n\n"

                    # 流式发送生成内容
                    for chunk in stream:
                        if await request.is_disconnected():
                            logger.info(f"[{client_ip}] Client disconnected")
                            return
                        yield f"data: {json.dumps({'type': 'chunk', 'content': chunk}, ensure_ascii=False)}\n\n"

                    # 结束标记
                    latency = round(time.time() - start_time, 2)
                    yield f"data: {json.dumps({'type': 'done', 'latency': latency}, ensure_ascii=False)}\n\n"
                    logger.info(f"[{client_ip}] Done in {latency}s, route={route_type}")

                except ConnectionError as e:
                    logger.error(f"[{client_ip}] Ollama connection error: {e}")
                    yield f"data: {json.dumps({'type': 'error', 'message': '模型服务不可用，请确认 Ollama 已启动'}, ensure_ascii=False)}\n\n"
                except Exception as e:
                    logger.error(f"[{client_ip}] Unexpected error: {e}", exc_info=True)
                    yield f"data: {json.dumps({'type': 'error', 'message': f'内部错误: {e}'}, ensure_ascii=False)}\n\n"

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",  # Nginx 不缓冲
            },
        )

    # ── 前端页面 ──
    @app.get("/", response_class=HTMLResponse)
    async def index():
        """返回前端页面"""
        html_path = Path(__file__).parent / "frontend.html"
        if not html_path.exists():
            raise HTTPException(status_code=404, detail="前端文件未找到")
        return HTMLResponse(html_path.read_text(encoding="utf-8"))

    return app
