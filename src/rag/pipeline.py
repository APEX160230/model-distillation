"""RAG 管线编排 — 检索 → 生成

P2.1: 支持 ConceptMapper/GraphRAG 结构化上下文传递。
将检索器的额外上下文（概念简述、方剂对比、药材列表）传给生成器，
帮助 1.5B 模型生成更准确的答案。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator, Optional

from src.rag.generate import Generator
from src.rag.retrieve import RetrievalResult, VectorRetriever
from src.rag.hybrid_retriever import HybridRetriever


@dataclass
class RAGResponse:
    """RAG 完整响应"""
    answer: str
    retrieved_docs: list[RetrievalResult]
    latency: float
    route_type: str = ""  # 查询路由类型
    context_extras: dict[str, Any] | None = None  # P2.1 额外上下文


class RAGPipeline:
    """RAG 管线

    P2.1 默认使用 HybridRetriever（概念映射 + GraphRAG + 查询路由 + BM25 + 向量）。
    设置 use_hybrid=False 可回退到纯向量检索（用于 P0/P1 对比）。

    用法：
        pipeline = RAGPipeline()
        response = pipeline.query("什么是阳明病？")
        print(response.answer)
        print(response.context_extras)  # 概念简述等
    """

    def __init__(
        self,
        chroma_path: str = "data/chroma",
        clauses_path: str = "data/processed/classics/shanghan_clauses.jsonl",
        model: str = "qwen25-15b-tcm",
        top_k: int = 5,
        generator=None,
        use_hybrid: bool = True,
    ) -> None:
        if use_hybrid:
            self._hybrid_retriever = HybridRetriever(
                chroma_path=chroma_path,
                clauses_path=clauses_path,
            )
            self._retriever = self._hybrid_retriever
        else:
            self._hybrid_retriever = None
            self._retriever = VectorRetriever(persist_dir=chroma_path)

        self._generator = generator or Generator(model=model)
        self._top_k = top_k

    def retrieve(self, question: str) -> list[RetrievalResult]:
        """仅检索，不生成"""
        return self._retriever.query(question, top_k=self._top_k)

    def query(
        self,
        question: str,
        temperature: float = 0.7,
        max_tokens: int = 512,
    ) -> RAGResponse:
        """完整 RAG 查询：检索 → 生成

        P2.1: 自动从 HybridRetriever 获取额外上下文并传给生成器。
        """
        import time
        start = time.time()

        docs = self.retrieve(question)

        # 获取路由类型和额外上下文（P2.1）
        route_type = ""
        context_extras: dict[str, Any] | None = None
        if self._hybrid_retriever:
            if self._hybrid_retriever.last_route:
                route_type = self._hybrid_retriever.last_route.query_type.value
            context_extras = self._hybrid_retriever.last_context or None

        answer = self._generator.generate(
            question, docs,
            temperature=temperature,
            max_tokens=max_tokens,
            context_extras=context_extras,
        )

        return RAGResponse(
            answer=answer,
            retrieved_docs=docs,
            latency=round(time.time() - start, 2),
            route_type=route_type,
            context_extras=context_extras,
        )

    def stream_query(
        self,
        question: str,
        temperature: float = 0.7,
        max_tokens: int = 512,
    ) -> tuple[list[RetrievalResult], Iterator[str], str, dict[str, Any] | None]:
        """流式 RAG 查询：先检索，再流式生成

        Returns:
            (docs, stream, route_type, context_extras)
        """
        docs = self.retrieve(question)

        route_type = ""
        context_extras: dict[str, Any] | None = None
        if self._hybrid_retriever:
            if self._hybrid_retriever.last_route:
                route_type = self._hybrid_retriever.last_route.query_type.value
            context_extras = self._hybrid_retriever.last_context or None

        stream = self._generator.stream_generate(
            question, docs,
            temperature=temperature,
            max_tokens=max_tokens,
            context_extras=context_extras,
        )
        return docs, stream, route_type, context_extras

    @property
    def retriever(self):
        """暴露检索器"""
        return self._retriever

    @property
    def hybrid_retriever(self) -> HybridRetriever | None:
        """暴露混合检索器（用于访问 graph_stats 等）"""
        return self._hybrid_retriever
