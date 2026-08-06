"""RAG 管线编排 — 检索 → 生成

将向量检索和生成模块组合为完整的 RAG 管线。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

from src.rag.generate import Generator
from src.rag.retrieve import VectorRetriever, RetrievalResult


@dataclass
class RAGResponse:
    """RAG 完整响应"""
    answer: str
    retrieved_docs: list[RetrievalResult]
    latency: float


class RAGPipeline:
    """RAG 管线

    用法：
        pipeline = RAGPipeline()
        response = pipeline.query("桂枝汤的组成")
        print(response.answer)
    """

    def __init__(
        self,
        chroma_path: str = "data/chroma",
        model: str = "qwen25-15b-tcm",
        top_k: int = 5,
        generator=None,
    ) -> None:
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
        """完整 RAG 查询：检索 → 生成"""
        import time
        start = time.time()

        docs = self.retrieve(question)
        answer = self._generator.generate(
            question, docs,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        return RAGResponse(
            answer=answer,
            retrieved_docs=docs,
            latency=round(time.time() - start, 2),
        )

    def stream_query(
        self,
        question: str,
        temperature: float = 0.7,
        max_tokens: int = 512,
    ) -> tuple[list[RetrievalResult], Iterator[str]]:
        """流式 RAG 查询：先检索，再流式生成"""
        docs = self.retrieve(question)
        stream = self._generator.stream_generate(
            question, docs,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return docs, stream

    @property
    def retriever(self) -> VectorRetriever:
        """暴露检索器（用于统计/管理）"""
        return self._retriever
