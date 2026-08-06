"""BM25 关键词检索模块

使用 jieba 中文分词 + BM25 算法实现精确关键词匹配检索。
解决向量搜索无法精确匹配 "桂枝汤主之" 等关键词的问题。

与 VectorRetriever 互补：
- VectorRetriever: 语义相似度，擅长理解同义/近义
- BM25Retriever: 关键词精确匹配，擅长专有名词匹配
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import jieba
from rank_bm25 import BM25Okapi

from src.rag.retrieve import RetrievalResult


@dataclass
class BM25Document:
    """BM25 索引文档"""
    clause_id: int
    chapter: str
    original_text: str
    tokens: list[str]


class BM25Retriever:
    """BM25 关键词检索器

    使用 jieba 分词对伤寒论条文建立 BM25 索引。
    适合精确关键词匹配，如 "桂枝汤" → 匹配包含 "桂枝汤" 的条文。

    用法：
        r = BM25Retriever()
        r.build_index("data/processed/classics/shanghan_clauses.jsonl")
        results = r.search("桂枝汤组成", top_k=5)
    """

    def __init__(self, jsonl_path: str | None = None) -> None:
        self._documents: list[BM25Document] = []
        self._bm25: BM25Okapi | None = None
        self._clause_to_idx: dict[int, int] = {}

        if jsonl_path:
            self.build_index(jsonl_path)

    def build_index(self, jsonl_path: str) -> None:
        """从 JSONL 文件构建 BM25 索引

        Args:
            jsonl_path: JSONL 文件路径，每行包含 clause_id 和 original_text
        """
        clauses = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    clauses.append(json.loads(line))

        self._documents = []
        self._clause_to_idx = {}

        # 添加自定义词典：方剂名和药材名
        from src.data.formulas_db import FORMULAS, get_all_herbs

        for formula in FORMULAS:
            jieba.add_word(formula.name)
            for herb in formula.herbs:
                jieba.add_word(herb)

        for i, c in enumerate(clauses):
            text = c["original_text"]
            tokens = list(jieba.cut(text))
            doc = BM25Document(
                clause_id=c["clause_id"],
                chapter=c.get("chapter", ""),
                original_text=text,
                tokens=tokens,
            )
            self._documents.append(doc)
            self._clause_to_idx[c["clause_id"]] = i

        # 构建 BM25 索引
        corpus = [doc.tokens for doc in self._documents]
        self._bm25 = BM25Okapi(corpus)

    def search(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        """BM25 关键词检索

        Args:
            query: 查询文本
            top_k: 返回条数

        Returns:
            RetrievalResult 列表，按 BM25 分数降序
        """
        if not self._bm25 or not self._documents:
            raise RuntimeError("BM25 index not built. Call build_index() first.")

        # 分词查询
        query_tokens = list(jieba.cut(query))
        # 过滤空 token 和停用词
        query_tokens = [t for t in query_tokens if t.strip() and len(t) > 0]

        if not query_tokens:
            return []

        # BM25 搜索
        scores = self._bm25.get_scores(query_tokens)

        # 获取 top_k 结果
        ranked_indices = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )[:top_k]

        results = []
        for idx in ranked_indices:
            doc = self._documents[idx]
            score = float(scores[idx])
            if score <= 0:
                continue  # 跳过零分结果
            results.append(RetrievalResult(
                id=f"clause_{doc.clause_id}",
                text=doc.original_text,
                metadata={
                    "clause_id": doc.clause_id,
                    "chapter": doc.chapter,
                    "original_text": doc.original_text,
                },
                distance=1.0 - score,  # 转换为 distance（越低越相似）
            ))

        return results

    def search_by_clause_ids(self, clause_ids: list[int]) -> list[RetrievalResult]:
        """直接按 clause_id 列表获取条文

        Args:
            clause_ids: 条文 ID 列表

        Returns:
            RetrievalResult 列表，保持输入顺序
        """
        results = []
        for cid in clause_ids:
            idx = self._clause_to_idx.get(cid)
            if idx is not None:
                doc = self._documents[idx]
                results.append(RetrievalResult(
                    id=f"clause_{doc.clause_id}",
                    text=doc.original_text,
                    metadata={
                        "clause_id": doc.clause_id,
                        "chapter": doc.chapter,
                        "original_text": doc.original_text,
                    },
                    distance=0.0,  # 精确匹配，distance=0
                ))
        return results

    @property
    def count(self) -> int:
        """索引中的文档数"""
        return len(self._documents)
