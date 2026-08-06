"""混合检索器 — 路由 + 概念映射 + GraphRAG + BM25 + 向量

P2.1 升级：集成 ConceptMapper 和 GraphRAG

根据查询类型选择最优检索策略：
- clause_id: 直接 ID 查询（100% 准确）
- formula: 方剂名 → 倒排索引 → 条文 + 向量补充
- herb: 药材名 → GraphRAG → 方剂列表 + 条文
- comparison: GraphRAG 方剂对比 + 双路检索 → RRF 融合
- semantic: ConceptMapper 概念映射 → BM25+向量扩展 RRF 融合

使用 RRF (Reciprocal Rank Fusion) 融合多路检索结果。
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from src.data.formulas_db import FORMULAS, get_formula_dict, get_herb_formulas
from src.rag.bm25 import BM25Retriever
from src.rag.concept_mapper import ConceptMapper, ConceptMapping
from src.rag.graph_rag import TCMKnowledgeGraph
from src.rag.query_router import QueryRouter, QueryType, RouteResult
from src.rag.retrieve import RetrievalResult, VectorRetriever


def reciprocal_rank_fusion(
    result_lists: list[list[RetrievalResult]],
    k: int = 60,
) -> list[RetrievalResult]:
    """Reciprocal Rank Fusion 融合多路检索结果

    RRF 公式: score(d) = Σ 1/(k + rank_i(d))

    Args:
        result_lists: 多路检索结果列表
        k: RRF 参数（默认 60），平滑常数

    Returns:
            融合后的 RetrievalResult 列表，按 RRF 分数降序
    """
    rrf_scores: dict[str, float] = defaultdict(float)
    best_result: dict[str, RetrievalResult] = {}

    for result_list in result_lists:
        for rank, result in enumerate(result_list):
            doc_id = result.id
            rrf_scores[doc_id] += 1.0 / (k + rank + 1)
            if doc_id not in best_result:
                best_result[doc_id] = result

    # 按 RRF 分数排序
    sorted_ids = sorted(rrf_scores.keys(), key=lambda d: rrf_scores[d], reverse=True)

    results = []
    for doc_id in sorted_ids:
        result = best_result[doc_id]
        # 用 RRF 分数替换 distance（越低越相似，所以用 1 - normalized_score）
        max_score = max(rrf_scores.values()) if rrf_scores else 1.0
        normalized = rrf_scores[doc_id] / max_score if max_score > 0 else 0
        results.append(RetrievalResult(
            id=result.id,
            text=result.text,
            metadata=result.metadata,
            distance=1.0 - normalized,
        ))

    return results


class HybridRetriever:
    """混合检索器

    P2.1: 整合 ConceptMapper + GraphRAG + QueryRouter + BM25 + VectorRetriever

    用法：
        r = HybridRetriever()
        results = r.query("什么是阳明病？", top_k=5)
        context = r.last_context  # 获取额外生成上下文
    """

    def __init__(
        self,
        chroma_path: str = "data/chroma",
        clauses_path: str = "data/processed/classics/shanghan_clauses.jsonl",
    ) -> None:
        self._router = QueryRouter()
        self._bm25 = BM25Retriever()
        self._vector = VectorRetriever(persist_dir=chroma_path)
        self._concept_mapper = ConceptMapper()
        self._graph = TCMKnowledgeGraph()
        self._clauses: dict[int, dict] = {}  # clause_id → clause data
        self._formula_dict = get_formula_dict()
        self._herb_formulas = get_herb_formulas()
        self._last_route: RouteResult | None = None
        self._last_context: dict[str, Any] = {}  # 额外生成上下文

        # 加载条文数据
        if Path(clauses_path).exists():
            self._load_clauses(clauses_path)
            self._bm25.build_index(clauses_path)

    def _load_clauses(self, jsonl_path: str) -> None:
        """加载条文数据到内存"""
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    c = json.loads(line)
                    self._clauses[c["clause_id"]] = c

    def build_index(self, jsonl_path: str) -> None:
        """构建所有索引（向量 + BM25）"""
        self._load_clauses(jsonl_path)
        self._bm25.build_index(jsonl_path)
        self._vector.build_index(jsonl_path)

    @property
    def last_route(self) -> RouteResult | None:
        """最近一次路由结果"""
        return self._last_route

    @property
    def last_context(self) -> dict[str, Any]:
        """最近一次查询的额外生成上下文

        可能包含：
        - concept_brief: 概念简述（来自 ConceptMapper）
        - formula_list: 方剂列表（来自 GraphRAG，药材查询时）
        - comparison_data: 方剂对比数据（来自 GraphRAG，对比查询时）
        """
        return self._last_context

    def query(self, text: str, top_k: int = 5) -> list[RetrievalResult]:
        """混合检索

        Args:
            text: 查询文本
            top_k: 返回条数

        Returns:
            RetrievalResult 列表
        """
        # 重置上下文
        self._last_context = {}

        route = self._router.route(text)
        self._last_route = route

        if route.query_type == QueryType.CLAUSE_ID:
            return self._search_by_clause_id(route.clause_id, top_k)

        elif route.query_type == QueryType.FORMULA:
            return self._search_by_formula(route.formula_name, top_k)

        elif route.query_type == QueryType.HERB:
            return self._search_by_herbs(route.herbs, top_k)

        elif route.query_type == QueryType.COMPARISON:
            return self._search_comparison(route.entities, top_k)

        else:  # SEMANTIC
            return self._search_semantic(text, route.rewritten_query or text, top_k)

    def _clause_to_result(self, clause_id: int, distance: float = 0.0) -> RetrievalResult | None:
        """将 clause_id 转换为 RetrievalResult"""
        c = self._clauses.get(clause_id)
        if c is None:
            return None
        return RetrievalResult(
            id=f"clause_{clause_id}",
            text=c["original_text"],
            metadata={
                "clause_id": clause_id,
                "chapter": c.get("chapter", ""),
                "original_text": c["original_text"],
            },
            distance=distance,
        )

    def _clauses_to_results(self, clause_ids: list[int], distance: float = 0.0) -> list[RetrievalResult]:
        """批量将 clause_id 列表转换为 RetrievalResult"""
        results = []
        for cid in clause_ids:
            result = self._clause_to_result(cid, distance=distance)
            if result:
                results.append(result)
        return results

    def _search_by_clause_id(
        self, clause_id: int | None, top_k: int
    ) -> list[RetrievalResult]:
        """条文号直接查询"""
        if clause_id is None:
            return []

        result = self._clause_to_result(clause_id, distance=0.0)
        if result:
            return [result]

        # 如果条文不存在，回退到语义搜索
        return self._search_semantic(f"伤寒论第{clause_id}条", f"伤寒论第{clause_id}条", top_k)

    def _search_by_formula(
        self, formula_name: str | None, top_k: int
    ) -> list[RetrievalResult]:
        """方剂名查询：倒排索引 + BM25 + 向量"""
        if not formula_name:
            return []

        results: list[RetrievalResult] = []

        # 1. 从方剂数据库获取出处条文（精确匹配）
        formula_info = self._formula_dict.get(formula_name)
        if formula_info:
            result = self._clause_to_result(formula_info.clause_id, distance=0.0)
            if result:
                results.append(result)
            # 存储方剂信息到上下文
            self._last_context["formula_info"] = {
                "name": formula_info.name,
                "herbs": formula_info.herbs,
                "syndrome": formula_info.syndrome,
                "brief": formula_info.brief,
            }

        # 2. BM25 搜索包含该方剂名的条文
        bm25_results = self._bm25.search(formula_name, top_k=top_k * 2)
        existing_ids = {r.id for r in results}
        for r in bm25_results:
            if r.id not in existing_ids:
                results.append(r)
                existing_ids.add(r.id)

        # 3. 如果结果不足，用向量补充
        if len(results) < top_k:
            vec_results = self._vector.query(
                f"{formula_name} 组成 主之", top_k=top_k * 2
            )
            for r in vec_results:
                if r.id not in existing_ids:
                    results.append(r)
                    existing_ids.add(r.id)
                    if len(results) >= top_k:
                        break

        return results[:top_k]

    def _search_by_herbs(self, herbs: list[str], top_k: int) -> list[RetrievalResult]:
        """药材查询：GraphRAG → 方剂列表 → 条文

        P2.1: 使用 GraphRAG 进行结构化查询，并将完整方剂列表存入上下文。
        """
        if not herbs:
            return []

        # 使用 GraphRAG 查询
        if len(herbs) == 1:
            graph_result = self._graph.query_by_herb(herbs[0])
        else:
            graph_result = self._graph.query_by_herbs_intersection(herbs)

        # 存储方剂列表到上下文（供生成器使用）
        self._last_context["herb_query"] = {
            "herbs": herbs,
            "formula_names": graph_result.formula_names,
            "formula_count": len(graph_result.formula_names),
        }

        # 获取条文
        results = self._clauses_to_results(graph_result.clause_ids, distance=0.0)

        # 如果结果不足，用 BM25 补充
        if len(results) < top_k:
            bm25_query = " ".join(herbs)
            bm25_results = self._bm25.search(bm25_query, top_k=top_k)
            existing_ids = {r.id for r in results}
            for r in bm25_results:
                if r.id not in existing_ids:
                    results.append(r)
                    existing_ids.add(r.id)
                    if len(results) >= top_k:
                        break

        return results[:top_k]

    def _search_comparison(
        self, entities: list[str], top_k: int
    ) -> list[RetrievalResult]:
        """对比查询：GraphRAG 方剂对比 + 双路检索 → RRF 融合

        P2.1: 使用 GraphRAG 获取结构化对比数据（组成差异、主治差异）。
        """
        if not entities or len(entities) < 2:
            return self._search_semantic(
                " ".join(entities) if entities else "",
                " ".join(entities) if entities else "",
                top_k,
            )

        # GraphRAG 获取对比数据
        comparison = self._graph.compare_formulas(entities[0], entities[1])
        if "error" not in comparison:
            self._last_context["comparison"] = comparison

        all_results: list[list[RetrievalResult]] = []

        for entity in entities[:2]:
            entity_results: list[RetrievalResult] = []

            # 方剂倒排索引
            formula_info = self._formula_dict.get(entity)
            if formula_info:
                result = self._clause_to_result(formula_info.clause_id, distance=0.0)
                if result:
                    entity_results.append(result)

            # BM25
            bm25_results = self._bm25.search(entity, top_k=top_k)
            existing = {r.id for r in entity_results}
            for r in bm25_results:
                if r.id not in existing:
                    entity_results.append(r)
                    existing.add(r.id)

            # 向量
            if len(entity_results) < top_k:
                vec_results = self._vector.query(entity, top_k=top_k)
                for r in vec_results:
                    if r.id not in existing:
                        entity_results.append(r)
                        existing.add(r.id)

            all_results.append(entity_results[:top_k])

        # RRF 融合
        return reciprocal_rank_fusion(all_results)[:top_k]

    def _search_semantic(
        self, original_query: str, rewritten_query: str, top_k: int
    ) -> list[RetrievalResult]:
        """语义查询：ConceptMapper → GraphRAG → BM25+向量 RRF 融合

        P2.1 升级：
        1. 先查 ConceptMapper，命中则直接返回映射条文
        2. 未命中则用 GraphRAG 查证候→方剂→条文
        3. 最后 fallback 到 BM25+向量 RRF 融合
        """
        # 1. ConceptMapper 概念映射
        concept = self._concept_mapper.lookup(original_query)
        if concept and concept.all_clauses:
            # 存储概念信息到上下文
            self._last_context["concept"] = {
                "concept": concept.concept,
                "brief": concept.brief,
                "related_formulas": concept.related_formulas,
                "defining_clauses": concept.defining_clauses,
                "treatment_clauses": concept.treatment_clauses,
            }

            results = self._clauses_to_results(concept.all_clauses, distance=0.0)

            # 用扩展关键词做 BM25 补充
            if len(results) < top_k and concept.expansion_keywords:
                expanded_query = " ".join(concept.expansion_keywords)
                bm25_results = self._bm25.search(expanded_query, top_k=top_k * 2)
                existing_ids = {r.id for r in results}
                for r in bm25_results:
                    if r.id not in existing_ids:
                        results.append(r)
                        existing_ids.add(r.id)
                        if len(results) >= top_k:
                            break

            return results[:top_k]

        # 2. GraphRAG 证候查询（模糊匹配）
        # 提取核心概念词
        core = rewritten_query.strip("？?。.，,的")
        graph_result = self._graph.query_by_syndrome(core)
        if graph_result.clause_ids:
            self._last_context["graph_syndrome"] = {
                "syndrome": core,
                "formulas": graph_result.formula_names,
            }
            results = self._clauses_to_results(graph_result.clause_ids, distance=0.0)
            if len(results) >= top_k:
                return results[:top_k]

            # 不足则补充 BM25+向量
            remaining = top_k - len(results)
            bm25_results = self._bm25.search(rewritten_query, top_k=remaining + top_k)
            vec_results = self._vector.query(rewritten_query, top_k=top_k * 2)
            fused = reciprocal_rank_fusion([bm25_results, vec_results])
            existing_ids = {r.id for r in results}
            for r in fused:
                if r.id not in existing_ids:
                    results.append(r)
                    existing_ids.add(r.id)
                    if len(results) >= top_k:
                        break
            return results[:top_k]

        # 3. Fallback: BM25 + 向量 RRF 融合（用扩展查询）
        expanded = self._concept_mapper.expand_query(original_query)
        bm25_results = self._bm25.search(expanded, top_k=top_k * 2)
        vec_results = self._vector.query(rewritten_query, top_k=top_k * 2)

        return reciprocal_rank_fusion([bm25_results, vec_results])[:top_k]

    def count(self) -> int:
        """索引中的文档数"""
        return len(self._clauses)

    @property
    def graph_stats(self) -> dict[str, int]:
        """知识图谱统计"""
        return {
            "nodes": self._graph.node_count,
            "edges": self._graph.edge_count,
            "formulas": self._graph.formula_count,
            "herbs": self._graph.herb_count,
            "syndromes": self._graph.syndrome_count,
        }
