"""评测指标模块

提供 recall@k、关键词命中率、延迟统计等评测指标计算。
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from collections import defaultdict


@dataclass
class EvalResult:
    """单条评测结果"""
    question_id: int
    category: str
    question: str
    answer: str
    expected_answer: str
    reference_clauses: list[int]
    retrieved_clauses: list[int]
    latency: float


def compute_recall_at_k(results: list[EvalResult], k: int = 5) -> float:
    """计算 recall@k

    在有 reference_clauses 的题目中，top-k 检索结果命中任一 reference 的比例。

    Args:
        results: 评测结果列表
        k: top-k

    Returns:
        recall@k，范围 [0, 1]
    """
    valid = [r for r in results if r.reference_clauses]
    if not valid:
        return 0.0
    hits = 0
    for r in valid:
        top_k = r.retrieved_clauses[:k]
        if any(ref in top_k for ref in r.reference_clauses):
            hits += 1
    return hits / len(valid)


def compute_keyword_accuracy(results: list[EvalResult]) -> float:
    """计算关键词命中率

    从 expected_answer 中提取 2+ 字中文关键词，检查是否出现在 answer 中。

    Args:
        results: 评测结果列表

    Returns:
        平均关键词命中率，范围 [0, 1]
    """
    scores = []
    for r in results:
        keywords = re.findall(r"[\u4e00-\u9fff]{2,}", r.expected_answer)
        keywords = list(dict.fromkeys(keywords))  # 去重保序
        if not keywords:
            scores.append(1.0)
            continue
        hits = sum(1 for kw in keywords if kw in r.answer)
        scores.append(hits / len(keywords))
    return sum(scores) / len(scores) if scores else 0.0


def compute_latency_stats(results: list[EvalResult]) -> dict[str, float]:
    """计算延迟统计

    Args:
        results: 评测结果列表

    Returns:
        包含 min, max, mean, p50, p95 的字典
    """
    if not results:
        return {"min": 0, "max": 0, "mean": 0, "p50": 0, "p95": 0}
    latencies = sorted(r.latency for r in results)
    n = len(latencies)

    def percentile(p: float) -> float:
        if n == 1:
            return latencies[0]
        rank = p * (n - 1)
        lower = int(math.floor(rank))
        upper = int(math.ceil(rank))
        if lower == upper:
            return latencies[lower]
        frac = rank - lower
        return latencies[lower] + frac * (latencies[upper] - latencies[lower])

    return {
        "min": latencies[0],
        "max": latencies[-1],
        "mean": round(sum(latencies) / n, 2),
        "p50": round(percentile(0.50), 2),
        "p95": round(percentile(0.95), 2),
    }


def compute_category_accuracy(results: list[EvalResult]) -> dict[str, float]:
    """按类别计算关键词命中率

    Args:
        results: 评测结果列表

    Returns:
        {类别: 平均关键词命中率}
    """
    cat_scores: dict[str, list[float]] = defaultdict(list)
    for r in results:
        keywords = re.findall(r"[\u4e00-\u9fff]{2,}", r.expected_answer)
        keywords = list(dict.fromkeys(keywords))
        if not keywords:
            cat_scores[r.category].append(1.0)
            continue
        hits = sum(1 for kw in keywords if kw in r.answer)
        cat_scores[r.category].append(hits / len(keywords))

    return {
        cat: round(sum(scores) / len(scores), 4)
        for cat, scores in cat_scores.items()
    }


def generate_report(results: list[EvalResult], stage: str = "P0") -> dict:
    """生成完整评测报告

    Args:
        results: 评测结果列表
        stage: 阶段标识（P0/P1/P2）

    Returns:
        完整报告字典
    """
    return {
        "stage": stage,
        "total_questions": len(results),
        "recall_at_5": round(compute_recall_at_k(results, k=5), 4),
        "keyword_accuracy": round(compute_keyword_accuracy(results), 4),
        "category_accuracy": compute_category_accuracy(results),
        "latency": compute_latency_stats(results),
        "details": [
            {
                "question_id": r.question_id,
                "category": r.category,
                "question": r.question,
                "answer": r.answer,
                "expected_answer": r.expected_answer,
                "reference_clauses": r.reference_clauses,
                "retrieved_clauses": r.retrieved_clauses,
                "latency": r.latency,
            }
            for r in results
        ],
    }
