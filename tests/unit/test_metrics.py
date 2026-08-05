"""评测指标模块测试"""
import pytest
from src.eval.metrics import (
    EvalResult,
    compute_recall_at_k,
    compute_keyword_accuracy,
    compute_latency_stats,
    compute_category_accuracy,
    generate_report,
)


def _make_result(**kwargs):
    """快速构造 EvalResult"""
    defaults = {
        "question_id": 1,
        "category": "test",
        "question": "q",
        "answer": "a",
        "expected_answer": "e",
        "reference_clauses": [],
        "retrieved_clauses": [],
        "latency": 1.0,
    }
    defaults.update(kwargs)
    return EvalResult(**defaults)


class TestRecallAtK:
    def test_perfect_recall(self):
        results = [_make_result(reference_clauses=[1], retrieved_clauses=[1, 2, 3])]
        assert compute_recall_at_k(results, k=5) == 1.0

    def test_zero_recall(self):
        results = [_make_result(reference_clauses=[1], retrieved_clauses=[2, 3, 4])]
        assert compute_recall_at_k(results, k=5) == 0.0

    def test_partial_recall(self):
        r1 = _make_result(reference_clauses=[1], retrieved_clauses=[1, 2])
        r2 = _make_result(reference_clauses=[5], retrieved_clauses=[1, 2])
        assert compute_recall_at_k([r1, r2], k=5) == 0.5

    def test_skip_questions_without_references(self):
        r1 = _make_result(reference_clauses=[], retrieved_clauses=[1, 2])
        r2 = _make_result(reference_clauses=[3], retrieved_clauses=[3, 4])
        assert compute_recall_at_k([r1, r2], k=5) == 1.0


class TestKeywordAccuracy:
    def test_all_keywords_present(self):
        # 关键词提取: re.findall(r"[\u4e00-\u9fff]{2,}", ...) 匹配连续中文
        # "桂枝汤、芍药、甘草" → ["桂枝汤", "芍药", "甘草"]
        r = _make_result(expected_answer="桂枝汤、芍药、甘草", answer="桂枝汤含有芍药和甘草")
        assert compute_keyword_accuracy([r]) == 1.0

    def test_no_keywords_present(self):
        r = _make_result(expected_answer="桂枝汤、芍药、甘草", answer="完全无关的内容")
        assert compute_keyword_accuracy([r]) == 0.0

    def test_partial_keywords(self):
        r = _make_result(expected_answer="桂枝汤、芍药、甘草", answer="桂枝汤用于治疗")
        # 关键词: 桂枝汤, 芍药, 甘草 — 命中 桂枝汤 = 1/3
        result = compute_keyword_accuracy([r])
        assert 0 < result < 1.0


class TestLatencyStats:
    def test_basic_stats(self):
        results = [_make_result(latency=l) for l in [5, 10, 15]]
        stats = compute_latency_stats(results)
        assert stats["min"] == 5
        assert stats["max"] == 15
        assert stats["mean"] == 10
        assert stats["p50"] == 10

    def test_empty_results(self):
        stats = compute_latency_stats([])
        assert stats["p50"] == 0
        assert stats["p95"] == 0


class TestCategoryAccuracy:
    def test_per_category_breakdown(self):
        r1 = _make_result(category="A", expected_answer="桂枝汤", answer="桂枝汤")
        r2 = _make_result(category="B", expected_answer="麻黄汤", answer="无关")
        result = compute_category_accuracy([r1, r2])
        assert result["A"] == 1.0
        assert result["B"] == 0.0
