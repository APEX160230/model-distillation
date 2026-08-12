"""评测运行器测试"""
from unittest.mock import MagicMock
import pytest
import json
import tempfile
from pathlib import Path

from src.eval.runner import EvalRunner
from src.eval.metrics import EvalResult
from src.rag.retrieve import RetrievalResult


def _make_mock_pipeline():
    """创建 mock pipeline"""
    pipeline = MagicMock()
    pipeline.query.return_value = MagicMock(
        answer="测试回答",
        retrieved_docs=[
            RetrievalResult(id="clause_1", text="太阳之为病", metadata={"clause_id": 1, "chapter": "test"}, distance=0.2),
        ],
        latency=1.5,
        route_type="formula",
        context_extras=None,
    )
    return pipeline


def _create_eval_file(tmp_path):
    """创建临时评测集"""
    eval_data = [
        {"id": 1, "category": "经典原文检索", "question": "伤寒论第1条原文是什么？",
         "expected_answer": "太阳之为病", "reference_clauses": [1]},
        {"id": 2, "category": "方剂查询", "question": "桂枝汤的组成是什么？",
         "expected_answer": "桂枝芍药甘草", "reference_clauses": [12]},
    ]
    filepath = tmp_path / "eval.jsonl"
    with open(filepath, "w", encoding="utf-8") as f:
        for item in eval_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    return str(filepath)


class TestEvalRunner:
    def test_run_all_questions(self, tmp_path):
        pipeline = _make_mock_pipeline()
        eval_path = _create_eval_file(tmp_path)
        runner = EvalRunner(pipeline, eval_path)
        results = runner.run()
        assert len(results) == 2

    def test_result_contains_retrieved_clauses(self, tmp_path):
        pipeline = _make_mock_pipeline()
        eval_path = _create_eval_file(tmp_path)
        runner = EvalRunner(pipeline, eval_path)
        results = runner.run()
        assert results[0].retrieved_clauses == [1]

    def test_result_contains_answer_and_latency(self, tmp_path):
        pipeline = _make_mock_pipeline()
        eval_path = _create_eval_file(tmp_path)
        runner = EvalRunner(pipeline, eval_path)
        results = runner.run()
        assert results[0].answer == "测试回答"
        assert results[0].latency == 1.5

    def test_generate_report(self, tmp_path):
        pipeline = _make_mock_pipeline()
        eval_path = _create_eval_file(tmp_path)
        runner = EvalRunner(pipeline, eval_path)
        results = runner.run()
        report = runner.generate_report(results, stage="P0")
        assert report["stage"] == "P0"
        assert report["total_questions"] == 2

    def test_save_report(self, tmp_path):
        pipeline = _make_mock_pipeline()
        eval_path = _create_eval_file(tmp_path)
        runner = EvalRunner(pipeline, eval_path)
        results = runner.run()
        output_path = str(tmp_path / "report.json")
        report = runner.save_report(results, output_path, stage="P0")
        assert Path(output_path).exists()
        with open(output_path, "r", encoding="utf-8") as f:
            saved = json.load(f)
        assert saved["total_questions"] == 2
