"""评测运行器 — 执行 50 题评测集

将评测集通过 RAG 管道运行，收集结果并生成报告。
"""
from __future__ import annotations

import json
from pathlib import Path

from src.eval.metrics import EvalResult, generate_report


class EvalRunner:
    """评测运行器

    使用示例:
        runner = EvalRunner(pipeline, "data/eval/eval_50.jsonl")
        results = runner.run()
        report = runner.save_report(results, "data/processed/p0_baseline_report.json")
    """

    def __init__(self, pipeline, eval_path: str) -> None:
        self._pipeline = pipeline
        self._eval_path = Path(eval_path)

    def _load_eval_data(self) -> list[dict]:
        """加载评测集 JSONL"""
        items = []
        with open(self._eval_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    items.append(json.loads(line))
        return items

    def run(self) -> list[EvalResult]:
        """执行评测

        Returns:
            EvalResult 列表
        """
        eval_data = self._load_eval_data()
        results: list[EvalResult] = []

        for i, item in enumerate(eval_data, 1):
            question = item["question"]
            category = item.get("category", "unknown")
            expected = item.get("expected_answer", "")
            ref_clauses = item.get("reference_clauses", [])

            print(f"  [{i}/{len(eval_data)}] {category}: {question}...")

            response = self._pipeline.query(question, temperature=0.3, max_tokens=512)

            retrieved_clauses = [doc.clause_id for doc in response.retrieved_docs]

            result = EvalResult(
                question_id=item.get("id", i),
                category=category,
                question=question,
                answer=response.answer,
                expected_answer=expected,
                reference_clauses=ref_clauses,
                retrieved_clauses=retrieved_clauses,
                latency=response.latency,
            )
            results.append(result)

            print(f"    -> 延迟 {response.latency}s, 检索到 {len(response.retrieved_docs)} 条")

        return results

    def generate_report(self, results: list[EvalResult], stage: str = "P0") -> dict:
        """生成报告"""
        return generate_report(results, stage=stage)

    def save_report(
        self,
        results: list[EvalResult],
        output_path: str,
        stage: str = "P0",
    ) -> dict:
        """生成并保存报告

        Args:
            results: 评测结果
            output_path: 输出文件路径
            stage: 阶段标识

        Returns:
            报告字典
        """
        report = self.generate_report(results, stage=stage)

        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        print(f"\n评测报告已保存: {output_path}")
        print(f"  阶段: {report['stage']}")
        print(f"  题数: {report['total_questions']}")
        print(f"  recall@5: {report['recall_at_5']}")
        print(f"  关键词命中率: {report['keyword_accuracy']}")
        print(f"  延迟 p50: {report['latency']['p50']}s, p95: {report['latency']['p95']}s")

        return report
