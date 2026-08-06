"""LLM-as-judge 评测模块

使用 Ollama 本地模型对答案质量进行 0-5 分评分，
替代 keyword_accuracy 对解释/对比类问题的误判。

评分维度：
- 正确性 (0-2分): 答案是否与经典原文/教材标准答案一致
- 完整性 (0-2分): 答案是否覆盖了关键要点
- 相关性 (0-1分): 答案是否切题，没有跑题或幻觉

总分 0-5 分，0.5 分为粒度。
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass

import requests


@dataclass
class JudgeResult:
    """评判结果"""
    score: float  # 0-5
    reasoning: str  # 评判理由
    raw_response: str  # 原始响应


class LLMJudge:
    """LLM 评判器

    使用 Ollama API 对答案质量进行评分。

    用法：
        judge = LLMJudge()
        result = judge.score("什么是阳明病？", "阳明病是...", "阳明病是胃家实...")
        print(result.score)  # 4.5
    """

    JUDGE_PROMPT = """你是一个中医经典知识评测专家。请对以下问答进行评分。

问题：{question}
标准答案：{expected}
待评答案：{answer}

评分标准（总分 0-5 分）：
- 正确性 (0-2分)：答案内容是否与标准答案/经典原文一致，有无错误或幻觉
- 完整性 (0-2分)：答案是否覆盖了关键要点（如病因、病机、主症、主方）
- 相关性 (0-1分)：答案是否切题，没有跑题

请按以下格式输出：
分数：X.X
理由：简短说明扣分原因

只输出上述两行，不要输出其他内容。"""

    def __init__(
        self,
        model: str = "qwen2.5:1.5b",
        base_url: str = "http://localhost:11434",
        timeout: int = 30,
    ) -> None:
        self._model = model
        self._base_url = base_url
        self._timeout = timeout

    def score(
        self,
        question: str,
        expected_answer: str,
        actual_answer: str,
    ) -> JudgeResult:
        """对单个问答进行评分

        Args:
            question: 问题
            expected_answer: 标准答案
            actual_answer: 待评答案

        Returns:
            JudgeResult 包含分数和理由
        """
        prompt = self.JUDGE_PROMPT.format(
            question=question,
            expected=expected_answer[:500],  # 截断防止 prompt 过长
            answer=actual_answer[:500],
        )

        try:
            response = requests.post(
                f"{self._base_url}/api/generate",
                json={
                    "model": self._model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.1,  # 低温度保证评分一致性
                        "num_predict": 200,
                    },
                },
                timeout=self._timeout,
            )
            response.raise_for_status()
            raw = response.json().get("response", "")

            # 解析分数
            score, reasoning = self._parse_response(raw)
            return JudgeResult(score=score, reasoning=reasoning, raw_response=raw)

        except Exception as e:
            return JudgeResult(
                score=-1,
                reasoning=f"Judge error: {e}",
                raw_response="",
            )

    def score_batch(
        self,
        items: list[dict[str, str]],
        delay: float = 0.1,
    ) -> list[JudgeResult]:
        """批量评分

        Args:
            items: [{"question": ..., "expected_answer": ..., "answer": ...}, ...]
            delay: 每次评分间隔（秒）

        Returns:
            JudgeResult 列表
        """
        results: list[JudgeResult] = []
        for i, item in enumerate(items):
            result = self.score(
                question=item["question"],
                expected_answer=item["expected_answer"],
                actual_answer=item["answer"],
            )
            results.append(result)
            if delay > 0:
                time.sleep(delay)
        return results

    def _parse_response(self, raw: str) -> tuple[float, str]:
        """解析 LLM 评分响应

        Returns:
            (score, reasoning) 元组
        """
        # 匹配 "分数：X.X" 或 "分数: X.X"
        score_match = re.search(r"分数[：:]\s*(\d+(?:\.\d+)?)", raw)
        if score_match:
            score = float(score_match.group(1))
            score = max(0.0, min(5.0, score))  # 钳位到 [0, 5]
        else:
            # 尝试匹配数字
            num_match = re.search(r"(\d+(?:\.\d+)?)\s*/\s*5", raw)
            if num_match:
                score = float(num_match.group(1))
            else:
                # 最后尝试匹配行首数字
                num_match = re.search(r"^(\d+(?:\.\d+)?)", raw.strip())
                score = float(num_match.group(1)) if num_match else 0.0
                score = max(0.0, min(5.0, score))

        # 匹配 "理由：..."
        reasoning_match = re.search(r"理由[：:]\s*(.+)", raw, re.DOTALL)
        reasoning = reasoning_match.group(1).strip() if reasoning_match else raw[:200]

        return score, reasoning

    def is_available(self) -> bool:
        """检查评判模型是否可用"""
        try:
            r = requests.get(f"{self._base_url}/api/tags", timeout=5)
            if r.status_code != 200:
                return False
            models = r.json().get("models", [])
            return any(m.get("name", "").startswith(self._model) for m in models)
        except Exception:
            return False
