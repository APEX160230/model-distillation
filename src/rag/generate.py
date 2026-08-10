"""生成模块 — Ollama 本地推理

调用 Ollama API 进行文本生成，支持同步和流式两种模式。
P2.1: 支持从 ConceptMapper/GraphRAG 传入结构化上下文。
"""
from __future__ import annotations

import json
import os
from typing import Any, Iterator

import requests

from src.rag.retrieve import RetrievalResult

SYSTEM_PROMPT = """你是一位中医老师，擅长用通俗易懂的方式讲解中医经典知识。
请用口语化的讲解风格，像老师讲课一样回答问题。
请根据提供的经典原文和参考信息回答问题。
注意事项：
- 引用经典原文时标注条文编号
- 解释方剂时列出完整组成
- 不提供具体诊疗建议
- 如果检索结果中没有相关信息，请如实说明"""


def format_retrieved_docs(docs: list[RetrievalResult]) -> str:
    """格式化检索结果为 prompt 上下文"""
    if not docs:
        return "（未检索到相关原文）"

    lines = []
    for doc in docs:
        lines.append(f"【第{doc.clause_id}条·{doc.chapter}】\n{doc.text}")
    return "\n\n".join(lines)


def format_context_extras(extras: dict[str, Any] | None) -> str:
    """格式化额外上下文（概念简述、方剂信息、对比数据等）

    P2.1: 将 ConceptMapper 和 GraphRAG 的结构化数据
    转化为自然语言提示，帮助 1.5B 模型生成更好的答案。
    """
    if not extras:
        return ""

    parts: list[str] = []

    # 概念简述
    concept = extras.get("concept")
    if concept:
        parts.append(f"【概念参考】{concept.get('brief', '')}")
        related = concept.get("related_formulas", [])
        if related:
            parts.append(f"相关方剂：{'、'.join(related)}")

    # 方剂信息（方剂查询时）
    formula_info = extras.get("formula_info")
    if formula_info:
        herbs = "、".join(formula_info.get("herbs", []))
        parts.append(f"【方剂信息】{formula_info.get('name', '')}：{herbs}")
        parts.append(f"主治：{formula_info.get('syndrome', '')}")
        parts.append(f"功效：{formula_info.get('brief', '')}")

    # 药材查询结果
    herb_query = extras.get("herb_query")
    if herb_query:
        formula_names = herb_query.get("formula_names", [])
        count = herb_query.get("formula_count", 0)
        herbs = "、".join(herb_query.get("herbs", []))
        if formula_names:
            parts.append(f"【药材查询】含{herbs}的方剂共{count}首：{'、'.join(formula_names)}")

    # 方剂对比数据
    comparison = extras.get("comparison")
    if comparison and "error" not in comparison:
        f1 = comparison.get("formula1", {})
        f2 = comparison.get("formula2", {})
        parts.append(f"【方剂对比】")
        parts.append(f"{f1.get('name', '')}：{'、'.join(f1.get('herbs', []))}，主治{f1.get('syndrome', '')}")
        parts.append(f"{f2.get('name', '')}：{'、'.join(f2.get('herbs', []))}，主治{f2.get('syndrome', '')}")
        only1 = comparison.get("herbs_only_in_1", [])
        only2 = comparison.get("herbs_only_in_2", [])
        common = comparison.get("common_herbs", [])
        if only1:
            parts.append(f"{f1.get('name', '')}独有：{'、'.join(only1)}")
        if only2:
            parts.append(f"{f2.get('name', '')}独有：{'、'.join(only2)}")
        if common:
            parts.append(f"共有药材：{'、'.join(common)}")

    # GraphRAG 证候查询
    graph_syndrome = extras.get("graph_syndrome")
    if graph_syndrome:
        formulas = graph_syndrome.get("formulas", [])
        if formulas:
            parts.append(f"【证候方剂】相关方剂：{'、'.join(formulas)}")

    # 超范围标记
    out_of_scope = extras.get("out_of_scope")
    if out_of_scope:
        parts.append(f"【范围提示】{out_of_scope.get('message', '')}")

    if not parts:
        return ""

    return "参考信息：\n" + "\n".join(parts)


def build_prompt(
    question: str,
    docs: list[RetrievalResult],
    context_extras: dict[str, Any] | None = None,
) -> str:
    """构建完整 prompt

    Args:
        question: 用户问题
        docs: 检索到的文档列表
        context_extras: 额外上下文（概念简述、方剂信息等）
    """
    context = format_retrieved_docs(docs)
    extras = format_context_extras(context_extras)

    prompt = f"请根据以下经典原文回答问题。\n\n经典原文：\n{context}"
    if extras:
        prompt += f"\n\n{extras}"
    prompt += f"\n\n问题：{question}"
    return prompt


class Generator:
    """Ollama 生成器

    使用 Ollama REST API 进行文本生成。
    P2.1: 支持从 ConceptMapper/GraphRAG 传入结构化上下文。

    使用示例:
        gen = Generator(model="qwen25-15b-tcm")
        answer = gen.generate("桂枝汤的组成", docs, context_extras={...})
    """

    OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

    def __init__(self, model: str = "qwen25-15b-tcm") -> None:
        self._model = model

    def generate(
        self,
        question: str,
        docs: list[RetrievalResult],
        temperature: float = 0.7,
        max_tokens: int = 512,
        context_extras: dict[str, Any] | None = None,
    ) -> str:
        """同步生成回答

        Args:
            question: 用户问题
            docs: 检索到的文档列表
            temperature: 采样温度
            max_tokens: 最大生成 token 数
            context_extras: 额外上下文（P2.1）

        Returns:
            生成的回答文本
        """
        prompt = build_prompt(question, docs, context_extras)
        response = requests.post(
            f"{self.OLLAMA_HOST}/api/chat",
            json={
                "model": self._model,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens,
                },
            },
            timeout=120,
        )
        response.raise_for_status()
        return response.json()["message"]["content"]

    def stream_generate(
        self,
        question: str,
        docs: list[RetrievalResult],
        temperature: float = 0.7,
        max_tokens: int = 512,
        context_extras: dict[str, Any] | None = None,
    ) -> Iterator[str]:
        """流式生成回答

        Args:
            question: 用户问题
            docs: 检索到的文档列表
            temperature: 采样温度
            max_tokens: 最大生成 token 数
            context_extras: 额外上下文（P2.1）

        Yields:
            生成的文本片段
        """
        prompt = build_prompt(question, docs, context_extras)
        response = requests.post(
            f"{self.OLLAMA_HOST}/api/chat",
            json={
                "model": self._model,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                "stream": True,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens,
                },
            },
            stream=True,
            timeout=120,
        )
        for line in response.iter_lines():
            if line:
                chunk = json.loads(line)
                content = chunk.get("message", {}).get("content", "")
                if content:
                    yield content
