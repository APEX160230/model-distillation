"""生成模块 — Ollama 本地推理

调用 Ollama API 进行文本生成，支持同步和流式两种模式。
"""
from __future__ import annotations

from typing import Iterator

import ollama

from src.rag.retrieve import RetrievalResult

SYSTEM_PROMPT = """你是一位中医老师，擅长用通俗易懂的方式讲解中医经典知识。
请根据提供的经典原文回答问题。
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


def build_prompt(question: str, docs: list[RetrievalResult]) -> str:
    """构建完整 prompt"""
    context = format_retrieved_docs(docs)
    return f"请根据以下经典原文回答问题。\n\n经典原文：\n{context}\n\n问题：{question}"


class Generator:
    """Ollama 生成器

    使用 ollama.chat API 进行文本生成。

    使用示例:
        gen = Generator(model="qwen25-15b-tcm")
        answer = gen.generate("桂枝汤的组成", docs)
    """

    def __init__(self, model: str = "qwen25-15b-tcm") -> None:
        self._model = model

    def generate(
        self,
        question: str,
        docs: list[RetrievalResult],
        temperature: float = 0.7,
        max_tokens: int = 512,
    ) -> str:
        """同步生成回答

        Args:
            question: 用户问题
            docs: 检索到的文档列表
            temperature: 采样温度
            max_tokens: 最大生成 token 数

        Returns:
            生成的回答文本
        """
        prompt = build_prompt(question, docs)
        response = ollama.chat(
            model=self._model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            options={
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        )
        return response["message"]["content"]

    def stream_generate(
        self,
        question: str,
        docs: list[RetrievalResult],
        temperature: float = 0.7,
        max_tokens: int = 512,
    ) -> Iterator[str]:
        """流式生成回答

        Args:
            question: 用户问题
            docs: 检索到的文档列表
            temperature: 采样温度
            max_tokens: 最大生成 token 数

        Yields:
            生成的文本片段
        """
        prompt = build_prompt(question, docs)
        stream = ollama.chat(
            model=self._model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            options={
                "temperature": temperature,
                "num_predict": max_tokens,
            },
            stream=True,
        )
        for chunk in stream:
            content = chunk["message"]["content"]
            if content:
                yield content
