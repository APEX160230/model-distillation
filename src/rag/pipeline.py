"""RAG 管线编排 — 检索 → 生成

P2.1: 支持 ConceptMapper/GraphRAG 结构化上下文传递。
将检索器的额外上下文（概念简述、方剂对比、药材列表）传给生成器，
帮助 1.5B 模型生成更准确的答案。

P0-4 安全护栏：
- 检索为空（0 条依据）时拒答，不调用模型，杜绝自由发挥
- 剂量/处方类查询直接拦截，符合 PRD「不提供诊疗建议」边界
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterator, Optional

from src.rag.generate import Generator, build_diagnosis_template
from src.rag.retrieve import RetrievalResult, VectorRetriever
from src.rag.hybrid_retriever import HybridRetriever
from src.rag.diagnosis import DiagnosisEngine

# ── 安全护栏（P0-4）────────────────────────────────────────

# 检索为空时的拒答话术
EMPTY_RETRIEVAL_MESSAGE = (
    "抱歉，我在经典知识库中没有检索到与这个问题相关的内容。"
    "本系统目前收录的是伤寒论相关的方证知识，您可以换个问法，"
    "或咨询专业中医师获取帮助。"
)

# 剂量/处方类查询关键词（命中即拦截，不检索不生成）
_DOSE_QUERY_PATTERN = re.compile(
    r"多少克|多少钱|多少两|几克|几钱|剂量|用量|怎么吃|怎么服|吃多少|"
    r"一次吃|每次吃|一日吃|开个方|开方|处方|"
    r"给我开|帮我开|药方|抓药|煎药|熬药|服用方法"
)

# 危象类查询关键词（命中即拒答并引导就医）
# 注意：仅匹配明确的急危重症表述，避免误伤中医证名（如"太阳中风证"）
_EMERGENCY_QUERY_PATTERN = re.compile(
    r"脑溢血|脑出血|急性心梗|心肌梗死|休克|大出血|昏迷不醒|抽搐不止|呼吸困难|"
    r"剧烈胸痛|胸口剧痛|快不行|急救|马上要晕|晕倒在地|中风了|脑中风|中风发作|急性中风|"
    r"流血不止|血止不住|止不住|鼻血不止|流鼻血.*不止|吐血|咯血"
)


def detect_dose_prescription_query(question: str) -> str | None:
    """检测剂量/处方类查询，命中返回拒答话术

    评测暴露：模型会编造剂量（"桂枝5钱"）甚至虚构处方。
    这类查询直接拦截，不进入检索和生成链路。
    """
    if _DOSE_QUERY_PATTERN.search(question):
        return (
            "本系统是中医经典知识问答工具，按产品定位不提供具体用药剂量和处方建议。"
            "相关剂量的记载请查阅《伤寒论》原文，具体用药请咨询专业中医师。"
        )
    if _EMERGENCY_QUERY_PATTERN.search(question):
        return (
            "您描述的情况可能属于急症，请立即前往医院急诊就医，不要延误治疗。"
            "本系统仅提供中医经典知识讲解，不能替代专业医疗判断。"
        )
    return None


@dataclass
class RAGResponse:
    """RAG 完整响应"""
    answer: str
    retrieved_docs: list[RetrievalResult]
    latency: float
    route_type: str = ""  # 查询路由类型
    context_extras: dict[str, Any] | None = None  # P2.1 额外上下文


class RAGPipeline:
    """RAG 管线

    P2.1 默认使用 HybridRetriever（概念映射 + GraphRAG + 查询路由 + BM25 + 向量）。
    设置 use_hybrid=False 可回退到纯向量检索（用于 P0/P1 对比）。

    用法：
        pipeline = RAGPipeline()
        response = pipeline.query("什么是阳明病？")
        print(response.answer)
        print(response.context_extras)  # 概念简述等
    """

    def __init__(
        self,
        chroma_path: str = "data/chroma",
        clauses_path: str = "data/processed/classics/shanghan_clauses.jsonl",
        model: str = "tcm-model",
        top_k: int = 5,
        generator=None,
        use_hybrid: bool = True,
    ) -> None:
        if use_hybrid:
            self._hybrid_retriever = HybridRetriever(
                chroma_path=chroma_path,
                clauses_path=clauses_path,
            )
            self._retriever = self._hybrid_retriever
        else:
            self._hybrid_retriever = None
            self._retriever = VectorRetriever(persist_dir=chroma_path)

        self._generator = generator or Generator(model=model)
        self._top_k = top_k
        self._diagnosis = DiagnosisEngine()
        self._lecture_retriever = None  # FR4 讲稿库（build_lecture_chroma 构建，可选）

    def retrieve(self, question: str) -> list[RetrievalResult]:
        """仅检索，不生成"""
        return self._retriever.query(question, top_k=self._top_k)

    def query(
        self,
        question: str,
        temperature: float = 0.7,
        max_tokens: int = 512,
    ) -> RAGResponse:
        """完整 RAG 查询：检索 → 生成

        P2.1: 自动从 HybridRetriever 获取额外上下文并传给生成器。
        P0-4: 剂量/处方查询直接拦截；检索为空拒答，不调用模型。
        """
        import time
        start = time.time()

        # P0-4 护栏 1：剂量/处方/危象类查询直接拦截
        rejection = detect_dose_prescription_query(question)
        if rejection:
            return RAGResponse(
                answer=rejection,
                retrieved_docs=[],
                latency=round(time.time() - start, 2),
                route_type="rejected",
                context_extras={"rejection": rejection},
            )

        # PRD v3.0 辨证分支：症状描述优先走辨证引擎（不依赖模型推理）
        diagnosis_engine = getattr(self, "_diagnosis", None)
        if diagnosis_engine:
            diagnosis = diagnosis_engine.diagnose(question)
            if diagnosis.status == "need_clarification":
                # 证据不足 → 追问（前端选择题），不调用模型
                return RAGResponse(
                    answer=diagnosis.question,
                    retrieved_docs=[],
                    latency=round(time.time() - start, 2),
                    route_type="diagnosis_clarify",
                    context_extras={"diagnosis": diagnosis.to_dict()},
                )

        docs = self.retrieve(question)

        # 获取路由类型和额外上下文（P2.1）
        route_type = ""
        context_extras: dict[str, Any] | None = None
        if self._hybrid_retriever:
            if self._hybrid_retriever.last_route:
                route_type = self._hybrid_retriever.last_route.query_type.value
            context_extras = self._hybrid_retriever.last_context or None

        # PRD v3.0：辨证成功 → 辨证结论注入上下文 + 讲稿素材
        if diagnosis_engine and diagnosis.status == "diagnosed":
            if context_extras is None:
                context_extras = {}
            context_extras["diagnosis"] = diagnosis.to_dict()
            self._inject_lectures(question, context_extras)

        # P0-4 护栏 2：检索为空且无有效上下文 → 拒答，不调用模型
        has_effective_context = bool(
            context_extras
            and any(
                k in context_extras
                for k in ("concept", "formula_info", "herb_query", "comparison",
                          "graph_syndrome", "formula_compositions", "diagnosis")
            )
        )
        if not docs:
            # 超范围查询（out_of_scope）→ 用其消息拒答
            if context_extras and context_extras.get("out_of_scope"):
                message = context_extras["out_of_scope"].get(
                    "message", EMPTY_RETRIEVAL_MESSAGE)
                return RAGResponse(
                    answer=message,
                    retrieved_docs=[],
                    latency=round(time.time() - start, 2),
                    route_type=route_type,
                    context_extras=context_extras,
                )
            if not has_effective_context:
                return RAGResponse(
                    answer=EMPTY_RETRIEVAL_MESSAGE,
                    retrieved_docs=[],
                    latency=round(time.time() - start, 2),
                    route_type=route_type,
                    context_extras=context_extras,
                )

        answer = self._generator.generate(
            question, docs,
            temperature=temperature,
            max_tokens=max_tokens,
            context_extras=context_extras,
        )

        # PRD v3.0：辨证成功 → 前两层模板拼接到回答前
        if diagnosis_engine and diagnosis.status == "diagnosed":
            template = build_diagnosis_template(diagnosis.to_dict(), docs)
            answer = template + "\n\n" + answer

        return RAGResponse(
            answer=answer,
            retrieved_docs=docs,
            latency=round(time.time() - start, 2),
            route_type=route_type,
            context_extras=context_extras,
        )

    def stream_query(
        self,
        question: str,
        temperature: float = 0.7,
        max_tokens: int = 512,
    ) -> tuple[list[RetrievalResult], Iterator[str], str, dict[str, Any] | None]:
        """流式 RAG 查询：先检索，再流式生成

        P0-4: 剂量/处方查询直接拦截；检索为空拒答，不调用模型。

        Returns:
            (docs, stream, route_type, context_extras)
        """
        # P0-4 护栏 1：剂量/处方/危象类查询直接拦截
        rejection = detect_dose_prescription_query(question)
        if rejection:
            def _rejection_stream():
                yield rejection
            return [], _rejection_stream(), "rejected", {"rejection": rejection}

        # PRD v3.0 辨证分支（流式）
        diagnosis_engine = getattr(self, "_diagnosis", None)
        diagnosis = None
        if diagnosis_engine:
            diagnosis = diagnosis_engine.diagnose(question)
            if diagnosis.status == "need_clarification":
                def _clarify_stream():
                    yield diagnosis.question
                return (
                    [],
                    _clarify_stream(),
                    "diagnosis_clarify",
                    {"diagnosis": diagnosis.to_dict()},
                )

        docs = self.retrieve(question)

        route_type = ""
        context_extras: dict[str, Any] | None = None
        if self._hybrid_retriever:
            if self._hybrid_retriever.last_route:
                route_type = self._hybrid_retriever.last_route.query_type.value
            context_extras = self._hybrid_retriever.last_context or None

        # PRD v3.0：辨证成功 → 辨证结论注入上下文 + 讲稿素材
        if diagnosis_engine and diagnosis and diagnosis.status == "diagnosed":
            if context_extras is None:
                context_extras = {}
            context_extras["diagnosis"] = diagnosis.to_dict()
            self._inject_lectures(question, context_extras)

        # P0-4 护栏 2：检索为空且无有效上下文 → 拒答，不调用模型
        has_effective_context = bool(
            context_extras
            and any(
                k in context_extras
                for k in ("concept", "formula_info", "herb_query", "comparison",
                          "graph_syndrome", "formula_compositions", "diagnosis")
            )
        )
        if not docs:
            # 超范围查询（out_of_scope）→ 用其消息拒答
            if context_extras and context_extras.get("out_of_scope"):
                message = context_extras["out_of_scope"].get(
                    "message", EMPTY_RETRIEVAL_MESSAGE)

                def _outofscope_stream():
                    yield message
                return [], _outofscope_stream(), route_type, context_extras
            if not has_effective_context:
                def _empty_stream():
                    yield EMPTY_RETRIEVAL_MESSAGE
                return [], _empty_stream(), route_type, context_extras

        stream = self._generator.stream_generate(
            question, docs,
            temperature=temperature,
            max_tokens=max_tokens,
            context_extras=context_extras,
        )

        # PRD v3.0：辨证成功 → 前两层模板（代码生成）+ 模型第三层讲解
        if diagnosis_engine and diagnosis and diagnosis.status == "diagnosed":
            template = build_diagnosis_template(diagnosis.to_dict(), docs)
            stream = self._stream_with_template(template, stream)

        return docs, stream, route_type, context_extras

    @staticmethod
    def _stream_with_template(template: str, stream: Iterator[str]) -> Iterator[str]:
        """在模型流前先输出模板层（辨证前两层，代码生成）"""
        yield template + "\n\n"
        for chunk in stream:
            yield chunk

    def _inject_lectures(self, question: str, context_extras: dict[str, Any]) -> None:
        """FR4: 检索倪师讲稿素材注入上下文（第三层讲解引用倪师原话）

        讲稿库（collection: lectures）由 scripts/build_lecture_chroma.py 构建。
        不可用（未构建/异常）时静默跳过，不影响主链路。
        """
        lecture_retriever = getattr(self, "_lecture_retriever", None)
        if not lecture_retriever:
            return
        try:
            hits = lecture_retriever.query(question, top_k=3)
            if hits:
                context_extras["lectures"] = [
                    {
                        "book": h.metadata.get("book", ""),
                        "topic": h.metadata.get("topic", ""),
                        "text": h.text,
                    }
                    for h in hits
                ]
        except Exception:
            # 讲稿库不可用时静默降级（条文检索与辨证链路不受影响）
            pass

    @property
    def retriever(self):
        """暴露检索器"""
        return self._retriever

    @property
    def hybrid_retriever(self) -> HybridRetriever | None:
        """暴露混合检索器（用于访问 graph_stats 等）"""
        return self._hybrid_retriever
