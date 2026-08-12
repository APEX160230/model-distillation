"""RAG 管线安全护栏测试（P0-4）

覆盖：
- 剂量/处方/危象类查询拦截（不检索不生成）
- 检索为空时拒答（不调用模型）
- 检索为空但有有效上下文时正常生成
"""
from unittest.mock import MagicMock

from src.rag.pipeline import (
    RAGPipeline,
    EMPTY_RETRIEVAL_MESSAGE,
    detect_dose_prescription_query,
)


class TestDetectDosePrescriptionQuery:
    """剂量/处方/危象类查询检测"""

    def test_dose_query_intercepted(self):
        for q in ["麻黄用多少克？", "桂枝汤的剂量是多少", "一次吃几克"]:
            assert detect_dose_prescription_query(q) is not None, q

    def test_prescription_query_intercepted(self):
        for q in ["感冒了给我开个方子", "帮我开个处方", "你给我开方"]:
            assert detect_dose_prescription_query(q) is not None, q

    def test_normal_query_not_intercepted(self):
        for q in ["桂枝汤的组成是什么", "什么是太阳中风证", "麻黄汤治什么病"]:
            assert detect_dose_prescription_query(q) is None, q


class TestPipelineGuards:
    """RAGPipeline 安全护栏"""

    def _make_pipeline(self, retriever=None, generator=None):
        pipeline = RAGPipeline.__new__(RAGPipeline)
        pipeline._hybrid_retriever = retriever or MagicMock()
        pipeline._retriever = pipeline._hybrid_retriever
        pipeline._generator = generator or MagicMock()
        pipeline._top_k = 5
        return pipeline

    def test_query_rejects_dose_question_without_calling_generator(self):
        pipeline = self._make_pipeline()
        resp = pipeline.query("麻黄用多少克")
        assert resp.route_type == "rejected"
        assert "不提供具体用药剂量" in resp.answer
        assert not pipeline._generator.generate.called

    def test_query_rejects_emergency_question(self):
        pipeline = self._make_pipeline()
        resp = pipeline.query("我中风了怎么办")
        assert resp.route_type == "rejected"
        assert "立即前往医院急诊" in resp.answer

    def test_query_empty_retrieval_returns_message(self):
        retriever = MagicMock()
        retriever.query.return_value = []
        retriever.last_route = None
        retriever.last_context = {}
        pipeline = self._make_pipeline(retriever=retriever)
        resp = pipeline.query("完全无关的问题测试")
        assert resp.answer == EMPTY_RETRIEVAL_MESSAGE
        assert not pipeline._generator.generate.called

    def test_query_empty_retrieval_with_context_generates(self):
        """检索空但有 concept 上下文时仍应生成（如 out_of_scope 之外的情况）"""
        retriever = MagicMock()
        retriever.query.return_value = []
        retriever.last_route = None
        retriever.last_context = {"concept": {"brief": "概念"}}
        gen = MagicMock()
        gen.generate.return_value = "正常回答"
        pipeline = self._make_pipeline(retriever=retriever, generator=gen)
        resp = pipeline.query("什么是太阳病")
        assert resp.answer == "正常回答"
        assert gen.generate.called

    def test_stream_query_rejects_dose_question(self):
        pipeline = self._make_pipeline()
        docs, stream, route_type, extras = pipeline.stream_query("麻黄用多少克")
        assert route_type == "rejected"
        assert docs == []
        assert "不提供具体用药剂量" in "".join(stream)
        assert not pipeline._generator.stream_generate.called

    def test_stream_query_empty_retrieval_returns_message(self):
        retriever = MagicMock()
        retriever.query.return_value = []
        retriever.last_route = None
        retriever.last_context = {}
        pipeline = self._make_pipeline(retriever=retriever)
        docs, stream, route_type, extras = pipeline.stream_query("无关问题")
        assert docs == []
        assert "".join(stream) == EMPTY_RETRIEVAL_MESSAGE
        assert not pipeline._generator.stream_generate.called
