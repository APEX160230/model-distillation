"""辨证链路接入管线测试（PRD v3.0 §5 FR3/FR5）

覆盖：
- 证据不足 → 追问（不调用模型）
- 辨证成功 → diagnosis 传入 generator（三层生成）
- 非症状问题（辨证 rejected）→ 降级走现有 RAG 链路
"""
from types import SimpleNamespace
from unittest.mock import MagicMock

from src.rag.diagnosis import DiagnosisEngine
from src.rag.pipeline import RAGPipeline


def _mock_doc(clause_id=35, text="太阳病，头痛发热，恶寒无汗者，麻黄汤主之", chapter="辨太阳病脉证并治"):
    return SimpleNamespace(clause_id=clause_id, text=text, chapter=chapter)


class TestPipelineDiagnosis:
    """辨证链路接入"""

    def _make_pipeline(self, retriever=None, generator=None):
        pipeline = RAGPipeline.__new__(RAGPipeline)
        pipeline._hybrid_retriever = retriever or MagicMock()
        pipeline._retriever = pipeline._hybrid_retriever
        pipeline._generator = generator or MagicMock()
        pipeline._top_k = 5
        pipeline._diagnosis = DiagnosisEngine()
        return pipeline

    def test_clarification_returns_question_without_generator(self):
        """症状证据不足 → 返回追问，不调用模型"""
        pipeline = self._make_pipeline()
        resp = pipeline.query("我有点头痛")
        assert resp.route_type == "diagnosis_clarify"
        assert resp.answer
        assert not pipeline._generator.generate.called

    def test_clarification_returns_structured_question(self):
        """追问问题与选项返回在 context_extras 中（前端选择题渲染）"""
        pipeline = self._make_pipeline()
        resp = pipeline.query("我有点头痛")
        diag = resp.context_extras.get("diagnosis", {})
        assert diag.get("status") == "need_clarification"
        assert diag.get("question")
        assert len(diag.get("options", [])) >= 2

    def test_diagnosed_passes_diagnosis_to_generator(self):
        """辨证成功 → generator 收到含 diagnosis 的 context_extras（检索空也放行）"""
        retriever = MagicMock()
        retriever.query.return_value = []
        retriever.last_route = None
        retriever.last_context = {}
        gen = MagicMock()
        gen.generate.return_value = "三层回答"
        pipeline = self._make_pipeline(retriever=retriever, generator=gen)

        resp = pipeline.query("头痛怕冷不出汗")

        assert gen.generate.called
        _args, kwargs = gen.generate.call_args
        extras = kwargs.get("context_extras") or {}
        diag = extras.get("diagnosis", {})
        assert diag.get("status") == "diagnosed"
        assert diag.get("syndrome") == "太阳伤寒"
        assert resp.answer == "三层回答"

    def test_rejected_falls_back_to_rag(self):
        """非症状问题（辨证 rejected）→ 降级走现有 RAG 链路"""
        retriever = MagicMock()
        retriever.query.return_value = [_mock_doc()]
        retriever.last_route = None
        retriever.last_context = {}
        gen = MagicMock()
        gen.generate.return_value = "知识回答"
        pipeline = self._make_pipeline(retriever=retriever, generator=gen)

        resp = pipeline.query("桂枝汤的组成是什么")

        assert gen.generate.called
        assert resp.answer == "知识回答"
        extras = resp.context_extras or {}
        assert "diagnosis" not in extras or extras["diagnosis"]["status"] != "diagnosed"

    def test_stream_clarification(self):
        """流式模式：证据不足 → 追问，不调用模型"""
        pipeline = self._make_pipeline()
        docs, stream, route_type, extras = pipeline.stream_query("我有点头痛")
        assert route_type == "diagnosis_clarify"
        assert docs == []
        assert "".join(stream)
        assert not pipeline._generator.stream_generate.called
