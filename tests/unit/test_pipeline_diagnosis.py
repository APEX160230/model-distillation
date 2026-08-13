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
        """辨证成功 → 三层回答完全由模板生成，不调用模型"""
        retriever = MagicMock()
        retriever.query.return_value = []
        retriever.last_route = None
        retriever.last_context = {}
        gen = MagicMock()
        gen.generate.return_value = "不应被调用"
        pipeline = self._make_pipeline(retriever=retriever, generator=gen)

        resp = pipeline.query("头痛怕冷不出汗")

        # 辨证链路不经过模型（锚点理论：模板+素材直引，严谨性 100%）
        assert not gen.generate.called
        assert "【辨证方向】" in resp.answer
        assert "【类方思路】" in resp.answer
        assert "麻黄汤类方" in resp.answer

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

    def test_diagnosed_stream_starts_with_template(self):
        """辨证流式：三层回答完全由模板生成，无模型流"""
        retriever = MagicMock()
        retriever.query.return_value = [_mock_doc()]
        retriever.last_route = None
        retriever.last_context = {}
        gen = MagicMock()
        gen.stream_generate.return_value = iter(["不应出现"])
        pipeline = self._make_pipeline(retriever=retriever, generator=gen)

        docs, stream, route_type, extras = pipeline.stream_query("头痛怕冷不出汗")

        full = "".join(stream)
        assert not gen.stream_generate.called
        assert "【辨证方向】" in full
        assert "【类方思路】" in full
        assert "麻黄汤类方" in full
        assert "不应出现" not in full

    def test_diagnosed_injects_lectures(self):
        """辨证成功时检索讲稿素材并注入 context_extras（FR4）"""
        retriever = MagicMock()
        retriever.query.return_value = [_mock_doc()]
        retriever.last_route = None
        retriever.last_context = {}
        gen = MagicMock()
        gen.generate.return_value = "讲解"
        lecture_retriever = MagicMock()
        lecture_retriever.query.return_value = [
            SimpleNamespace(text="太阳伤寒是寒邪束表，毛孔紧闭。", metadata={"book": "伤寒", "topic": "太阳伤寒讲解"}),
        ]
        pipeline = self._make_pipeline(retriever=retriever, generator=gen)
        pipeline._lecture_retriever = lecture_retriever

        resp = pipeline.query("头痛怕冷不出汗")

        # 讲稿素材直引进第三层
        assert "【倪师讲解】" in resp.answer
        assert "太阳伤寒是寒邪束表" in resp.answer
        extras = resp.context_extras or {}
        lectures = extras.get("lectures", [])
        assert len(lectures) == 1
        assert lectures[0]["book"] == "伤寒"
        assert lecture_retriever.query.called

    def test_diagnosed_lecture_failure_silent(self):
        """讲稿库不可用时静默跳过，用条文素材兜底"""
        retriever = MagicMock()
        retriever.query.return_value = [_mock_doc()]
        retriever.last_route = None
        retriever.last_context = {}
        gen = MagicMock()
        gen.generate.return_value = "讲解"
        lecture_retriever = MagicMock()
        lecture_retriever.query.side_effect = Exception("chroma 不可用")
        pipeline = self._make_pipeline(retriever=retriever, generator=gen)
        pipeline._lecture_retriever = lecture_retriever

        resp = pipeline.query("头痛怕冷不出汗")

        assert resp.answer  # 主链路不受影响
        assert "【倪师讲解】" in resp.answer  # 条文素材兜底
        extras = resp.context_extras or {}
        assert "lectures" not in extras or extras["lectures"] == []
