"""FastAPI 接口测试"""
import pytest
from unittest.mock import MagicMock
from fastapi.testclient import TestClient

from src.serve.api import create_app
from src.rag.retrieve import RetrievalResult


def _make_mock_pipeline():
    """创建 mock pipeline（对齐当前接口：stream_query 返回 4 元组）"""
    pipeline = MagicMock()
    pipeline.retriever.count.return_value = 330
    pipeline._generator._model = "tcm-model"
    pipeline.hybrid_retriever.graph_stats = {
        "nodes": 269,
        "edges": 518,
        "formulas": 71,
        "herbs": 62,
        "syndromes": 71,
    }

    docs = [
        RetrievalResult(id="clause_1", text="太阳之为病", metadata={"clause_id": 1, "chapter": "test"}, distance=0.2),
    ]

    def _stream(*args, **kwargs):
        return docs, iter(["回答", "内容"]), "formula", {"formula_info": {"name": "桂枝汤"}}

    pipeline.stream_query.side_effect = _stream
    return pipeline


@pytest.fixture
def client():
    pipeline = _make_mock_pipeline()
    app = create_app(pipeline)
    return TestClient(app)


class TestHealth:
    def test_returns_ok(self, client):
        resp = client.get("/api/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["graph_stats"]["nodes"] == 269
        assert data["graph_stats"]["formulas"] == 71
        assert data["model"] == "tcm-model"


class TestGraphStats:
    def test_returns_stats(self, client):
        resp = client.get("/api/graph/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert "nodes" in data
        assert "formulas" in data
        assert "herbs" in data


class TestChat:
    def test_streaming_response(self, client):
        resp = client.post("/api/chat", json={"question": "什么是太阳病"})
        assert resp.status_code == 200
        body = resp.text
        assert "retrieved" in body
        assert "chunk" in body
        assert "done" in body

    def test_empty_question_returns_400(self, client):
        # 空串被 Pydantic min_length=1 拦截返回 422；纯空格经 strip 后由业务层返回 400
        resp = client.post("/api/chat", json={"question": "   "})
        assert resp.status_code == 400
        assert "问题不能为空" in resp.json()["detail"]


class TestFrontend:
    def test_frontend_served(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        assert "html" in resp.headers["content-type"].lower()
