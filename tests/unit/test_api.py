"""FastAPI 接口测试"""
import pytest
from unittest.mock import MagicMock
from fastapi.testclient import TestClient

from src.serve.api import create_app
from src.rag.retrieve import RetrievalResult


def _make_mock_pipeline():
    """创建 mock pipeline"""
    pipeline = MagicMock()
    pipeline.retriever.count.return_value = 330

    docs = [
        RetrievalResult(id="clause_1", text="太阳之为病", metadata={"clause_id": 1, "chapter": "test"}, distance=0.2),
    ]

    def _stream(*args, **kwargs):
        return docs, iter(["回答", "内容"])

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
        assert data["vector_store_count"] == 330
        assert data["model"] == "qwen2.5:1.5b"


class TestGraphStats:
    def test_returns_stats(self, client):
        resp = client.get("/api/graph/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert "total_nodes" in data
        assert "formula_count" in data
        assert "herb_count" in data


class TestChat:
    def test_streaming_response(self, client):
        resp = client.post("/api/chat", json={"question": "什么是太阳病"})
        assert resp.status_code == 200
        body = resp.text
        assert "retrieved" in body
        assert "chunk" in body
        assert "done" in body

    def test_empty_question_returns_400(self, client):
        resp = client.post("/api/chat", json={"question": ""})
        assert resp.status_code == 400
        assert "问题不能为空" in resp.json()["detail"]


class TestFrontend:
    def test_frontend_served(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        assert "html" in resp.headers["content-type"].lower()
