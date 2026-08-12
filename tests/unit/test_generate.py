"""生成模块测试"""
import json
from unittest.mock import patch, MagicMock
import pytest

from src.rag.retrieve import RetrievalResult
from src.rag.generate import Generator, format_retrieved_docs, build_prompt


def _make_doc(clause_id=1, text="太阳之为病", chapter="辨太阳病"):
    return RetrievalResult(
        id=f"clause_{clause_id}",
        text=text,
        metadata={"clause_id": clause_id, "chapter": chapter},
        distance=0.3,
    )


class TestFormatRetrievedDocs:
    def test_empty_docs(self):
        assert "未检索到" in format_retrieved_docs([])

    def test_single_doc(self):
        doc = _make_doc(clause_id=1, text="太阳之为病")
        result = format_retrieved_docs([doc])
        assert "第1条" in result
        assert "太阳之为病" in result

    def test_multiple_docs(self):
        docs = [_make_doc(clause_id=1), _make_doc(clause_id=2)]
        result = format_retrieved_docs(docs)
        assert "第1条" in result
        assert "第2条" in result


class TestBuildPrompt:
    def test_contains_question(self):
        prompt = build_prompt("什么是太阳病", [_make_doc()])
        assert "什么是太阳病" in prompt

    def test_contains_context(self):
        prompt = build_prompt("test", [_make_doc(text="太阳之为病")])
        assert "太阳之为病" in prompt


class TestGenerator:
    """Generator 已改用 requests 调用 Ollama REST API，测试 mock requests.post"""

    @staticmethod
    def _mock_response(content: str | None = None, stream_chunks: list[dict] | None = None):
        """构造 mock 的 requests.Response"""
        resp = MagicMock()
        if content is not None:
            resp.json.return_value = {"message": {"content": content}}
        if stream_chunks is not None:
            resp.iter_lines.return_value = [
                json.dumps(chunk, ensure_ascii=False).encode() for chunk in stream_chunks
            ]
        return resp

    @patch("src.rag.generate.requests.post")
    def test_generate_returns_string(self, mock_post):
        mock_post.return_value = self._mock_response(content="这是回答")
        gen = Generator(model="test-model")
        result = gen.generate("问题", [_make_doc()])
        assert isinstance(result, str)
        assert result == "这是回答"

    @patch("src.rag.generate.requests.post")
    def test_generate_calls_ollama(self, mock_post):
        mock_post.return_value = self._mock_response(content="回答")
        gen = Generator(model="test-model")
        gen.generate("问题", [_make_doc()])
        assert mock_post.called
        assert mock_post.call_args.args[0].endswith("/api/chat")

    @patch("src.rag.generate.requests.post")
    def test_generate_passes_system_prompt(self, mock_post):
        mock_post.return_value = self._mock_response(content="回答")
        gen = Generator(model="test-model")
        gen.generate("问题", [_make_doc()])
        payload = mock_post.call_args.kwargs["json"]
        messages = payload["messages"]
        assert any(m["role"] == "system" for m in messages)

    @patch("src.rag.generate.requests.post")
    def test_generate_temperature(self, mock_post):
        mock_post.return_value = self._mock_response(content="回答")
        gen = Generator(model="test-model")
        gen.generate("问题", [_make_doc()], temperature=0.1)
        payload = mock_post.call_args.kwargs["json"]
        assert payload["options"].get("temperature") == 0.1

    @patch("src.rag.generate.requests.post")
    def test_stream_generate_yields_chunks(self, mock_post):
        mock_post.return_value = self._mock_response(
            stream_chunks=[
                {"message": {"content": " chunk1"}},
                {"message": {"content": " chunk2"}},
                {"message": {"content": ""}},
            ]
        )
        gen = Generator(model="test-model")
        chunks = list(gen.stream_generate("问题", [_make_doc()]))
        assert "chunk1" in chunks[0]
        assert "chunk2" in chunks[1]
        assert len(chunks) == 2  # empty chunk filtered

    @patch("src.rag.generate.requests.post")
    def test_generate_max_tokens(self, mock_post):
        mock_post.return_value = self._mock_response(content="回答")
        gen = Generator(model="test-model")
        gen.generate("问题", [_make_doc()], max_tokens=128)
        payload = mock_post.call_args.kwargs["json"]
        assert payload["options"].get("num_predict") == 128
