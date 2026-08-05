"""生成模块测试"""
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
    @patch("src.rag.generate.ollama.chat")
    def test_generate_returns_string(self, mock_chat):
        mock_chat.return_value = {"message": {"content": "这是回答"}}
        gen = Generator(model="test-model")
        result = gen.generate("问题", [_make_doc()])
        assert isinstance(result, str)
        assert result == "这是回答"

    @patch("src.rag.generate.ollama.chat")
    def test_generate_calls_ollama(self, mock_chat):
        mock_chat.return_value = {"message": {"content": "回答"}}
        gen = Generator(model="test-model")
        gen.generate("问题", [_make_doc()])
        assert mock_chat.called

    @patch("src.rag.generate.ollama.chat")
    def test_generate_passes_system_prompt(self, mock_chat):
        mock_chat.return_value = {"message": {"content": "回答"}}
        gen = Generator(model="test-model")
        gen.generate("问题", [_make_doc()])
        call_args = mock_chat.call_args
        messages = call_args.kwargs.get("messages", call_args.args[1] if len(call_args.args) > 1 else [])
        assert any(m["role"] == "system" for m in messages)

    @patch("src.rag.generate.ollama.chat")
    def test_generate_temperature(self, mock_chat):
        mock_chat.return_value = {"message": {"content": "回答"}}
        gen = Generator(model="test-model")
        gen.generate("问题", [_make_doc()], temperature=0.1)
        call_args = mock_chat.call_args
        options = call_args.kwargs.get("options", {})
        assert options.get("temperature") == 0.1

    @patch("src.rag.generate.ollama.chat")
    def test_stream_generate_yields_chunks(self, mock_chat):
        mock_chat.return_value = iter([
            {"message": {"content": " chunk1"}},
            {"message": {"content": " chunk2"}},
            {"message": {"content": ""}},
        ])
        gen = Generator(model="test-model")
        chunks = list(gen.stream_generate("问题", [_make_doc()]))
        assert "chunk1" in chunks[0]
        assert "chunk2" in chunks[1]
        assert len(chunks) == 2  # empty chunk filtered

    @patch("src.rag.generate.ollama.chat")
    def test_generate_max_tokens(self, mock_chat):
        mock_chat.return_value = {"message": {"content": "回答"}}
        gen = Generator(model="test-model")
        gen.generate("问题", [_make_doc()], max_tokens=128)
        call_args = mock_chat.call_args
        options = call_args.kwargs.get("options", {})
        assert options.get("num_predict") == 128
