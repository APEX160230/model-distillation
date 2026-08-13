"""生成模块测试"""
import json
from unittest.mock import patch, MagicMock
import pytest

from src.rag.retrieve import RetrievalResult
from src.rag.generate import (
    Generator,
    SYSTEM_PROMPT,
    format_retrieved_docs,
    build_prompt,
    format_context_extras,
    verify_clause_numbers,
    apply_safety_filter,
)


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


class TestSystemPrompt:
    """P0-4: SYSTEM_PROMPT 强化约束"""

    def test_contains_fact_fidelity_rule(self):
        assert "不得增删改换" in SYSTEM_PROMPT

    def test_contains_no_fabricate_clause_rule(self):
        assert "不得编造不存在的条文编号" in SYSTEM_PROMPT

    def test_contains_no_dose_rule(self):
        assert "剂量" in SYSTEM_PROMPT

    def test_contains_empty_retrieval_rule(self):
        assert "未收录相关内容" in SYSTEM_PROMPT


class TestVerifyClauseNumbers:
    """P0-4: 条文编号交叉校验"""

    def test_keeps_valid_clause_refs(self):
        answer = "太阳之为病【第1条】，桂枝汤主之【第12条】"
        docs = [_make_doc(clause_id=1), _make_doc(clause_id=12)]
        assert "【第1条】" in verify_clause_numbers(answer, docs)
        assert "【第12条】" in verify_clause_numbers(answer, docs)

    def test_removes_fabricated_clause_refs(self):
        # 模型编造了第 99 条（不在检索结果中）→ 删除引用标记
        answer = "此证见【第99条】所述"
        docs = [_make_doc(clause_id=1), _make_doc(clause_id=12)]
        result = verify_clause_numbers(answer, docs)
        assert "第99条" not in result
        assert "此证见所述" in result

    def test_empty_docs_returns_unchanged(self):
        answer = "见第1条"
        assert verify_clause_numbers(answer, []) == answer

    def test_plain_clause_ref_checked(self):
        # 不带【】的"第X条"也要校验
        answer = "正如第 35 条所说"
        docs = [_make_doc(clause_id=12)]
        assert "第" not in verify_clause_numbers(answer, docs)


class TestApplySafetyFilter:
    """P0-4: 输出侧安全过滤"""

    def test_appends_disclaimer_on_dose(self):
        answer = "麻黄三钱，桂枝二钱"
        result = apply_safety_filter(answer)
        assert "不构成诊疗建议" in result
        assert "咨询专业中医师" in result

    def test_appends_emergency_notice(self):
        answer = "若出现心梗症状"
        result = apply_safety_filter(answer)
        assert "立即前往医院急诊" in result

    def test_no_keyword_no_change(self):
        answer = "桂枝汤调和营卫，解肌发汗"
        assert apply_safety_filter(answer) == answer

    def test_avoids_duplicate_disclaimer(self):
        answer = "麻黄三钱"
        once = apply_safety_filter(answer)
        twice = apply_safety_filter(once)
        assert once.count("不构成诊疗建议") == 1
        assert twice.count("不构成诊疗建议") == 1

    def test_prescribing_action_triggers_disclaimer(self):
        """给病人开药/吃点药等处方动作语气 → 追加免责（线上实测暴露）"""
        for bad in ["你给病人吃点药", "可以开点药吃", "给你开个药", "建议用药治疗"]:
            result = apply_safety_filter(bad)
            assert "不构成诊疗建议" in result, bad

    def test_empty_answer(self):
        assert apply_safety_filter("") == ""


class TestFormatContextExtras:
    """P0-4: 症状→证候→方剂路径注入组成"""

    def test_formula_compositions_rendered(self):
        extras = {
            "formula_compositions": [
                {"name": "麻黄汤", "herbs": ["麻黄", "桂枝", "甘草", "杏仁"], "syndrome": "太阳伤寒证"},
                {"name": "桂枝汤", "herbs": ["桂枝", "芍药", "甘草", "生姜", "大枣"], "syndrome": "太阳中风证"},
            ]
        }
        result = format_context_extras(extras)
        assert "【方剂组成】" in result
        assert "麻黄汤：麻黄、桂枝、甘草、杏仁" in result
        assert "主治太阳伤寒证" in result
        assert "原文照抄" in result


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
    def test_stream_appends_disclaimer_suffix(self, mock_post):
        """P0-4: 流式内容含剂量时收尾追加免责声明"""
        mock_post.return_value = self._mock_response(
            stream_chunks=[
                {"message": {"content": "麻黄"}},
                {"message": {"content": "三钱。"}},
            ]
        )
        gen = Generator(model="test-model")
        chunks = list(gen.stream_generate("问题", [_make_doc()]))
        full = "".join(chunks)
        assert "不构成诊疗建议" in full
        # 原始流式内容未被改写，免责以追加形式出现
        assert "麻黄三钱。" in full

    @patch("src.rag.generate.requests.post")
    def test_stream_no_disclaimer_without_dose(self, mock_post):
        mock_post.return_value = self._mock_response(
            stream_chunks=[{"message": {"content": "桂枝汤调和营卫。"}}]
        )
        gen = Generator(model="test-model")
        chunks = list(gen.stream_generate("问题", [_make_doc()]))
        full = "".join(chunks)
        assert "不构成诊疗建议" not in full

    @patch("src.rag.generate.requests.post")
    def test_generate_verify_removes_fabricated_clause(self, mock_post):
        """P0-4: 同步生成时删除编造的条文编号"""
        mock_post.return_value = self._mock_response(
            content="此证见【第99条】，太阳之为病【第1条】"
        )
        gen = Generator(model="test-model")
        docs = [_make_doc(clause_id=1)]
        result = gen.generate("问题", docs)
        assert "第99条" not in result
        assert "【第1条】" in result

    @patch("src.rag.generate.requests.post")
    def test_generate_can_disable_verify(self, mock_post):
        mock_post.return_value = self._mock_response(content="见【第99条】")
        gen = Generator(model="test-model")
        result = gen.generate("问题", [_make_doc(clause_id=1)], verify=False)
        assert "第99条" in result

    @patch("src.rag.generate.requests.post")
    def test_generate_max_tokens(self, mock_post):
        mock_post.return_value = self._mock_response(content="回答")
        gen = Generator(model="test-model")
        gen.generate("问题", [_make_doc()], max_tokens=128)
        payload = mock_post.call_args.kwargs["json"]
        assert payload["options"].get("num_predict") == 128
