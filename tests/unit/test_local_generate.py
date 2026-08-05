"""测试 LocalGenerator — 使用 mock 验证接口正确性"""
import pytest
from unittest.mock import MagicMock, patch, PropertyMock
import torch
from src.rag.local_generate import LocalGenerator
from src.rag.retrieve import RetrievalResult


def _make_doc(clause_id=1, text="太阳之为病，脉浮，头项强痛而恶寒。", chapter="辨太阳病脉证并治上"):
    return RetrievalResult(
        id=f"clause_{clause_id}",
        text=text,
        metadata={"clause_id": clause_id, "chapter": chapter},
        distance=0.2,
    )


def _make_mock_inputs(seq_len=5):
    """创建模拟的 tokenizer 输出（有 .to() 方法）"""
    inputs = MagicMock()
    inputs.__getitem__ = MagicMock(return_value=torch.tensor([[0] * seq_len]))
    inputs.to = MagicMock(return_value=inputs)
    inputs.__contains__ = MagicMock(return_value=True)
    return inputs


class TestLocalGeneratorInit:
    def test_init_with_defaults(self):
        gen = LocalGenerator("output_merged")
        assert gen._model_path == "output_merged"
        assert gen._device == "cpu"
        assert gen._model is None
        assert gen._tokenizer is None

    def test_init_with_custom_params(self):
        gen = LocalGenerator("models/my_model", device="cuda", dtype=torch.float16)
        assert gen._model_path == "models/my_model"
        assert gen._device == "cuda"
        assert gen._dtype == torch.float16


class TestLocalGeneratorGenerate:
    @patch.object(LocalGenerator, "model", new_callable=PropertyMock)
    @patch.object(LocalGenerator, "tokenizer", new_callable=PropertyMock)
    def test_generate_returns_string(self, mock_tok, mock_model):
        """测试 generate 返回字符串"""
        mock_tokenizer = MagicMock()
        mock_tok.return_value = mock_tokenizer
        mock_tokenizer.apply_chat_template.return_value = "test prompt"
        mock_tokenizer.pad_token_id = 0
        mock_tokenizer.eos_token_id = 1
        mock_tokenizer.decode.return_value = "桂枝汤主治太阳中风"

        # tokenizer() 返回有 .to() 方法的 mock
        mock_tokenizer.return_value = _make_mock_inputs(5)

        mock_m = MagicMock()
        mock_model.return_value = mock_m
        # generate 返回 [batch, seq_len]，前 5 个是 input，后面是生成
        mock_m.generate.return_value = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7]])

        gen = LocalGenerator("output_merged")
        result = gen.generate("桂枝汤主治什么", [_make_doc()])
        assert isinstance(result, str)
        assert "桂枝汤" in result

    @patch.object(LocalGenerator, "model", new_callable=PropertyMock)
    @patch.object(LocalGenerator, "tokenizer", new_callable=PropertyMock)
    def test_generate_with_empty_docs(self, mock_tok, mock_model):
        """测试空文档列表"""
        mock_tokenizer = MagicMock()
        mock_tok.return_value = mock_tokenizer
        mock_tokenizer.apply_chat_template.return_value = "test"
        mock_tokenizer.pad_token_id = 0
        mock_tokenizer.eos_token_id = 1
        mock_tokenizer.decode.return_value = "无相关信息"
        mock_tokenizer.return_value = _make_mock_inputs(2)

        mock_m = MagicMock()
        mock_model.return_value = mock_m
        mock_m.generate.return_value = torch.tensor([[0, 1, 2, 3]])

        gen = LocalGenerator("output_merged")
        result = gen.generate("测试问题", [])
        assert isinstance(result, str)

    @patch.object(LocalGenerator, "model", new_callable=PropertyMock)
    @patch.object(LocalGenerator, "tokenizer", new_callable=PropertyMock)
    def test_generate_calls_model_generate(self, mock_tok, mock_model):
        """测试 generate 确实调用了 model.generate"""
        mock_tokenizer = MagicMock()
        mock_tok.return_value = mock_tokenizer
        mock_tokenizer.apply_chat_template.return_value = "prompt"
        mock_tokenizer.pad_token_id = 0
        mock_tokenizer.eos_token_id = 1
        mock_tokenizer.decode.return_value = "回答"
        mock_tokenizer.return_value = _make_mock_inputs(3)

        mock_m = MagicMock()
        mock_model.return_value = mock_m
        mock_m.generate.return_value = torch.tensor([[0, 1, 2, 3, 4]])

        gen = LocalGenerator("output_merged")
        gen.generate("问题", [_make_doc()])
        mock_m.generate.assert_called_once()


class TestLocalGeneratorInterface:
    """验证 LocalGenerator 与 Generator 接口一致"""

    def test_has_generate_method(self):
        gen = LocalGenerator("output_merged")
        assert hasattr(gen, "generate")
        assert callable(gen.generate)

    def test_has_stream_generate_method(self):
        gen = LocalGenerator("output_merged")
        assert hasattr(gen, "stream_generate")
        assert callable(gen.stream_generate)

    def test_generate_signature_matches_generator(self):
        """generate 方法签名应与 Generator 一致"""
        import inspect
        gen_sig = inspect.signature(LocalGenerator.generate)
        params = list(gen_sig.parameters.keys())
        assert "question" in params
        assert "docs" in params
        assert "temperature" in params
        assert "max_tokens" in params
