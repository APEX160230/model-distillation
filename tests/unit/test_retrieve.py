"""向量检索模块测试

TDD: 先写失败测试，再实现 src/rag/retrieve.py
"""
import json
from pathlib import Path

import numpy as np
import pytest

from src.rag.retrieve import VectorRetriever


class FakeEmbedder:
    """确定性假嵌入器（CI 无网络时替代真实 bge 模型）

    基于关键词匹配构造 one-hot 风格向量：
    - 语义测试（桂枝汤→第12条、麻黄汤→第35条）通过关键词命中近似保持
    - 完全确定性，无网络/模型依赖
    """

    KEYWORDS = ["桂枝", "麻黄", "太阳", "中风", "伤寒", "脉浮", "发热", "恶寒"]

    def __init__(self, dim: int = 8):
        self._dim = dim

    def encode(self, text: str) -> np.ndarray:
        return self.encode_batch([text])[0]

    def encode_batch(self, texts: list[str]) -> np.ndarray:
        vecs = []
        for t in texts:
            v = np.zeros(self._dim, dtype=np.float32)
            for i, kw in enumerate(self.KEYWORDS[: self._dim]):
                if kw in t:
                    v[i] = 1.0
            if v.sum() == 0:
                v[0] = 0.1  # 无关键词命中时给微弱基线，避免零向量
            vecs.append(v)
        return np.stack(vecs)


def make_retriever(tmp_path, embedder=None, **kwargs):
    """构造检索器，默认注入假嵌入器（不下载真实模型）"""
    kwargs.setdefault("persist_dir", str(tmp_path / "chroma"))
    kwargs.setdefault("embedder", embedder or FakeEmbedder())
    return VectorRetriever(**kwargs)


@pytest.fixture
def sample_clauses(tmp_path):
    """创建临时条辨数据用于测试"""
    clauses = [
        {"clause_id": 1, "clause_id_cn": "一", "chapter": "辨太阳病脉证并治上",
         "original_text": "太阳之为病，脉浮，头项强痛而恶寒。"},
        {"clause_id": 2, "clause_id_cn": "二", "chapter": "辨太阳病脉证并治上",
         "original_text": "太阳病，发热，汗出，恶风，脉缓者，名为中风。"},
        {"clause_id": 3, "clause_id_cn": "三", "chapter": "辨太阳病脉证并治上",
         "original_text": "太阳病，或已发热，或未发热，必恶寒，体痛，呕逆，脉阴阳俱紧者，名为伤寒。"},
        {"clause_id": 12, "clause_id_cn": "一二", "chapter": "辨太阳病脉证并治上",
         "original_text": "太阳中风，阳浮而阴弱，阳浮者热自发，阴弱者汗自出，啬啬恶寒，淅淅恶风，翕翕发热，鼻鸣干呕者，桂枝汤主之。"},
        {"clause_id": 35, "clause_id_cn": "三五", "chapter": "辨太阳病脉证并治上",
         "original_text": "太阳病，头痛发热，身疼腰痛，骨节疼痛，恶风无汗而喘者，麻黄汤主之。"},
    ]
    filepath = tmp_path / "test_clauses.jsonl"
    with open(filepath, "w", encoding="utf-8") as f:
        for c in clauses:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    return filepath


class TestRetrieverInit:
    """测试检索器初始化"""

    def test_init_with_persist_dir(self, tmp_path):
        """指定持久化目录初始化"""
        retriever = make_retriever(tmp_path)
        assert retriever.persist_dir.exists()

    def test_init_default_collection_name(self, tmp_path):
        """默认 collection 名为 shanghan"""
        retriever = make_retriever(tmp_path)
        assert retriever.collection_name == "shanghan"


class TestBuildIndex:
    """测试建库"""

    def test_build_from_jsonl(self, sample_clauses, tmp_path):
        """从 JSONL 文件建库"""
        retriever = make_retriever(tmp_path)
        retriever.build_index(str(sample_clauses))
        assert retriever.count() == 5

    def test_build_idempotent(self, sample_clauses, tmp_path):
        """重复建库不翻倍"""
        retriever = make_retriever(tmp_path)
        retriever.build_index(str(sample_clauses))
        retriever.build_index(str(sample_clauses))
        assert retriever.count() == 5

    def test_build_stores_metadata(self, sample_clauses, tmp_path):
        """建库后存储 clause_id 和 chapter 元数据"""
        retriever = make_retriever(tmp_path)
        retriever.build_index(str(sample_clauses))
        results = retriever.query("太阳病", top_k=1)
        assert len(results) == 1
        item = results[0]
        assert item.clause_id > 0
        assert item.chapter != ""
        assert item.text != ""
        assert item.score > 0


class TestQuery:
    """测试查询"""

    def test_query_returns_top_k(self, sample_clauses, tmp_path):
        """查询返回 top_k 条结果"""
        retriever = make_retriever(tmp_path)
        retriever.build_index(str(sample_clauses))
        results = retriever.query("太阳病", top_k=3)
        assert len(results) == 3

    def test_query_results_sorted_by_score(self, sample_clauses, tmp_path):
        """结果按相似度降序排列"""
        retriever = make_retriever(tmp_path)
        retriever.build_index(str(sample_clauses))
        results = retriever.query("太阳病", top_k=5)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_query桂枝汤_retrieves_correct_clause(self, sample_clauses, tmp_path):
        """查询桂枝汤应返回第 12 条"""
        retriever = make_retriever(tmp_path)
        retriever.build_index(str(sample_clauses))
        results = retriever.query("桂枝汤主治什么", top_k=1)
        assert results[0].clause_id == 12

    def test_query麻黄汤_retrieves_correct_clause(self, sample_clauses, tmp_path):
        """查询麻黄汤应返回第 35 条"""
        retriever = make_retriever(tmp_path)
        retriever.build_index(str(sample_clauses))
        results = retriever.query("麻黄汤主治什么", top_k=1)
        assert results[0].clause_id == 35

    def test_query_empty_raises(self, tmp_path):
        """空查询抛出 ValueError"""
        retriever = make_retriever(tmp_path)
        with pytest.raises(ValueError, match="empty"):
            retriever.query("", top_k=3)


class TestPersist:
    """测试持久化"""

    def test_persist_and_reload(self, sample_clauses, tmp_path):
        """建库后重新加载，数据不丢"""
        persist_dir = str(tmp_path / "chroma")

        # 第一次: 建库
        r1 = make_retriever(tmp_path, persist_dir=persist_dir)
        r1.build_index(str(sample_clauses))
        assert r1.count() == 5

        # 第二次: 重新加载
        r2 = make_retriever(tmp_path, persist_dir=persist_dir)
        assert r2.count() == 5
        results = r2.query("太阳病脉浮", top_k=1)
        assert len(results) == 1
