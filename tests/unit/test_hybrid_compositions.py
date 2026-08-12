"""HybridRetriever 症状路径方剂组成注入测试（P0-4）

覆盖：SEMANTIC 分支（症状→证候→方剂）检索到方剂名时，
从 formulas_db 取回组成注入 last_context，避免模型凭记忆编造组成。
"""
from types import SimpleNamespace
from unittest.mock import MagicMock

from src.rag.hybrid_retriever import HybridRetriever


def _make_retriever():
    """构造只含 _enrich_formula_compositions 所需依赖的实例"""
    r = HybridRetriever.__new__(HybridRetriever)
    r._formula_dict = {
        "麻黄汤": SimpleNamespace(
            name="麻黄汤",
            herbs=["麻黄", "桂枝", "甘草", "杏仁"],
            syndrome="太阳伤寒证",
            brief="发汗解表，宣肺平喘",
            clause_id=35,
        ),
        "桂枝汤": SimpleNamespace(
            name="桂枝汤",
            herbs=["桂枝", "芍药", "甘草", "生姜", "大枣"],
            syndrome="太阳中风证",
            brief="调和营卫，解肌发汗",
            clause_id=12,
        ),
    }
    return r


class TestEnrichFormulaCompositions:
    def test_returns_compositions_for_known_formulas(self):
        r = _make_retriever()
        comps = r._enrich_formula_compositions(["麻黄汤", "桂枝汤"])
        assert len(comps) == 2
        assert comps[0]["herbs"] == ["麻黄", "桂枝", "甘草", "杏仁"]
        assert comps[1]["herbs"] == ["桂枝", "芍药", "甘草", "生姜", "大枣"]

    def test_skips_unknown_formulas(self):
        r = _make_retriever()
        comps = r._enrich_formula_compositions(["不存在的汤", "麻黄汤"])
        assert len(comps) == 1
        assert comps[0]["name"] == "麻黄汤"

    def test_empty_list_returns_empty(self):
        r = _make_retriever()
        assert r._enrich_formula_compositions([]) == []
