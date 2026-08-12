"""GraphRAG 图谱查询测试（PRD v3.0 图谱重建前置）"""
from src.rag.graph_rag import TCMKnowledgeGraph


class TestGraphRAG:
    def setup_method(self):
        self.kg = TCMKnowledgeGraph()

    def test_query_by_syndrome_no_error(self):
        """query_by_syndrome 不应抛异常（修复 seen_clause 拼写 bug 回归测试）"""
        result = self.kg.query_by_syndrome("少阳证")
        assert isinstance(result.formula_names, list)
        assert isinstance(result.clause_ids, list)

    def test_query_by_syndrome_returns_formulas(self):
        """证候查询能返回方剂"""
        result = self.kg.query_by_syndrome("少阳证")
        assert result.formula_names

    def test_query_by_herb_no_error(self):
        """药材反查正常"""
        result = self.kg.query_by_herb("桂枝")
        assert result.formula_names
