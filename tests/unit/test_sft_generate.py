"""SFT 数据生成器测试"""
import json
import os
import tempfile
import pytest
from pathlib import Path

from src.data.sft_generate import SFTGeneratorV2, SFTSample, to_chinese_num
from src.data.formulas_db import FORMULAS, get_formula_dict, get_herb_formulas


# ==================== to_chinese_num 测试 ====================

class TestToChineseNum:
    def test_single_digit(self):
        assert to_chinese_num(1) == "一"
        assert to_chinese_num(9) == "九"

    def test_ten(self):
        assert to_chinese_num(10) == "十"

    def test_teens(self):
        assert to_chinese_num(11) == "十一"
        assert to_chinese_num(15) == "十五"
        assert to_chinese_num(19) == "十九"

    def test_tens(self):
        assert to_chinese_num(20) == "二十"
        assert to_chinese_num(35) == "三十五"
        assert to_chinese_num(99) == "九十九"

    def test_hundreds(self):
        assert to_chinese_num(100) == "一百"
        assert to_chinese_num(105) == "一百零五"
        assert to_chinese_num(110) == "一百一十"
        assert to_chinese_num(200) == "二百"
        assert to_chinese_num(330) == "三百三十"


# ==================== formulas_db 测试 ====================

class TestFormulasDB:
    def test_formula_count(self):
        assert len(FORMULAS) >= 50, f"方剂数据库应至少50首，实际{len(FORMULAS)}"

    def test_all_formulas_have_herbs(self):
        for f in FORMULAS:
            assert len(f.herbs) >= 1, f"{f.name} 应至少1味药"
            assert f.clause_id > 0, f"{f.name} 应有出处条文"
            assert f.syndrome, f"{f.name} 应有主治证候"

    def test_formula_dict(self):
        d = get_formula_dict()
        assert "桂枝汤" in d
        assert "麻黄汤" in d
        assert d["桂枝汤"].herbs == ["桂枝", "芍药", "甘草", "生姜", "大枣"]

    def test_herb_formulas(self):
        hf = get_herb_formulas()
        assert "桂枝" in hf
        assert "桂枝汤" in hf["桂枝"]
        assert "麻黄汤" in hf["桂枝"]  # 麻黄汤也含桂枝

    def test_eval_formulas_covered(self):
        """验证评测集中的方剂都在数据库中"""
        eval_formulas = [
            "桂枝汤", "麻黄汤", "小柴胡汤", "大承气汤", "五苓散",
            "真武汤", "炙甘草汤", "四逆汤", "白虎汤", "乌梅丸",
        ]
        d = get_formula_dict()
        for name in eval_formulas:
            assert name in d, f"评测方剂 {name} 不在数据库中"


# ==================== SFT 生成器测试 ====================

class TestSFTGenerator:
    @pytest.fixture
    def generator(self):
        base = Path(__file__).parent.parent.parent
        clauses_path = str(base / "data" / "processed" / "classics" / "shanghan_clauses.jsonl")
        return SFTGeneratorV2(clauses_path)

    def test_loads_clauses(self, generator):
        assert len(generator.clauses) == 330

    def test_clause_retrieval(self, generator):
        samples = generator.gen_clause_retrieval()
        assert len(samples) >= 400, f"原文检索类应>=400条，实际{len(samples)}"
        assert all(s.category == "经典原文检索" for s in samples)
        # 验证每条都有 instruction 和 output
        for s in samples[:10]:
            assert len(s.instruction) > 5
            assert len(s.output) > 10

    def test_formula_query(self, generator):
        samples = generator.gen_formula_query()
        assert len(samples) >= 200, f"方剂查询类应>=200条，实际{len(samples)}"
        assert all(s.category == "方剂查询" for s in samples)
        # 验证方剂组成正确
        guizhi_samples = [s for s in samples if "桂枝汤" in s.instruction and "组成" in s.instruction]
        if guizhi_samples:
            s = guizhi_samples[0]
            assert "桂枝" in s.output
            assert "芍药" in s.output

    def test_herb_association(self, generator):
        samples = generator.gen_herb_association()
        assert len(samples) >= 100, f"药材关联类应>=100条，实际{len(samples)}"
        assert all(s.category == "药材关联" for s in samples)

    def test_concept_explanation(self, generator):
        samples = generator.gen_concept_explanation()
        assert len(samples) >= 50, f"经典解释类应>=50条，实际{len(samples)}"
        assert all(s.category == "经典解释" for s in samples)
        # 验证太阳病解释包含关键信息
        taiyang = [s for s in samples if "太阳病" in s.instruction and "什么" in s.instruction]
        if taiyang:
            assert "脉浮" in taiyang[0].output

    def test_comprehensive(self, generator):
        samples = generator.gen_comprehensive()
        assert len(samples) >= 50, f"综合问答类应>=50条，实际{len(samples)}"
        assert all(s.category == "综合问答" for s in samples)

    def test_generate_all(self, generator):
        samples = generator.generate_all()
        assert len(samples) >= 800, f"总样本应>=800条，实际{len(samples)}"
        # 验证类别覆盖
        categories = set(s.category for s in samples)
        assert categories == {"经典原文检索", "方剂查询", "药材关联", "经典解释", "综合问答"}

    def test_save(self, generator, tmp_path):
        samples = generator.gen_clause_retrieval()[:10]
        output_path = str(tmp_path / "test_sft.jsonl")
        stats = generator.save(samples, output_path)

        assert stats["total"] == 10
        assert os.path.exists(output_path)

        with open(output_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        assert len(lines) == 10

        first = json.loads(lines[0])
        assert "instruction" in first
        assert "input" in first
        assert "output" in first
        assert "category" in first
