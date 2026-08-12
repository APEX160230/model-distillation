"""辨证判定引擎测试（PRD v3.0 §5 FR3）

覆盖：
- 症状提取（口语 → 术语 → 投票）
- 证型投票判定（太阳伤寒/中风、少阳、阳明热）
- 关键鉴别追问（有汗/无汗 分叉时）
- 低置信拒答（图谱外场景）
"""
from src.rag.diagnosis import DiagnosisEngine


class TestExtractSymptoms:
    """口语文本 → 术语症状提取"""

    def setup_method(self):
        self.engine = DiagnosisEngine()

    def test_colloquial_to_term(self):
        """怕冷→恶寒、不出汗→无汗 口语归一化"""
        s = self.engine.extract_symptoms("我这两天头痛，怕冷，不出汗")
        assert "头痛" in s
        assert "恶寒" in s
        assert "无汗" in s

    def test_direct_classical_term(self):
        """文言症状直接命中"""
        s = self.engine.extract_symptoms("口苦咽干目眩")
        assert "口苦" in s
        assert "咽干" in s
        assert "目眩" in s

    def test_unknown_symptom_ignored(self):
        """图谱外的症状词不产生噪音"""
        s = self.engine.extract_symptoms("我耳鸣，最近还头晕")
        # 耳鸣/头晕 不在投票表 → 不出现或忽略，但不应报错
        assert isinstance(s, list)

    def test_negation_not_extracted(self):
        """"不出汗"不应提取出"汗出"（负向子串回归测试）"""
        s = self.engine.extract_symptoms("我这两天头痛，怕冷，不出汗")
        assert "汗出" not in s
        assert "无汗" in s


class TestDiagnose:
    """证型投票判定"""

    def setup_method(self):
        self.engine = DiagnosisEngine()

    def test_sun_tai_shang_han(self):
        """头痛+恶寒+无汗 → 太阳伤寒，麻黄汤类方"""
        r = self.engine.diagnose("我这两天头痛，怕冷，不出汗")
        assert r.status == "diagnosed"
        assert r.syndrome == "太阳伤寒"
        assert r.family == "麻黄汤类方"
        assert set(r.evidence) >= {"头痛", "恶寒", "无汗"}

    def test_sun_tai_zhong_feng(self):
        """发热+汗出+恶风 → 太阳中风，桂枝汤类方"""
        r = self.engine.diagnose("发烧了，有出汗，有点怕风")
        assert r.status == "diagnosed"
        assert r.syndrome == "太阳中风"
        assert r.family == "桂枝汤类方"

    def test_shao_yang(self):
        """口苦+咽干+目眩 → 少阳证，柴胡汤类方"""
        r = self.engine.diagnose("嘴里发苦，嗓子干，头晕目眩")
        assert r.status == "diagnosed"
        assert r.syndrome == "少阳证"
        assert r.family == "柴胡汤类方"

    def test_yang_ming_heat(self):
        """大热+大汗+大渴 → 阳明热证（覆盖发热的干扰票）"""
        r = self.engine.diagnose("发高烧，出大汗，口渴想喝凉的")
        assert r.status == "diagnosed"
        assert r.syndrome == "阳明热证"

    def test_evidence_insufficient_asks_question(self):
        """只有头痛（1票）→ 追问而非硬判"""
        r = self.engine.diagnose("我有点头痛")
        assert r.status == "need_clarification"
        assert r.question is not None
        assert len(r.options) >= 2

    def test_parallel_scores_asks_differential(self):
        """恶寒+头痛：伤寒/中风并列 2 票 → 追问汗出与否"""
        r = self.engine.diagnose("怕冷，头痛")
        assert r.status == "need_clarification"
        assert "出汗" in r.question

    def test_rejected_when_no_syndrome_match(self):
        """图谱外症状 → 低置信拒答"""
        r = self.engine.diagnose("我耳鸣，听力下降")
        assert r.status == "rejected"
        assert r.reason is not None

    def test_diagnose_symptoms_list_direct(self):
        """直接传症状列表（追问二次判定用）"""
        r = self.engine.diagnose_symptoms(["恶寒", "无汗", "头痛"])
        assert r.status == "diagnosed"
        assert r.syndrome == "太阳伤寒"

    def test_resolve_with_answer(self):
        """追问答案回填后能判定：怕冷+头痛+没出汗 → 太阳伤寒"""
        r = self.engine.diagnose_symptoms(["恶寒", "头痛", "无汗"])
        assert r.status == "diagnosed"
        assert r.syndrome == "太阳伤寒"
