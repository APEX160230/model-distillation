"""辨证判定引擎（PRD v3.0 §5 FR3）

职责：用户症状 → 证型方向判定。判定完全基于数据（图谱投票），不依赖模型推理。

流程：
1. extract_symptoms: 口语文本 → 术语症状集合（复用 concept_mapper + 直接扫描症状表）
2. diagnose_symptoms: 症状交集投票 → 证型判定 / 关键鉴别追问 / 低置信拒答

锚点原则：判定结论来自 SYMPTOM_SYNDROME 映射表（可解释、可校验），
1.5B 模型不参与此环节。
"""
from __future__ import annotations

from dataclasses import dataclass, field

from src.data.symptom_mapping import (
    DIFFERENTIAL_QUESTIONS,
    SYNDROME_BRIEF,
    SYNDROME_FAMILY,
    SYMPTOM_SYNDROME,
)
from src.rag.concept_mapper import ConceptMapper


@dataclass
class DiagnosisResult:
    """辨证判定结果"""
    status: str  # "diagnosed" | "need_clarification" | "rejected"
    syndrome: str | None = None
    syndrome_brief: str | None = None
    family: str | None = None  # 类方标签（轻提示）
    evidence: list[str] = field(default_factory=list)  # 命中的症状
    question: str | None = None  # 追问问题
    options: list[str] = field(default_factory=list)
    reason: str | None = None  # 拒答原因

    def to_dict(self) -> dict:
        """结构化输出（供 pipeline 注入 context_extras / API 返回）"""
        return {
            "status": self.status,
            "syndrome": self.syndrome,
            "brief": self.syndrome_brief,
            "family": self.family,
            "evidence": self.evidence,
            "question": self.question,
            "options": self.options,
            "reason": self.reason,
        }


class DiagnosisEngine:
    """辨证判定引擎"""

    def __init__(self) -> None:
        self._mapper = ConceptMapper()
        self._symptom_terms = set(SYMPTOM_SYNDROME.keys())

    def extract_symptoms(self, text: str) -> list[str]:
        """从文本提取图谱内症状术语

        两条路径：
        1. concept_mapper 的口语→文言归一化（怕冷→恶寒）
        2. 直接扫描文本中出现的图谱术语（往来寒热等长词）
        """
        symptoms: list[str] = []
        seen: set[str] = set()

        # 路径 1：concept_mapper 归一化结果
        for term in self._mapper.extract_symptoms(text):
            if term in self._symptom_terms and term not in seen:
                symptoms.append(term)
                seen.add(term)

        # 路径 2：直接扫描图谱复合术语（长度 ≥ 4，避免"不出汗"误匹配"汗出"）
        # 短术语由 concept_mapper 词典精确归一化（不出汗→无汗），不走子串扫描
        for term in sorted(
            (t for t in self._symptom_terms if len(t) >= 4),
            key=len,
            reverse=True,
        ):
            if term in text and term not in seen:
                symptoms.append(term)
                seen.add(term)

        return symptoms

    def diagnose(self, text: str) -> DiagnosisResult:
        """端到端：文本 → 判定"""
        return self.diagnose_symptoms(self.extract_symptoms(text))

    def diagnose_symptoms(self, symptoms: list[str]) -> DiagnosisResult:
        """症状集合 → 投票判定"""
        # 投票：证型 t 得分 = 命中的症状数
        votes: dict[str, int] = {}
        evidence: list[str] = []
        for s in symptoms:
            candidates = SYMPTOM_SYNDROME.get(s)
            if not candidates:
                continue
            evidence.append(s)
            for t in candidates:
                votes[t] = votes.get(t, 0) + 1

        if not votes:
            return DiagnosisResult(
                status="rejected",
                reason="这个情况我不太确定，建议咨询专业中医师或前往医院就诊。",
            )

        # 排序：得分降序
        ranked = sorted(votes.items(), key=lambda kv: kv[1], reverse=True)
        top_syndrome, top_score = ranked[0]
        second_score = ranked[1][1] if len(ranked) > 1 else 0

        if top_score >= 2 and top_score - second_score >= 1:
            return DiagnosisResult(
                status="diagnosed",
                syndrome=top_syndrome,
                syndrome_brief=SYNDROME_BRIEF.get(top_syndrome),
                family=SYNDROME_FAMILY.get(top_syndrome),
                evidence=evidence,
            )

        # 证据不足或并列 → 关键鉴别追问
        top_syndromes = [t for t, s in ranked if s == top_score]
        question = self._pick_question(top_syndromes)
        if question:
            return DiagnosisResult(
                status="need_clarification",
                evidence=evidence,
                question=str(question["question"]),
                options=list(question["options"]),
                reason="症状信息不足，需要进一步确认。",
            )

        # 无匹配鉴别问题 → 通用追问（引导补充典型症状），不拒答
        generic = self._generic_question(top_syndromes)
        return DiagnosisResult(
            status="need_clarification",
            evidence=evidence,
            question=generic,
            options=[],
            reason="症状信息不足，需要进一步确认。",
        )

    def _generic_question(self, top_syndromes: list[str]) -> str:
        """构造通用追问：列出候选证型的典型症状引导用户补充"""
        hints: list[str] = []
        for t in top_syndromes[:2]:
            for s, candidates in SYMPTOM_SYNDROME.items():
                if t in candidates and s not in hints:
                    hints.append(s)
                if len(hints) >= 6:
                    break
            if len(hints) >= 6:
                break
        hint_text = "、".join(hints) if hints else "其他不适"
        return f"能再详细说说您还有哪些不舒服吗？比如：{hint_text} 等表现。"

    def _pick_question(self, top_syndromes: list[str]) -> dict | None:
        """选择能区分当前候选证型的鉴别问题"""
        for q in DIFFERENTIAL_QUESTIONS:
            related: set[str] = set()
            for terms in q["resolves"].values():
                for term in terms:
                    related.update(SYMPTOM_SYNDROME.get(term, []))
            if related & set(top_syndromes):
                return q
        return None
