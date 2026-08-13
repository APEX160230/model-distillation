"""生成模块 — Ollama 本地推理

调用 Ollama API 进行文本生成，支持同步和流式两种模式。
P2.1: 支持从 ConceptMapper/GraphRAG 传入结构化上下文。
"""
from __future__ import annotations

import json
import os
import re
from typing import Any, Iterator

import requests

from src.rag.retrieve import RetrievalResult
from src.data.symptom_mapping import (
    SYNDROME_SUGGESTIONS,
    SEEK_CARE_GUIDANCE,
)

SYSTEM_PROMPT = """你是一位中医老师，擅长用通俗易懂的方式讲解中医经典知识。
请用口语化的讲解风格，像老师讲课一样回答问题，同时保持正式得体。

回答规则（必须严格遵守）：
1. 【方剂信息】【组成】等参考信息中的方剂组成、主治是经过核实的权威内容，
   引用时必须原文照抄，不得增删改换任何一味药。
2. 引用经典原文时，只能引用参考信息中出现的条文，编号以【第X条】为准，
   不得编造不存在的条文编号。
3. 参考信息中没有的内容，不要编造。特别是：
   - 不知道方剂的用量剂量时，明确说明"原书记载的剂量用法需查阅原文，此处不提供"
   - 不提供具体诊疗建议，不针对个人病情开方
4. 如果检索结果为空或与问题无关，如实说明"知识库中未收录相关内容"。
5. 回答控制在 200-500 字，结构清晰，不要跑题。"""

# 辨证模式 SYSTEM_PROMPT（PRD v3.0 §5 FR5 三层回答）
# 前两层（辨证方向/类方思路）由代码模板生成（build_diagnosis_template），
# 模型只生成第三层【讲解与调理】——最大程度压缩 1.5B 自由发挥空间
DIAGNOSIS_SYSTEM_PROMPT = """你是一位中医老师，擅长用通俗易懂的方式讲解中医经典知识。
用户身体不适，系统已给出辨证方向与类方思路，你只需生成【讲解与调理】这一层内容：

【讲解与调理】要求：
1. 引用检索到的经典原文（标注【第X条】），用倪海厦讲课的口吻解释这个方向
2. 补充生活调理注意事项和就医引导
3. 只依据检索到的经典原文，不要添加原文之外的内容
4. 严禁给出方剂组成、剂量或开具处方
5. 控制在 200-350 字

示例：
【讲解与调理】
倪师讲太阳伤寒时说：「太阳病，头痛发热，身疼腰痛，骨节疼痛，恶风无汗而喘者，麻黄汤主之。」
这种情况是寒邪束表、毛孔紧闭，汗发不出来，所以人会怕冷、身上疼。生活上注意保暖、喝温热的水、
早点休息、别再吹风受凉。如果发热持续不退，或出现胸闷、喘促，请及时去医院。"""


def format_retrieved_docs(docs: list[RetrievalResult]) -> str:
    """格式化检索结果为 prompt 上下文"""
    if not docs:
        return "（未检索到相关原文）"

    lines = []
    for doc in docs:
        lines.append(f"【第{doc.clause_id}条·{doc.chapter}】\n{doc.text}")
    return "\n\n".join(lines)


def build_diagnosis_template(diagnosis: dict, docs: list[RetrievalResult] | None = None) -> str:
    """辨证前两层模板（代码生成，不经模型，保证严谨）

    第一层【辨证方向】：图谱结论 + 判定依据 + 条文引用
    第二层【类方思路】：类方标签 + 免责声明
    第三层【讲解与调理】由模型生成后拼接。
    """
    syndrome = diagnosis.get("syndrome", "")
    brief = diagnosis.get("brief", "")
    evidence = diagnosis.get("evidence", [])
    family = diagnosis.get("family", "")

    layer1 = f"【辨证方向】\n初步判断偏「{syndrome}」方向"
    if brief:
        layer1 += f"（{brief}）"
    if evidence:
        layer1 += f"。依据：您提到的{'、'.join(evidence)}同时指向这个方向"
    if docs:
        cids = [d.clause_id for d in docs[:3] if getattr(d, "clause_id", None)]
        if cids:
            layer1 += f"，与《伤寒论》第{'、'.join(map(str, cids))}条所述相符"
    layer1 += "。"

    parts = [layer1]
    if family:
        parts.append(
            f"【类方思路】\n传统上这类情况多从「{family}」思路考虑"
            "（仅作中医知识参考，不构成用药建议）。"
        )
    return "\n\n".join(parts)


DIAGNOSIS_DISCLAIMER = (
    "\n\n若症状持续或加重，请及时就医。"
    "本内容仅供中医知识学习参考，不构成诊疗建议。"
)

# 剂量脱敏：讲稿原文中的剂量记载（"麻黄三两""桂枝二两去皮"）不直引给用户，
# 避免被误认为用药建议。替换为占位标记。
_DOSE_MENTION_PATTERN = re.compile(
    r"[0-9０-９一二三四五六七八九十百千万两半]+\s*(克|钱|两|斤|g|G|ml|毫升|片|粒|枚|付|剂|碗|升|合|斗)"
)

# 煎服信号：素材中出现即截断（"上四味，以水九升……温服八合"属于实操指导，不直引）
_COOKING_SIGNAL = re.compile(r"上四味|以水|煮取|温服|煎服|去滓|内诸药|为散|杵捣|杵为")


def _sanitize_lecture_text(text: str) -> str:
    """讲稿素材安全化：剂量脱敏 + 煎服方法截断

    讲稿原文常含"麻黄三两""上四味，以水九升，煮取二升半……"等
    实操指导，直引给用户有被误认为用药建议的风险，统一处理。
    """
    text = _DOSE_MENTION_PATTERN.sub("〔剂量从略〕", text)
    m = _COOKING_SIGNAL.search(text)
    if m:
        text = text[: m.start()] + "（煎服方法从略）"
    return text


def build_lecture_layer(
    lectures: list[dict] | None = None,
    docs: list[RetrievalResult] | None = None,
) -> str:
    """第三层【倪师讲解】：讲稿素材直引（锚点理论，不经模型）

    素材为倪师讲稿原文片段（检索自 lectures collection），原文照抄，
    不经过模型组织——杜绝 1.5B 自由发挥导致的幻觉。
    剂量记载脱敏、煎服方法截断（避免被误认为用药建议）。
    无讲稿素材时回退为条文原文引用。
    """
    parts = ["【倪师讲解】"]
    if lectures:
        for lec in lectures[:3]:
            text = _sanitize_lecture_text((lec.get("text", "") or "").strip())[:150]
            if not text:
                continue
            book = lec.get("book", "")
            prefix = f"（《{book}》节选）" if book else ""
            parts.append(f"{prefix}{text}")
    elif docs:
        parts.append("倪师讲稿库中暂未检索到直接相关的讲解，以下为经典原文参考：")
        for doc in docs[:3]:
            cid = getattr(doc, "clause_id", None)
            head = f"【第{cid}条】" if cid else ""
            parts.append(f"{head}{doc.text[:120]}")
    else:
        parts.append("（暂无相关讲解素材，建议咨询专业中医师。）")
    return "\n".join(parts)


def build_diagnosis_full_answer(
    diagnosis: dict,
    docs: list[RetrievalResult] | None = None,
    lectures: list[dict] | None = None,
) -> str:
    """完整辨证回答（全部代码生成，不经模型）

    五层结构（用户真实诉求"我能怎么办"的核心答案）：
    1. 【辨证方向】图谱结论 + 依据（学术判断）
    2. 【类方思路】类方标签 + 免责（不构成用药建议）
    3. 【倪师讲解】讲稿原文直引（学术依据）
    4. 【调理建议】基于证型的通用生活建议（用户最需要——"我该怎么办"）
    5. 【就医引导】什么情况必须就医

    前 3 层是"知识/判断"，第 4 层是"建议"（锚点理论扩展：
    通用生活建议是中医常识，不涉及方药剂量，模板生成零幻觉）。
    严谨性 100%——所有内容均来自代码模板或讲稿原文，无模型生成。
    """
    layers = [
        build_diagnosis_template(diagnosis, docs),
        build_lecture_layer(lectures, docs),
    ]
    syndrome = diagnosis.get("syndrome", "")
    suggestions = SYNDROME_SUGGESTIONS.get(syndrome, [])
    if suggestions:
        items = "；".join(suggestions)
        layers.append(f"【调理建议】\n{items}")
    layers.append(f"【就医引导】\n{SEEK_CARE_GUIDANCE}")
    return "\n\n".join(layers) + DIAGNOSIS_DISCLAIMER


def format_context_extras(extras: dict[str, Any] | None) -> str:
    """格式化额外上下文（概念简述、方剂信息、对比数据等）

    P2.1: 将 ConceptMapper 和 GraphRAG 的结构化数据
    转化为自然语言提示，帮助 1.5B 模型生成更好的答案。
    """
    if not extras:
        return ""

    parts: list[str] = []

    # 概念简述
    concept = extras.get("concept")
    if concept:
        parts.append(f"【概念参考】{concept.get('brief', '')}")
        related = concept.get("related_formulas", [])
        if related:
            parts.append(f"相关方剂：{'、'.join(related)}")

    # 方剂信息（方剂查询时）
    formula_info = extras.get("formula_info")
    if formula_info:
        herbs = "、".join(formula_info.get("herbs", []))
        parts.append(f"【方剂信息】{formula_info.get('name', '')}：{herbs}")
        parts.append(f"主治：{formula_info.get('syndrome', '')}")
        parts.append(f"功效：{formula_info.get('brief', '')}")

    # 药材查询结果
    herb_query = extras.get("herb_query")
    if herb_query:
        formula_names = herb_query.get("formula_names", [])
        count = herb_query.get("formula_count", 0)
        herbs = "、".join(herb_query.get("herbs", []))
        if formula_names:
            parts.append(f"【药材查询】含{herbs}的方剂共{count}首：{'、'.join(formula_names)}")

    # 方剂对比数据
    comparison = extras.get("comparison")
    if comparison and "error" not in comparison:
        f1 = comparison.get("formula1", {})
        f2 = comparison.get("formula2", {})
        parts.append(f"【方剂对比】")
        parts.append(f"{f1.get('name', '')}：{'、'.join(f1.get('herbs', []))}，主治{f1.get('syndrome', '')}")
        parts.append(f"{f2.get('name', '')}：{'、'.join(f2.get('herbs', []))}，主治{f2.get('syndrome', '')}")
        only1 = comparison.get("herbs_only_in_1", [])
        only2 = comparison.get("herbs_only_in_2", [])
        common = comparison.get("common_herbs", [])
        if only1:
            parts.append(f"{f1.get('name', '')}独有：{'、'.join(only1)}")
        if only2:
            parts.append(f"{f2.get('name', '')}独有：{'、'.join(only2)}")
        if common:
            parts.append(f"共有药材：{'、'.join(common)}")

    # GraphRAG 证候查询
    graph_syndrome = extras.get("graph_syndrome")
    if graph_syndrome:
        formulas = graph_syndrome.get("formulas", [])
        if formulas:
            parts.append(f"【证候方剂】相关方剂：{'、'.join(formulas)}")

    # 方剂组成列表（症状→证候→方剂路径注入，P0-4）
    compositions = extras.get("formula_compositions")
    if compositions:
        parts.append("【方剂组成】（以下组成经核实，引用时必须原文照抄）")
        for c in compositions:
            herbs = "、".join(c.get("herbs", []))
            line = f"{c.get('name', '')}：{herbs}"
            if c.get("syndrome"):
                line += f"（主治{c.get('syndrome')}）"
            parts.append(line)

    # 辨证结论（PRD v3.0 三层回答第一二层）
    diagnosis = extras.get("diagnosis")
    if diagnosis and diagnosis.get("status") == "diagnosed":
        parts.append("【辨证结论】（系统判定，引用时必须原文照抄）")
        syndrome = diagnosis.get("syndrome", "")
        brief = diagnosis.get("brief", "")
        if brief:
            parts.append(f"证型方向：{syndrome}（{brief}）")
        else:
            parts.append(f"证型方向：{syndrome}")
        evidence = diagnosis.get("evidence", [])
        if evidence:
            parts.append(f"判定依据：{'、'.join(evidence)}")
        family = diagnosis.get("family")
        if family:
            parts.append(f"类方思路：传统多从「{family}」思路考虑（仅作知识参考，不构成用药建议）")

    # 倪师讲稿素材（FR4：第三层讲解引用倪师原话）
    lectures = extras.get("lectures")
    if lectures:
        parts.append("【倪师讲稿】（倪海厦讲课原文，引用时原文照抄，剂量记载省略不引用）")
        for lec in lectures:
            text = (lec.get("text", "") or "").strip()[:200]
            if not text:
                continue
            book = lec.get("book", "")
            topic = lec.get("topic", "")
            prefix = f"《{book}》" if book else ""
            if topic:
                prefix += f"「{topic}」"
            parts.append(f"{prefix}：{text}")

    # 超范围标记
    out_of_scope = extras.get("out_of_scope")
    if out_of_scope:
        parts.append(f"【范围提示】{out_of_scope.get('message', '')}")

    if not parts:
        return ""

    return "参考信息：\n" + "\n".join(parts)


def build_prompt(
    question: str,
    docs: list[RetrievalResult],
    context_extras: dict[str, Any] | None = None,
) -> str:
    """构建完整 prompt

    Args:
        question: 用户问题
        docs: 检索到的文档列表
        context_extras: 额外上下文（概念简述、方剂信息等）
    """
    context = format_retrieved_docs(docs)
    extras = format_context_extras(context_extras)

    prompt = f"请根据以下经典原文回答问题。\n\n经典原文：\n{context}"
    if extras:
        prompt += f"\n\n{extras}"
    prompt += f"\n\n问题：{question}"
    return prompt


class Generator:
    """Ollama 生成器

    使用 Ollama REST API 进行文本生成。
    P2.1: 支持从 ConceptMapper/GraphRAG 传入结构化上下文。

    使用示例:
        gen = Generator(model="tcm-model")
        answer = gen.generate("桂枝汤的组成", docs, context_extras={...})
    """

    OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

    def __init__(self, model: str = "tcm-model") -> None:
        self._model = model

    def _host(self) -> str:
        """运行时读取 OLLAMA_HOST（支持热更新，测试可 monkeypatch）"""
        return os.environ.get("OLLAMA_HOST", self.OLLAMA_HOST)

    def generate(
        self,
        question: str,
        docs: list[RetrievalResult],
        temperature: float = 0.7,
        max_tokens: int = 512,
        context_extras: dict[str, Any] | None = None,
        verify: bool = True,
        safe_filter: bool = True,
    ) -> str:
        """同步生成回答

        Args:
            question: 用户问题
            docs: 检索到的文档列表
            temperature: 采样温度
            max_tokens: 最大生成 token 数
            context_extras: 额外上下文（P2.1）
            verify: 是否对回答做条文编号交叉校验（P0-4）
            safe_filter: 是否对回答做输出侧安全过滤（P0-4）
        """
        prompt = build_prompt(question, docs, context_extras)
        system_prompt = (
            DIAGNOSIS_SYSTEM_PROMPT
            if context_extras and context_extras.get("diagnosis")
            else SYSTEM_PROMPT
        )
        response = requests.post(
            f"{self._host()}/api/chat",
            json={
                "model": self._model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens,
                },
            },
            timeout=120,
        )
        response.raise_for_status()
        answer = response.json()["message"]["content"]
        if verify:
            answer = verify_clause_numbers(answer, docs)
        if safe_filter:
            answer = apply_safety_filter(answer)
        return answer

    def stream_generate(
        self,
        question: str,
        docs: list[RetrievalResult],
        temperature: float = 0.7,
        max_tokens: int = 512,
        context_extras: dict[str, Any] | None = None,
        verify: bool = True,
        safe_filter: bool = True,
    ) -> Iterator[str]:
        """流式生成回答

        Args:
            question: 用户问题
            docs: 检索到的文档列表
            temperature: 采样温度
            max_tokens: 最大生成 token 数
            context_extras: 额外上下文（P2.1）
            verify: 是否对回答做条文编号交叉校验（P0-4，流式在收尾时校验）
            safe_filter: 是否对回答做输出侧安全过滤（P0-4，流式在收尾时追加免责）

        Yields:
            生成的文本片段（流式片段 + 可能的收尾修正片段）
        """
        prompt = build_prompt(question, docs, context_extras)
        system_prompt = (
            DIAGNOSIS_SYSTEM_PROMPT
            if context_extras and context_extras.get("diagnosis")
            else SYSTEM_PROMPT
        )
        response = requests.post(
            f"{self._host()}/api/chat",
            json={
                "model": self._model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                "stream": True,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens,
                },
            },
            stream=True,
            timeout=120,
        )
        response.raise_for_status()
        collected: list[str] = []
        for line in response.iter_lines():
            if line:
                chunk = json.loads(line)
                content = chunk.get("message", {}).get("content", "")
                if content:
                    collected.append(content)
                    yield content

        # 流式已发出的内容无法撤回，编号校验无法修正已发内容；
        # 安全过滤通过追加免责声明实现（P0-4）
        if collected and safe_filter:
            full = "".join(collected)
            filtered = apply_safety_filter(full)
            suffix = filtered[len(full):]
            if suffix:
                yield suffix


# ── 输出侧安全护栏（P0-4）──────────────────────────────────

# 条文编号引用模式：匹配【第X条】或第X条
_CLAUSE_REF_PATTERN = re.compile(r"[【\[]\s*第\s*(\d+)\s*条\s*[】\]]|第\s*(\d+)\s*条")


def verify_clause_numbers(answer: str, docs: list[RetrievalResult]) -> str:
    """条文编号交叉校验：删除回答中检索结果里不存在的条文编号引用

    模型生成时可能编造条文编号（如把第 23 条内容标成第 12 条）。
    此函数扫描回答中的「第X条」引用，凡编号不在本次检索结果中的，
    删除该引用标记（保留其余文本），避免误导用户。

    Args:
        answer: 模型生成的回答
        docs: 本次检索结果

    Returns:
        校验后的回答
    """
    if not docs:
        return answer

    valid_ids = {d.clause_id for d in docs if hasattr(d, "clause_id") and d.clause_id is not None}

    def _replace(match: re.Match) -> str:
        g1, g2 = match.group(1), match.group(2)
        num = int(g1 or g2)
        if num in valid_ids:
            return match.group(0)
        # 编号不在检索结果中 → 删除引用标记，避免编造
        return ""

    return _CLAUSE_REF_PATTERN.sub(_replace, answer)


# 剂量/处方关键词（数字+单位，或明确的剂量引导词）
# 数字支持阿拉伯数字与中文数字（中医剂量常用"三钱""二钱""十二枚"）
_CN_NUM = r"[0-9０-９一二三四五六七八九十百千万两半]"
_DOSE_PATTERN = re.compile(
    rf"{_CN_NUM}+\s*(克|钱|两|斤|g|G|ml|毫升|片|粒|枚|付|剂|碗|升)"
    r"|剂量|用量|用法|一次\s*\d+|每次\s*\d+|开方|处方|抓药|煎服|水煎服"
    r"|吃点药|开点药|开个药|给你开|用药治疗|用药建议|吃这个药"
)
# 危象关键词：需要立即就医
_EMERGENCY_PATTERN = re.compile(r"脑溢血|脑出血|心梗|心肌梗死|休克|大出血|昏迷|抽搐|呼吸困难|剧烈胸痛")
# 免责声明
_DISCLAIMER = "\n\n（以上内容仅为中医经典知识讲解，不构成诊疗建议。具体用药请咨询专业中医师。）"


def apply_safety_filter(answer: str) -> str:
    """输出侧安全过滤：检测剂量/处方/危象关键词，追加免责声明

    评测中模型曾直接给出具体剂量（如"桂枝5钱"）甚至编造处方，
    违反 PRD「不提供诊疗建议」边界。检测到剂量类表述时追加免责声明；
    检测到危象症状时追加就医提醒。

    Args:
        answer: 模型生成的回答

    Returns:
        过滤后的回答（可能追加免责声明）
    """
    if not answer:
        return answer

    append = []
    if _DOSE_PATTERN.search(answer):
        append.append(_DISCLAIMER)
    if _EMERGENCY_PATTERN.search(answer):
        append.append("\n\n（若您或家人出现上述急危重症表现，请立即前往医院急诊就医。）")

    if not append:
        return answer

    # 避免重复追加
    result = answer
    for suffix in append:
        if suffix not in result:
            result += suffix
    return result
