"""概念映射器 — 现代中医概念到经典条文的精确映射

解决 semantic route 的核心问题：现代汉语概念查询（如"什么是阳明病"）
与文言文条文之间的语义鸿沟。

映射策略：
1. 六经病 → 提纲条文 + 代表方剂条文
2. 证候 → 定义条文 + 主治方剂条文
3. 概念 → 相关关键词扩展（用于 BM25 fallback）

数据来源：伤寒论原文（公版）+ 中医教材标准分类。
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ConceptMapping:
    """概念映射结果"""
    concept: str
    # 定义条文（"X之为病" 提纲条文）
    defining_clauses: list[int] = field(default_factory=list)
    # 代表方剂条文（"X汤主之" 治疗条文）
    treatment_clauses: list[int] = field(default_factory=list)
    # 相关方剂名（用于 BM25 查询扩展）
    related_formulas: list[str] = field(default_factory=list)
    # 相关关键词（文言文，用于 BM25 查询扩展）
    expansion_keywords: list[str] = field(default_factory=list)
    # 概念简述（用于增强 generation prompt）
    brief: str = ""

    @property
    def all_clauses(self) -> list[int]:
        """所有相关条文（去重保序）"""
        seen: set[int] = set()
        result: list[int] = []
        for cid in self.defining_clauses + self.treatment_clauses:
            if cid not in seen:
                result.append(cid)
                seen.add(cid)
        return result


# ── 概念映射表 ──────────────────────────────────────────────
# 数据来源：伤寒论原文 + 中医教材标准分类

CONCEPT_MAP: dict[str, ConceptMapping] = {
    # ── 六经病 ──
    "太阳病": ConceptMapping(
        concept="太阳病",
        defining_clauses=[1],
        treatment_clauses=[12, 35],
        related_formulas=["桂枝汤", "麻黄汤"],
        expansion_keywords=["太阳", "脉浮", "头项强痛", "恶寒"],
        brief="太阳病为外感病初期，主表证，以脉浮、头项强痛、恶寒为提纲。",
    ),
    "阳明病": ConceptMapping(
        concept="阳明病",
        defining_clauses=[195],
        treatment_clauses=[223, 208],
        related_formulas=["白虎汤", "大承气汤", "小承气汤", "调胃承气汤"],
        expansion_keywords=["阳明", "胃家实", "大热", "大汗", "大渴", "潮热"],
        brief="阳明病为外感病阳热亢盛阶段，分经证（白虎汤）和腑证（承气汤）。",
    ),
    "少阳病": ConceptMapping(
        concept="少阳病",
        defining_clauses=[277],
        treatment_clauses=[96],
        related_formulas=["小柴胡汤"],
        expansion_keywords=["少阳", "口苦", "咽干", "目眩", "往来寒热", "胸胁苦满"],
        brief="少阳病为邪犯少阳胆经，枢机不利，以口苦咽干目眩为提纲，小柴胡汤主之。",
    ),
    "太阴病": ConceptMapping(
        concept="太阴病",
        defining_clauses=[287],
        treatment_clauses=[386],
        related_formulas=["理中丸", "四逆汤"],
        expansion_keywords=["太阴", "腹满", "吐", "食不下", "自利", "时腹自痛"],
        brief="太阴病为脾阳虚衰、寒湿内盛，以腹满而吐、食不下、自利为提纲。",
    ),
    "少阴病": ConceptMapping(
        concept="少阴病",
        defining_clauses=[295],
        treatment_clauses=[29, 317],
        related_formulas=["四逆汤", "通脉四逆汤", "真武汤", "黄连阿胶汤"],
        expansion_keywords=["少阴", "脉微细", "但欲寐", "下利清谷"],
        brief="少阴病为心肾阳衰或阴虚火旺，以脉微细、但欲寐为提纲，分寒化证和热化证。",
    ),
    "厥阴病": ConceptMapping(
        concept="厥阴病",
        defining_clauses=[],  # 伤寒论中厥阴篇无"之为病"提纲条文
        treatment_clauses=[338],
        related_formulas=["乌梅丸", "当归四逆汤"],
        expansion_keywords=["厥阴", "消渴", "气上撞心", "心中疼热", "饥而不欲食", "吐蛔"],
        brief="厥阴病为邪入厥阴、寒热错杂阶段，以消渴、气上撞心、心中疼热为特征，乌梅丸主之。",
    ),

    # ── 证候 ──
    "太阳中风证": ConceptMapping(
        concept="太阳中风证",
        defining_clauses=[12],
        treatment_clauses=[12],
        related_formulas=["桂枝汤"],
        expansion_keywords=["太阳中风", "阳浮而阴弱", "汗自出", "恶风", "桂枝汤主之"],
        brief="太阳中风证为外感风邪所致表虚证，以发热汗出恶风脉浮缓为特征，桂枝汤主之。",
    ),
    "太阳伤寒证": ConceptMapping(
        concept="太阳伤寒证",
        defining_clauses=[3],  # "太阳病，或已发热，或未发热，必恶寒，体痛，呕逆，脉阴阳俱紧者，名为伤寒。"
        treatment_clauses=[35],
        related_formulas=["麻黄汤"],
        expansion_keywords=["太阳伤寒", "无汗", "身疼痛", "脉浮紧", "麻黄汤主之"],
        brief="太阳伤寒证为外感寒邪所致表实证，以恶寒无汗身疼痛脉浮紧为特征，麻黄汤主之。",
    ),
    "蓄水证": ConceptMapping(
        concept="蓄水证",
        defining_clauses=[71],
        treatment_clauses=[71],
        related_formulas=["五苓散"],
        expansion_keywords=["小便不利", "消渴", "烦渴", "水入则吐", "五苓散主之", "脉浮"],
        brief="蓄水证为太阳病邪传膀胱、气化不利、水停下焦，以小便不利、口渴为特征，五苓散主之。",
    ),
    "蓄血证": ConceptMapping(
        concept="蓄血证",
        defining_clauses=[106, 124],
        treatment_clauses=[106, 124],
        related_formulas=["桃核承气汤", "抵当汤"],
        expansion_keywords=["蓄血", "少腹急结", "少腹硬满", "如狂", "发狂", "小便自利", "桃核承气汤", "抵当汤"],
        brief="蓄血证为邪热内传、瘀血结于下焦，以少腹急结/硬满、如狂/发狂、小便自利为特征。",
    ),
    "结胸证": ConceptMapping(
        concept="结胸证",
        defining_clauses=[135, 142],
        treatment_clauses=[135, 152],
        related_formulas=["大陷胸汤", "小陷胸汤"],
        expansion_keywords=["结胸", "心下痛", "按之石硬", "脉沉而紧", "大陷胸汤主之", "小结胸", "按之则痛"],
        brief="结胸证为邪热与痰水结于心下，以心下硬满疼痛为特征，分大结胸（大陷胸汤）和小结胸（小陷胸汤）。",
    ),
    "痞证": ConceptMapping(
        concept="痞证",
        defining_clauses=[149, 166],
        treatment_clauses=[149, 169],
        related_formulas=["半夏泻心汤", "大黄黄连泻心汤", "生姜泻心汤", "甘草泻心汤"],
        expansion_keywords=["心下痞", "按之濡", "但气痞耳", "半夏泻心汤", "泻心汤"],
        brief="痞证为脾胃不和、寒热错杂所致心下痞满而不痛，以半夏泻心汤等泻心汤类主之。",
    ),
    "脏结": ConceptMapping(
        concept="脏结",
        defining_clauses=[142],
        treatment_clauses=[142],
        related_formulas=[],
        expansion_keywords=["脏结", "如结胸状", "饮食如故", "时时下利", "寸脉浮", "关脉小细沉紧"],
        brief="脏结为脏气虚寒结滞，状如结胸而饮食如故、时时下利，属难治之证。",
    ),
    "虚烦证": ConceptMapping(
        concept="虚烦证",
        defining_clauses=[76],
        treatment_clauses=[76],
        related_formulas=["栀子豉汤"],
        expansion_keywords=["虚烦", "不得眠", "反复颠倒", "心中懊憹", "栀子豉汤主之"],
        brief="虚烦证为汗吐下后余热留扰胸膈，以虚烦不得眠、心中懊憹为特征，栀子豉汤主之。",
    ),
    "悬饮": ConceptMapping(
        concept="悬饮",
        defining_clauses=[152],
        treatment_clauses=[152],
        related_formulas=["十枣汤"],
        expansion_keywords=["悬饮", "心下痞硬满", "引胁下痛", "十枣汤主之"],
        brief="悬饮为水饮停聚于胸胁，以心下痞硬满、引胁下痛为特征，十枣汤主之。",
    ),
    "脾约": ConceptMapping(
        concept="脾约",
        defining_clauses=[247],
        treatment_clauses=[247],
        related_formulas=["麻子仁丸"],
        expansion_keywords=["脾约", "大便则硬", "小便数", "趺阳脉浮而涩", "麻子仁丸主之"],
        brief="脾约为胃强脾弱、津液偏渗所致大便硬，以小便数、大便硬为特征，麻子仁丸主之。",
    ),
}


# ── 口语→文言同义词映射 ─────────────────────────────────────
# 解决真实用户说"怕冷"但经典原文写"恶寒"的词汇鸿沟
COLLOQUIAL_TO_CLASSICAL: dict[str, str] = {
    "怕冷": "恶寒",
    "怕风": "恶风",
    "发烧": "发热",
    "烧不退": "潮热",
    "不出汗": "无汗",
    "没汗": "无汗",
    "没出汗": "无汗",
    "不汗出": "无汗",
    "出汗": "汗出",
    "拉肚子": "下利",
    "拉稀": "下利",
    "肚子痛": "腹痛",
    "肚子疼": "腹痛",
    "肚子胀": "腹满",
    "肚子硬": "腹硬",
    "手脚凉": "手足厥逆",
    "手脚冷": "手足厥冷",
    "手冷": "手足冷",
    "脚凉": "足寒",
    "嗓子干": "咽干",
    "嗓子疼": "咽痛",
    "嘴苦": "口苦",
    "嘴巴苦": "口苦",
    "嘴里苦": "口苦",
    "口干": "口渴",
    "口渴": "大渴",
    "口渴欲饮": "消渴",
    "大汗": "大汗出",
    "出汗多": "汗自出",
    "脉很大": "脉洪大",
    "脉大": "脉大",
    "脉弱": "脉微",
    "脉很细": "脉微细",
    "心口堵": "心下痞",
    "心口胀": "心下满",
    "心口痛": "心下痛",
    "心口硬": "心下硬",
    "胸口闷": "胸胁苦满",
    "两胁胀": "胸胁苦满",
    "想吐": "欲呕",
    "干呕": "干呕",
    "恶心": "喜呕",
    "不想吃": "不欲食",
    "吃不下": "食不下",
    "没精神": "但欲寐",
    "犯困": "但欲寐",
    "烦": "心烦",
    "烦躁": "烦躁",
    "身上疼": "身疼痛",
    "骨头疼": "骨节疼痛",
    "脖子僵": "项强",
    "头痛": "头痛",
    "眼睛花": "目眩",
    "感冒": "伤寒",
    # ── 高频口语补充（PRD v3.0 辨证输入理解） ──
    "流鼻涕": "鼻流清涕",
    "流清鼻涕": "鼻流清涕",
    "鼻塞": "鼻塞",
    "打喷嚏": "喷嚏",
    "咳嗽": "咳嗽",
    "咳痰": "咳痰",
    "有痰": "咳痰",
    "嗓子痛": "咽痛",
    "咽喉痛": "咽痛",
    "喉咙痛": "咽痛",
    "嗓子哑": "音哑",
    "鼻血": "鼻衄",
    "流鼻血": "鼻衄",
    "头晕": "头眩",
    "晕": "目眩",
    "没力气": "神疲",
    "乏力": "神疲",
    "累": "神疲",
    "没胃口": "不欲食",
    "不想吃饭": "食不下",
    "胃胀": "心下满",
    "胃痛": "心下痛",
    "反酸": "吞酸",
    "打嗝": "噫气",
    "口臭": "口臭",
    "嘴干": "口干",
    "嘴唇干": "唇干",
    "眼睛干": "目干",
    "眼屎多": "目眵多",
    "耳鸣": "耳鸣",
    "听力下降": "耳聋",
    "失眠": "不寐",
    "睡不着": "不寐",
    "睡不好": "不寐",
    "多梦": "多梦",
    "盗汗": "盗汗",
    "手脚心热": "手足心热",
    "腰酸": "腰酸",
    "腰痛": "腰痛",
    "腰疼": "腰痛",
    "膝盖冷": "膝冷",
    "腿肿": "足肿",
    "水肿": "浮肿",
    "尿频": "小便频数",
    "尿少": "小便不利",
    "尿黄": "小便黄",
    "大便干": "大便硬",
    "大便稀": "下利",
    "大便不成形": "便溏",
    "便血": "便血",
    "月经不调": "月经不调",
    "痛经": "经行腹痛",
    "忽冷忽热": "往来寒热",
    "一阵冷一阵热": "往来寒热",
    "心慌": "心悸",
    "胸闷": "胸闷",
    "气短": "短气",
    "出虚汗": "汗自出",
    "冷汗": "冷汗出",
}

# ── 症状→证候映射 ──────────────────────────────────────────
# 用户描述症状而非证候名时，通过症状组合推断证候
# key = 症状组合的 frozenset，value = ConceptMapping 概念名
SYMPTOM_TO_CONCEPT: list[tuple[list[str], str]] = [
    # 太阳伤寒表实证
    (["头痛", "无汗", "恶寒"], "太阳伤寒证"),
    (["头痛", "无汗", "身疼痛"], "太阳伤寒证"),
    (["恶寒", "无汗", "身疼"], "太阳伤寒证"),
    (["不出汗", "怕冷", "身疼"], "太阳伤寒证"),
    (["感冒", "不出汗"], "太阳伤寒证"),
    (["身上疼", "怕冷", "没汗"], "太阳伤寒证"),
    (["伤寒", "无汗"], "太阳伤寒证"),
    (["恶寒", "无汗"], "太阳伤寒证"),
    # 太阳中风表虚证
    (["发热", "汗出", "恶风"], "太阳中风证"),
    (["发烧", "出汗", "怕风"], "太阳中风证"),
    (["汗出", "恶风", "脉浮"], "太阳中风证"),
    # 少阳病
    (["口苦", "咽干", "目眩"], "少阳病"),
    (["口苦", "咽干"], "少阳病"),
    (["嘴苦", "嗓子干"], "少阳病"),
    (["往来寒热", "胸胁苦满"], "少阳病"),
    (["口苦", "目眩"], "少阳病"),
    # 阳明经证
    (["大热", "大汗", "大渴"], "阳明病"),
    (["发热", "汗出", "口渴", "脉洪大"], "阳明病"),
    (["发烧", "口渴", "出大汗", "脉很大"], "阳明病"),
    (["身热", "汗自出", "不恶寒"], "阳明病"),
    # 阳明腑证
    (["潮热", "腹满", "便秘"], "阳明病"),
    (["腹满", "大便硬", "潮热"], "阳明病"),
    # 少阴寒化证
    (["下利", "手足厥逆", "脉微"], "少阴病"),
    (["下利清谷", "手足厥逆"], "少阴病"),
    (["拉肚子", "手脚凉"], "少阴病"),
    (["腹痛", "下利", "四肢沉重"], "少阴病"),
    # 太阴病
    (["腹满", "吐", "食不下", "自利"], "太阴病"),
    (["肚子胀", "吃不下", "拉肚子"], "太阴病"),
    # 痞证
    (["心下痞", "满而不痛"], "痞证"),
    (["心下痞满"], "痞证"),
    (["心口堵"], "痞证"),
    # 结胸证
    (["心下痛", "按之石硬"], "结胸证"),
    (["心下硬满", "疼痛"], "结胸证"),
    # 蓄水证
    (["小便不利", "口渴", "水入则吐"], "蓄水证"),
    (["小便不利", "消渴"], "蓄水证"),
    # 蓄血证
    (["少腹急结", "如狂"], "蓄血证"),
    (["少腹硬满", "发狂"], "蓄血证"),
]

# ── 超范围关键词 ────────────────────────────────────────────
# 这些现代疾病名/闲聊主题不在伤寒论知识范围内
OUT_OF_SCOPE_KEYWORDS: list[str] = [
    "高血压", "糖尿病", "癌症", "肿瘤", "冠心病", "心脏病",
    "失眠", "抑郁症", "焦虑", "痛风", "风湿", "类风湿",
    "颈椎病", "腰椎间盘", "骨质增生", "骨质疏松",
    "肝炎", "肝硬化", "肾炎", "肾虚", "前列腺",
    "湿疹", "荨麻疹", "银屑病", "白癜风",
    "近视", "白内障", "青光眼",
    "甲状腺", "乳腺", "子宫", "卵巢",
    "减肥", "美容", "祛痘", "祛斑",
    "肺癌", "胃癌", "肝癌", "肠癌",
    "艾滋病", "新冠", "新冠肺炎",
    # 闲聊/无关话题（评测 Q10 科幻电影暴露）
    "电影", "电视剧", "音乐", "游戏", "股票", "基金", "足球", "篮球",
    "编程", "代码", "美食", "旅游", "天气", "新闻", "明星",
]


class ConceptMapper:
    """概念映射器

    将现代中医概念查询映射到经典条文，解决语义鸿沟问题。
    P2.2: 增加口语→文言同义词扩展和症状→证候推断。

    用法：
        mapper = ConceptMapper()
        result = mapper.lookup("什么是阳明病？")
        if result:
            print(result.all_clauses)  # [195, 223, 208]
        # 口语扩展
        expanded = mapper.expand_colloquial("头痛发烧怕冷不出汗")
        # 症状推断
        result = mapper.lookup_by_symptoms(["口苦", "咽干", "目眩"])
    """

    def __init__(self) -> None:
        self._map = CONCEPT_MAP.copy()
        self._colloquial = COLLOQUIAL_TO_CLASSICAL.copy()
        self._symptom_map = SYMPTOM_TO_CONCEPT.copy()
        self._out_of_scope = set(OUT_OF_SCOPE_KEYWORDS)

    def lookup(self, query: str) -> ConceptMapping | None:
        """查询概念映射

        Args:
            query: 用户查询文本

        Returns:
            匹配到的 ConceptMapping，未匹配返回 None
        """
        # 去除问题词，提取核心概念
        core = query
        question_words = [
            "什么是", "解释一下", "解释", "是什么", "是什么呢",
            "请问", "的", "？", "?", "证", "病",
        ]
        for word in question_words:
            core = core.replace(word, "")
        core = core.strip()

        # 精确匹配
        for concept, mapping in self._map.items():
            if concept == core or concept == query:
                return mapping

        # 包含匹配（查询包含概念名）
        for concept, mapping in self._map.items():
            if concept in query:
                return mapping

        # 反向匹配（概念名是查询的子串）
        for concept, mapping in self._map.items():
            # 去掉"证"/"病"后比较
            stripped = concept.rstrip("证病")
            if stripped and stripped in core and len(stripped) >= 2:
                return mapping

        return None

    def expand_query(self, query: str) -> str:
        """扩展查询，加入文言文关键词

        用于 BM25 fallback：将现代概念查询扩展为文言文关键词，
        提高关键词匹配的命中率。

        Args:
            query: 原始查询

        Returns:
            扩展后的查询字符串
        """
        mapping = self.lookup(query)
        if mapping and mapping.expansion_keywords:
            # 原始查询 + 扩展关键词
            expanded = query
            for kw in mapping.expansion_keywords:
                if kw not in expanded:
                    expanded += " " + kw
            return expanded
        return query

    @property
    def concepts(self) -> list[str]:
        """所有已映射概念"""
        return list(self._map.keys())

    def expand_colloquial(self, query: str) -> str:
        """口语→文言同义词扩展

        将查询中的口语词汇替换/补充为文言文对应词，
        解决 BM25 关键词匹配的词汇鸿沟。

        "头痛发烧怕冷不出汗" → "头痛 发热 恶寒 无汗"

        Args:
            query: 原始查询

        Returns:
            扩展后的查询（原始词 + 文言同义词）
        """
        expanded = query
        for colloquial, classical in self._colloquial.items():
            if colloquial in query:
                if classical not in expanded:
                    expanded += " " + classical
        return expanded

    def lookup_by_symptoms(self, symptoms: list[str]) -> ConceptMapping | None:
        """通过症状组合推断证候

        当用户描述症状而非证候名时，通过症状匹配推断最可能的证候。

        Args:
            symptoms: 症状列表（如 ["口苦", "咽干", "目眩"]）

        Returns:
            匹配到的 ConceptMapping，未匹配返回 None
        """
        symptom_set = set(symptoms)
        best_match = None
        best_score = 0

        for required_symptoms, concept_name in self._symptom_map:
            required_set = set(required_symptoms)
            overlap = len(symptom_set & required_set)
            if overlap == 0:
                continue
            # 匹配度 = 匹配症状数 / 需要症状数
            score = overlap / len(required_set)
            if score > best_score and score >= 0.5:
                best_score = score
                best_match = concept_name

        if best_match:
            return self._map.get(best_match)
        return None

    def extract_symptoms(self, query: str) -> list[str]:
        """从查询中提取已知症状词

        扫描查询中的所有口语和文言症状词。

        Args:
            query: 用户查询

        Returns:
            提取到的症状列表（文言文形式）
        """
        symptoms: list[str] = []
        # 检查文言症状词（排除否定修饰："不出汗"不应提取"汗出"）
        classical_symptoms = [
            "头痛", "发热", "恶寒", "恶风", "无汗", "汗出", "口苦",
            "咽干", "目眩", "往来寒热", "胸胁苦满", "心下痞", "心下满",
            "心下痛", "腹痛", "腹满", "下利", "手足厥逆", "脉浮",
            "脉浮紧", "脉浮缓", "脉洪大", "脉微", "脉微细", "但欲寐",
            "烦躁", "口渴", "消渴", "大汗", "潮热", "小便不利",
            "不欲食", "食不下", "干呕", "喜呕", "项强", "身疼痛",
            "骨节疼痛", "头项强痛",
            # 辨证投票表补充短术语（PRD v3.0）
            "大热", "大渴", "便秘", "谵语", "自利", "四肢厥逆",
            "心中烦", "不得卧", "如狂", "发狂", "小便数", "虚烦",
            "鼻鸣干呕", "默默不欲饮食", "时腹自痛", "喜温喜按",
        ]
        for s in classical_symptoms:
            if s in query and not self._is_negated(query, s):
                symptoms.append(s)

        # 检查口语症状词，转成文言（同样排除否定修饰）
        for colloquial, classical in self._colloquial.items():
            if colloquial in query and not self._is_negated(query, colloquial):
                if classical not in symptoms:
                    symptoms.append(classical)

        return symptoms

    @staticmethod
    def _is_negated(query: str, term: str) -> bool:
        """检查 term 在 query 中的匹配是否被否定词修饰

        "不出汗" 中的 "出汗"/"汗出" 都被 "不" 否定，不应提取。
        """
        idx = query.find(term)
        while idx != -1:
            prev = query[idx - 1] if idx > 0 else ""
            if prev in ("不", "没", "未", "别", "无"):
                return True
            idx = query.find(term, idx + 1)
        return False

    def is_out_of_scope(self, query: str) -> bool:
        """检查查询是否超出伤寒论知识范围

        Args:
            query: 用户查询

        Returns:
            True 如果查询超出系统知识范围
        """
        for kw in self._out_of_scope:
            if kw in query:
                return True
        return False
