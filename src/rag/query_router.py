"""查询路由器 — 识别查询类型并提取关键实体

支持 5 种查询类型：
1. clause_id: "伤寒论第1条原文是什么？" → 直接 ID 查询
2. formula:   "桂枝汤的组成是什么？" → 方剂名提取 + 倒排索引
3. herb:      "含有桂枝的方剂有哪些？" → 药材名提取 + 倒排索引
4. comparison: "桂枝汤和麻黄汤的区别" → 双实体提取 + 多路检索
5. semantic:  "什么是太阳中风证？" → 查询改写 + 混合检索
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from src.data.formulas_db import FORMULAS, get_formula_dict, get_herb_formulas


class QueryType(str, Enum):
    """查询类型"""
    CLAUSE_ID = "clause_id"
    FORMULA = "formula"
    HERB = "herb"
    COMPARISON = "comparison"
    SEMANTIC = "semantic"


@dataclass
class RouteResult:
    """路由结果"""
    query_type: QueryType = QueryType.SEMANTIC
    original_query: str = ""
    # CLAUSE_ID
    clause_id: Optional[int] = None
    # FORMULA
    formula_name: Optional[str] = None
    # HERB
    herbs: list[str] = field(default_factory=list)
    # COMPARISON
    entities: list[str] = field(default_factory=list)
    # SEMANTIC (改写后的查询)
    rewritten_query: Optional[str] = None

    def __repr__(self) -> str:
        parts = [f"type={self.query_type.value}"]
        if self.clause_id is not None:
            parts.append(f"clause_id={self.clause_id}")
        if self.formula_name:
            parts.append(f"formula={self.formula_name}")
        if self.herbs:
            parts.append(f"herbs={self.herbs}")
        if self.entities:
            parts.append(f"entities={self.entities}")
        if self.rewritten_query:
            parts.append(f"rewritten={self.rewritten_query}")
        return f"Route({', '.join(parts)})"


class QueryRouter:
    """查询路由器

    根据查询文本的模式匹配，判断查询类型并提取关键实体。
    优先级：clause_id > formula > herb > comparison > semantic

    用法：
        router = QueryRouter()
        route = router.route("伤寒论第1条原文是什么？")
        # route.query_type == QueryType.CLAUSE_ID
        # route.clause_id == 1
    """

    # 条文号正则：匹配 "第X条" 或 "第XXX条"
    _CLAUSE_ID_PATTERN = re.compile(r"第(\d+)条")

    # 问题模式正则（用于语义改写时去除）
    _QUESTION_PATTERNS = [
        r"什么是",
        r"解释一下",
        r"解释",
        r"的组成是什么",
        r"的组成",
        r"原文是什么",
        r"原文",
        r"有哪些",
        r"有什么",
        r"是什么",
        r"区别是什么",
        r"区别",
        r"在主治上的",
        r"在功能上的",
    ]

    def __init__(self) -> None:
        self._formula_dict = get_formula_dict()
        self._herb_formulas = get_herb_formulas()
        # 按名称长度降序排列，优先匹配长名（如"桂枝加葛根汤" > "桂枝汤"）
        self._formula_names = sorted(
            self._formula_dict.keys(), key=len, reverse=True
        )
        self._herb_names = sorted(
            self._herb_formulas.keys(), key=len, reverse=True
        )

    def route(self, query: str) -> RouteResult:
        """路由查询

        Args:
            query: 用户查询文本

        Returns:
            RouteResult 包含查询类型和提取的实体
        """
        result = RouteResult(original_query=query)

        # 1. 条文号查询
        clause_id = self._detect_clause_id(query)
        if clause_id is not None:
            result.query_type = QueryType.CLAUSE_ID
            result.clause_id = clause_id
            return result

        # 2. 对比查询（优先于方剂，避免"X和Y的区别"被误判为方剂查询）
        entities = self._detect_comparison(query)
        if entities:
            result.query_type = QueryType.COMPARISON
            result.entities = entities
            return result

        # 3. 方剂组成查询
        formula = self._detect_formula(query)
        if formula:
            result.query_type = QueryType.FORMULA
            result.formula_name = formula
            return result

        # 4. 药材关联查询
        herbs = self._detect_herbs(query)
        if herbs:
            result.query_type = QueryType.HERB
            result.herbs = herbs
            return result

        # 5. 语义查询（改写后走混合检索）
        result.query_type = QueryType.SEMANTIC
        result.rewritten_query = self._rewrite_query(query)
        return result

    def _detect_clause_id(self, query: str) -> Optional[int]:
        """检测条文号查询，提取条文编号"""
        match = self._CLAUSE_ID_PATTERN.search(query)
        if match:
            return int(match.group(1))
        return None

    def _detect_formula(self, query: str) -> Optional[str]:
        """检测方剂查询，提取方剂名

        匹配条件：查询中包含已知方剂名，且查询模式符合"组成/方剂"类问题
        """
        # 先检查是否是组成类查询
        formula_indicators = ["组成", "方剂", "配方", "药物", "有哪些药"]
        is_formula_query = any(ind in query for ind in formula_indicators)

        # 也匹配 "X汤" 直接出现
        if not is_formula_query:
            # 如果查询直接以方剂名开头，也算
            for name in self._formula_names:
                if query.startswith(name):
                    is_formula_query = True
                    break

        if not is_formula_query:
            return None

        # 在查询中查找已知方剂名
        for name in self._formula_names:
            if name in query:
                return name

        return None

    def _detect_herbs(self, query: str) -> list[str]:
        """检测药材查询，提取药材名

        匹配条件：查询中包含 "含有" / "含" / "用" 等关键词 + 已知药材名
        """
        herb_indicators = ["含有", "含", "用到", "用的", "有.*药"]
        is_herb_query = any(ind in query for ind in herb_indicators)

        if not is_herb_query:
            return []

        found_herbs = []
        for name in self._herb_names:
            if name in query and name not in found_herbs:
                found_herbs.append(name)

        return found_herbs

    def _detect_comparison(self, query: str) -> list[str]:
        """检测对比查询，提取两个实体

        匹配模式："X和Y的区别" / "X与Y有什么不同" / "X和Y"
        """
        comparison_indicators = ["区别", "不同", "差异", "对比", "比较"]
        is_comparison = any(ind in query for ind in comparison_indicators)

        if not is_comparison:
            return []

        # 尝试用已知方剂名分割
        # 先找所有在查询中出现的方剂名
        found_formulas = []
        for name in self._formula_names:
            if name in query:
                found_formulas.append(name)

        if len(found_formulas) >= 2:
            return found_formulas[:2]

        # 如果没找到足够的方剂名，尝试用 "和"/"与"/"跟" 分割
        split_patterns = [r"和", r"与", r"跟", r"以及"]
        for pattern in split_patterns:
            parts = re.split(pattern, query)
            if len(parts) >= 2:
                # 从每部分提取可能的实体
                entities = []
                for part in parts:
                    part = part.strip()
                    # 去除问题词
                    for q_pattern in self._QUESTION_PATTERNS:
                        part = re.sub(q_pattern, "", part)
                    part = part.strip("？?，,的")
                    if part and len(part) >= 2:
                        entities.append(part)
                if len(entities) >= 2:
                    return entities[:2]

        return []

    def _rewrite_query(self, query: str) -> str:
        """改写语义查询，去除问题词，保留核心概念

        例如：
        - "什么是太阳中风证？" → "太阳中风"
        - "解释一下太阳蓄水证" → "太阳蓄水"
        - "什么是阳明病？" → "阳明病"
        """
        rewritten = query

        # 去除问题模式
        for pattern in self._QUESTION_PATTERNS:
            rewritten = re.sub(pattern, "", rewritten)

        # 去除标点和多余空格
        rewritten = rewritten.strip("？?，,。.的！!")

        # 如果改写后为空，返回原文
        if not rewritten:
            return query

        # 去掉尾部的 "证" 和 "病" 时保留核心词用于搜索
        # 但不删除，因为 "太阳中风" 比 "太阳中风证" 更容易匹配原文
        # 实际上保留 "证" 和 "病" 也没问题，BM25 会处理

        return rewritten
