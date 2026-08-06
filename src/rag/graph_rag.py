"""GraphRAG — 基于 NetworkX 的知识图谱检索

构建 Formula-Herb-Syndrome-Clause 四类节点的知识图谱，
支持多跳关系查询，补充向量检索无法处理的结构化关系。

节点类型：
- formula: 方剂节点 (name, herbs, clause_id, syndrome, brief)
- herb: 药材节点 (name)
- syndrome: 证候节点 (name)
- clause: 条文节点 (clause_id, text, chapter)

边类型：
- CONTAINS: formula → herb (方剂含有药材)
- TREATS: formula → syndrome (方剂主治证候)
- APPEARS_IN: formula → clause (方剂出处条文)

查询模式：
1. syndrome → formulas → clauses (证候查询)
2. herb → formulas → clauses (药材查询，补充倒排索引)
3. formula1 vs formula2 → 组成差异 + 主治差异 (对比查询增强)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import networkx as nx

from src.data.formulas_db import FORMULAS, FormulaInfo, get_formula_dict, get_herb_formulas


@dataclass
class GraphQueryResult:
    """图谱查询结果"""
    clause_ids: list[int]
    formula_names: list[str]
    metadata: dict[str, Any]  # 额外信息（如组成差异、主治差异等）


class TCMKnowledgeGraph:
    """中医知识图谱

    从 formulas_db 构建 NetworkX 图谱，支持多跳查询。

    用法：
        kg = TCMKnowledgeGraph()
        result = kg.query_by_syndrome("少阳证")
        print(result.clause_ids)  # [96]
        print(result.formula_names)  # ["小柴胡汤"]
    """

    def __init__(self) -> None:
        self._graph = nx.DiGraph()
        self._formula_dict = get_formula_dict()
        self._herb_formulas = get_herb_formulas()
        self._build()

    def _build(self) -> None:
        """从 formulas_db 构建知识图谱"""
        # 添加方剂节点
        for f in FORMULAS:
            self._graph.add_node(
                f"formula:{f.name}",
                type="formula",
                name=f.name,
                herbs=f.herbs,
                clause_id=f.clause_id,
                syndrome=f.syndrome,
                brief=f.brief,
            )

            # 添加证候节点 + TREATS 边
            if f.syndrome:
                syndrome_node = f"syndrome:{f.syndrome}"
                self._graph.add_node(syndrome_node, type="syndrome", name=f.syndrome)
                self._graph.add_edge(f"formula:{f.name}", syndrome_node, relation="TREATS")

            # 添加条文节点 + APPEARS_IN 边
            clause_node = f"clause:{f.clause_id}"
            self._graph.add_node(clause_node, type="clause", clause_id=f.clause_id)
            self._graph.add_edge(f"formula:{f.name}", clause_node, relation="APPEARS_IN")

            # 添加药材节点 + CONTAINS 边
            for herb in f.herbs:
                herb_node = f"herb:{herb}"
                if herb_node not in self._graph:
                    self._graph.add_node(herb_node, type="herb", name=herb)
                self._graph.add_edge(f"formula:{f.name}", herb_node, relation="CONTAINS")

    def query_by_syndrome(self, syndrome: str) -> GraphQueryResult:
        """通过证候查询：证候 → 方剂 → 条文

        Args:
            syndrome: 证候名称（如"少阳证"、"蓄水证"）

        Returns:
            匹配的方剂和条文列表
        """
        # 模糊匹配证候节点
        syndrome_nodes = [
            n for n, d in self._graph.nodes(data=True)
            if d.get("type") == "syndrome" and syndrome in d.get("name", "")
        ]

        clause_ids: list[int] = []
        formula_names: list[str] = []
        seen_clauses: set[int] = set()

        for sn in syndrome_nodes:
            # 找到治疗该证候的方剂
            for formula_node in self._graph.predecessors(sn):
                data = self._graph.nodes[formula_node]
                if data.get("type") != "formula":
                    continue
                fname = data["name"]
                if fname not in formula_names:
                    formula_names.append(fname)

                cid = data.get("clause_id")
                if cid and cid not in seen_clause:
                    clause_ids.append(cid)
                    seen_clause.add(cid)

        return GraphQueryResult(
            clause_ids=clause_ids,
            formula_names=formula_names,
            metadata={"syndrome": syndrome},
        )

    def query_by_herb(self, herb: str) -> GraphQueryResult:
        """通过药材查询：药材 → 方剂 → 条文

        Args:
            herb: 药材名称

        Returns:
            含该药材的方剂和条文列表
        """
        herb_node = f"herb:{herb}"
        if herb_node not in self._graph:
            return GraphQueryResult(clause_ids=[], formula_names=[], metadata={})

        clause_ids: list[int] = []
        formula_names: list[str] = []
        seen_clause: set[int] = set()

        # 找到含有该药材的方剂
        for formula_node in self._graph.predecessors(herb_node):
            data = self._graph.nodes[formula_node]
            if data.get("type") != "formula":
                continue
            fname = data["name"]
            formula_names.append(fname)

            cid = data.get("clause_id")
            if cid and cid not in seen_clause:
                clause_ids.append(cid)
                seen_clause.add(cid)

        return GraphQueryResult(
            clause_ids=clause_ids,
            formula_names=formula_names,
            metadata={"herb": herb, "count": len(formula_names)},
        )

    def query_by_herbs_intersection(self, herbs: list[str]) -> GraphQueryResult:
        """多药材交集查询：同时含所有药材的方剂

        Args:
            herbs: 药材名列表

        Returns:
            同时含所有药材的方剂和条文列表
        """
        if not herbs:
            return GraphQueryResult(clause_ids=[], formula_names=[], metadata={})

        # 收集每个药材对应的方剂集合
        herb_formula_sets: list[set[str]] = []
        for herb in herbs:
            result = self.query_by_herb(herb)
            herb_formula_sets.append(set(result.formula_names))

        if not herb_formula_sets or not all(herb_formula_sets):
            return GraphQueryResult(clause_ids=[], formula_names=[], metadata={"herbs": herbs})

        # 取交集
        intersection = herb_formula_sets[0]
        for s in herb_formula_sets[1:]:
            intersection &= s

        clause_ids: list[int] = []
        seen: set[int] = set()
        for fname in sorted(intersection):
            f = self._formula_dict.get(fname)
            if f and f.clause_id not in seen:
                clause_ids.append(f.clause_id)
                seen.add(f.clause_id)

        return GraphQueryResult(
            clause_ids=clause_ids,
            formula_names=sorted(intersection),
            metadata={"herbs": herbs, "intersection_count": len(intersection)},
        )

    def compare_formulas(self, name1: str, name2: str) -> dict[str, Any]:
        """方剂对比：返回组成差异和主治差异

        Args:
            name1: 方剂1名称
            name2: 方剂2名称

        Returns:
            对比信息字典
        """
        f1 = self._formula_dict.get(name1)
        f2 = self._formula_dict.get(name2)

        if not f1 or not f2:
            return {"error": f"Formula not found: {name1 if not f1 else name2}"}

        herbs1 = set(f1.herbs)
        herbs2 = set(f2.herbs)

        return {
            "formula1": {
                "name": f1.name,
                "herbs": f1.herbs,
                "clause_id": f1.clause_id,
                "syndrome": f1.syndrome,
                "brief": f1.brief,
            },
            "formula2": {
                "name": f2.name,
                "herbs": f2.herbs,
                "clause_id": f2.clause_id,
                "syndrome": f2.syndrome,
                "brief": f2.brief,
            },
            "herbs_only_in_1": sorted(herbs1 - herbs2),
            "herbs_only_in_2": sorted(herbs2 - herbs1),
            "common_herbs": sorted(herbs1 & herbs2),
            "clause_ids": [f1.clause_id, f2.clause_id],
        }

    def get_formula_info(self, name: str) -> FormulaInfo | None:
        """获取方剂信息"""
        return self._formula_dict.get(name)

    def get_all_herbs(self) -> list[str]:
        """获取所有药材列表"""
        return sorted(self._herb_formulas.keys())

    @property
    def node_count(self) -> int:
        return self._graph.number_of_nodes()

    @property
    def edge_count(self) -> int:
        return self._graph.number_of_edges()

    @property
    def formula_count(self) -> int:
        return sum(1 for _, d in self._graph.nodes(data=True) if d.get("type") == "formula")

    @property
    def herb_count(self) -> int:
        return sum(1 for _, d in self._graph.nodes(data=True) if d.get("type") == "herb")

    @property
    def syndrome_count(self) -> int:
        return sum(1 for _, d in self._graph.nodes(data=True) if d.get("type") == "syndrome")
