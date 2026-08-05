"""知识图谱构建模块

从伤寒论方剂数据构建 NetworkX 知识图谱，支持多跳查询和 JSON 持久化。
基于 Day 2 验证脚本 data/kg_test.py 的成熟逻辑重构，扩展至 113 方剂。
"""
import json
import networkx as nx

from src.data.formulas_data import (
    FORMULAS,
    SYNDROME_FORMULA,
    FORMULA_DERIVATIONS,
    CLAUSE_FORMULA,
)


def build_graph() -> nx.DiGraph:
    """构建伤寒论知识图谱"""
    G = nx.DiGraph()

    # 1. 添加方剂节点 + 药材节点 + CONTAINS 边
    seen_formulas = set()
    for formula, herbs in FORMULAS.items():
        if formula in seen_formulas:
            continue
        seen_formulas.add(formula)
        G.add_node(formula, type="formula", source="伤寒论")
        for herb in herbs:
            if herb not in G:
                G.add_node(herb, type="herb")
            G.add_edge(formula, herb, relation="contains")

    # 2. 添加证型节点 + TREATS 边
    for syndrome, formulas in SYNDROME_FORMULA.items():
        if syndrome not in G:
            G.add_node(syndrome, type="syndrome")
        for formula in formulas:
            if formula in G:
                G.add_edge(syndrome, formula, relation="treats")

    # 3. 添加方剂加减关系 DERIVED_FROM 边
    for derived, base, change in FORMULA_DERIVATIONS:
        if derived not in G:
            G.add_node(derived, type="formula", source="伤寒论")
        if base not in G:
            G.add_node(base, type="formula", source="伤寒论")
        G.add_edge(derived, base, relation="derived_from", change=change)

    # 4. 添加条文节点 + MENTIONS 边
    for clause_num, formula in CLAUSE_FORMULA:
        clause_id = f"第{clause_num}条"
        if clause_id not in G:
            G.add_node(clause_id, type="text", number=clause_num)
        if formula in G:
            G.add_edge(clause_id, formula, relation="mentions")

    return G


# ========== 查询函数 ==========

def query_herbs_in_formula(G: nx.DiGraph, formula_name: str) -> list[str]:
    """查询某方剂包含哪些药材"""
    if formula_name not in G:
        return []
    herbs = [n for n in G.successors(formula_name)
             if G.nodes[n].get("type") == "herb"]
    return sorted(herbs)


def query_formulas_containing_herb(G: nx.DiGraph, herb_name: str) -> list[str]:
    """查询哪些方剂含有某药材"""
    if herb_name not in G:
        return []
    formulas = [n for n in G.predecessors(herb_name)
                if G.nodes[n].get("type") == "formula"]
    return sorted(formulas)


def query_common_herbs(G: nx.DiGraph, f1: str, f2: str) -> list[str]:
    """查询两个方剂的共同药材"""
    herbs1 = set(query_herbs_in_formula(G, f1))
    herbs2 = set(query_herbs_in_formula(G, f2))
    return sorted(herbs1 & herbs2)


def query_herb_difference(G: nx.DiGraph, f1: str, f2: str) -> tuple[list[str], list[str]]:
    """查询两个方剂的药材差异"""
    herbs1 = set(query_herbs_in_formula(G, f1))
    herbs2 = set(query_herbs_in_formula(G, f2))
    return sorted(herbs1 - herbs2), sorted(herbs2 - herbs1)


def query_formulas_with_both_herbs(G: nx.DiGraph, h1: str, h2: str) -> list[str]:
    """查询同时含两种药材的方剂"""
    f1 = set(query_formulas_containing_herb(G, h1))
    f2 = set(query_formulas_containing_herb(G, h2))
    return sorted(f1 & f2)


def query_formulas_for_syndrome(G: nx.DiGraph, syndrome: str) -> list[str]:
    """查询某证型对应的方剂"""
    if syndrome not in G:
        return []
    formulas = [n for n in G.successors(syndrome)
                if G.nodes[n].get("type") == "formula"]
    return sorted(formulas)


def query_derivations(G: nx.DiGraph, formula: str) -> list[tuple[str, str]]:
    """查询某方剂的加减变化"""
    if formula not in G:
        return []
    result = []
    for n in G.predecessors(formula):
        if G.nodes[n].get("type") == "formula":
            edge = G.edges[n, formula]
            if edge.get("relation") == "derived_from":
                result.append((n, edge.get("change", "")))
    return result


# ========== 持久化 ==========

def save_graph(G: nx.DiGraph, path: str) -> None:
    """保存图谱为 JSON"""
    data = nx.node_link_data(G)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_graph(path: str) -> nx.DiGraph:
    """从 JSON 加载图谱"""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return nx.node_link_graph(data)


def graph_stats(G: nx.DiGraph) -> dict:
    """获取图谱统计信息"""
    types = {}
    for _, data in G.nodes(data=True):
        t = data.get("type", "unknown")
        types[t] = types.get(t, 0) + 1
    return {
        "total_nodes": G.number_of_nodes(),
        "total_edges": G.number_of_edges(),
        "formula_count": types.get("formula", 0),
        "herb_count": types.get("herb", 0),
        "syndrome_count": types.get("syndrome", 0),
        "text_count": types.get("text", 0),
    }
