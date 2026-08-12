"""
Day 2 可行性验证：迷你中医知识图谱构建 + 查询测试

从伤寒论前 20 个方剂构建知识图谱：
- 实体：方剂(20) + 药材(~25) = ~45 节点
- 关系：方剂 → 包含 → 药材
- 查询测试：
  1. 桂枝汤包含哪些药？
  2. 麻黄汤和桂枝汤的共同药物是什么？
  3. 哪些方剂含有桂枝？
  4. 哪些方剂同时含有麻黄和桂枝？
  5. 桂枝汤和桂枝加附子汤的药物差异？
"""

import networkx as nx
import time
import json

# ============================================================
# 1. 知识图谱数据（从伤寒论原文提取）
# ============================================================

# 方剂 → 药材组成（简体）
FORMULAS = {
    "桂枝汤": ["桂枝", "芍药", "甘草", "生姜", "大枣"],
    "桂枝加葛根汤": ["葛根", "麻黄", "芍药", "生姜", "甘草", "大枣", "桂枝"],
    "桂枝加附子汤": ["桂枝", "芍药", "甘草", "生姜", "大枣", "附子"],
    "桂枝去芍药汤": ["桂枝", "甘草", "生姜", "大枣"],
    "桂枝去芍药加附子汤": ["桂枝", "甘草", "生姜", "大枣", "附子"],
    "桂枝麻黄各半汤": ["桂枝", "芍药", "生姜", "甘草", "麻黄", "大枣", "杏仁"],
    "桂枝二麻黄一汤": ["桂枝", "芍药", "麻黄", "生姜", "杏仁", "甘草", "大枣"],
    "白虎加人参汤": ["知母", "石膏", "甘草", "粳米", "人参"],
    "桂枝二越婢一汤": ["桂枝", "芍药", "麻黄", "甘草", "大枣", "生姜", "石膏"],
    "桂枝去桂加茯苓白术汤": ["芍药", "甘草", "生姜", "白术", "茯苓", "大枣"],
    "甘草干姜汤": ["甘草", "干姜"],
    "芍药甘草汤": ["芍药", "甘草"],
    "调胃承气汤": ["大黄", "甘草", "芒硝"],
    "四逆汤": ["甘草", "干姜", "附子"],
    "葛根汤": ["葛根", "麻黄", "桂枝", "生姜", "甘草", "芍药", "大枣"],
    "葛根加半夏汤": ["葛根", "麻黄", "甘草", "芍药", "桂枝", "生姜", "半夏", "大枣"],
    "葛根黄芩黄连汤": ["葛根", "甘草", "黄芩", "黄连"],
    "麻黄汤": ["麻黄", "桂枝", "甘草", "杏仁"],
    "大青龙汤": ["麻黄", "桂枝", "甘草", "杏仁", "生姜", "大枣", "石膏"],
    "小青龙汤": ["麻黄", "芍药", "细辛", "干姜", "甘草", "桂枝", "五味子", "半夏"],
}

# 证型 → 方剂（主治关系）
SYNDROME_FORMULA = {
    "太阳中风": ["桂枝汤"],
    "太阳伤寒": ["麻黄汤"],
    "表虚证": ["桂枝汤", "桂枝加附子汤"],
    "表实证": ["麻黄汤", "大青龙汤"],
    "少阳证": [],  # 小柴胡汤在后面的条文，这里先不列
    "阳明热证": ["白虎加人参汤"],
    "太阴虚寒": ["四逆汤", "甘草干姜汤"],
}


def build_graph():
    """构建知识图谱"""
    G = nx.DiGraph()

    # 添加方剂节点
    for formula in FORMULAS:
        G.add_node(formula, type="formula", source="伤寒论")

    # 添加药材节点 + 包含关系
    all_herbs = set()
    for formula, herbs in FORMULAS.items():
        for herb in herbs:
            if herb not in all_herbs:
                G.add_node(herb, type="herb")
                all_herbs.add(herb)
            G.add_edge(formula, herb, relation="contains")

    # 添加证型节点 + 主治关系
    for syndrome, formulas in SYNDROME_FORMULA.items():
        G.add_node(syndrome, type="syndrome")
        for formula in formulas:
            if formula in G:
                G.add_edge(syndrome, formula, relation="treats")

    return G


def query_herbs_in_formula(G, formula_name):
    """查询：某方剂包含哪些药？"""
    if formula_name not in G:
        return []
    herbs = [n for n in G.successors(formula_name) if G.nodes[n].get("type") == "herb"]
    return sorted(herbs)


def query_formulas_containing_herb(G, herb_name):
    """查询：哪些方剂含有某药？"""
    if herb_name not in G:
        return []
    formulas = [n for n in G.predecessors(herb_name) if G.nodes[n].get("type") == "formula"]
    return sorted(formulas)


def query_common_herbs(G, formula1, formula2):
    """查询：两个方剂的共同药物"""
    herbs1 = set(query_herbs_in_formula(G, formula1))
    herbs2 = set(query_herbs_in_formula(G, formula2))
    return sorted(herbs1 & herbs2)


def query_herb_difference(G, formula1, formula2):
    """查询：两个方剂的药物差异"""
    herbs1 = set(query_herbs_in_formula(G, formula1))
    herbs2 = set(query_herbs_in_formula(G, formula2))
    only_in_1 = sorted(herbs1 - herbs2)
    only_in_2 = sorted(herbs2 - herbs1)
    return only_in_1, only_in_2


def query_formulas_with_both_herbs(G, herb1, herb2):
    """查询：同时含有两种药的方剂"""
    formulas1 = set(query_formulas_containing_herb(G, herb1))
    formulas2 = set(query_formulas_containing_herb(G, herb2))
    return sorted(formulas1 & formulas2)


def run_tests():
    """运行查询测试"""
    G = build_graph()

    print(f"知识图谱构建完成")
    print(f"  节点数: {G.number_of_nodes()}")
    print(f"  边数: {G.number_of_edges()}")
    formula_count = sum(1 for n in G.nodes if G.nodes[n].get("type") == "formula")
    herb_count = sum(1 for n in G.nodes if G.nodes[n].get("type") == "herb")
    syndrome_count = sum(1 for n in G.nodes if G.nodes[n].get("type") == "syndrome")
    print(f"  方剂: {formula_count}, 药材: {herb_count}, 证型: {syndrome_count}")
    print()

    # 估算内存占用
    import sys
    graph_size = sys.getsizeof(G)
    print(f"  图谱对象内存: ~{graph_size / 1024:.1f} KB")
    print()

    results = []

    # 测试 1: 桂枝汤包含哪些药？
    print("=" * 60)
    print("测试 1: 桂枝汤包含哪些药？")
    t0 = time.perf_counter()
    herbs = query_herbs_in_formula(G, "桂枝汤")
    t1 = time.perf_counter()
    print(f"  结果: {herbs}")
    print(f"  耗时: {(t1-t0)*1000:.2f} ms")
    expected = sorted(["桂枝", "芍药", "甘草", "生姜", "大枣"])
    passed = herbs == expected
    print(f"  预期: {expected}")
    print(f"  通过: {'✅' if passed else '❌'}")
    results.append(("桂枝汤包含哪些药", passed, (t1-t0)*1000))

    # 测试 2: 麻黄汤和桂枝汤的共同药物
    print("\n" + "=" * 60)
    print("测试 2: 麻黄汤和桂枝汤的共同药物是什么？")
    t0 = time.perf_counter()
    common = query_common_herbs(G, "麻黄汤", "桂枝汤")
    t1 = time.perf_counter()
    print(f"  结果: {common}")
    print(f"  耗时: {(t1-t0)*1000:.2f} ms")
    expected = sorted(["甘草", "桂枝"])
    passed = common == expected
    print(f"  预期: {expected}")
    print(f"  通过: {'✅' if passed else '❌'}")
    results.append(("麻黄汤和桂枝汤的共同药物", passed, (t1-t0)*1000))

    # 测试 3: 哪些方剂含有桂枝？
    print("\n" + "=" * 60)
    print("测试 3: 哪些方剂含有桂枝？")
    t0 = time.perf_counter()
    formulas = query_formulas_containing_herb(G, "桂枝")
    t1 = time.perf_counter()
    print(f"  结果: {formulas}")
    print(f"  耗时: {(t1-t0)*1000:.2f} ms")
    print(f"  数量: {len(formulas)}")
    passed = len(formulas) >= 10  # 桂枝是高频药
    print(f"  通过: {'✅' if passed else '❌'} (预期≥10)")
    results.append(("哪些方剂含有桂枝", passed, (t1-t0)*1000))

    # 测试 4: 同时含有麻黄和桂枝的方剂
    print("\n" + "=" * 60)
    print("测试 4: 哪些方剂同时含有麻黄和桂枝？")
    t0 = time.perf_counter()
    formulas = query_formulas_with_both_herbs(G, "麻黄", "桂枝")
    t1 = time.perf_counter()
    print(f"  结果: {formulas}")
    print(f"  耗时: {(t1-t0)*1000:.2f} ms")
    expected = sorted(["桂枝二麻黄一汤", "桂枝二越婢一汤", "桂枝麻黄各半汤",
                      "大青龙汤", "小青龙汤", "葛根汤", "葛根加半夏汤",
                      "桂枝加葛根汤", "麻黄汤"])
    passed = set(formulas) == set(expected)
    print(f"  预期: {expected}")
    print(f"  通过: {'✅' if passed else '❌'}")
    results.append(("同时含麻黄和桂枝的方剂", passed, (t1-t0)*1000))

    # 测试 5: 桂枝汤和桂枝加附子汤的药物差异
    print("\n" + "=" * 60)
    print("测试 5: 桂枝汤和桂枝加附子汤的药物差异？")
    t0 = time.perf_counter()
    only_1, only_2 = query_herb_difference(G, "桂枝汤", "桂枝加附子汤")
    t1 = time.perf_counter()
    print(f"  桂枝汤独有: {only_1}")
    print(f"  桂枝加附子汤独有: {only_2}")
    print(f"  耗时: {(t1-t0)*1000:.2f} ms")
    passed = only_1 == [] and only_2 == ["附子"]
    print(f"  预期: 桂枝汤独有=[], 桂枝加附子汤独有=['附子']")
    print(f"  通过: {'✅' if passed else '❌'}")
    results.append(("桂枝汤vs桂枝加附子汤差异", passed, (t1-t0)*1000))

    # 测试 6: 查询含有甘草的方剂数量
    print("\n" + "=" * 60)
    print("测试 6: 哪些方剂含有甘草？（甘草是伤寒论最高频药）")
    t0 = time.perf_counter()
    formulas = query_formulas_containing_herb(G, "甘草")
    t1 = time.perf_counter()
    print(f"  结果: {formulas}")
    print(f"  数量: {len(formulas)}")
    print(f"  耗时: {(t1-t0)*1000:.2f} ms")
    passed = len(formulas) >= 15  # 甘草是最高频
    print(f"  通过: {'✅' if passed else '❌'} (预期≥15)")
    results.append(("含有甘草的方剂", passed, (t1-t0)*1000))

    # 总结
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    passed_count = sum(1 for _, p, _ in results if p)
    total_count = len(results)
    max_time = max(t for _, _, t in results)
    avg_time = sum(t for _, _, t in results) / total_count

    for name, passed, t in results:
        status = "✅" if passed else "❌"
        print(f"  {status} {name} ({t:.2f}ms)")

    print(f"\n  通过率: {passed_count}/{total_count}")
    print(f"  最大耗时: {max_time:.2f} ms")
    print(f"  平均耗时: {avg_time:.2f} ms")
    print(f"  100ms 内: {'✅' if max_time < 100 else '❌'}")

    # 判定
    print(f"\n{'='*60}")
    if passed_count == total_count and max_time < 100:
        print("🎉 Day 2 验证通过！知识图谱方案可行。")
    elif passed_count == total_count:
        print("⚠️ 查询正确但性能不达标，需要优化")
    else:
        print("❌ 有查询失败，需要检查数据")
    print(f"{'='*60}")

    # 保存图谱为 JSON（验证可持久化）
    graph_data = nx.node_link_data(G)
    graph_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "raw", "knowledge_graph.json")
    with open(graph_path, "w", encoding="utf-8") as f:
        json.dump(graph_data, f, ensure_ascii=False, indent=2)
    print(f"\n图谱已保存到: {graph_path}")

    # 验证从 JSON 重新加载
    t0 = time.perf_counter()
    with open(graph_path, "r", encoding="utf-8") as f:
        loaded_data = json.load(f)
    G2 = nx.node_link_graph(loaded_data)
    t1 = time.perf_counter()
    print(f"从 JSON 重新加载: {(t1-t0)*1000:.2f} ms, 节点: {G2.number_of_nodes()}, 边: {G2.number_of_edges()}")


if __name__ == "__main__":
    import os
    print("=" * 60)
    print("Day 2: 迷你中医知识图谱验证")
    print("=" * 60)
    print()
    run_tests()
