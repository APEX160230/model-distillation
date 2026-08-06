"""P2.1 评测脚本 — ConceptMapper + GraphRAG + LLM-as-judge

对比 P0（纯向量+原始模型）、P1（纯向量+微调模型）、P2.0（混合检索+微调模型）、
P2.1（概念映射+GraphRAG+LLM judge）

三项改进：
1. ConceptMapper: 现代 TCM 概念 → 经典条文精确映射，解决 semantic route 语义鸿沟
2. GraphRAG: 知识图谱多跳查询，药材/证候/方剂对比的结构化检索
3. LLM-as-judge: 用 qwen2.5:1.5b 对答案质量打 0-5 分，替代 keyword_accuracy

用法：
    python scripts/run_p2_1_eval.py

产出：
    data/processed/p2_1_report.json          — P2.1 评测报告（含 judge 分数）
    data/processed/p0_p1_p2_p21_comparison.json — 四阶段对比
"""
import os
import sys
import json
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

from src.rag.pipeline import RAGPipeline
from src.rag.generate import Generator
from src.eval.runner import EvalRunner
from src.eval.metrics import EvalResult, generate_report, compute_judge_scores
from src.eval.llm_judge import LLMJudge


def load_report(name: str) -> dict | None:
    """加载历史报告"""
    path = PROJECT_ROOT / "data" / "processed" / name
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def generate_four_way_comparison(
    p0: dict | None, p1: dict | None, p2: dict | None, p21: dict
) -> dict:
    """生成 P0 vs P1 vs P2.0 vs P2.1 四阶段对比"""
    reports = {"p0": p0, "p1": p1, "p2": p2, "p2.1": p21}

    comparison = {
        "stages": {
            "p0": p0["stage"] if p0 else "N/A",
            "p1": p1["stage"] if p1 else "N/A",
            "p2": p2["stage"] if p2 else "N/A",
            "p2.1": p21["stage"],
        },
        "metrics": {},
    }

    # 核心指标
    for metric in ["recall_at_5", "keyword_accuracy"]:
        row = {}
        for stage, report in reports.items():
            row[stage] = report.get(metric, 0) if report else None
        p0_val = row.get("p0", 0) or 0
        p21_val = row.get("p2.1", 0) or 0
        delta = p21_val - p0_val
        row["delta_p0_to_p21"] = round(delta, 4)
        row["improvement_pct"] = round(
            (delta / p0_val * 100) if p0_val > 0 else 0, 1
        )
        comparison["metrics"][metric] = row

    # LLM judge 分数（P2.1 独有）
    if "llm_judge" in p21:
        comparison["metrics"]["llm_judge_mean"] = {
            "p0": None, "p1": None, "p2": None,
            "p2.1": p21["llm_judge"]["mean"],
        }
        comparison["metrics"]["llm_judge_median"] = {
            "p0": None, "p1": None, "p2": None,
            "p2.1": p21["llm_judge"]["median"],
        }

    # 延迟
    for key in ["p50", "p95", "mean"]:
        row = {}
        for stage, report in reports.items():
            row[stage] = report.get("latency", {}).get(key, 0) if report else None
        comparison["metrics"][f"latency_{key}"] = row

    # 分类对比
    cat_comparison = {}
    all_cats = set()
    for report in reports.values():
        if report:
            cats = report.get("category_accuracy", {})
            if isinstance(cats, list):
                cats = {c["category"]: c["accuracy"] for c in cats}
            all_cats.update(cats.keys())

    for cat in sorted(all_cats):
        row = {}
        for stage, report in reports.items():
            if report:
                cats = report.get("category_accuracy", {})
                if isinstance(cats, list):
                    cats = {c["category"]: c["accuracy"] for c in cats}
                row[stage] = cats.get(cat, 0)
            else:
                row[stage] = None
        cat_comparison[cat] = row

    comparison["category_comparison"] = cat_comparison

    # LLM judge 分类对比
    if "llm_judge" in p21:
        cat_judge = p21["llm_judge"].get("category_mean", {})
        comparison["judge_category"] = cat_judge

    # 路由分布（P2.1 独有）
    if "details" in p21:
        route_dist = {}
        for d in p21["details"]:
            rt = d.get("route_type", "unknown")
            route_dist[rt] = route_dist.get(rt, 0) + 1
        comparison["route_distribution"] = route_dist

    # context_extras 命中统计（P2.1 独有）
    if "details" in p21:
        context_hits = {"concept": 0, "formula_info": 0, "herb_query": 0,
                        "comparison": 0, "graph_syndrome": 0}
        for d in p21["details"]:
            ctx = d.get("context_extras", {})
            if ctx:
                for key in context_hits:
                    if key in ctx:
                        context_hits[key] += 1
        comparison["context_extras_hits"] = context_hits

    return comparison


def main():
    print("=" * 60)
    print("P2.1 评测: ConceptMapper + GraphRAG + LLM-as-judge")
    print("=" * 60)

    # 检查 Ollama
    ollama_path = r"C:\Users\23919\AppData\Local\Programs\Ollama\ollama.exe"
    if not Path(ollama_path).exists():
        print(f"ERROR: Ollama 未找到: {ollama_path}")
        sys.exit(1)

    # 1. 创建 Generator（Ollama 微调模型）
    print("\n[1/4] 连接 Ollama 微调模型...")
    generator = Generator(model="qwen25-15b-tcm")

    # 2. 创建 RAG Pipeline（P2.1 HybridRetriever）
    print("\n[2/4] 构建 P2.1 混合检索管道（ConceptMapper + GraphRAG）...")
    pipeline = RAGPipeline(
        chroma_path=str(PROJECT_ROOT / "data" / "chroma"),
        clauses_path=str(PROJECT_ROOT / "data" / "processed" / "classics" / "shanghan_clauses.jsonl"),
        top_k=5,
        generator=generator,
        use_hybrid=True,
    )
    print(f"索引文档数: {pipeline.retriever.count()}")

    # 打印图谱统计
    if pipeline.hybrid_retriever:
        stats = pipeline.hybrid_retriever.graph_stats
        print(f"知识图谱: {stats['nodes']} 节点, {stats['edges']} 边 "
              f"({stats['formulas']} 方剂, {stats['herbs']} 药材, {stats['syndromes']} 证候)")

    # 3. 运行 50 题评测
    print("\n[3/4] 运行 50 题评测...")
    eval_path = str(PROJECT_ROOT / "data" / "eval" / "eval_50.jsonl")
    runner = EvalRunner(pipeline, eval_path)

    t_start = time.time()
    results = runner.run()
    eval_time = time.time() - t_start
    print(f"\n评测耗时: {eval_time:.1f}s ({eval_time/60:.1f} min)")

    # 保存初步报告（不含 judge）
    p21_report = runner.generate_report(results, stage="P2.1-ConceptGraph")
    p21_report_path = str(PROJECT_ROOT / "data" / "processed" / "p2_1_report.json")

    # 4. LLM-as-judge 评分
    print("\n[4/4] LLM-as-judge 评分中...")
    print("-" * 50)

    judge = LLMJudge(model="qwen2.5:1.5b")
    if not judge.is_available():
        print("WARNING: qwen2.5:1.5b 模型不可用，跳过 LLM judge 评分")
        print("  请运行: ollama pull qwen2.5:1.5b")
    else:
        judge_items = []
        for r in results:
            judge_items.append({
                "question": r.question,
                "expected_answer": r.expected_answer,
                "answer": r.answer,
            })

        judge_results = judge.score_batch(judge_items, delay=0.1)

        for i, (r, jr) in enumerate(zip(results, judge_results), 1):
            r.judge_score = jr.score
            r.judge_reasoning = jr.reasoning
            score_str = f"{jr.score:.1f}" if jr.score >= 0 else "ERR"
            print(f"  [{i}/{len(results)}] {r.category}: {score_str}/5.0 — {r.question[:30]}...")

        judge_stats = compute_judge_scores(results)
        print(f"\nLLM Judge 统计:")
        print(f"  平均分: {judge_stats['mean']:.2f} / 5.0")
        print(f"  中位数: {judge_stats['median']:.2f} / 5.0")
        print(f"  分类均分:")
        for cat, score in sorted(judge_stats["category_mean"].items()):
            print(f"    {cat}: {score:.2f}")

    # 重新生成含 judge 的报告
    p21_report = generate_report(results, stage="P2.1-ConceptGraph")

    # 保存 P2.1 报告
    with open(p21_report_path, "w", encoding="utf-8") as f:
        json.dump(p21_report, f, ensure_ascii=False, indent=2)
    print(f"\nP2.1 报告已保存: {p21_report_path}")

    # 四阶段对比
    print("\n" + "=" * 60)
    print("P0 vs P1 vs P2.0 vs P2.1 四阶段对比")
    print("=" * 60)

    p0 = load_report("p0_baseline_report.json")
    p1 = load_report("p1_lora_report.json")
    p2 = load_report("p2_hybrid_report.json")

    comparison = generate_four_way_comparison(p0, p1, p2, p21_report)

    comparison_path = str(
        PROJECT_ROOT / "data" / "processed" / "p0_p1_p2_p21_comparison.json"
    )
    with open(comparison_path, "w", encoding="utf-8") as f:
        json.dump(comparison, f, ensure_ascii=False, indent=2)

    # 打印对比表
    print(f"\n{'指标':<25} {'P0':>10} {'P1':>10} {'P2.0':>10} {'P2.1':>10} {'P0→P2.1':>10} {'提升%':>10}")
    print("-" * 85)
    for m, vals in comparison["metrics"].items():
        p0_v = vals.get("p0") or 0
        p1_v = vals.get("p1") or 0
        p2_v = vals.get("p2") or 0
        p21_v = vals.get("p2.1") or 0

        if "latency" in m:
            print(f"{m:<25} {p0_v:>10.2f} {p1_v:>10.2f} {p2_v:>10.2f} {p21_v:>10.2f}")
        elif "llm_judge" in m:
            # judge 分数只有 P2.1 有
            p0_s = "-" if vals.get("p0") is None else f"{p0_v:.2f}"
            p1_s = "-" if vals.get("p1") is None else f"{p1_v:.2f}"
            p2_s = "-" if vals.get("p2") is None else f"{p2_v:.2f}"
            print(f"{m:<25} {p0_s:>10} {p1_s:>10} {p2_s:>10} {p21_v:>10.2f}")
        else:
            delta = vals.get("delta_p0_to_p21", 0)
            pct = vals.get("improvement_pct", 0)
            sign = "+" if delta >= 0 else ""
            print(
                f"{m:<25} {p0_v:>10.4f} {p1_v:>10.4f} {p2_v:>10.4f} {p21_v:>10.4f} "
                f"{sign}{delta:>9.4f} {sign}{pct:>9.1f}%"
            )

    # 分类对比
    print(f"\n{'分类':<20} {'P0':>10} {'P1':>10} {'P2.0':>10} {'P2.1':>10}")
    print("-" * 60)
    for cat, vals in comparison["category_comparison"].items():
        p0_v = vals.get("p0", 0) or 0
        p1_v = vals.get("p1", 0) or 0
        p2_v = vals.get("p2", 0) or 0
        p21_v = vals.get("p2.1", 0) or 0
        print(f"{cat:<20} {p0_v:>10.4f} {p1_v:>10.4f} {p2_v:>10.4f} {p21_v:>10.4f}")

    # LLM judge 分类
    if "judge_category" in comparison:
        print(f"\n{'分类':<20} {'Judge 分数':>10}")
        print("-" * 30)
        for cat, score in sorted(comparison["judge_category"].items()):
            print(f"{cat:<20} {score:>10.2f}")

    # 路由分布
    if "route_distribution" in comparison:
        print(f"\n{'路由类型':<20} {'题数':>5}")
        print("-" * 25)
        for rt, count in sorted(comparison["route_distribution"].items()):
            print(f"{rt:<20} {count:>5}")

    # context_extras 命中
    if "context_extras_hits" in comparison:
        print(f"\n{'上下文类型':<20} {'命中次数':>8}")
        print("-" * 28)
        for ctx_type, count in comparison["context_extras_hits"].items():
            if count > 0:
                print(f"{ctx_type:<20} {count:>8}")

    print(f"\n对比报告已保存: {comparison_path}")
    print(f"\n总耗时: {eval_time + (time.time() - t_start):.1f}s")


if __name__ == "__main__":
    main()
