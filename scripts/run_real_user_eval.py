"""真实用户场景评测 — 模拟普通用户的自然问法

与 P2.1 模板评测对比：
- 模板评测：50题，精确匹配路由（"第12条原文" / "桂枝汤组成" / "含有桂枝的方剂"）
- 真实评测：25题，口语化/模糊/症状描述/超范围

目的：发现系统在非理想输入下的真实表现
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
from src.rag.query_router import QueryRouter


def main():
    print("=" * 60)
    print("真实用户场景评测 — 自然语言、口语化、模糊查询")
    print("=" * 60)

    ollama_path = r"C:\Users\23919\AppData\Local\Programs\Ollama\ollama.exe"
    if not Path(ollama_path).exists():
        print(f"ERROR: Ollama 未找到: {ollama_path}")
        sys.exit(1)

    # 1. 构建管道
    print("\n[1/3] 构建混合检索管道...")
    generator = Generator(model="qwen25-15b-tcm")
    pipeline = RAGPipeline(
        chroma_path=str(PROJECT_ROOT / "data" / "chroma"),
        clauses_path=str(PROJECT_ROOT / "data" / "processed" / "classics" / "shanghan_clauses.jsonl"),
        top_k=5,
        generator=generator,
        use_hybrid=True,
    )
    print(f"索引文档数: {pipeline.retriever.count()}")
    if pipeline.hybrid_retriever:
        stats = pipeline.hybrid_retriever.graph_stats
        print(f"知识图谱: {stats['nodes']} 节点, {stats['edges']} 边")

    # 2. 加载真实用户评测集
    eval_path = str(PROJECT_ROOT / "data" / "eval" / "eval_real_user.jsonl")
    runner = EvalRunner(pipeline, eval_path)

    # 先打印路由分析
    print("\n[2/3] 路由分析（评测前）...")
    router = QueryRouter()
    items = runner._load_eval_data()
    print(f"{'ID':>3} {'路由':12s} {'预期路由':12s} 问题")
    print("-" * 80)
    for item in items:
        q = item["question"]
        route = router.route(q)
        # 预期路由
        cat = item.get("category", "")
        if "症状" in cat or "口语" in cat or "多概念" in cat or "超范围" in cat:
            expected = "semantic?"
        elif "自然条文" in cat:
            expected = "clause_id?"
        elif "方剂用法" in cat:
            expected = "formula"
        elif "药材自然" in cat:
            expected = "herb?"
        elif "辨证对比" in cat:
            expected = "comparison?"
        else:
            expected = "?"
        flag = "✓" if expected.startswith(route.query_type.value[:4]) else "✗"
        print(f"{item.get('id',0):3d} {route.query_type.value:12s} {expected:12s} {flag} {q[:40]}")

    # 3. 运行评测
    print(f"\n[3/3] 运行 {len(items)} 题真实用户评测...")
    t_start = time.time()
    results = runner.run()
    eval_time = time.time() - t_start
    print(f"\n评测耗时: {eval_time:.1f}s ({eval_time/60:.1f} min)")

    # 4. LLM judge 评分
    print("\nLLM-as-judge 评分中...")
    print("-" * 50)

    judge = LLMJudge(model="qwen2.5:1.5b")
    if judge.is_available():
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
    else:
        print("WARNING: qwen2.5:1.5b 模型不可用，跳过 LLM judge")

    # 5. 生成报告
    report = generate_report(results, stage="P2.1-RealUser")

    # 添加路由分析
    route_analysis = []
    for r in results:
        route_analysis.append({
            "question_id": r.question_id,
            "category": r.category,
            "question": r.question,
            "route_type": r.route_type,
            "judge_score": r.judge_score,
        })
    report["route_analysis"] = route_analysis

    report_path = str(PROJECT_ROOT / "data" / "processed" / "real_user_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n报告已保存: {report_path}")

    # 6. 打印详细结果
    print("\n" + "=" * 80)
    print("真实用户评测详细结果")
    print("=" * 80)

    print(f"\n{'ID':>3} {'类别':8s} {'路由':12s} {'recall':6s} {'judge':6s} 问题")
    print("-" * 100)
    for r in results:
        recall_hit = "✓" if _check_recall(r) else "✗"
        judge_s = f"{r.judge_score:.1f}" if r.judge_score >= 0 else "?"
        print(f"{r.question_id:3d} {r.category[:6]:8s} {r.route_type:12s} {recall_hit:6s} {judge_s:6s} {r.question[:40]}")

    # 7. 对比模板评测
    print("\n" + "=" * 60)
    print("模板评测 vs 真实用户评测对比")
    print("=" * 60)

    # 加载 P2.1 模板报告
    p21_path = PROJECT_ROOT / "data" / "processed" / "p2_1_report.json"
    if p21_path.exists():
        with open(p21_path, "r", encoding="utf-8") as f:
            p21 = json.load(f)

        print(f"\n{'指标':<25} {'模板评测(50题)':>15} {'真实用户(25题)':>15}")
        print("-" * 55)
        print(f"{'recall@5':<25} {p21.get('recall_at_5',0)*100:>14.1f}% {report.get('recall_at_5',0)*100:>14.1f}%")
        print(f"{'keyword_accuracy':<25} {p21.get('keyword_accuracy',0)*100:>14.1f}% {report.get('keyword_accuracy',0)*100:>14.1f}%")
        if 'llm_judge' in p21 and 'llm_judge' in report:
            print(f"{'LLM judge 均分':<25} {p21['llm_judge']['mean']:>15.2f} {report['llm_judge']['mean']:>15.2f}")
        print(f"{'延迟 p50 (s)':<25} {p21.get('latency',{}).get('p50',0):>15.2f} {report.get('latency',{}).get('p50',0):>15.2f}")

    # 8. 分类分析
    print(f"\n{'分类':<15} {'题数':>5} {'recall':>8} {'judge':>8}")
    print("-" * 36)
    cat_results = {}
    for r in results:
        cat = r.category
        if cat not in cat_results:
            cat_results[cat] = {"count": 0, "recall": 0, "judge": []}
        cat_results[cat]["count"] += 1
        cat_results[cat]["recall"] += 1 if _check_recall(r) else 0
        if r.judge_score >= 0:
            cat_results[cat]["judge"].append(r.judge_score)

    for cat, d in sorted(cat_results.items()):
        recall_pct = d["recall"] / d["count"] * 100
        judge_avg = sum(d["judge"]) / len(d["judge"]) if d["judge"] else 0
        print(f"{cat:<15} {d['count']:>5} {recall_pct:>7.1f}% {judge_avg:>8.2f}")

    # 9. 路由正确率
    print(f"\n路由分析:")
    route_correct = 0
    route_total = 0
    for ra in route_analysis:
        cat = ra["category"]
        rt = ra["route_type"]
        route_total += 1
        # 判断路由是否合理
        if "方剂用法" in cat and rt == "formula":
            route_correct += 1
        elif "自然条文" in cat and rt in ("clause_id", "semantic", "formula"):
            route_correct += 1
        elif "药材自然" in cat and rt == "herb":
            route_correct += 1
        elif "辨证对比" in cat and rt == "comparison":
            route_correct += 1
        elif ("症状" in cat or "口语" in cat or "多概念" in cat or "超范围" in cat) and rt == "semantic":
            route_correct += 1
    print(f"  合理路由: {route_correct}/{route_total} ({route_correct/route_total*100:.0f}%)")

    # 10. 打印答案样例（低分题）
    print("\n低分题答案（judge < 3.0）:")
    print("-" * 80)
    low_score = [r for r in results if r.judge_score < 3.0 and r.judge_score >= 0]
    for r in low_score[:5]:
        print(f"\nQ{r.question_id} ({r.route_type}): {r.question}")
        print(f"  Judge: {r.judge_score:.1f}/5")
        print(f"  期望: {r.expected_answer[:150]}")
        print(f"  实际: {r.answer[:200]}")
        print(f"  检索: {r.retrieved_clauses}")

    print(f"\n总耗时: {time.time() - t_start:.1f}s")


def _check_recall(result: EvalResult) -> bool:
    """检查 recall 是否命中"""
    if not result.reference_clauses:
        return True  # 无参考条文的题目（药材关联等），默认通过
    ref_set = set(result.reference_clauses)
    ret_set = set(result.retrieved_clauses[:5])
    return bool(ref_set & ret_set)


if __name__ == "__main__":
    main()
