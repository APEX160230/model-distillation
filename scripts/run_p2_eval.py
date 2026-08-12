"""P2.0 评测脚本 — HybridRetriever + Ollama 推理

对比 P0（纯向量+原始模型）、P1（纯向量+微调模型）、P2.0（混合检索+微调模型）

用法：
    python scripts/run_p2_eval.py

产出：
    data/processed/p2_hybrid_report.json — P2.0 评测报告
    data/processed/p0_p1_p2_comparison.json — 三阶段对比
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


def load_report(name: str) -> dict | None:
    """加载历史报告"""
    path = PROJECT_ROOT / "data" / "processed" / name
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def generate_three_way_comparison(
    p0: dict | None, p1: dict | None, p2: dict
) -> dict:
    """生成 P0 vs P1 vs P2 三阶段对比"""
    comparison = {
        "stages": {
            "p0": p0["stage"] if p0 else "N/A",
            "p1": p1["stage"] if p1 else "N/A",
            "p2": p2["stage"],
        },
        "metrics": {},
    }

    reports = {"p0": p0, "p1": p1, "p2": p2}

    # 核心指标
    for metric in ["recall_at_5", "keyword_accuracy"]:
        row = {}
        for stage, report in reports.items():
            row[stage] = report.get(metric, 0) if report else None
        p0_val = row.get("p0", 0) or 0
        p2_val = row.get("p2", 0) or 0
        delta = p2_val - p0_val
        row["delta_p0_to_p2"] = round(delta, 4)
        row["improvement_pct"] = round(
            (delta / p0_val * 100) if p0_val > 0 else 0, 1
        )
        comparison["metrics"][metric] = row

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

    # 路由分布（P2 独有）
    if "details" in p2:
        route_dist = {}
        for d in p2["details"]:
            rt = d.get("route_type", "unknown")
            route_dist[rt] = route_dist.get(rt, 0) + 1
        comparison["route_distribution"] = route_dist

    return comparison


def main():
    print("=" * 60)
    print("P2.0 评测: HybridRetriever + Ollama 微调模型")
    print("=" * 60)

    # 检查 Ollama 模型（环境变量 OLLAMA_PATH 优先，其次 PATH 自动探测）
    ollama_path = os.environ.get("OLLAMA_PATH")
    if not ollama_path:
        for _dir in os.environ.get("PATH", "").split(os.pathsep):
            for _name in ("ollama.exe", "ollama"):
                _candidate = os.path.join(_dir, _name)
                if os.path.exists(_candidate):
                    ollama_path = _candidate
                    break
            if ollama_path:
                break
    if not ollama_path:
        print("ERROR: Ollama 未找到，请设置环境变量 OLLAMA_PATH 或加入 PATH")
        sys.exit(1)

    # 1. 创建 Generator（Ollama）
    print("\n[1/3] 连接 Ollama...")
    generator = Generator(model="qwen25-15b-tcm")

    # 2. 创建 RAG Pipeline（HybridRetriever）
    print("\n[2/3] 构建混合检索管道...")
    pipeline = RAGPipeline(
        chroma_path=str(PROJECT_ROOT / "data" / "chroma"),
        clauses_path=str(PROJECT_ROOT / "data" / "processed" / "classics" / "shanghan_clauses.jsonl"),
        top_k=5,
        generator=generator,
        use_hybrid=True,
    )
    print(f"索引文档数: {pipeline.retriever.count()}")

    # 3. 运行评测
    print("\n[3/3] 运行 50 题评测...")
    eval_path = str(PROJECT_ROOT / "data" / "eval" / "eval_50.jsonl")
    runner = EvalRunner(pipeline, eval_path)

    t_start = time.time()
    results = runner.run()
    total_time = time.time() - t_start

    # 保存 P2 报告
    p2_report_path = str(PROJECT_ROOT / "data" / "processed" / "p2_hybrid_report.json")
    report = runner.save_report(results, p2_report_path, stage="P2-Hybrid")

    print(f"\n总耗时: {total_time:.1f}s ({total_time/60:.1f} min)")

    # 三阶段对比
    print("\n" + "=" * 60)
    print("P0 vs P1 vs P2.0 对比")
    print("=" * 60)

    p0 = load_report("p0_baseline_report.json")
    p1 = load_report("p1_lora_report.json")
    comparison = generate_three_way_comparison(p0, p1, report)

    comparison_path = str(
        PROJECT_ROOT / "data" / "processed" / "p0_p1_p2_comparison.json"
    )
    with open(comparison_path, "w", encoding="utf-8") as f:
        json.dump(comparison, f, ensure_ascii=False, indent=2)

    # 打印对比表
    print(f"\n{'指标':<25} {'P0':>10} {'P1':>10} {'P2.0':>10} {'P0→P2':>10} {'提升%':>10}")
    print("-" * 75)
    for m, vals in comparison["metrics"].items():
        if "latency" in m:
            p0_v = vals.get("p0", 0) or 0
            p1_v = vals.get("p1", 0) or 0
            p2_v = vals.get("p2", 0) or 0
            print(f"{m:<25} {p0_v:>10.2f} {p1_v:>10.2f} {p2_v:>10.2f}")
        else:
            p0_v = vals.get("p0", 0) or 0
            p1_v = vals.get("p1", 0) or 0
            p2_v = vals.get("p2", 0) or 0
            delta = vals.get("delta_p0_to_p2", 0)
            pct = vals.get("improvement_pct", 0)
            sign = "+" if delta >= 0 else ""
            print(
                f"{m:<25} {p0_v:>10.4f} {p1_v:>10.4f} {p2_v:>10.4f} {sign}{delta:>9.4f} {sign}{pct:>9.1f}%"
            )

    print(f"\n{'分类':<20} {'P0':>10} {'P1':>10} {'P2.0':>10}")
    print("-" * 50)
    for cat, vals in comparison["category_comparison"].items():
        p0_v = vals.get("p0", 0) or 0
        p1_v = vals.get("p1", 0) or 0
        p2_v = vals.get("p2", 0) or 0
        print(f"{cat:<20} {p0_v:>10.4f} {p1_v:>10.4f} {p2_v:>10.4f}")

    # 路由分布
    if "route_distribution" in comparison:
        print(f"\n{'路由类型':<20} {'题数':>5}")
        print("-" * 25)
        for rt, count in sorted(comparison["route_distribution"].items()):
            print(f"{rt:<20} {count:>5}")

    print(f"\n对比报告已保存: {comparison_path}")


if __name__ == "__main__":
    main()
