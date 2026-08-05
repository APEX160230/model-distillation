"""P1 评测脚本 — 使用 LoRA 微调后的模型运行 50 题评测

用法：
    python scripts/run_p1_eval.py

产出：
    data/processed/p1_lora_report.json — P1 评测报告
    data/processed/p0_vs_p1_comparison.json — P0 vs P1 对比
"""
import os
import sys
import json
import time
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import torch
from src.rag.pipeline import RAGPipeline
from src.rag.local_generate import LocalGenerator
from src.eval.runner import EvalRunner


def load_p0_report() -> dict | None:
    """加载 P0 报告用于对比"""
    p0_path = PROJECT_ROOT / "data" / "processed" / "p0_baseline_report.json"
    if p0_path.exists():
        with open(p0_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def compare_reports(p0: dict | None, p1: dict) -> dict:
    """生成 P0 vs P1 对比报告"""
    comparison = {
        "p0_stage": p0["stage"] if p0 else "N/A",
        "p1_stage": p1["stage"],
        "metrics": {},
    }

    if p0:
        metrics = ["recall_at_5", "keyword_accuracy"]
        for m in metrics:
            p0_val = p0.get(m, 0)
            p1_val = p1.get(m, 0)
            delta = p1_val - p0_val
            comparison["metrics"][m] = {
                "p0": p0_val,
                "p1": p1_val,
                "delta": round(delta, 4),
                "improvement_pct": round((delta / p0_val * 100) if p0_val > 0 else 0, 1),
            }

        # 延迟对比
        for key in ["p50", "p95", "mean"]:
            p0_val = p0.get("latency", {}).get(key, 0)
            p1_val = p1.get("latency", {}).get(key, 0)
            comparison["metrics"][f"latency_{key}"] = {
                "p0": p0_val,
                "p1": p1_val,
                "delta": round(p1_val - p0_val, 2),
            }

        # 分类准确率对比
        p0_cats = {c["category"]: c["accuracy"] for c in p0.get("category_accuracy", [])}
        p1_cats = {c["category"]: c["accuracy"] for c in p1.get("category_accuracy", [])}
        comparison["category_comparison"] = {}
        for cat in sorted(set(p0_cats) | set(p1_cats)):
            p0_val = p0_cats.get(cat, 0)
            p1_val = p1_cats.get(cat, 0)
            comparison["category_comparison"][cat] = {
                "p0": p0_val,
                "p1": p1_val,
                "delta": round(p1_val - p0_val, 4),
            }
    else:
        comparison["note"] = "P0 报告不存在，无法对比"

    return comparison


def main():
    print("=" * 60)
    print("P1 评测: LoRA 微调后模型 + 50 题评测集")
    print("=" * 60)

    merged_dir = PROJECT_ROOT / "output_merged"
    if not merged_dir.exists():
        print(f"ERROR: 微调模型不存在: {merged_dir}")
        print("请先运行训练: python scripts/train_lora_cpu.py")
        sys.exit(1)

    # 检查训练统计
    stats_path = PROJECT_ROOT / "output_lora" / "training_stats.json"
    if stats_path.exists():
        with open(stats_path, "r", encoding="utf-8") as f:
            stats = json.load(f)
        print(f"\n训练信息:")
        print(f"  基座: {stats['model']}")
        print(f"  LoRA rank: {stats['lora_rank']}, alpha: {stats['lora_alpha']}")
        print(f"  训练样本: {stats['train_samples']}")
        print(f"  Final loss: {stats['final_loss']}")
        print(f"  训练时间: {stats.get('train_time_minutes', 'N/A')} min")

    # 1. 创建 LocalGenerator
    print("\n[1/3] 加载微调模型...")
    generator = LocalGenerator(
        model_path=str(merged_dir),
        device="cpu",
        dtype=torch.bfloat16,
    )

    # 2. 创建 RAG Pipeline
    print("\n[2/3] 构建 RAG 管道...")
    pipeline = RAGPipeline(
        chroma_path=str(PROJECT_ROOT / "data" / "chroma"),
        top_k=5,
        generator=generator,
    )
    print(f"向量库文档数: {pipeline.retriever.count()}")

    # 3. 运行评测
    print("\n[3/3] 运行 50 题评测...")
    eval_path = str(PROJECT_ROOT / "data" / "eval" / "eval_50.jsonl")
    runner = EvalRunner(pipeline, eval_path)

    t_start = time.time()
    results = runner.run()
    total_time = time.time() - t_start

    # 保存 P1 报告
    p1_report_path = str(PROJECT_ROOT / "data" / "processed" / "p1_lora_report.json")
    report = runner.save_report(results, p1_report_path, stage="P1-LoRA")

    print(f"\n总耗时: {total_time:.1f}s ({total_time/60:.1f} min)")

    # 对比 P0
    print("\n" + "=" * 60)
    print("P0 vs P1 对比")
    print("=" * 60)

    p0 = load_p0_report()
    comparison = compare_reports(p0, report)

    comparison_path = str(PROJECT_ROOT / "data" / "processed" / "p0_vs_p1_comparison.json")
    with open(comparison_path, "w", encoding="utf-8") as f:
        json.dump(comparison, f, ensure_ascii=False, indent=2)

    if p0:
        print(f"\n{'指标':<25} {'P0':>10} {'P1':>10} {'变化':>10} {'提升%':>10}")
        print("-" * 65)
        for m, vals in comparison["metrics"].items():
            delta = vals["delta"]
            pct = vals.get("improvement_pct", 0)
            sign = "+" if delta >= 0 else ""
            print(f"{m:<25} {vals['p0']:>10.4f} {vals['p1']:>10.4f} {sign}{delta:>9.4f} {sign}{pct:>9.1f}%")

        print(f"\n{'分类':<20} {'P0':>10} {'P1':>10} {'变化':>10}")
        print("-" * 50)
        for cat, vals in comparison["category_comparison"].items():
            delta = vals["delta"]
            sign = "+" if delta >= 0 else ""
            print(f"{cat:<20} {vals['p0']:>10.4f} {vals['p1']:>10.4f} {sign}{delta:>9.4f}")
    else:
        print("P0 报告不存在，无法对比")

    print(f"\n对比报告已保存: {comparison_path}")


if __name__ == "__main__":
    main()
