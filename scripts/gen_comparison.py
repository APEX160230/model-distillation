"""生成 P0 vs P1 对比报告（不重跑评测）"""
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

p0_path = PROJECT_ROOT / "data" / "processed" / "p0_baseline_report.json"
p1_path = PROJECT_ROOT / "data" / "processed" / "p1_lora_report.json"

with open(p0_path, "r", encoding="utf-8") as f:
    p0 = json.load(f)
with open(p1_path, "r", encoding="utf-8") as f:
    p1 = json.load(f)

comparison = {
    "p0_stage": p0["stage"],
    "p1_stage": p1["stage"],
    "metrics": {},
}

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

for key in ["p50", "p95", "mean"]:
    p0_val = p0.get("latency", {}).get(key, 0)
    p1_val = p1.get("latency", {}).get(key, 0)
    comparison["metrics"][f"latency_{key}"] = {
        "p0": p0_val,
        "p1": p1_val,
        "delta": round(p1_val - p0_val, 2),
    }

p0_cats = p0.get("category_accuracy", {})
p1_cats = p1.get("category_accuracy", {})
comparison["category_comparison"] = {}
for cat in sorted(set(p0_cats) | set(p1_cats)):
    p0_val = p0_cats.get(cat, 0)
    p1_val = p1_cats.get(cat, 0)
    comparison["category_comparison"][cat] = {
        "p0": p0_val,
        "p1": p1_val,
        "delta": round(p1_val - p0_val, 4),
    }

out_path = PROJECT_ROOT / "data" / "processed" / "p0_vs_p1_comparison.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(comparison, f, ensure_ascii=False, indent=2)

print("P0 vs P1 对比报告已生成:", out_path)
print()

header = f"{'指标':<25} {'P0':>10} {'P1':>10} {'变化':>10} {'提升%':>10}"
print(header)
print("-" * 65)
for m, vals in comparison["metrics"].items():
    delta = vals["delta"]
    pct = vals.get("improvement_pct", 0)
    sign = "+" if delta >= 0 else ""
    print(f"{m:<25} {vals['p0']:>10.4f} {vals['p1']:>10.4f} {sign}{delta:>9.4f} {sign}{pct:>9.1f}%")

print()
cat_header = f"{'分类':<20} {'P0':>10} {'P1':>10} {'变化':>10}"
print(cat_header)
print("-" * 50)
for cat, vals in comparison["category_comparison"].items():
    delta = vals["delta"]
    sign = "+" if delta >= 0 else ""
    print(f"{cat:<20} {vals['p0']:>10.4f} {vals['p1']:>10.4f} {sign}{delta:>9.4f}")
