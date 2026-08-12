"""真实场景辨证实测（PRD v3.0 §7 验证策略）

15-20 例真实症状描述，人工标注预期证型方向，
检查辨证引擎判定正确性 + 追问合理性 + 拒答恰当性。

用法：
    .venv/Scripts/python.exe scripts/eval_diagnosis_scenarios.py

验收标准（PRD v3.0 §7）：
- 致命错误 = 0（编方剂/给剂量由护栏保证，此处检查辨证方向）
- 辨证方向正确率 ≥ 80%（核心层场景）
- 证据不足时合理追问，不硬判
"""
from src.rag.diagnosis import DiagnosisEngine

# (场景描述, 预期证型方向(可为列表), 备注)
SCENARIOS: list[tuple[str, str | list[str] | None, str]] = [
    # ── 核心层：太阳病 ──
    ("我这两天头痛，怕冷，不出汗", "太阳伤寒", "典型太阳伤寒"),
    ("昨晚吹了空调，浑身疼，怕冷，一点汗都没有", "太阳伤寒", "感冒场景"),
    ("发烧，怕风，一摸身上有汗", "太阳中风", "典型太阳中风"),
    ("昨天淋雨了，现在发烧，身上有汗，怕风", "太阳中风", "淋雨感冒"),
    # ── 少阳病 ──
    ("嘴里发苦，嗓子干，还有点头晕", "少阳证", "口苦咽干目眩提纲"),
    ("一阵冷一阵热，胸口和两边肋骨发闷", "少阳证", "往来寒热+胸胁苦满"),
    # ── 阳明病 ──
    ("发高烧，出大汗，口渴想喝凉的", "阳明热证", "大热大汗大渴"),
    ("肚子胀，好几天没大便了，晚上说胡话", "阳明腑实", "腹满便秘谵语"),
    # ── 太阴/少阴 ──
    ("拉肚子，肚子疼，手脚冰凉", ["太阴虚寒", "少阴寒化"], "下利+腹痛+厥逆，二证皆属里寒方向"),
    ("整天没精神，只想躺着，拉肚子拉的是没消化的东西，手脚冷", "少阴寒化", "但欲寐+下利清谷+厥逆"),
    # ── 其他证型 ──
    ("心口堵得慌，按着不痛", "痞证", "心下痞满而不痛"),
    ("小便少，口渴，喝水就吐", "蓄水证", "小便不利+水入则吐"),
    # ── 追问场景（证据不足） ──
    ("我有点头痛", None, "单症状 → 应追问"),
    ("拉肚子，肚子疼", None, "证据不足 → 应追问"),
    # ── 拒答场景（图谱外） ──
    ("我最近耳鸣，听力下降", None, "图谱外 → 应拒答"),
    ("想问问减肥怎么吃", None, "非辨证 → 应拒答/降级"),
]


def main() -> None:
    engine = DiagnosisEngine()
    print(f"{'场景':<38} {'预期':<8} {'实际':<12} 状态")
    print("-" * 80)
    correct = 0
    judged = 0
    for desc, expected, note in SCENARIOS:
        r = engine.diagnose(desc)
        if expected is not None:
            judged += 1
            expect_list = expected if isinstance(expected, list) else [expected]
            ok = r.status == "diagnosed" and r.syndrome in expect_list
            if ok:
                correct += 1
            exp_text = "/".join(expect_list)
            mark = "✓" if ok else "✗"
            print(f"{desc[:34]:<38} {exp_text:<10} {str(r.syndrome or ''):<12} {r.status} {mark} {note}")
        else:
            expect = "追问" if "追问" in note else "拒答"
            ok = (expect == "追问" and r.status == "need_clarification") or \
                 (expect == "拒答" and r.status == "rejected")
            mark = "✓" if ok else "✗"
            print(f"{desc[:34]:<38} {expect:<8} {r.status:<12} {'':<0} {mark} {note}")
            if ok:
                correct += 1
            judged += 1

    print("-" * 80)
    print(f"正确率: {correct}/{judged} = {correct / judged * 100:.0f}%（目标 ≥ 80%）")


if __name__ == "__main__":
    main()
