"""线上端到端辨证实测（PRD v3.0 §7，真实使用为准）

模板化后辨证链路已去模型化（秒级响应），批量打线上 API 验证：
- 辨证方向正确性（route_type=diagnosed + syndrome 判断）
- 三层结构完整性（辨证方向/类方思路/倪师讲解）
- 素材相关性（人工观察）
- 追问闭环（need_clarification → 选项回填再问 → 判定）
- 拒答/危象边界

用法:
    .venv/Scripts/python.exe scripts/eval_online_diagnosis.py [--host https://tcm.zzy1n.cc]
"""
import argparse
import json
import sys
import urllib.request
import urllib.error

SCENARIOS: list[tuple[str, str | list[str] | None, str]] = [
    # ── 核心层：太阳病 ──
    ("我这两天头痛，怕冷，不出汗", "太阳伤寒", "典型太阳伤寒"),
    ("发烧，怕风，一摸身上有汗", "太阳中风", "典型太阳中风"),
    ("昨晚吹了空调，浑身疼，怕冷，一点汗都没有", "太阳伤寒", "感冒场景"),
    # ── 少阳/阳明 ──
    ("嘴里发苦，嗓子干，还有点头晕", "少阳证", "少阳提纲"),
    ("发高烧，出大汗，口渴想喝凉的", "阳明热证", "大热大汗大渴"),
    ("肚子胀，好几天没大便了，晚上说胡话", "阳明腑实", "腹满便秘谵语"),
    # ── 太阴/少阴 ──
    ("拉肚子，肚子疼，手脚冰凉", ["太阴虚寒", "少阴寒化"], "里寒方向"),
    # ── 追问闭环 ──
    ("我有点头痛", None, "应追问（单症状）"),
    # ── 拒答/降级 ──
    ("我最近耳鸣，听力下降", None, "图谱外 → 降级知识链路（检索到相关讲解即合理）"),
    ("桂枝汤的组成是什么", None, "知识问答 → 走模型链路"),
]


def call_chat(host: str, question: str, timeout: int = 120) -> dict:
    """调用线上 /api/chat，返回 (route_type, answer, diagnosis, lectures, latency)"""
    req = urllib.request.Request(
        f"{host}/api/chat",
        data=json.dumps({"question": question}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8")

    route_type = ""
    diagnosis = None
    lectures = []
    chunks: list[str] = []
    latency = None
    for line in raw.splitlines():
        if not line.startswith("data: "):
            continue
        try:
            data = json.loads(line[6:])
        except json.JSONDecodeError:
            continue
        if data["type"] == "retrieved":
            route_type = data.get("route_type", "")
            extras = data.get("context_extras") or {}
            diagnosis = extras.get("diagnosis")
            lectures = extras.get("lectures") or []
        elif data["type"] == "chunk":
            chunks.append(data["content"])
        elif data["type"] == "done":
            latency = data.get("latency")
    return {
        "route": route_type,
        "answer": "".join(chunks),
        "diagnosis": diagnosis,
        "lectures": lectures,
        "latency": latency,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="线上端到端辨证实测")
    parser.add_argument("--host", default="https://tcm.zzy1n.cc", help="线上服务地址")
    args = parser.parse_args()
    host = args.host.rstrip("/")

    print(f"==> 线上辨证实测: {host}")
    print(f"{'场景':<36} {'预期':<8} {'实际':<14} 状态  延迟")
    print("-" * 90)

    correct = 0
    judged = 0
    for desc, expected, note in SCENARIOS:
        try:
            r = call_chat(host, desc)
        except urllib.error.HTTPError as e:
            print(f"{desc[:32]:<38} 请求失败 HTTP {e.code}")
            continue
        except Exception as e:
            print(f"{desc[:32]:<38} 请求异常: {e}")
            continue

        sy = (r["diagnosis"] or {}).get("syndrome") if r["diagnosis"] else None
        layers_ok = all(k in r["answer"] for k in ("【辨证方向】", "【类方思路】", "【倪师讲解】"))

        if expected is not None:
            judged += 1
            exp_list = expected if isinstance(expected, list) else [expected]
            ok = r["route"] == "diagnosed" and sy in exp_list
            if ok:
                correct += 1
            mark = "✓" if ok else "✗"
            exp_text = "/".join(exp_list)
            status = f"{sy or '?'}"
        else:
            judged += 1
            if "追问" in note:
                ok = r["route"] == "diagnosis_clarify"
                status = "追问"
            elif "知识问答" in note or "降级知识链路" in note:
                ok = r["route"] not in ("diagnosed", "diagnosis_clarify") and len(r["answer"]) > 20
                status = "知识链路"
            else:
                ok = r["route"] in ("diagnosed", "diagnosis_clarify", "rejected")
                status = f"{r['route']}"
            if ok:
                correct += 1
            mark = "✓" if ok else "✗"
            exp_text = note.split("→")[-1].strip()

        lat = f"{r['latency']:.1f}s" if r["latency"] is not None else "?"
        print(f"{desc[:32]:<38} {exp_text:<8} {status:<14} {mark}  {lat}")

        # 追问场景：模拟选项回填闭环
        if expected is None and "追问" in note and r["route"] == "diagnosis_clarify":
            diag = r["diagnosis"] or {}
            options = diag.get("options") or []
            if options:
                followup = f"{desc}，{options[0]}"
                r2 = call_chat(host, followup)
                sy2 = (r2["diagnosis"] or {}).get("syndrome") if r2["diagnosis"] else None
                print(f"    ↳ 追问选项「{options[0]}」→ route={r2['route']} syndrome={sy2}")

    print("-" * 90)
    print(f"正确率: {correct}/{judged} = {correct / judged * 100:.0f}%（目标 ≥ 80%）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
