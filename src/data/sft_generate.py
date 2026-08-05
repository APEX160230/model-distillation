"""SFT 训练数据生成器 v2 — 从结构化数据生成高质量 instruction-response 对

数据来源：
- 330 条伤寒论原文 (shanghan_clauses.jsonl)
- 60+ 核心方剂数据库 (formulas_db.py)

生成策略：模板 + 变体，覆盖 5 个评测类别，分布接近评测集 (30/20/20/20/10)
目标：2000+ 条高质量训练样本
"""
import json
import random
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass

from src.data.formulas_db import FORMULAS, get_formula_dict, get_herb_formulas

random.seed(42)


def to_chinese_num(n: int) -> str:
    """阿拉伯数字转中文数字"""
    cn = "零一二三四五六七八九"
    if n < 10:
        return cn[n]
    if n < 20:
        return "十" + (cn[n % 10] if n % 10 else "")
    if n < 100:
        tens, ones = divmod(n, 10)
        return cn[tens] + "十" + (cn[ones] if ones else "")
    if n < 1000:
        h, rest = divmod(n, 100)
        r = cn[h] + "百"
        if rest >= 10:
            t, o = divmod(rest, 10)
            r += (cn[t] if t else "零") + "十" + (cn[o] if o else "")
        elif rest > 0:
            r += "零" + cn[rest]
        return r
    return str(n)


@dataclass
class SFTSample:
    instruction: str
    input: str
    output: str
    category: str = ""

    def to_dict(self) -> dict:
        return {
            "instruction": self.instruction,
            "input": self.input,
            "output": self.output,
            "category": self.category,
        }


class SFTGeneratorV2:
    """从结构化数据生成 SFT 训练样本（v2，使用完整方剂数据库）"""

    def __init__(self, clauses_path: str):
        with open(clauses_path, "r", encoding="utf-8") as f:
            self.clauses = [json.loads(line) for line in f]
        self.clause_by_id = {c["clause_id"]: c for c in self.clauses}

        self.formula_dict = get_formula_dict()
        self.herb_formulas = get_herb_formulas()
        self.all_herbs = sorted(self.herb_formulas.keys())

        # 条文 → 提到的方剂
        self.clause_formulas: dict[int, list[str]] = {}
        for c in self.clauses:
            mentioned = [f.name for f in FORMULAS if f.name in c["original_text"]]
            if mentioned:
                self.clause_formulas[c["clause_id"]] = mentioned

        # 方剂 → 相关条文
        self.formula_clauses: dict[str, list[int]] = defaultdict(list)
        for cid, formulas in self.clause_formulas.items():
            for f in formulas:
                self.formula_clauses[f].append(cid)

    # ==================== 1. 经典原文检索 (目标 ~600) ====================

    def gen_clause_retrieval(self) -> list[SFTSample]:
        samples = []

        for c in self.clauses:
            cid = c["clause_id"]
            text = c["original_text"]
            cn_num = to_chinese_num(cid)
            chapter = c["chapter"]

            # 变体 1: 直接问原文（每条都生成）
            q1 = random.choice([
                f"伤寒论第{cid}条原文是什么？",
                f"请说出伤寒论第{cid}条的内容。",
                f"伤寒论第{cn_num}条的原文是什么？",
                f"请引用伤寒论第{cid}条原文。",
                f"伤寒论第{cid}条说了什么？",
            ])
            a1 = random.choice([
                f"伤寒论第{cid}条原文：\n{text}",
                f"伤寒论第{cid}条（{chapter}）原文如下：\n{text}",
                f"根据《伤寒论》第{cid}条，原文为：\n{text}",
                f"伤寒论第{cid}条：\n{text}",
            ])
            samples.append(SFTSample(q1, "", a1, "经典原文检索"))

            # 变体 2: 按内容反查编号（只对前 200 条生成，控制数量）
            if cid <= 200 and len(text) >= 10:
                snippet = text[:15]
                q2 = random.choice([
                    f'\u201c{snippet}\u2026\u201d是伤寒论第几条？',
                    f'伤寒论哪一条提到了\u201c{snippet}\u201d？',
                    f'\u201c{snippet}\u2026\u201d这条原文出自伤寒论第几条？',
                ])
                a2 = f"这段内容出自伤寒论第{cid}条（{chapter}）。原文完整内容为：\n{text}"
                samples.append(SFTSample(q2, "", a2, "经典原文检索"))

        return samples

    # ==================== 2. 方剂查询 (目标 ~400) ====================

    def gen_formula_query(self) -> list[SFTSample]:
        samples = []

        for f in FORMULAS:
            herb_str = "\u3001".join(f.herbs)
            clause = self.clause_by_id.get(f.clause_id, {})

            # 变体 1: 问组成
            q1 = random.choice([
                f"{f.name}的组成是什么？",
                f"{f.name}由哪些药物组成？",
                f"{f.name}的药物组成有哪些？",
                f"请说出{f.name}的组成。",
                f"{f.name}有哪些药味？",
            ])
            a1 = random.choice([
                f"{f.name}的组成为：{herb_str}。",
                f"{f.name}由{herb_str}组成。",
                f"{f.name}的药物组成包括：{herb_str}。",
                f"{f.name}的组成：{herb_str}。{f.brief}。",
            ])
            samples.append(SFTSample(q1, "", a1, "方剂查询"))

            # 变体 2: 问主治
            q2 = random.choice([
                f"{f.name}主治什么证候？",
                f"{f.name}的适应证是什么？",
                f"{f.name}用于治疗什么病证？",
                f"{f.name}的功效是什么？",
            ])
            a2 = f"{f.name}主治{f.syndrome}。{f.brief}。"
            if clause:
                a2 += f"\n相关条文见于伤寒论第{f.clause_id}条。"
            samples.append(SFTSample(q2, "", a2, "方剂查询"))

            # 变体 3: 问出处条文
            if f.clause_id in self.clause_by_id:
                q3 = random.choice([
                    f"{f.name}出自伤寒论哪一条？",
                    f"伤寒论中哪一条提到了{f.name}？",
                    f"{f.name}在伤寒论中的依据是什么？",
                ])
                clause_text = self.clause_by_id[f.clause_id]["original_text"]
                a3 = f"{f.name}见于伤寒论第{f.clause_id}条。原文：\n{clause_text}"
                samples.append(SFTSample(q3, "", a3, "方剂查询"))

            # 变体 4: 组成 + 主治综合
            q4 = random.choice([
                f"请详细介绍{f.name}的组成、功效和主治。",
                f"{f.name}的组成和主治证候分别是什么？",
            ])
            a4 = f"{f.name}\n组成：{herb_str}\n主治：{f.syndrome}\n功效：{f.brief}"
            if clause:
                a4 += f"\n出处：伤寒论第{f.clause_id}条"
            samples.append(SFTSample(q4, "", a4, "方剂查询"))

        return samples

    # ==================== 3. 药材关联 (目标 ~400) ====================

    def gen_herb_association(self) -> list[SFTSample]:
        samples = []

        # 单药材 → 方剂
        for herb in self.all_herbs:
            formulas = self.herb_formulas[herb]
            f_str = "\u3001".join(formulas)
            q = random.choice([
                f"含有{herb}的方剂有哪些？",
                f"哪些方剂中含有{herb}？",
                f"{herb}在伤寒论中出现在哪些方剂里？",
                f"伤寒论中用到{herb}的方剂有哪些？",
                f"含有{herb}的经方有哪些？",
            ])
            a = random.choice([
                f"含有{herb}的方剂有：{f_str}。",
                f"在伤寒论中，含有{herb}的方剂包括：{f_str}。",
                f"{herb}出现在以下方剂中：{f_str}。",
                f"含有{herb}的方剂共{len(formulas)}首：{f_str}。",
            ])
            samples.append(SFTSample(q, "", a, "药材关联"))

        # 双药材 → 方剂
        herb_list = sorted(self.all_herbs)
        for i, h1 in enumerate(herb_list):
            for h2 in herb_list[i + 1:]:
                f1 = set(self.herb_formulas[h1])
                f2 = set(self.herb_formulas[h2])
                common = sorted(f1 & f2)
                if len(common) >= 2:
                    c_str = "\u3001".join(common)
                    q = random.choice([
                        f"同时含有{h1}和{h2}的方剂有哪些？",
                        f"哪些方剂中既有{h1}又有{h2}？",
                        f"{h1}和{h2}同时出现在哪些方剂中？",
                    ])
                    a = f"同时含有{h1}和{h2}的方剂有：{c_str}。"
                    samples.append(SFTSample(q, "", a, "药材关联"))

        return samples

    # ==================== 4. 经典解释 (目标 ~400) ====================

    def gen_concept_explanation(self) -> list[SFTSample]:
        samples = []

        # 六经病定义
        six_meridian_defs = [
            ("太阳病", 1, "太阳之为病，脉浮，头项强痛而恶寒。",
             "太阳病是外感病的初期阶段，邪犯肌表，正邪交争于表。主要表现为脉浮、头项强痛而恶寒。太阳病分为太阳中风（表虚证）和太阳伤寒（表实证）两大类型。"),
            ("阳明病", 208, "阳明病，脉迟，虽汗出不恶寒者，其身必重，短气，腹满而喘，有潮热者，此外欲解，可攻里也。手足濈然汗出者，此大便已硬也，大承气汤主之。",
             "阳明病是外感病发展过程中，阳热亢盛、胃肠燥热的阶段。分为阳明经证（白虎汤证）和阳明腑证（承气汤证），主要表现为大热、大汗、大渴、脉洪大。"),
            ("少阳病", 96, "伤寒五六日中风，往来寒热，胸胁苦满，嘿嘿不欲饮食，心烦喜呕，或胸中烦而不呕，或渴，或腹中痛，或胁下痞硬，或心下悸、小便不利，或不渴、身有微热，或咳者，小柴胡汤主之。",
             "少阳病是邪犯少阳胆经，枢机不利所致。主要表现为往来寒热、胸胁苦满、默默不欲饮食、心烦喜呕，以小柴胡汤为主方。"),
            ("太阴病", 273, "太阴之为病，腹满而吐，食不下，自利益甚，时腹自痛。若下之，必胸下结硬。",
             "太阴病是脾阳虚衰、寒湿内盛所致。主要表现为腹满而吐、食不下、自利益甚、时腹自痛，以理中丸、四逆汤类温中散寒。"),
            ("少阴病", 281, "少阴之为病，脉微细，但欲寐也。",
             "少阴病是心肾阳衰或阴虚火旺所致。分为寒化证（四逆汤证）和热化证（黄连阿胶汤证），主要表现为脉微细、但欲寐。"),
            ("厥阴病", 338, "伤寒脉微而厥，至七八日肤冷，其人躁无暂安时者，此为脏厥，非蛔厥也。蛔厥者，其人当吐蛔。今病者静，而复时烦者，此为脏寒。蛔上入其膈，故烦，须臾复止，得食而呕，又烦者，蛔闻食臭出，其人常自吐蛔。蛔厥者，乌梅丸主之。又主久利。",
             "厥阴病是邪入厥阴、寒热错杂的阶段。主要表现为消渴、气上撞心、心中疼热、饥而不欲食、食则吐蛔，以乌梅丸为主方。"),
        ]

        for name, cid, text, explanation in six_meridian_defs:
            for q in [
                f"什么是{name}？",
                f"{name}的定义是什么？",
                f"{name}的主要表现是什么？",
                f"请解释{name}的概念。",
                f"{name}的证候特点是什么？",
            ]:
                a = f"{explanation}\n\n相关条文：伤寒论第{cid}条：\n{text}"
                samples.append(SFTSample(q, "", a, "经典解释"))

        # 证型解释
        syndrome_defs = [
            ("太阳中风证", 2, "太阳病，发热，汗出，恶风，脉缓者，名为中风。",
             "太阳中风证是外感风邪所致的表虚证，主要表现为发热、汗出、恶风、脉浮缓，以桂枝汤为主方。"),
            ("太阳伤寒证", 3, "太阳病，或已发热，或未发热，必恶寒，体痛，呕逆，脉阴阳俱紧者，名曰伤寒。",
             "太阳伤寒证是外感寒邪所致的表实证，主要表现为恶寒、无汗、身疼痛、脉浮紧，以麻黄汤为主方。"),
            ("蓄水证", 71, "太阳病，发汗后，大汗出，胃中干，烦躁不得眠，欲得饮水者，少少与饮之，令胃气和则愈。若脉浮，小便不利，微热消渴者，五苓散主之。",
             "蓄水证是太阳病邪传膀胱、气化不利、水停下焦所致。主要表现为小便不利、口渴、烦渴欲饮水、水入则吐，以五苓散为主方。"),
            ("蓄血证", 106, "太阳病不解，热结膀胱，其人如狂，血自下，下者愈。其外不解者，尚未可攻，当先解其外；外解已，但少腹急结者，乃可攻之，宜桃核承气汤。",
             "蓄血证是邪热内传、瘀血结于下焦所致。主要表现为少腹急结或硬满、如狂或发狂、小便自利，以桃核承气汤或抵当汤为主方。"),
            ("结胸证", 135, "伤寒六七日，结胸热实，脉沉而紧，心下痛，按之石硬者，大陷胸汤主之。",
             "结胸证是邪热与痰水结于心下所致。主要表现为心下硬满疼痛、按之石硬，分为大结胸（大陷胸汤）和小结胸（小陷胸汤）。"),
            ("痞证", 149, "伤寒五六日，呕而发热者，柴胡汤证具，而以他药下之，柴胡证仍在者，复与柴胡汤。此虽已下之，不为逆，必蒸蒸而振，却发热汗出而解。若心下满而硬痛者，此为结胸也，大陷胸汤主之。但满而不痛者，此为痞，柴胡不中与之，宜半夏泻心汤。",
             "痞证是脾胃不和、寒热错杂所致心下痞满但不痛的证候。主要表现为心下满而不痛，以半夏泻心汤等泻心汤类为主方。"),
        ]

        for name, cid, text, explanation in syndrome_defs:
            for q in [
                f"什么是{name}？",
                f"{name}的表现是什么？",
                f"请解释{name}。",
                f"{name}的病因病机是什么？",
                f"{name}应该用什么方剂治疗？",
            ]:
                a = f"{explanation}\n\n相关条文：伤寒论第{cid}条：\n{text}"
                samples.append(SFTSample(q, "", a, "经典解释"))

        # 从含方剂的条文生成"什么情况下用X汤"
        for cid, formulas in self.clause_formulas.items():
            if cid > 150:
                continue
            clause = self.clause_by_id[cid]
            text = clause["original_text"]
            for formula_name in formulas[:1]:
                f_info = self.formula_dict.get(formula_name)
                if not f_info:
                    continue
                q = random.choice([
                    f"什么情况下应该使用{formula_name}？",
                    f"{formula_name}的适用证候是什么？",
                    f"伤寒论中{formula_name}用于治疗什么？",
                    f"{formula_name}的辨证要点是什么？",
                ])
                a = f"根据伤寒论第{cid}条，{formula_name}的适用情况为：\n{text}"
                if f_info.brief:
                    a += f"\n\n{f_info.brief}。"
                samples.append(SFTSample(q, "", a, "经典解释"))

        return samples

    # ==================== 5. 综合问答 (目标 ~200) ====================

    def gen_comprehensive(self) -> list[SFTSample]:
        samples = []
        fdict = self.formula_dict

        # 方剂对比（组成）
        compare_pairs = [
            ("桂枝汤", "麻黄汤"),
            ("桂枝汤", "桂枝加葛根汤"),
            ("桂枝汤", "桂枝加附子汤"),
            ("大承气汤", "小承气汤"),
            ("大承气汤", "调胃承气汤"),
            ("小承气汤", "调胃承气汤"),
            ("小柴胡汤", "大柴胡汤"),
            ("白虎汤", "白虎加人参汤"),
            ("四逆汤", "通脉四逆汤"),
            ("四逆汤", "真武汤"),
            ("半夏泻心汤", "生姜泻心汤"),
            ("半夏泻心汤", "甘草泻心汤"),
            ("生姜泻心汤", "甘草泻心汤"),
            ("大黄黄连泻心汤", "附子泻心汤"),
            ("大青龙汤", "小青龙汤"),
            ("桂枝汤", "桂枝加芍药汤"),
            ("五苓散", "茯苓甘草汤"),
            ("桃核承气汤", "抵当汤"),
            ("大陷胸汤", "小陷胸汤"),
            ("当归四逆汤", "当归四逆加吴茱萸生姜汤"),
            ("黄芩汤", "黄芩加半夏生姜汤"),
            ("桂枝麻黄各半汤", "桂枝二麻黄一汤"),
            ("白虎汤", "大承气汤"),
            ("理中丸", "四逆汤"),
        ]

        for f1_name, f2_name in compare_pairs:
            f1 = fdict.get(f1_name)
            f2 = fdict.get(f2_name)
            if not f1 or not f2:
                continue

            h1, h2 = set(f1.herbs), set(f2.herbs)
            common = h1 & h2
            only1, only2 = h1 - h2, h2 - h1

            # 组成对比
            q1 = f"{f1_name}和{f2_name}在组成上有什么区别？"
            a1_parts = [f"{f1_name}和{f2_name}的组成对比如下：\n"]
            a1_parts.append(f"{f1_name}组成：{'、'.join(f1.herbs)}\n")
            a1_parts.append(f"{f2_name}组成：{'、'.join(f2.herbs)}\n")
            if common:
                a1_parts.append(f"共同药物：{'、'.join(sorted(common))}\n")
            if only1:
                a1_parts.append(f"{f1_name}独有：{'、'.join(sorted(only1))}\n")
            if only2:
                a1_parts.append(f"{f2_name}独有：{'、'.join(sorted(only2))}")
            samples.append(SFTSample(q1, "", "".join(a1_parts), "综合问答"))

            # 主治对比
            q2 = f"{f1_name}和{f2_name}在主治上有什么区别？"
            a2 = f"{f1_name}主治{f1.syndrome}，{f2_name}主治{f2.syndrome}。\n"
            a2 += f"{f1_name}：{f1.brief}。\n"
            a2 += f"{f2_name}：{f2.brief}。\n"
            a2 += f"出处：{f1_name}见于第{f1.clause_id}条，{f2_name}见于第{f2.clause_id}条。"
            samples.append(SFTSample(q2, "", a2, "综合问答"))

            # 综合对比
            q3 = f"请比较{f1_name}和{f2_name}的异同。"
            a3 = f"{f1_name}与{f2_name}比较：\n"
            a3 += f"1. 组成：{f1_name}含{'、'.join(f1.herbs)}；{f2_name}含{'、'.join(f2.herbs)}。\n"
            a3 += f"2. 主治：{f1_name}主治{f1.syndrome}；{f2_name}主治{f2.syndrome}。\n"
            a3 += f"3. 功效：{f1.brief}；{f2.brief}。\n"
            if common:
                a3 += f"4. 共同药物：{'、'.join(sorted(common))}。"
            samples.append(SFTSample(q3, "", a3, "综合问答"))

        # 证型对比
        syndrome_compare = [
            ("太阳中风证", "太阳伤寒证",
             "太阳中风证（表虚证）：发热、汗出、恶风、脉浮缓，治以桂枝汤调和营卫。",
             "太阳伤寒证（表实证）：恶寒、无汗、身疼痛、脉浮紧，治以麻黄汤发汗解表。",
             "两者鉴别要点在于：有汗无汗、脉缓脉紧。"),
            ("阳明经证", "阳明腑证",
             "阳明经证：大热、大汗、大渴、脉洪大，无腹满便秘，治以白虎汤清热生津。",
             "阳明腑证：潮热、谵语、腹满痛、便秘，治以承气汤类通下热结。",
             "两者区别在于有无燥屎内结。"),
            ("少阴寒化证", "少阴热化证",
             "少阴寒化证：脉微细、但欲寐、四肢厥冷、下利清谷，治以四逆汤回阳救逆。",
             "少阴热化证：心烦不得卧、口燥咽干，治以黄连阿胶汤滋阴清热。",
             "两者区别在于阳气衰微与阴虚火旺。"),
        ]

        for s1, s2, desc1, desc2, key in syndrome_compare:
            q = f"{s1}和{s2}有什么区别？"
            a = f"{s1}和{s2}的区别：\n{desc1}\n{desc2}\n{key}"
            samples.append(SFTSample(q, "", a, "综合问答"))

        return samples

    # ==================== 汇总 ====================

    def generate_all(self) -> list[SFTSample]:
        all_samples = []
        all_samples.extend(self.gen_clause_retrieval())
        all_samples.extend(self.gen_formula_query())
        all_samples.extend(self.gen_herb_association())
        all_samples.extend(self.gen_concept_explanation())
        all_samples.extend(self.gen_comprehensive())
        random.shuffle(all_samples)
        return all_samples

    def save(self, samples: list[SFTSample], output_path: str) -> dict:
        from collections import Counter
        with open(output_path, "w", encoding="utf-8") as f:
            for s in samples:
                f.write(json.dumps(s.to_dict(), ensure_ascii=False) + "\n")
        return {
            "total": len(samples),
            "by_category": dict(Counter(s.category for s in samples)),
        }


if __name__ == "__main__":
    base = Path(__file__).parent.parent.parent
    clauses_path = str(base / "data" / "processed" / "classics" / "shanghan_clauses.jsonl")
    output_path = str(base / "data" / "processed" / "sft_train_p1.jsonl")

    gen = SFTGeneratorV2(clauses_path)
    samples = gen.generate_all()
    stats = gen.save(samples, output_path)

    print("=== SFT 训练数据生成完成 (v2) ===")
    print(f"输出: {output_path}")
    print(f"总样本数: {stats['total']}")
    print(f"\n按类别分布:")
    for cat, cnt in sorted(stats["by_category"].items(), key=lambda x: -x[1]):
        pct = cnt / stats["total"] * 100
        print(f"  {cat}: {cnt} ({pct:.1f}%)")

    # 评测集分布对比
    print(f"\n评测集分布对比:")
    eval_dist = {"经典原文检索": 30, "方剂查询": 20, "药材关联": 20, "经典解释": 20, "综合问答": 10}
    for cat in ["经典原文检索", "方剂查询", "药材关联", "经典解释", "综合问答"]:
        train_pct = stats["by_category"].get(cat, 0) / stats["total"] * 100
        eval_pct = eval_dist[cat]
        print(f"  {cat}: 训练{train_pct:.1f}% vs 评测{eval_pct}%")
