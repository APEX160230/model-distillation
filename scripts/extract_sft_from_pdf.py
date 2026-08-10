"""
PDF -> SFT 训练数据提取管道

将倪海厦人纪系列 5 本 PDF 讲稿提取为符合 PRD §4.3 要求的 instruction-output 格式 SFT 数据。
保留倪师口语化讲解风格（"什么意思呢""你想想看""诸位"等）。

流程: PDF全文提取 -> 清洗去噪 -> 主题分段 -> 指令生成 -> 质量过滤 -> 输出JSONL
"""

import fitz  # PyMuPDF
import json
import re
import os
import sys
from pathlib import Path
from collections import Counter

# ============================================================
# 配置
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PDF_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 5 本人纪 PDF
PDF_FILES = {
    "针灸": "人纪01针灸2026.4.pdf",
    "内经": "人纪02内经2026.4.pdf",
    "本草": "人纪03本草2026.4.pdf",
    "伤寒": "人纪04伤寒.pdf",
    "金匮": "人纪05金匮2026.4.pdf",
}

# 需要清洗的页眉模式
HEADER_PATTERNS = [
    r"微信公众号：岐黄圣贤智慧",
    r"岐黄传承道法自然",
    r"电子文稿，仅限自用，公益传播，请勿商用。",
    r"V\d{6}\s*版",
    r"人纪\w+V\d+",
    r"伤寒V\d+",
    r"金匮V\d+",
    r"本草V\d+",
    r"针灸V\d+",
    r"内经V\d+",
]

# 非教学内容模式（需要过滤的噪音段落）
NOISE_PATTERNS = [
    r"^www\.qhsxzh\.com",
    r"^微信公众号",
    r"^岐黄传承",
    r"^电子文稿",
    r"QR码|二维码",
    r"^V\d{6}",
    r"^\d+$",  # 纯页码
]

# 条文编号模式（伤寒论/金匮要略的条辨编号）
CLAUSE_PATTERN = re.compile(
    r"^([一二三四五六七八九十百千]+)[：:]\s*(.+)"
)

# 集数标记（视频集数）
EPISODE_PATTERN = re.compile(r"\[第(\d+)\s*集\]")

# 口语化风格标记
ORAL_MARKERS = [
    "什么意思呢", "你想想看", "诸位", "就是说", "所以说", "好不好",
    "简单讲", "实际上", "就这样子", "你错了", "我告诉", "临床",
    "我们在", "为什么", "意思就是", "对不对", "了解", "知道",
    "这个方子", "这个条辨", "这个观念", "这个意思", "这个道理",
    "经方", "张仲景", "倪师", "处方",
]

# 方剂名（从 formulas_db 关键方剂提取）
FORMULAS = [
    "桂枝汤", "麻黄汤", "葛根汤", "小柴胡汤", "大柴胡汤",
    "白虎汤", "白虎加人参汤", "承气汤", "调胃承气汤", "小承气汤", "大承气汤",
    "理中丸", "四逆汤", "真武汤", "附子汤", "桃花汤",
    "黄连阿胶汤", "猪苓汤", "五苓散", "茵陈蒿汤",
    "麻杏石甘汤", "葛根黄芩黄连汤", "黄芩汤",
    "桂枝加葛根汤", "桂枝加附子汤", "桂枝麻黄各半汤",
    "小建中汤", "炙甘草汤", "茯苓四逆汤",
    "乌梅丸", "麻黄升麻汤", "干姜附子汤", "茯苓桂枝白术甘草汤",
    "桂枝去桂加茯苓白术汤", "厚朴生姜半夏甘草人参汤",
    "大青龙汤", "小青龙汤", "桂枝二麻黄一汤",
    "文蛤散", "三物白散", "十枣汤", "瓜蒂散",
    "旋覆代赭汤", "桂枝人参汤", "白虎加桂枝汤",
    "新加汤", "桂枝加芍药生姜各一两人参三两新加汤",
    "葛根加半夏汤", "桂枝加厚朴杏子汤",
]

# 证型/病名
SYNDROMES = [
    "太阳病", "阳明病", "少阳病", "太阴病", "少阴病", "厥阴病",
    "太阳中风", "太阳伤寒", "太阳温病", "中风", "伤寒", "温病",
    "风温", "结胸", "痞证", "蓄水证", "蓄血证",
    "脏结", "热入血室", "合病", "并病",
]

# 药材名
HERBS = [
    "桂枝", "麻黄", "芍药", "甘草", "生姜", "大枣",
    "葛根", "柴胡", "黄芩", "半夏", "人参", "干姜",
    "附子", "茯苓", "白术", "黄连", "大黄", "芒硝",
    "杏仁", "石膏", "知母", "粳米", "厚朴", "枳实",
    "当归", "川芎", "生地黄", "阿胶", "鸡子黄",
    "猪苓", "泽泻", "滑石", "茵陈", "栀子", "黄柏",
    "细辛", "五味子", "桂枝", "防风", "羌活",
]

# 中医概念
CONCEPTS = [
    "营卫不和", "传经", "六经", "辨证", "经方", "时方",
    "阴阳", "表里", "寒热", "虚实", "脉象",
    "恶寒", "恶风", "发热", "汗出", "无汗", "头痛",
    "体痛", "呕逆", "下利", "烦躁", "口渴", "心下痞",
    "但热不寒", "往来寒热", "四肢厥逆",
]


# ============================================================
# Step 1: PDF 全文提取
# ============================================================

def extract_pdf_text(pdf_path: str) -> list[dict]:
    """提取 PDF 全文，返回每页文本的列表"""
    doc = fitz.open(pdf_path)
    pages = []
    for i in range(len(doc)):
        text = doc[i].get_text("text")
        pages.append({"page": i + 1, "text": text})
    doc.close()
    return pages


# ============================================================
# Step 2: 清洗去噪
# ============================================================

def clean_text(text: str) -> str:
    """清洗单页文本：去除页眉、页脚、噪音行"""
    lines = text.split("\n")
    cleaned = []
    for line in lines:
        stripped = line.strip()
        # 跳过空行
        if not stripped:
            continue
        # 跳过页眉模式
        if any(re.search(p, stripped) for p in HEADER_PATTERNS):
            continue
        # 跳过噪音模式
        if any(re.search(p, stripped) for p in NOISE_PATTERNS):
            continue
        cleaned.append(stripped)
    return "\n".join(cleaned)


def clean_pages(pages: list[dict]) -> str:
    """清洗所有页面并合并为连续文本"""
    cleaned_parts = []
    for page in pages:
        text = clean_text(page["text"])
        if text.strip():
            cleaned_parts.append(text)
    return "\n\n".join(cleaned_parts)


# ============================================================
# Step 3: 主题分段
# ============================================================

def segment_text(text: str, book_name: str) -> list[dict]:
    """
    将连续文本按主题切分为段落。
    切分策略:
    1. 优先在条文编号处切分（伤寒/金匮）
    2. 在集数标记处切分
    3. 话题转换词处切分（"我们再来看""接下来""现在讲"等）
    4. 超长段落（>600字）在句子边界处二次切分
    """
    # 话题转换词（在这些词前面切分）
    TRANSITION_PATTERNS = [
        r"我们再来看", r"接下来", r"现在讲", r"现在我们",
        r"诸位[现在，]", r"好，[那这个]", r"这个条辨",
        r"再来[看是]", r"下面[这个看]", r"然后[我我们这]",
    ]
    
    # 句子结束标记
    SENTENCE_END = re.compile(r"[。！？\n]")
    
    segments = []
    
    # 将文本按行分割（每行是一个潜在的切分单元）
    all_lines = [l.strip() for l in text.split("\n") if l.strip()]
    
    # 将行分组为段落
    paragraphs = []
    current_para = []
    current_para_len = 0
    
    for line in all_lines:
        # 检查是否是切分点
        clause_match = CLAUSE_PATTERN.match(line)
        episode_match = EPISODE_PATTERN.search(line)
        is_transition = any(re.search(p, line[:30]) for p in TRANSITION_PATTERNS)
        
        # 在切分点处结束当前段落
        if (clause_match or episode_match or is_transition) and current_para_len >= 50:
            paragraphs.append("\n".join(current_para))
            current_para = []
            current_para_len = 0
        
        current_para.append(line)
        current_para_len += len(line)
        
        # 段落达到目标长度时结束
        if current_para_len >= 500:
            paragraphs.append("\n".join(current_para))
            current_para = []
            current_para_len = 0
    
    # 最后一段
    if current_para and current_para_len >= 50:
        paragraphs.append("\n".join(current_para))
    
    # 二次切分：仍然过长的段落按句子边界切分
    final_segments = []
    for para in paragraphs:
        if len(para) <= 800:
            final_segments.append(para)
        else:
            # 按句子切分
            sentences = re.split(r"(?<=[。！？])", para)
            current_chunk = []
            current_len = 0
            for sent in sentences:
                if not sent.strip():
                    continue
                current_chunk.append(sent)
                current_len += len(sent)
                if current_len >= 500:
                    final_segments.append("".join(current_chunk))
                    current_chunk = []
                    current_len = 0
            if current_chunk and current_len >= 80:
                final_segments.append("".join(current_chunk))
    
    # 转换为 segment dict
    segments = []
    for seg_text in final_segments:
        seg_text = seg_text.strip()
        if len(seg_text) >= 80:
            segments.append({
                "text": seg_text,
                "book": book_name,
                "char_count": len(seg_text),
            })
    
    return segments


# ============================================================
# Step 4: 指令生成
# ============================================================

def generate_instruction(segment: dict) -> str:
    """
    根据段落内容关键词生成自然的问题（instruction）。
    
    关键改进：只看段落前 400 字来确定主题，避免段落中顺带提到的关键词干扰。
    优先级:
    1. 条文编号 → "请解释伤寒论第X条的含义。"
    2. 方剂名 → "请讲解{方剂}的组成和主治。"
    3. 证型名 → "什么是{证型}？其病机和临床表现是什么？"
    4. 药材名 → "请讲解{药材}的功效和临床应用。"
    5. 症状/概念 → "请解释{概念}的含义和病机。"
    6. 默认 → 根据内容主题生成通用问题
    """
    text = segment["text"]
    book = segment["book"]
    # 只看前 400 字确定主题
    head = text[:400]
    
    # 1. 条文编号
    clause_match = CLAUSE_PATTERN.match(text.split("\n")[0])
    if clause_match:
        clause_num = clause_match.group(1)
        clause_text = clause_match.group(2)[:60]
        book_ref = "伤寒论" if book == "伤寒" else "金匮要略" if book == "金匮" else "中医经典"
        return f"请解释{book_ref}第{clause_num}条：「{clause_text}」"
    
    # 2. 方剂名（在前 400 字中找）
    for formula in FORMULAS:
        if formula in head:
            if any(kw in head for kw in ["组成", "由", "药物", "配方"]):
                return f"请讲解{formula}的组成和主治。"
            elif any(kw in head for kw in ["临床", "用", "治", "证"]):
                return f"{formula}在临床上如何应用？主治什么证？"
            else:
                return f"请讲解{formula}的组方原理和临床应用。"
    
    # 3. 证型/病名（在前 400 字中找）
    for syndrome in SYNDROMES:
        if syndrome in head:
            if any(kw in head for kw in ["病机", "原因", "为什么", "什么意思"]):
                return f"什么是{syndrome}？其病机是什么？"
            elif any(kw in head for kw in ["症状", "表现", "临床", "证"]):
                return f"{syndrome}的临床表现和辨证要点是什么？"
            else:
                return f"请讲解{syndrome}的病机和治疗原则。"
    
    # 4. 药材名（在前 400 字中找）
    for herb in HERBS:
        if herb in head:
            if any(kw in head for kw in ["功效", "作用", "功能"]):
                return f"请讲解{herb}的功效和临床应用。"
            elif any(kw in head for kw in ["配伍", "搭配", "合"]):
                return f"{herb}在经方中常与哪些药材配伍？"
            else:
                return f"请讲解{herb}在经方中的应用。"
    
    # 5. 症状/概念（在前 400 字中找）
    for concept in CONCEPTS:
        if concept in head:
            return f"请解释{concept}的含义和临床意义。"
    
    # 6. 默认：根据书本主题生成
    book_topics = {
        "针灸": "针灸经络和穴位知识",
        "内经": "黄帝内经的理论",
        "本草": "中药本草知识",
        "伤寒": "伤寒论的辨证论治",
        "金匮": "金匮要略的杂病辨治",
    }
    topic = book_topics.get(book, "中医知识")
    return f"请讲解以下{topic}。"


# ============================================================
# Step 5: 质量过滤
# ============================================================

def is_quality_segment(segment: dict) -> bool:
    """质量过滤：判断段落是否适合作为 SFT 训练数据"""
    text = segment["text"]
    
    # 太短（< 100字）—— 信息量不足
    if len(text) < 100:
        return False
    
    # 太长（> 2000字）—— 超出 1.5B 模型生成能力
    if len(text) > 2000:
        return False
    
    # 目录/索引页检测：大量短行包含"第X 集"或"条辨"编号
    lines = text.split("\n")
    toc_lines = sum(1 for l in lines if re.match(r"^第\d+\s*集$", l.strip()) or re.match(r"^条辨\d", l.strip()))
    if toc_lines > 5:
        return False
    
    # 纯条文无讲解（只有原文没有倪师讲解）
    if len(lines) <= 2 and not any(m in text for m in ORAL_MARKERS):
        return False
    
    # 噪音内容（版权、广告等）
    noise_keywords = ["版权", "购买", "扫码", "微信扫码", "打印店", "客服", "目录", "视频目录"]
    if any(kw in text for kw in noise_keywords):
        return False
    
    # 必须包含至少一个中医关键词
    tcm_keywords = FORMULAS + SYNDROMES + HERBS + CONCEPTS + [
        "伤寒", "金匮", "经方", "辨证", "处方", "条辨", "脉",
        "汤", "丸", "散", "证", "病", "药", "穴", "经", "络",
    ]
    if not any(kw in text for kw in tcm_keywords):
        return False
    
    return True


def count_oral_markers(text: str) -> int:
    """统计口语化标记数量"""
    return sum(1 for m in ORAL_MARKERS if m in text)


# ============================================================
# Step 6: 主流程
# ============================================================

def process_pdf(pdf_path: str, book_name: str) -> list[dict]:
    """处理单本 PDF，返回 SFT 数据列表"""
    print(f"\n{'='*60}")
    print(f"处理: {book_name} ({os.path.basename(pdf_path)})")
    print(f"{'='*60}")
    
    # Step 1: 提取
    pages = extract_pdf_text(pdf_path)
    print(f"  提取: {len(pages)} 页")
    
    # Step 2: 清洗
    text = clean_pages(pages)
    char_count = len(text)
    print(f"  清洗后: {char_count} 字符")
    
    # Step 3: 分段
    segments = segment_text(text, book_name)
    print(f"  分段: {len(segments)} 段")
    
    # Step 4+5: 生成指令 + 质量过滤
    sft_data = []
    oral_count = 0
    for seg in segments:
        if not is_quality_segment(seg):
            continue
        instruction = generate_instruction(seg)
        oral_markers = count_oral_markers(seg["text"])
        if oral_markers > 0:
            oral_count += 1
        
        sft_data.append({
            "instruction": instruction,
            "input": "",
            "output": seg["text"],
            "book": book_name,
            "char_count": seg["char_count"],
            "oral_markers": oral_markers,
        })
    
    print(f"  质量过滤后: {len(sft_data)} 条")
    print(f"  含口语标记: {oral_count}/{len(sft_data)} ({oral_count/max(len(sft_data),1)*100:.0f}%)")
    
    # 长度统计
    if sft_data:
        lengths = [d["char_count"] for d in sft_data]
        print(f"  长度: min={min(lengths)}, max={max(lengths)}, avg={sum(lengths)//len(lengths)}")
    
    return sft_data


def main():
    # 支持单本测试或全量运行
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        # 测试模式：只处理伤寒前 20 页
        print("=== 测试模式：伤寒前 20 页 ===")
        pdf_path = str(PDF_DIR / PDF_FILES["伤寒"])
        pages = extract_pdf_text(pdf_path)[:20]
        text = clean_pages(pages)
        segments = segment_text(text, "伤寒")
        
        sft_data = []
        for seg in segments:
            if not is_quality_segment(seg):
                continue
            instruction = generate_instruction(seg)
            sft_data.append({
                "instruction": instruction,
                "input": "",
                "output": seg["text"],
                "book": "伤寒",
                "char_count": seg["char_count"],
                "oral_markers": count_oral_markers(seg["text"]),
            })
        
        print(f"\n产出: {len(sft_data)} 条 SFT 数据")
        print(f"含口语标记: {sum(1 for d in sft_data if d['oral_markers']>0)}/{len(sft_data)}")
        
        # 展示样本
        print("\n=== 样本展示 ===")
        for i, d in enumerate(sft_data[:5]):
            print(f"\n--- 样本 {i+1} ---")
            print(f"instruction: {d['instruction']}")
            print(f"oral_markers: {d['oral_markers']}")
            print(f"char_count: {d['char_count']}")
            print(f"output (前300字): {d['output'][:300]}...")
        
        # 写入测试输出
        test_output = OUTPUT_DIR / "sft_test_sample.jsonl"
        with open(test_output, "w", encoding="utf-8") as f:
            for d in sft_data:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
        print(f"\n测试样本写入: {test_output}")
        return
    
    # 全量模式：处理所有 5 本
    print("=" * 60)
    print("全量提取 5 本人纪 PDF 的 SFT 数据")
    print("=" * 60)
    
    all_data = []
    stats = {}
    for book_name, pdf_file in PDF_FILES.items():
        pdf_path = str(PDF_DIR / pdf_file)
        if not os.path.exists(pdf_path):
            print(f"\n跳过: {book_name} (文件不存在: {pdf_file})")
            continue
        data = process_pdf(pdf_path, book_name)
        all_data.extend(data)
        stats[book_name] = {
            "count": len(data),
            "oral_pct": sum(1 for d in data if d["oral_markers"] > 0) / max(len(data), 1) * 100,
        }
    
    # 去重（基于 output 前 100 字的相似度）
    seen = set()
    unique_data = []
    for d in all_data:
        key = d["output"][:100]
        if key not in seen:
            seen.add(key)
            unique_data.append(d)
    
    print(f"\n{'='*60}")
    print(f"总计: {len(all_data)} 条 -> 去重后: {len(unique_data)} 条")
    print(f"{'='*60}")
    
    # 统计
    print("\n各书统计:")
    for book, s in stats.items():
        print(f"  {book}: {s['count']} 条, 口语标记 {s['oral_pct']:.0f}%")
    
    oral_total = sum(1 for d in unique_data if d["oral_markers"] > 0)
    print(f"\n口语标记覆盖率: {oral_total}/{len(unique_data)} ({oral_total/max(len(unique_data),1)*100:.0f}%)")
    
    # 长度分布
    lengths = [d["char_count"] for d in unique_data]
    if lengths:
        print(f"长度: min={min(lengths)}, max={max(lengths)}, avg={sum(lengths)//len(lengths)}")
        print(f"  <200字: {sum(1 for l in lengths if l<200)}")
        print(f"  200-500字: {sum(1 for l in lengths if 200<=l<500)}")
        print(f"  500-800字: {sum(1 for l in lengths if 500<=l<800)}")
        print(f"  800-1200字: {sum(1 for l in lengths if 800<=l<1200)}")
        print(f"  >1200字: {sum(1 for l in lengths if l>=1200)}")
    
    # 输出 JSONL
    output_path = OUTPUT_DIR / "sft_train_pdf.jsonl"
    with open(output_path, "w", encoding="utf-8") as f:
        for d in unique_data:
            # 移除内部统计字段，只保留 SFT 格式字段
            clean_d = {
                "instruction": d["instruction"],
                "input": d["input"],
                "output": d["output"],
                "book": d["book"],
            }
            f.write(json.dumps(clean_d, ensure_ascii=False) + "\n")
    
    print(f"\n输出: {output_path}")
    
    # 数据统计报告
    report = {
        "total_entries": len(unique_data),
        "by_book": stats,
        "oral_marker_coverage": f"{oral_total}/{len(unique_data)} ({oral_total/max(len(unique_data),1)*100:.1f}%)",
        "length_stats": {
            "min": min(lengths) if lengths else 0,
            "max": max(lengths) if lengths else 0,
            "avg": sum(lengths)//len(lengths) if lengths else 0,
        },
        "length_distribution": {
            "<200": sum(1 for l in lengths if l < 200),
            "200-500": sum(1 for l in lengths if 200 <= l < 500),
            "500-800": sum(1 for l in lengths if 500 <= l < 800),
            "800-1200": sum(1 for l in lengths if 800 <= l < 1200),
            ">1200": sum(1 for l in lengths if l >= 1200),
        },
    }
    report_path = OUTPUT_DIR / "sft_pdf_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"报告: {report_path}")


if __name__ == "__main__":
    main()
