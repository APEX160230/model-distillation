"""SFT 数据格式化模块

将倪海厦讲稿文本转为 instruction-output 对，用于 LoRA 微调。
采用规则法（PRD §15 决策：先试规则，质量不够再上 GPT）。
"""
import re
import json
from dataclasses import dataclass


@dataclass
class SFTSample:
    """单条 SFT 训练样本"""
    instruction: str
    input: str
    output: str


# 方剂编号正则：匹配行首 "15 桂枝汤" 格式
_FORMULA_TITLE_RE = re.compile(r'^\d{1,3}\s+(.+?汤)', re.MULTILINE)

# 条辨编号正则：匹配 "一：" "二：" 格式（伤寒论原文格式）和 "第X条" 格式
_CLAUSE_RE = re.compile(r'^([一二三四五六七八九十百零]{1,4})[：:]', re.MULTILINE)
_CLAUSE_ALT_RE = re.compile(r'第([一二三四五六七八九十百零\d]+)条')

# 六经病名
_SIX_MERIDIANS = ["太阳病", "阳明病", "少阳病", "太阴病", "少阴病", "厥阴病"]

# 常见药材
_HERBS = [
    "桂枝", "麻黄", "芍药", "甘草", "生姜", "大枣", "人参", "黄芩",
    "半夏", "干姜", "附子", "白术", "茯苓", "大黄", "芒硝", "杏仁",
    "石膏", "知母", "柴胡", "葛根", "黄连", "细辛", "五味子",
]

# 方剂名匹配（用于检测段落中是否讨论方剂，不含"小""大"等宽泛前缀）
_FORMULA_MENTION_RE = re.compile(
    r'((?:桂枝|麻黄|葛根|柴胡|白虎|承气|四逆|青龙|陷胸|泻心|理中|真武|附子|'
    r'茯苓|甘草|芍药|栀子|茵陈|黄连|黄芩|半夏|橘皮|枳实|厚朴|猪苓|桃核|'
    r'抵挡|乌梅|旋覆|当归|牡蛎|龙骨)'
    r'(?:加|去|二|各)?)'
    r'汤'
)


def detect_content_type(text: str) -> str:
    """检测段落内容类型

    优先级：条文标题 > 方剂标题 > 证型 > 药材 > 通用
    判断依据是段落开头格式，而非文中任意位置出现的关键词。
    """
    # 条辨编号在段首
    if _CLAUSE_RE.match(text.strip()) or _CLAUSE_ALT_RE.search(text[:20]):
        return "条文"

    # 方剂编号在段首（如 "15 桂枝汤"）
    if _FORMULA_TITLE_RE.match(text.strip()):
        return "方剂"

    # 六经病名在段首或前 50 字内
    for meridian in _SIX_MERIDIANS:
        if meridian in text[:50] and ("证" in text[:100] or "病" in text[:100]):
            return "证型"

    # 药材在前 50 字内
    for herb in _HERBS:
        if herb in text[:50]:
            return "药材"

    return "通用"


def generate_instruction(text: str) -> str:
    """根据段落内容生成指令"""
    content_type = detect_content_type(text)

    if content_type == "条文":
        # 中文数字条辨编号
        match = _CLAUSE_RE.match(text.strip())
        if match:
            num = match.group(1)
            return f"请解释伤寒论第{num}条的原文含义。"
        # 第X条格式
        match = _CLAUSE_ALT_RE.search(text[:20])
        if match:
            num = match.group(1)
            return f"请解释伤寒论第{num}条的原文含义。"

    if content_type == "方剂":
        # 方剂编号标题（如 "15 桂枝汤"）
        match = _FORMULA_TITLE_RE.match(text.strip())
        if match:
            formula = match.group(1)
            return f"请解释{formula}的组成和主治。"

    if content_type == "证型":
        for meridian in _SIX_MERIDIANS:
            if meridian in text[:50]:
                return f"请解释{meridian}的证候特点和治疗原则。"

    if content_type == "药材":
        found_herbs = [h for h in _HERBS if h in text[:50]]
        if found_herbs:
            return f"请介绍{found_herbs[0]}在伤寒论中的应用。"

    # 默认：取第一句作为主题
    first_sentence = re.split(r'[。！？\n]', text)[0]
    if len(first_sentence) > 5:
        return f"请讲解以下中医知识：{first_sentence[:30]}。"
    return "请讲解相关的中医知识。"


def _split_long_paragraph(text: str, max_length: int = 800) -> list[str]:
    """将过长段落按句子切分为多个子段落

    讲稿 PDF 每页可能是连续文本（无空行），需要按句号切分。
    """
    if len(text) <= max_length:
        return [text]

    # 按句号切分，保留句号
    sentences = re.split(r'(?<=。)', text)
    chunks = []
    current = ""
    for sent in sentences:
        if len(current) + len(sent) > max_length and current:
            chunks.append(current.strip())
            current = sent
        else:
            current += sent
    if current.strip():
        chunks.append(current.strip())
    return chunks


def format_sft_pairs(
    text: str,
    min_length: int = 50,
    max_length: int = 800,
) -> list[SFTSample]:
    """将讲稿文本转为 SFT 样本

    按段落切分，对过长段落按句子进一步切分，生成 instruction-output 对。
    过短段落丢弃，过长段落截断。
    """
    # 先按双换行切分段落，再对每个段落按句子切分
    raw_paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    paragraphs = []
    for para in raw_paragraphs:
        paragraphs.extend(_split_long_paragraph(para, max_length))

    samples = []
    for para in paragraphs:
        if len(para) < min_length:
            continue
        if len(para) > max_length:
            para = para[:max_length]

        instruction = generate_instruction(para)
        samples.append(SFTSample(
            instruction=instruction,
            input="",
            output=para,
        ))

    return samples


def filter_and_dedup(samples: list[SFTSample]) -> list[SFTSample]:
    """过滤和去重

    基于 output 前 100 字去重，避免高度相似样本。
    """
    seen = set()
    filtered = []
    for s in samples:
        key = s.output[:100]
        if key in seen:
            continue
        seen.add(key)
        filtered.append(s)
    return filtered


def save_jsonl(samples: list[SFTSample], output_path: str) -> int:
    """保存为 JSONL 格式（Alpaca 格式）"""
    with open(output_path, 'w', encoding='utf-8') as f:
        for s in samples:
            row = {
                "instruction": s.instruction,
                "input": s.input,
                "output": s.output,
            }
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
    return len(samples)
