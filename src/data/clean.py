"""文本清洗模块

去除 PDF 提取文本中的页眉页脚、水印、推广信息，并按方剂名切分章节。
"""
import re
from dataclasses import dataclass

from src.data.extract import Page


@dataclass
class CleanedChapter:
    """切分后的章节"""
    title: str
    content: str
    source_pages: list[int]


# 噪声正则模式
NOISE_PATTERNS = [
    r'www\.qhsxzh\.com',
    r'https?://[^\s]+',
    r'微信公众号[:：]?\s*\S*',
    r'关注公众号[:：]?\s*\S*',
    r'岐黄传承道法自然',
    r'伤寒V\d+',
    r'电子文稿[，,].*',
    r'公益传播[，,].*',
    r'—\s*\d+\s*—',
    r'-\s*\d+\s*-',
    r'第\s*\d+\s*页',
    r'^\s*\d+\s*$',
    r'版权所有',
    r'翻版必究',
    # 目录行：含连续点号和页码
    r'^.*\.{3,}\d+\s*$',
]

# 章节标题正则：匹配行首的条辨编号或方剂编号
# 真实文本格式： "一：太阳之为病..." 或 "15 桂枝汤" 或 "164 半夏泻心汤"
_CHAPTER_RE = re.compile(
    r'^(?:'
    r'(?:[一二三四五六七八九十百零]{1,4})[：:]'  # 中文数字条辨编号
    r'|\d{1,3}\s+[\u4e00-\u9fff].*?汤'  # 编号+方剂名
    r')',
    re.MULTILINE,
)


def remove_noise(text: str) -> str:
    """移除页眉页脚、水印、推广信息"""
    for pattern in NOISE_PATTERNS:
        text = re.sub(pattern, '', text, flags=re.MULTILINE)
    return text


def clean_page(text: str) -> str:
    """清洗单页文本：去噪 + 合并空行 + 去行首尾空格"""
    text = remove_noise(text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    lines = [line.strip() for line in text.split('\n')]
    return '\n'.join(lines).strip()


def clean_pages(pages: list[Page]) -> list[Page]:
    """批量清洗页面，返回新的 Page 列表"""
    return [Page(number=p.number, text=clean_page(p.text)) for p in pages]


def segment_by_formula(text: str) -> list[CleanedChapter]:
    """按条辨编号或方剂编号切分文本为章节

    真实文本中条辨用"一："格式，方剂用"15 桂枝汤"格式。
    只匹配行首的编号格式，避免正文中提到方剂名时误切。
    """
    if not text.strip():
        return []

    matches = list(_CHAPTER_RE.finditer(text))
    if len(matches) == 0:
        return [CleanedChapter(title="未分类", content=text.strip(), source_pages=[])]

    chapters = []
    for i, match in enumerate(matches):
        title = match.group(0).strip()
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        content = text[start:end].strip()
        if content:
            chapters.append(CleanedChapter(
                title=title,
                content=content,
                source_pages=[],
            ))
    return chapters
