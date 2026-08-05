"""PDF 批量提取模块

从倪海厦讲稿 PDF 中提取文本，生成结构化结果和质量报告。
基于 Day 1 验证脚本 data/extract_test.py 的成熟逻辑重构。
"""
from dataclasses import dataclass, field
from pathlib import Path
import re
import json

import fitz  # PyMuPDF


@dataclass
class Page:
    """单页提取结果"""
    number: int
    text: str


@dataclass
class ExtractionResult:
    """单个 PDF 的提取结果"""
    pdf_path: str
    pdf_name: str
    page_count: int
    pages: list[Page] = field(default_factory=list)


# 中医关键词（用于质量检测）
TCM_KEYWORDS = [
    "桂枝", "麻黄", "伤寒", "太阳", "阳明", "少阳",
    "太阴", "少阴", "厥阴", "芍药", "甘草", "生姜",
    "大枣", "人参", "黄芩", "半夏", "干姜", "附子",
    "白术", "茯苓", "大黄", "芒硝", "杏仁", "石膏",
]

# 正常字符范围（中文 + 全角标点 + ASCII 可见 + 通用标点）
_NORMAL_CHAR_RE = re.compile(
    r'[\u4e00-\u9fff\u3000-\u303f\uff00-\uffef\u2010-\u2027a-zA-Z0-9\s\.\,\;\:\!\?\(\)\[\]\{\}\"\'\-\/\%\+\u00C0-\u024F]'
)


def extract_pdf(pdf_path: str) -> ExtractionResult:
    """提取单个 PDF 的全部页面文本"""
    path = Path(pdf_path)
    doc = fitz.open(str(path))
    pages = [
        Page(number=i + 1, text=doc[i].get_text("text"))
        for i in range(len(doc))
    ]
    result = ExtractionResult(
        pdf_path=str(path),
        pdf_name=path.name,
        page_count=len(doc),
        pages=pages,
    )
    doc.close()
    return result


def extract_all_pdfs(pdf_dir: str) -> dict[str, ExtractionResult]:
    """批量提取目录下所有 PDF"""
    pdf_paths = sorted(Path(pdf_dir).glob("*.pdf"))
    if not pdf_paths:
        raise FileNotFoundError(f"目录 {pdf_dir} 下没有 PDF 文件")
    return {p.name: extract_pdf(str(p)) for p in pdf_paths}


def generate_quality_report(
    results: dict[str, ExtractionResult], output_path: str
) -> dict:
    """生成提取质量报告"""
    report: dict = {"files": {}}
    for name, result in results.items():
        all_text = "\n".join(p.text for p in result.pages)
        found = [kw for kw in TCM_KEYWORDS if kw in all_text]
        missing = [kw for kw in TCM_KEYWORDS if kw not in all_text]
        normal_count = len(_NORMAL_CHAR_RE.findall(all_text))
        garbage_ratio = 1 - (normal_count / max(len(all_text), 1))
        report["files"][name] = {
            "pages": result.page_count,
            "total_chars": len(all_text),
            "avg_chars_per_page": round(len(all_text) / max(result.page_count, 1)),
            "keyword_hits": f"{len(found)}/{len(TCM_KEYWORDS)}",
            "missing_keywords": missing,
            "garbage_ratio": round(garbage_ratio, 4),
            "verdict": "pass" if len(found) >= 20 and garbage_ratio < 0.01 else "warning",
        }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    return report
