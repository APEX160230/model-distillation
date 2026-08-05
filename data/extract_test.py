"""
Day 1 可行性验证脚本：PDF 下载 + 文本提取质量检测

测试目标：
1. 从 qhsxzh.com 下载倪注伤寒论 PDF
2. 用 PyMuPDF 和 pdfplumber 提取文本
3. 检查中医文本提取质量（药名、方剂、古文断句）
"""

import os
import sys
import time
import json

import requests

# 路径配置
DATA_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(DATA_DIR, "raw")
os.makedirs(RAW_DIR, exist_ok=True)

# 下载源
PDF_URL = "https://www.qhsxzh.com/wp-content/uploads/2026/05/人纪04伤寒2026.4.pdf"
PDF_PATH = os.path.join(RAW_DIR, "人纪04伤寒.pdf")

WIKISOURCE_URL = "https://zh.wikisource.org/zh/傷寒論"
WIKISOURCE_PATH = os.path.join(RAW_DIR, "伤寒论_原文.txt")


def download_pdf():
    """下载倪注伤寒论 PDF"""
    print(f"[1/4] 下载 PDF: {PDF_URL}")
    print(f"      保存到: {PDF_PATH}")

    if os.path.exists(PDF_PATH) and os.path.getsize(PDF_PATH) > 1_000_000:
        print(f"      已存在 ({os.path.getsize(PDF_PATH) / 1024 / 1024:.1f} MB)，跳过下载")
        return True

    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept": "application/pdf,*/*",
        }
        resp = requests.get(PDF_URL, headers=headers, timeout=60, stream=True)
        resp.raise_for_status()

        total = int(resp.headers.get("content-length", 0))
        print(f"      文件大小: {total / 1024 / 1024:.1f} MB")

        downloaded = 0
        with open(PDF_PATH, "wb") as f:
            for chunk in resp.iter_content(chunk_size=65536):
                f.write(chunk)
                downloaded += len(chunk)

        print(f"      下载完成: {downloaded / 1024 / 1024:.1f} MB")
        return True

    except Exception as e:
        print(f"      下载失败: {e}")
        return False


def fetch_wikisource_text():
    """从维基文库获取伤寒论原文"""
    print(f"\n[2/4] 获取维基文库伤寒论原文")

    if os.path.exists(WIKISOURCE_PATH) and os.path.getsize(WIKISOURCE_PATH) > 10000:
        print(f"      已存在 ({os.path.getsize(WIKISOURCE_PATH)} bytes)，跳过")
        return True

    try:
        from bs4 import BeautifulSoup

        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        resp = requests.get(WIKISOURCE_URL, headers=headers, timeout=30)
        resp.raise_for_status()
        resp.encoding = "utf-8"

        soup = BeautifulSoup(resp.text, "html.parser")

        # 维基文库的正文内容在 mw-parser-output div 里
        content = soup.find("div", {"class": "mw-parser-output"})
        if not content:
            print("      未找到正文内容 div")
            return False

        # 提取纯文本
        text = content.get_text(separator="\n", strip=True)

        # 清理维基文库的导航/页脚等
        lines = text.split("\n")
        clean_lines = []
        skip_sections = ["导航", "页脚", "编辑", "查看", "last edited", "Wikisource"]
        for line in lines:
            if any(s in line.lower() for s in [s.lower() for s in skip_sections]):
                continue
            if line.strip():
                clean_lines.append(line)

        clean_text = "\n".join(clean_lines)

        with open(WIKISOURCE_PATH, "w", encoding="utf-8") as f:
            f.write(clean_text)

        print(f"      获取成功: {len(clean_text)} 字符, {len(clean_lines)} 行")
        print(f"      保存到: {WIKISOURCE_PATH}")
        return True

    except Exception as e:
        print(f"      获取失败: {e}")
        return False


def extract_pdf_pymupdf():
    """用 PyMuPDF 提取 PDF 文本"""
    print(f"\n[3/4] PyMuPDF 提取测试")

    if not os.path.exists(PDF_PATH):
        print("      PDF 不存在，跳过")
        return None

    try:
        import fitz  # PyMuPDF

        doc = fitz.open(PDF_PATH)
        total_pages = len(doc)
        print(f"      总页数: {total_pages}")

        # 提取前 5 页作为样本
        sample_pages = min(5, total_pages)
        extracted_texts = []

        for i in range(sample_pages):
            page = doc[i]
            text = page.get_text("text")
            extracted_texts.append(text)
            char_count = len(text)
            print(f"      第 {i+1} 页: {char_count} 字符")

        doc.close()

        # 保存提取的样本
        sample_path = os.path.join(RAW_DIR, "pymupdf_sample.txt")
        with open(sample_path, "w", encoding="utf-8") as f:
            for i, text in enumerate(extracted_texts):
                f.write(f"\n{'='*60}\n")
                f.write(f"=== PyMuPDF 第 {i+1} 页 ===\n")
                f.write(f"{'='*60}\n\n")
                f.write(text)

        print(f"      样本保存到: {sample_path}")
        return extracted_texts

    except Exception as e:
        print(f"      提取失败: {e}")
        return None


def extract_pdf_pdfplumber():
    """用 pdfplumber 提取 PDF 文本"""
    print(f"\n[4/4] pdfplumber 提取测试")

    if not os.path.exists(PDF_PATH):
        print("      PDF 不存在，跳过")
        return None

    try:
        import pdfplumber

        extracted_texts = []

        with pdfplumber.open(PDF_PATH) as pdf:
            total_pages = len(pdf.pages)
            print(f"      总页数: {total_pages}")

            sample_pages = min(5, total_pages)
            for i in range(sample_pages):
                page = pdf.pages[i]
                text = page.extract_text() or ""
                extracted_texts.append(text)
                char_count = len(text)
                print(f"      第 {i+1} 页: {char_count} 字符")

        # 保存提取的样本
        sample_path = os.path.join(RAW_DIR, "pdfplumber_sample.txt")
        with open(sample_path, "w", encoding="utf-8") as f:
            for i, text in enumerate(extracted_texts):
                f.write(f"\n{'='*60}\n")
                f.write(f"=== pdfplumber 第 {i+1} 页 ===\n")
                f.write(f"{'='*60}\n\n")
                f.write(text)

        print(f"      样本保存到: {sample_path}")
        return extracted_texts

    except Exception as e:
        print(f"      提取失败: {e}")
        return None


def quality_report(pymupdf_texts, pdfplumber_texts):
    """生成质量报告"""
    print(f"\n{'='*60}")
    print("质量分析报告")
    print(f"{'='*60}\n")

    report = {
        "pdf_file": os.path.basename(PDF_PATH),
        "wikisource_file": os.path.basename(WIKISOURCE_PATH) if os.path.exists(WIKISOURCE_PATH) else None,
    }

    # PDF 提取质量
    if pymupdf_texts:
        total_chars = sum(len(t) for t in pymupdf_texts)
        avg_chars = total_chars / len(pymupdf_texts)
        report["pymupdf"] = {
            "sample_pages": len(pymupdf_texts),
            "total_chars": total_chars,
            "avg_chars_per_page": round(avg_chars, 0),
        }
        print(f"PyMuPDF:")
        print(f"  样本页数: {len(pymupdf_texts)}")
        print(f"  总字符: {total_chars}")
        print(f"  平均每页: {avg_chars:.0f} 字符")

        # 检查中医关键词
        all_text = "\n".join(pymupdf_texts)
        tcm_keywords = ["桂枝", "麻黄", "伤寒", "太阳", "阳明", "少阳", "太阴", "少阴", "厥阴",
                        "芍药", "甘草", "生姜", "大枣", "人参", "黄芩", "半夏", "干姜"]
        found = [kw for kw in tcm_keywords if kw in all_text]
        missing = [kw for kw in tcm_keywords if kw not in all_text]
        report["pymupdf"]["tcm_keywords_found"] = found
        report["pymupdf"]["tcm_keywords_missing"] = missing
        print(f"  中医关键词命中: {len(found)}/{len(tcm_keywords)}")
        if missing:
            print(f"  缺失关键词: {missing}")

        # 检查乱码（非中文非ASCII非标点的字符占比）
        import re
        normal_chars = len(re.findall(r'[\u4e00-\u9fff\u3000-\u303f\uff00-\uffefa-zA-Z0-9\s\.\,\;\:\!\?\(\)\[\]\{\}\"\'\-\/\(\)]', all_text))
        garbage_ratio = 1 - (normal_chars / max(len(all_text), 1))
        report["pymupdf"]["garbage_ratio"] = round(garbage_ratio, 4)
        print(f"  乱码比例: {garbage_ratio:.2%}")

        # 显示第一页前500字预览
        print(f"\n  第1页预览（前500字）:")
        preview = pymupdf_texts[0][:500] if pymupdf_texts else ""
        for line in preview.split("\n")[:10]:
            print(f"    {line}")

    if pdfplumber_texts:
        total_chars = sum(len(t) for t in pdfplumber_texts)
        avg_chars = total_chars / len(pdfplumber_texts)
        report["pdfplumber"] = {
            "sample_pages": len(pdfplumber_texts),
            "total_chars": total_chars,
            "avg_chars_per_page": round(avg_chars, 0),
        }
        print(f"\npdfplumber:")
        print(f"  样本页数: {len(pdfplumber_texts)}")
        print(f"  总字符: {total_chars}")
        print(f"  平均每页: {avg_chars:.0f} 字符")

    # 维基文库原文
    if os.path.exists(WIKISOURCE_PATH):
        with open(WIKISOURCE_PATH, "r", encoding="utf-8") as f:
            wiki_text = f.read()
        report["wikisource"] = {
            "total_chars": len(wiki_text),
        }
        print(f"\n维基文库伤寒论原文:")
        print(f"  总字符: {len(wiki_text)}")

    # 结论
    print(f"\n{'='*60}")
    print("结论")
    print(f"{'='*60}")

    if pymupdf_texts and pymupdf_texts[0]:
        avg_chars = sum(len(t) for t in pymupdf_texts) / len(pymupdf_texts)
        if avg_chars > 500:
            print("✅ PDF 提取质量良好 — 文本型 PDF，PyMuPDF 可直接提取")
            report["verdict"] = "pass"
        elif avg_chars > 100:
            print("⚠️ PDF 提取质量中等 — 可能是扫描版，需要检查 OCR 必要性")
            report["verdict"] = "warning"
        else:
            print("❌ PDF 提取质量差 — 几乎无文本，需要 OCR")
            report["verdict"] = "fail"
    else:
        print("❌ 无法提取 PDF 文本")
        report["verdict"] = "fail"

    # 保存报告
    report_path = os.path.join(RAW_DIR, "extraction_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n报告保存到: {report_path}")


if __name__ == "__main__":
    print("=" * 60)
    print("Day 1: PDF 提取可行性验证")
    print("=" * 60)
    print()

    # Step 1: 下载 PDF
    pdf_ok = download_pdf()

    # Step 2: 获取维基文库原文
    wiki_ok = fetch_wikisource_text()

    # Step 3: PyMuPDF 提取
    pymupdf_texts = extract_pdf_pymupdf() if pdf_ok else None

    # Step 4: pdfplumber 提取
    pdfplumber_texts = extract_pdf_pdfplumber() if pdf_ok else None

    # 质量报告
    quality_report(pymupdf_texts, pdfplumber_texts)
