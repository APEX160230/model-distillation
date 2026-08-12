"""
提取 PDF 正文内容页（方剂条文部分）做深度质量检查
"""
import fitz
import os

PDF_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "raw", "人纪04伤寒.pdf")
OUTPUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "raw", "content_sample.txt")

doc = fitz.open(PDF_PATH)
total = len(doc)

# 提取正文内容页：跳过前言目录，取第 15、50、100、150、200 页
sample_pages = [14, 49, 99, 149, 199]  # 0-indexed

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    for page_num in sample_pages:
        if page_num >= total:
            continue
        page = doc[page_num]
        text = page.get_text("text")
        f.write(f"\n{'='*60}\n")
        f.write(f"=== 第 {page_num + 1} 页（正文内容）===\n")
        f.write(f"{'='*60}\n\n")
        f.write(text)
        f.write("\n")

doc.close()
print(f"正文样本保存到: {OUTPUT_PATH}")
print(f"总页数: {total}")
