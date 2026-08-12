# M1 数据流水线 实现计划

> **面向 AI 代理的工作者：** 必需子技能：使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 逐任务实现此计划。步骤使用复选框（`- [ ]`）语法来跟踪进度。

**目标：** 构建 M1 数据流水线——从 5 本 PDF 提取文本、清洗分章、生成 SFT 训练集、扩展知识图谱至 113 方剂、建立 50 题评测集，为 P0/P1/P2 三阶段对比打下数据地基。

**架构：** 离线流水线分四条支线并行推进：(1) PDF→清洗→SFT 训练集 (2) 伤寒论原文→知识图谱 (3) 伤寒论原文→向量库（M2 阶段）(4) 人工编写评测集。本计划覆盖 (1)(2)(4)，各模块通过文件接口通信，可独立运行和测试。

**技术栈：** Python 3.13 / uv / PyMuPDF / NetworkX / pandas / pytest

---

## 文件结构

| 文件 | 职责 | 状态 |
|------|------|------|
| `pyproject.toml` | 依赖管理（uv） | 新建 |
| `.gitignore` | 忽略原始数据、模型权重 | 新建 |
| `src/__init__.py` | 包标识 | 新建 |
| `src/data/__init__.py` | 数据子包 | 新建 |
| `src/data/extract.py` | PDF 批量提取 + 质量报告 | 新建 |
| `src/data/clean.py` | 文本清洗（去噪、分章） | 新建 |
| `src/data/sft_format.py` | SFT 数据格式化（规则法） | 新建 |
| `src/data/kg_build.py` | 知识图谱构建 + 查询 + 持久化 | 新建 |
| `src/data/formulas_data.py` | 伤寒论 113 方剂数据 | 新建 |
| `src/eval/__init__.py` | 评测子包 | 新建 |
| `src/eval/dataset.py` | 评测集加载器 | 新建 |
| `data/eval/eval_50.jsonl` | 50 题评测集 | 新建 |
| `tests/conftest.py` | pytest 公共 fixture | 新建 |
| `tests/unit/test_extract.py` | PDF 提取测试 | 新建 |
| `tests/unit/test_clean.py` | 文本清洗测试 | 新建 |
| `tests/unit/test_sft_format.py` | SFT 格式化测试 | 新建 |
| `tests/unit/test_kg_build.py` | 知识图谱测试 | 新建 |
| `tests/unit/test_eval_dataset.py` | 评测集测试 | 新建 |
| `README.md` | 更新项目结构 | 修改 |

**已有文件（参考，不修改）：**
- `data/extract_test.py` — Day 1 验证脚本，extract.py 的参考实现
- `data/kg_test.py` — Day 2 验证脚本，kg_build.py 的参考实现
- `data/raw/knowledge_graph.json` — 20 方剂迷你图谱

---

## 任务 1：项目脚手架

**文件：**
- 创建：`pyproject.toml`
- 创建：`.gitignore`
- 创建：`src/__init__.py`、`src/data/__init__.py`、`src/eval/__init__.py`
- 创建：`tests/__init__.py`、`tests/unit/__init__.py`
- 创建：`tests/conftest.py`

- [ ] **步骤 1：创建 pyproject.toml**

```toml
[project]
name = "tcm-assistant"
version = "0.1.0"
description = "中医经典知识检索与理解助手"
requires-python = ">=3.11"
dependencies = [
    "pymupdf>=1.24",
    "networkx>=3.0",
    "pandas>=2.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
]

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["."]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

- [ ] **步骤 2：创建 .gitignore**

```gitignore
# 原始数据
data/raw/
data/processed/

# 模型权重
models/
*.gguf
outputs/

# Python
__pycache__/
*.pyc
.venv/
venv/
.env

# IDE
.vscode/
.idea/

# OS
.DS_Store
Thumbs.db
```

- [ ] **步骤 3：创建包标识文件**

`src/__init__.py`:
```python
"""中医经典知识检索与理解助手"""
```

`src/data/__init__.py`:
```python
"""数据处理模块"""
```

`src/eval/__init__.py`:
```python
"""评测模块"""
```

`tests/__init__.py`、`tests/unit/__init__.py`：空文件。

- [ ] **步骤 4：创建 conftest.py**

```python
"""pytest 公共 fixture"""
import pytest
import fitz
from pathlib import Path


@pytest.fixture
def sample_pdf(tmp_path):
    """创建一个小的测试 PDF（2 页，含中医文本）"""
    pdf_path = tmp_path / "test.pdf"
    doc = fitz.open()
    page1 = doc.new_page()
    page1.insert_text((72, 72), "桂枝汤主治太阳中风证。\n由桂枝、芍药、甘草、生姜、大枣组成。")
    page2 = doc.new_page()
    page2.insert_text((72, 72), "麻黄汤主治太阳伤寒证。\n由麻黄、桂枝、甘草、杏仁组成。")
    doc.save(str(pdf_path))
    doc.close()
    return str(pdf_path)


@pytest.fixture
def sample_pdf_dir(tmp_path):
    """创建包含 2 个小 PDF 的目录"""
    for name, content in [("a.pdf", "太阳病"), ("b.pdf", "阳明病")]:
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((72, 72), content)
        doc.save(str(tmp_path / name))
        doc.close()
    return str(tmp_path)


@pytest.fixture
def noisy_text():
    """模拟 PDF 提取后的含噪文本"""
    return (
        "桂枝汤主治太阳中风证。\n"
        "www.qhsxzh.com\n"
        "微信公众号：qhsxzh\n"
        "— 12 —\n"
        "第12页\n"
        "由桂枝、芍药、甘草、生姜、大枣组成。\n"
        "版权所有 翻版必究\n"
    )


@pytest.fixture
def lecture_text():
    """模拟倪海厦讲稿段落（用于 SFT 测试）"""
    return (
        "桂枝汤是伤寒论第一方，由桂枝三两、芍药三两、甘草二两、生姜三两、大枣十二枚组成。"
        "这个方子治的是太阳中风证——什么意思呢，就是你受了风邪，营卫不和，"
        "出现头痛、发热、汗出、恶风这些症状。桂枝解肌发汗，芍药敛阴和营，"
        "一散一收，调和营卫。配上甘草和中，生姜散寒，大枣补虚。\n\n"
        "麻黄汤呢，由麻黄、桂枝、甘草、杏仁四味药组成，治的是太阳伤寒证。"
        "这个方子和桂枝汤的区别在于，桂枝汤走和法，麻黄汤走汗法。"
        "你想想看，桂枝汤用芍药敛阴，麻黄汤用麻黄杏仁开腠理、降肺气，方向不同。"
    )
```

- [ ] **步骤 5：安装依赖并验证**

运行：
```bash
cd C:\Users\23919\WorkBuddy\端侧模型蒸馏
C:\Users\23919\.workbuddy\binaries\python\versions\3.13.12\python.exe -m pip install -e ".[dev]"
```
预期：安装成功，无报错。

- [ ] **步骤 6：验证 pytest 可运行**

运行：
```bash
C:\Users\23919\.workbuddy\binaries\python\versions\3.13.12\python.exe -m pytest --co
```
预期：`no tests ran`（还没有测试文件），不报 import 错误。

- [ ] **步骤 7：Commit**

```bash
git add pyproject.toml .gitignore src/ tests/conftest.py tests/__init__.py tests/unit/__init__.py
git commit -m "chore: 项目脚手架 — pyproject.toml + 目录结构 + conftest"
```

---

## 任务 2：PDF 批量提取

**文件：**
- 创建：`src/data/extract.py`
- 创建：`tests/unit/test_extract.py`

- [ ] **步骤 1：编写失败的测试**

`tests/unit/test_extract.py`:
```python
"""PDF 提取模块测试"""
import json
from pathlib import Path

from src.data.extract import (
    extract_pdf,
    extract_all_pdfs,
    generate_quality_report,
    ExtractionResult,
    Page,
)


class TestExtractPdf:
    def test_returns_extraction_result(self, sample_pdf):
        result = extract_pdf(sample_pdf)
        assert isinstance(result, ExtractionResult)

    def test_page_count(self, sample_pdf):
        result = extract_pdf(sample_pdf)
        assert result.page_count == 2

    def test_pages_have_text(self, sample_pdf):
        result = extract_pdf(sample_pdf)
        assert len(result.pages) == 2
        assert all(isinstance(p, Page) for p in result.pages)
        assert all(len(p.text) > 0 for p in result.pages)

    def test_page_numbers_sequential(self, sample_pdf):
        result = extract_pdf(sample_pdf)
        assert [p.number for p in result.pages] == [1, 2]

    def test_contains_tcm_content(self, sample_pdf):
        result = extract_pdf(sample_pdf)
        all_text = "\n".join(p.text for p in result.pages)
        assert "桂枝" in all_text
        assert "麻黄" in all_text


class TestExtractAllPdfs:
    def test_returns_dict(self, sample_pdf_dir):
        results = extract_all_pdfs(sample_pdf_dir)
        assert isinstance(results, dict)
        assert len(results) == 2

    def test_raises_on_empty_dir(self, tmp_path):
        import pytest
        with pytest.raises(FileNotFoundError):
            extract_all_pdfs(str(tmp_path))


class TestQualityReport:
    def test_generates_report(self, sample_pdf_dir, tmp_path):
        results = extract_all_pdfs(sample_pdf_dir)
        report_path = str(tmp_path / "report.json")
        report = generate_quality_report(results, report_path)
        assert "files" in report
        assert Path(report_path).exists()

    def test_report_has_stats(self, sample_pdf_dir, tmp_path):
        results = extract_all_pdfs(sample_pdf_dir)
        report = generate_quality_report(results, str(tmp_path / "r.json"))
        for name, stats in report["files"].items():
            assert "pages" in stats
            assert "total_chars" in stats
            assert "keyword_hits" in stats
            assert "garbage_ratio" in stats
```

- [ ] **步骤 2：运行测试验证失败**

运行：`C:\Users\23919\.workbuddy\binaries\python\versions\3.13.12\python.exe -m pytest tests/unit/test_extract.py -v`
预期：FAIL，报错 `ModuleNotFoundError: No module named 'src.data.extract'`

- [ ] **步骤 3：编写实现代码**

`src/data/extract.py`:
```python
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

# 正常字符范围（中文 + ASCII 可见 + 常见标点）
_NORMAL_CHAR_RE = re.compile(
    r'[\u4e00-\u9fff\u3000-\u303fa-zA-Z0-9\s\.\,\;\:\!\?\(\)\[\]\{\}\"\'\-\/]'
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
```

- [ ] **步骤 4：运行测试验证通过**

运行：`C:\Users\23919\.workbuddy\binaries\python\versions\3.13.12\python.exe -m pytest tests/unit/test_extract.py -v`
预期：全部 PASS

- [ ] **步骤 5：Commit**

```bash
git add src/data/extract.py tests/unit/test_extract.py
git commit -m "feat: PDF 批量提取模块 — extract.py + 质量报告"
```

---

## 任务 3：文本清洗

**文件：**
- 创建：`src/data/clean.py`
- 创建：`tests/unit/test_clean.py`

- [ ] **步骤 1：编写失败的测试**

`tests/unit/test_clean.py`:
```python
"""文本清洗模块测试"""
from src.data.clean import (
    remove_noise,
    clean_page,
    clean_pages,
    segment_by_formula,
    CleanedChapter,
)
from src.data.extract import Page


class TestRemoveNoise:
    def test_removes_url(self, noisy_text):
        result = remove_noise(noisy_text)
        assert "www.qhsxzh.com" not in result

    def test_removes_wechat(self, noisy_text):
        result = remove_noise(noisy_text)
        assert "微信公众号" not in result

    def test_removes_page_number(self, noisy_text):
        result = remove_noise(noisy_text)
        assert "— 12 —" not in result
        assert "第12页" not in result

    def test_removes_copyright(self, noisy_text):
        result = remove_noise(noisy_text)
        assert "版权所有" not in result
        assert "翻版必究" not in result

    def test_preserves_content(self, noisy_text):
        result = remove_noise(noisy_text)
        assert "桂枝汤主治太阳中风证" in result
        assert "由桂枝、芍药、甘草、生姜、大枣组成" in result


class TestCleanPage:
    def test_strips_whitespace(self, noisy_text):
        result = clean_page(noisy_text)
        assert not result.startswith("\n")
        assert not result.endswith("\n")

    def test_collapses_blank_lines(self):
        text = "第一行\n\n\n\n\n第二行"
        result = clean_page(text)
        assert "\n\n\n" not in result

    def test_strips_line_ends(self):
        text = "  桂枝汤  \n  麻黄汤  "
        result = clean_page(text)
        lines = result.split("\n")
        assert lines[0] == "桂枝汤"
        assert lines[1] == "麻黄汤"


class TestCleanPages:
    def test_returns_new_pages(self):
        pages = [Page(number=1, text="  桂枝  \nwww.qhsxzh.com"), Page(number=2, text="麻黄")]
        result = clean_pages(pages)
        assert len(result) == 2
        assert result[0].number == 1
        assert "www.qhsxzh.com" not in result[0].text
        assert result[0].text.startswith("桂枝")


class TestSegmentByFormula:
    def test_splits_by_formula_name(self):
        text = (
            "桂枝汤由桂枝、芍药、甘草、生姜、大枣组成。\n"
            "这个方子治太阳中风证。\n\n"
            "麻黄汤由麻黄、桂枝、甘草、杏仁组成。\n"
            "这个方子治太阳伤寒证。"
        )
        chapters = segment_by_formula(text)
        assert len(chapters) >= 2
        assert isinstance(chapters[0], CleanedChapter)
        assert any("桂枝汤" in c.title for c in chapters)
        assert any("麻黄汤" in c.title for c in chapters)

    def test_empty_text_returns_empty(self):
        assert segment_by_formula("") == []
```

- [ ] **步骤 2：运行测试验证失败**

运行：`C:\Users\23919\.workbuddy\binaries\python\versions\3.13.12\python.exe -m pytest tests/unit/test_clean.py -v`
预期：FAIL，`ModuleNotFoundError`

- [ ] **步骤 3：编写实现代码**

`src/data/clean.py`:
```python
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
    r'—\s*\d+\s*—',
    r'-\s*\d+\s*-',
    r'第\s*\d+\s*页',
    r'^\s*\d+\s*$',
    r'版权所有',
    r'翻版必究',
]

# 方剂名模式（常见伤寒论方剂）
_FORMULA_RE = re.compile(
    r'((?:桂枝|麻黄|葛根|柴胡|白虎|承气|四逆|青龙|陷胸|泻心|理中|真武|附子|'
    r'茯苓|甘草|芍药|栀子|茵陈|黄连|黄芩|半夏|橘皮|枳实|厚朴|猪苓|桃核|'
    r'抵挡|乌梅|旋覆|当归|牡蛎|龙骨|赤石脂|禹余粮|竹叶石膏|烧裈散|'
    r'枳实栀子|蜜煎导|猪胆汁|炙甘草|牡蛎泽泻|白虎加|调胃|'
    r'桃核承气|大黄黄连|附子泻心|生姜泻心|甘草泻心|干姜黄芩黄连|'
    r'麻黄升麻|桂枝附子|去桂加白术|甘草附子|桂枝人参|黄芩汤|黄连汤|'
    r'旋覆代赭|小|大)'
    r'(?:加|去|二|各|汤)?)'
    r'汤'
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
    """按方剂名切分文本为章节"""
    if not text.strip():
        return []

    # 查找所有方剂名位置
    matches = list(_FORMULA_RE.finditer(text))
    if len(matches) == 0:
        return [CleanedChapter(title="未分类", content=text.strip(), source_pages=[])]

    chapters = []
    for i, match in enumerate(matches):
        title = match.group(0)
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
```

- [ ] **步骤 4：运行测试验证通过**

运行：`C:\Users\23919\.workbuddy\binaries\python\versions\3.13.12\python.exe -m pytest tests/unit/test_clean.py -v`
预期：全部 PASS

- [ ] **步骤 5：Commit**

```bash
git add src/data/clean.py tests/unit/test_clean.py
git commit -m "feat: 文本清洗模块 — 去噪 + 分章"
```

---

## 任务 4：SFT 数据构建

**文件：**
- 创建：`src/data/sft_format.py`
- 创建：`tests/unit/test_sft_format.py`

- [ ] **步骤 1：编写失败的测试**

`tests/unit/test_sft_format.py`:
```python
"""SFT 数据格式化测试"""
import json
from pathlib import Path

from src.data.sft_format import (
    detect_content_type,
    generate_instruction,
    format_sft_pairs,
    filter_and_dedup,
    save_jsonl,
    SFTSample,
)


class TestDetectContentType:
    def test_detects_formula(self):
        text = "桂枝汤由桂枝、芍药、甘草、生姜、大枣组成。"
        assert detect_content_type(text) == "方剂"

    def test_detects_clause(self):
        text = "第12条：太阳中风，阳浮而阴弱。"
        assert detect_content_type(text) == "条文"

    def test_detects_syndrome(self):
        text = "太阳病，发热汗出，恶风脉缓者，名为中风。"
        assert detect_content_type(text) == "证型"

    def test_detects_herb(self):
        text = "桂枝这味药，能解肌发表，温通经脉。"
        assert detect_content_type(text) == "药材"

    def test_defaults_to_general(self):
        text = "中医讲究辨证论治，整体观念。"
        assert detect_content_type(text) == "通用"


class TestGenerateInstruction:
    def test_formula_instruction(self):
        text = "桂枝汤是伤寒论第一方，治太阳中风证。"
        instruction = generate_instruction(text)
        assert "桂枝汤" in instruction

    def test_clause_instruction(self):
        text = "第12条原文：太阳中风，阳浮而阴弱。"
        instruction = generate_instruction(text)
        assert "第12条" in instruction or "12" in instruction

    def test_syndrome_instruction(self):
        text = "太阳病，发热汗出，恶风脉缓者，名为中风。"
        instruction = generate_instruction(text)
        assert "太阳病" in instruction


class TestFormatSftPairs:
    def test_returns_samples(self, lecture_text):
        samples = format_sft_pairs(lecture_text)
        assert len(samples) > 0
        assert all(isinstance(s, SFTSample) for s in samples)

    def test_each_sample_has_instruction_and_output(self, lecture_text):
        samples = format_sft_pairs(lecture_text)
        for s in samples:
            assert len(s.instruction) > 0
            assert len(s.output) > 0

    def test_filters_short_paragraphs(self):
        text = "短。\n\n桂枝汤是伤寒论第一方，由桂枝、芍药、甘草、生姜、大枣组成。这个方子治太阳中风证。"
        samples = format_sft_pairs(text, min_length=50)
        assert all(len(s.output) >= 50 for s in samples)

    def test_truncates_long_paragraphs(self):
        long_text = "桂枝汤" * 500
        samples = format_sft_pairs(long_text, max_length=800)
        assert all(len(s.output) <= 800 for s in samples)


class TestFilterAndDedup:
    def test_removes_duplicates(self):
        samples = [
            SFTSample(instruction="a", input="", output="桂枝汤治太阳中风"),
            SFTSample(instruction="b", input="", output="桂枝汤治太阳中风"),  # 重复
            SFTSample(instruction="c", input="", output="麻黄汤治太阳伤寒"),
        ]
        result = filter_and_dedup(samples)
        assert len(result) == 2


class TestSaveJsonl:
    def test_writes_jsonl(self, tmp_path):
        samples = [
            SFTSample(instruction="什么是桂枝汤？", input="", output="桂枝汤是伤寒论第一方。"),
        ]
        path = str(tmp_path / "train.jsonl")
        count = save_jsonl(samples, path)
        assert count == 1
        assert Path(path).exists()
        with open(path, encoding="utf-8") as f:
            line = json.loads(f.readline())
            assert line["instruction"] == "什么是桂枝汤？"
            assert line["output"] == "桂枝汤是伤寒论第一方。"
```

- [ ] **步骤 2：运行测试验证失败**

运行：`C:\Users\23919\.workbuddy\binaries\python\versions\3.13.12\python.exe -m pytest tests/unit/test_sft_format.py -v`
预期：FAIL，`ModuleNotFoundError`

- [ ] **步骤 3：编写实现代码**

`src/data/sft_format.py`:
```python
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


# 方剂名正则（复用 clean.py 的模式）
_FORMULA_RE = re.compile(
    r'((?:桂枝|麻黄|葛根|柴胡|白虎|承气|四逆|青龙|陷胸|泻心|理中|真武|附子|'
    r'茯苓|甘草|芍药|栀子|茵陈|黄连|黄芩|半夏|橘皮|枳实|厚朴|猪苓|桃核|'
    r'抵挡|乌梅|旋覆|当归|牡蛎|龙骨)'
    r'(?:加|去|二|各|汤)?)'
    r'汤'
)

# 条辨编号正则
_CLAUSE_RE = re.compile(r'第([一二三四五六七八九十百零\d]+)条')

# 六经病名
_SIX_MERIDIANS = ["太阳病", "阳明病", "少阳病", "太阴病", "少阴病", "厥阴病"]

# 常见药材
_HERBS = [
    "桂枝", "麻黄", "芍药", "甘草", "生姜", "大枣", "人参", "黄芩",
    "半夏", "干姜", "附子", "白术", "茯苓", "大黄", "芒硝", "杏仁",
    "石膏", "知母", "柴胡", "葛根", "黄连", "细辛", "五味子",
]


def detect_content_type(text: str) -> str:
    """检测段落内容类型"""
    if _CLAUSE_RE.search(text):
        return "条文"
    if _FORMULA_RE.search(text):
        return "方剂"
    for meridian in _SIX_MERIDIANS:
        if meridian in text and ("证" in text or "病" in text):
            return "证型"
    for herb in _HERBS:
        if herb in text:
            return "药材"
    return "通用"


def generate_instruction(text: str) -> str:
    """根据段落内容生成指令"""
    content_type = detect_content_type(text)

    if content_type == "条文":
        match = _CLAUSE_RE.search(text)
        if match:
            num = match.group(1)
            return f"请解释伤寒论第{num}条的原文含义。"

    if content_type == "方剂":
        formulas = _FORMULA_RE.findall(text)
        unique = list(dict.fromkeys(formulas))
        if len(unique) >= 2:
            return f"请比较{unique[0]}汤和{unique[1]}汤的组成和主治区别。"
        if unique:
            return f"请解释{unique[0]}汤的组成和主治。"

    if content_type == "证型":
        for meridian in _SIX_MERIDIANS:
            if meridian in text:
                return f"请解释{meridian}的证候特点和治疗原则。"

    if content_type == "药材":
        found_herbs = [h for h in _HERBS if h in text]
        if found_herbs:
            return f"请介绍{found_herbs[0]}在伤寒论中的应用。"

    # 默认：取第一句作为主题
    first_sentence = re.split(r'[。！？\n]', text)[0]
    if len(first_sentence) > 5:
        return f"请讲解以下中医知识：{first_sentence[:30]}。"
    return "请讲解相关的中医知识。"


def format_sft_pairs(
    text: str,
    min_length: int = 50,
    max_length: int = 800,
) -> list[SFTSample]:
    """将讲稿文本转为 SFT 样本

    按段落切分，生成 instruction-output 对。
    过短段落丢弃，过长段落截断。
    """
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]

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
```

- [ ] **步骤 4：运行测试验证通过**

运行：`C:\Users\23919\.workbuddy\binaries\python\versions\3.13.12\python.exe -m pytest tests/unit/test_sft_format.py -v`
预期：全部 PASS

- [ ] **步骤 5：Commit**

```bash
git add src/data/sft_format.py tests/unit/test_sft_format.py
git commit -m "feat: SFT 数据构建模块 — 规则法 instruction 生成 + 去重"
```

---

## 任务 5：知识图谱构建

**文件：**
- 创建：`src/data/formulas_data.py`
- 创建：`src/data/kg_build.py`
- 创建：`tests/unit/test_kg_build.py`

- [ ] **步骤 1：编写失败的测试**

`tests/unit/test_kg_build.py`:
```python
"""知识图谱构建测试"""
import json
from pathlib import Path

from src.data.kg_build import (
    build_graph,
    query_herbs_in_formula,
    query_formulas_containing_herb,
    query_common_herbs,
    query_herb_difference,
    query_formulas_with_both_herbs,
    query_formulas_for_syndrome,
    save_graph,
    load_graph,
    graph_stats,
)


class TestBuildGraph:
    def test_returns_graph(self):
        G = build_graph()
        assert G.number_of_nodes() > 0
        assert G.number_of_edges() > 0

    def test_has_formula_nodes(self):
        G = build_graph()
        formulas = [n for n, d in G.nodes(data=True) if d.get("type") == "formula"]
        assert len(formulas) >= 113

    def test_has_herb_nodes(self):
        G = build_graph()
        herbs = [n for n, d in G.nodes(data=True) if d.get("type") == "herb"]
        assert len(herbs) >= 80

    def test_has_syndrome_nodes(self):
        G = build_graph()
        syndromes = [n for n, d in G.nodes(data=True) if d.get("type") == "syndrome"]
        assert len(syndromes) >= 10

    def test_has_text_nodes(self):
        G = build_graph()
        texts = [n for n, d in G.nodes(data=True) if d.get("type") == "text"]
        assert len(texts) >= 50


class TestQueryHerbsInFormula:
    def test_guizhi_tang(self):
        G = build_graph()
        herbs = query_herbs_in_formula(G, "桂枝汤")
        assert set(herbs) == {"桂枝", "芍药", "甘草", "生姜", "大枣"}

    def test_mahuang_tang(self):
        G = build_graph()
        herbs = query_herbs_in_formula(G, "麻黄汤")
        assert set(herbs) == {"麻黄", "桂枝", "甘草", "杏仁"}

    def test_nonexistent_formula(self):
        G = build_graph()
        assert query_herbs_in_formula(G, "不存在的方") == []


class TestQueryFormulasContainingHerb:
    def test_guizhi(self):
        G = build_graph()
        formulas = query_formulas_containing_herb(G, "桂枝")
        assert "桂枝汤" in formulas
        assert "麻黄汤" in formulas

    def test_gancao_most_frequent(self):
        G = build_graph()
        formulas = query_formulas_containing_herb(G, "甘草")
        assert len(formulas) >= 50  # 甘草是伤寒论最高频药


class TestQueryCommonHerbs:
    def test_guizhi_vs_mahuang(self):
        G = build_graph()
        common = query_common_herbs(G, "桂枝汤", "麻黄汤")
        assert set(common) == {"桂枝", "甘草"}


class TestQueryHerbDifference:
    def test_guizhi_vs_jiafuzi(self):
        G = build_graph()
        only_1, only_2 = query_herb_difference(G, "桂枝汤", "桂枝加附子汤")
        assert only_1 == []
        assert only_2 == ["附子"]


class TestQueryFormulasWithBothHerbs:
    def test_mahuang_and_guizhi(self):
        G = build_graph()
        formulas = query_formulas_with_both_herbs(G, "麻黄", "桂枝")
        assert "麻黄汤" in formulas
        assert "大青龙汤" in formulas


class TestQueryFormulasForSyndrome:
    def test_taiyang_zhongfeng(self):
        G = build_graph()
        formulas = query_formulas_for_syndrome(G, "太阳中风")
        assert "桂枝汤" in formulas


class TestPersistence:
    def test_save_and_load(self, tmp_path):
        G = build_graph()
        path = str(tmp_path / "kg.json")
        save_graph(G, path)
        assert Path(path).exists()

        G2 = load_graph(path)
        assert G2.number_of_nodes() == G.number_of_nodes()
        assert G2.number_of_edges() == G.number_of_edges()

        # 验证加载后查询仍然正确
        herbs = query_herbs_in_formula(G2, "桂枝汤")
        assert set(herbs) == {"桂枝", "芍药", "甘草", "生姜", "大枣"}


class TestGraphStats:
    def test_returns_dict(self):
        G = build_graph()
        stats = graph_stats(G)
        assert "total_nodes" in stats
        assert "total_edges" in stats
        assert "formula_count" in stats
        assert "herb_count" in stats
        assert "syndrome_count" in stats
        assert stats["formula_count"] >= 113
```

- [ ] **步骤 2：运行测试验证失败**

运行：`C:\Users\23919\.workbuddy\binaries\python\versions\3.13.12\python.exe -m pytest tests/unit/test_kg_build.py -v`
预期：FAIL，`ModuleNotFoundError`

- [ ] **步骤 3：编写 formulas_data.py（伤寒论 113 方剂数据）**

`src/data/formulas_data.py`:
```python
"""伤寒论 113 方剂 — 药材组成数据

数据来源：伤寒论原文（公版）+ 倪海厦人纪·伤寒讲稿校验。
已验证前 20 方剂（Day 2 可行性验证），其余 93 方剂从伤寒论原文整理。
"""

# 方剂 → 药材组成
FORMULAS: dict[str, list[str]] = {
    # === 前 20 方剂（Day 2 已验证） ===
    "桂枝汤": ["桂枝", "芍药", "甘草", "生姜", "大枣"],
    "桂枝加葛根汤": ["葛根", "麻黄", "芍药", "生姜", "甘草", "大枣", "桂枝"],
    "桂枝加附子汤": ["桂枝", "芍药", "甘草", "生姜", "大枣", "附子"],
    "桂枝去芍药汤": ["桂枝", "甘草", "生姜", "大枣"],
    "桂枝去芍药加附子汤": ["桂枝", "甘草", "生姜", "大枣", "附子"],
    "桂枝麻黄各半汤": ["桂枝", "芍药", "生姜", "甘草", "麻黄", "大枣", "杏仁"],
    "桂枝二麻黄一汤": ["桂枝", "芍药", "麻黄", "生姜", "杏仁", "甘草", "大枣"],
    "白虎加人参汤": ["知母", "石膏", "甘草", "粳米", "人参"],
    "桂枝二越婢一汤": ["桂枝", "芍药", "麻黄", "甘草", "大枣", "生姜", "石膏"],
    "桂枝去桂加茯苓白术汤": ["芍药", "甘草", "生姜", "白术", "茯苓", "大枣"],
    "甘草干姜汤": ["甘草", "干姜"],
    "芍药甘草汤": ["芍药", "甘草"],
    "调胃承气汤": ["大黄", "甘草", "芒硝"],
    "四逆汤": ["甘草", "干姜", "附子"],
    "葛根汤": ["葛根", "麻黄", "桂枝", "生姜", "甘草", "芍药", "大枣"],
    "葛根加半夏汤": ["葛根", "麻黄", "甘草", "芍药", "桂枝", "生姜", "半夏", "大枣"],
    "葛根黄芩黄连汤": ["葛根", "甘草", "黄芩", "黄连"],
    "麻黄汤": ["麻黄", "桂枝", "甘草", "杏仁"],
    "大青龙汤": ["麻黄", "桂枝", "甘草", "杏仁", "生姜", "大枣", "石膏"],
    "小青龙汤": ["麻黄", "芍药", "细辛", "干姜", "甘草", "桂枝", "五味子", "半夏"],

    # === 第 21-60 方剂 ===
    "桂枝加厚朴杏子汤": ["桂枝", "芍药", "甘草", "生姜", "大枣", "厚朴", "杏仁"],
    "干姜附子汤": ["干姜", "附子"],
    "桂枝加芍药生姜各一两人参三两新加汤": ["桂枝", "芍药", "甘草", "人参", "大枣", "生姜"],
    "麻黄杏仁甘草石膏汤": ["麻黄", "杏仁", "甘草", "石膏"],
    "桂枝甘草汤": ["桂枝", "甘草"],
    "茯苓桂枝白术甘草汤": ["茯苓", "桂枝", "白术", "甘草"],
    "芍药甘草附子汤": ["芍药", "甘草", "附子"],
    "茯苓四逆汤": ["茯苓", "人参", "附子", "甘草", "干姜"],
    "五苓散": ["猪苓", "泽泻", "白术", "茯苓", "桂枝"],
    "茯苓桂枝甘草大枣汤": ["茯苓", "桂枝", "甘草", "大枣"],
    "桂枝甘草龙骨牡蛎汤": ["桂枝", "甘草", "龙骨", "牡蛎"],
    "桂枝去芍药加蜀漆牡蛎龙骨救逆汤": ["桂枝", "甘草", "生姜", "大枣", "牡蛎", "蜀漆", "龙骨"],
    "桂枝加桂汤": ["桂枝", "芍药", "甘草", "生姜", "大枣"],
    "桂枝甘草龙骨牡蛎汤": ["桂枝", "甘草", "龙骨", "牡蛎"],
    "抵当汤": ["水蛭", "虻虫", "桃仁", "大黄"],
    "抵当丸": ["水蛭", "虻虫", "桃仁", "大黄"],
    "大陷胸丸": ["大黄", "葶苈子", "芒硝", "杏仁"],
    "大陷胸汤": ["大黄", "芒硝", "甘遂"],
    "小陷胸汤": ["黄连", "半夏", "瓜蒌"],
    "文蛤散": ["文蛤"],
    "白散": ["桔梗", "贝母", "巴豆"],
    "柴胡桂枝汤": ["柴胡", "桂枝", "芍药", "黄芩", "人参", "甘草", "半夏", "大枣", "生姜"],
    "柴胡桂枝干姜汤": ["柴胡", "桂枝", "干姜", "瓜蒌根", "黄芩", "牡蛎", "甘草"],
    "半夏泻心汤": ["半夏", "黄芩", "干姜", "人参", "甘草", "黄连", "大枣"],
    "十枣汤": ["芫花", "甘遂", "大戟", "大枣"],
    "大黄黄连泻心汤": ["大黄", "黄连"],
    "附子泻心汤": ["大黄", "黄连", "黄芩", "附子"],
    "生姜泻心汤": ["生姜", "甘草", "人参", "干姜", "黄芩", "半夏", "黄连", "大枣"],
    "甘草泻心汤": ["甘草", "黄芩", "干姜", "半夏", "大枣", "黄连"],
    "赤石脂禹余粮汤": ["赤石脂", "禹余粮"],
    "旋覆代赭汤": ["旋覆花", "人参", "生姜", "代赭石", "甘草", "半夏", "大枣"],
    "桂枝人参汤": ["桂枝", "甘草", "白术", "人参", "干姜"],
    "瓜蒂散": ["瓜蒂", "赤小豆"],
    "黄芩汤": ["黄芩", "芍药", "甘草", "大枣"],
    "黄芩加半夏生姜汤": ["黄芩", "芍药", "甘草", "大枣", "半夏", "生姜"],
    "黄连汤": ["黄连", "甘草", "干姜", "桂枝", "人参", "半夏", "大枣"],
    "桂枝附子汤": ["桂枝", "附子", "生姜", "甘草", "大枣"],
    "白术附子汤": ["白术", "附子", "甘草", "生姜", "大枣"],
    "甘草附子汤": ["甘草", "附子", "白术", "桂枝"],
    "炙甘草汤": ["甘草", "生姜", "人参", "生地黄", "桂枝", "阿胶", "麦门冬", "麻仁", "大枣"],

    # === 第 61-100 方剂 ===
    "大承气汤": ["大黄", "厚朴", "枳实", "芒硝"],
    "小承气汤": ["大黄", "厚朴", "枳实"],
    "猪苓汤": ["猪苓", "茯苓", "泽泻", "阿胶", "滑石"],
    "蜜煎导方": ["蜂蜜"],
    "猪胆汁方": ["猪胆汁"],
    "茵陈蒿汤": ["茵陈蒿", "栀子", "大黄"],
    "栀子柏皮汤": ["栀子", "甘草", "黄柏"],
    "麻黄连翘赤小豆汤": ["麻黄", "连翘", "杏仁", "赤小豆", "大枣", "生梓白皮", "生姜", "甘草"],
    "桂枝加芍药汤": ["桂枝", "芍药", "甘草", "生姜", "大枣"],
    "桂枝加大黄汤": ["桂枝", "芍药", "甘草", "生姜", "大枣", "大黄"],
    "麻黄升麻汤": ["麻黄", "升麻", "当归", "知母", "黄芩", "萎蕤", "芍药", "天门冬", "桂枝", "茯苓", "甘草", "石膏", "白术", "干姜"],
    "干姜黄芩黄连人参汤": ["干姜", "黄芩", "黄连", "人参"],
    "桂枝麻黄各半汤": ["桂枝", "芍药", "生姜", "甘草", "麻黄", "大枣", "杏仁"],
    "白头翁汤": ["白头翁", "黄连", "黄柏", "秦皮"],
    "四逆加人参汤": ["甘草", "干姜", "附子", "人参"],
    "理中丸": ["人参", "干姜", "甘草", "白术"],
    "通脉四逆汤": ["甘草", "干姜", "附子"],
    "通脉四逆加猪胆汁汤": ["甘草", "干姜", "附子", "猪胆汁"],
    "茯苓四逆汤": ["茯苓", "人参", "附子", "甘草", "干姜"],
    "真武汤": ["茯苓", "芍药", "白术", "生姜", "附子"],
    "附子汤": ["附子", "茯苓", "人参", "白术", "芍药"],
    "桃花汤": ["赤石脂", "干姜", "粳米"],
    "猪肤汤": ["猪肤", "蜂蜜", "白粉"],
    "甘草汤": ["甘草"],
    "桔梗汤": ["桔梗", "甘草"],
    "苦酒汤": ["半夏", "鸡子白", "苦酒"],
    "半夏散及汤": ["半夏", "桂枝", "甘草"],
    "乌梅丸": ["乌梅", "细辛", "干姜", "黄连", "当归", "附子", "蜀椒", "桂枝", "人参", "黄柏"],
    "当归四逆汤": ["当归", "桂枝", "芍药", "细辛", "甘草", "通草", "大枣"],
    "当归四逆加吴茱萸生姜汤": ["当归", "芍药", "甘草", "通草", "桂枝", "细辛", "生姜", "吴茱萸", "大枣"],
    "麻黄汤": ["麻黄", "桂枝", "甘草", "杏仁"],  # 重复，构建时去重
    "白虎汤": ["知母", "石膏", "甘草", "粳米"],
    "竹叶石膏汤": ["竹叶", "石膏", "半夏", "麦门冬", "人参", "甘草", "粳米"],
    "烧裈散": ["烧裈"],

    # === 第 101-113 方剂 ===
    "枳实栀子豉汤": ["枳实", "栀子", "豆豉"],
    "牡蛎泽泻散": ["牡蛎", "泽泻", "蜀漆", "葶苈子", "商陆根", "海藻", "瓜蒌根"],
    "禹余粮丸": ["禹余粮"],
    "芍药甘草附子汤": ["芍药", "甘草", "附子"],  # 重复，构建时去重
    "小柴胡汤": ["柴胡", "黄芩", "人参", "半夏", "甘草", "生姜", "大枣"],
    "大柴胡汤": ["柴胡", "黄芩", "芍药", "半夏", "生姜", "枳实", "大枣", "大黄"],
    "柴胡加芒硝汤": ["柴胡", "黄芩", "人参", "甘草", "生姜", "半夏", "大枣", "芒硝"],
    "柴胡加龙骨牡蛎汤": ["柴胡", "龙骨", "黄芩", "生姜", "铅丹", "人参", "桂枝", "茯苓", "半夏", "大黄", "牡蛎", "大枣"],
    "桃核承气汤": ["桃仁", "大黄", "桂枝", "甘草", "芒硝"],
    "厚朴生姜半夏甘草人参汤": ["厚朴", "生姜", "半夏", "甘草", "人参"],
    "栀子豉汤": ["栀子", "豆豉"],
    "栀子甘草豉汤": ["栀子", "甘草", "豆豉"],
    "栀子生姜豉汤": ["栀子", "生姜", "豆豉"],
    "栀子厚朴汤": ["栀子", "厚朴", "枳实"],
    "栀子干姜汤": ["栀子", "干姜"],
}

# 证型 → 方剂（主治关系）
SYNDROME_FORMULA: dict[str, list[str]] = {
    "太阳中风": ["桂枝汤"],
    "太阳伤寒": ["麻黄汤"],
    "太阳温病": ["麻黄杏仁甘草石膏汤"],
    "表虚证": ["桂枝汤", "桂枝加附子汤"],
    "表实证": ["麻黄汤", "大青龙汤"],
    "阳明热证": ["白虎汤", "白虎加人参汤"],
    "阳明腑实证": ["大承气汤", "小承气汤", "调胃承气汤"],
    "少阳证": ["小柴胡汤", "大柴胡汤"],
    "太阴虚寒": ["理中丸", "四逆汤"],
    "少阴热化证": ["黄连阿胶汤"],
    "少阴寒化证": ["四逆汤", "通脉四逆汤", "白通汤"],
    "厥阴证": ["乌梅丸"],
    "蓄水证": ["五苓散"],
    "蓄血证": ["抵当汤", "桃核承气汤"],
    "结胸证": ["大陷胸汤", "小陷胸汤", "大陷胸丸"],
    "痞证": ["半夏泻心汤", "生姜泻心汤", "甘草泻心汤"],
    "湿热发黄": ["茵陈蒿汤", "栀子柏皮汤", "麻黄连翘赤小豆汤"],
}

# 方剂加减关系：(衍生方, 基础方, 变化说明)
FORMULA_DERIVATIONS: list[tuple[str, str, str]] = [
    ("桂枝加葛根汤", "桂枝汤", "加葛根、麻黄"),
    ("桂枝加附子汤", "桂枝汤", "加附子"),
    ("桂枝去芍药汤", "桂枝汤", "去芍药"),
    ("桂枝去芍药加附子汤", "桂枝去芍药汤", "加附子"),
    ("桂枝麻黄各半汤", "桂枝汤", "合麻黄汤减量"),
    ("桂枝二麻黄一汤", "桂枝汤", "合麻黄汤，桂二麻一"),
    ("桂枝二越婢一汤", "桂枝汤", "合越婢汤，桂二越一"),
    ("桂枝加厚朴杏子汤", "桂枝汤", "加厚朴、杏仁"),
    ("桂枝加芍药汤", "桂枝汤", "倍芍药"),
    ("桂枝加大黄汤", "桂枝加芍药汤", "加大黄"),
    ("桂枝加桂汤", "桂枝汤", "加桂二两"),
    ("桂枝去桂加茯苓白术汤", "桂枝汤", "去桂加茯苓、白术"),
    ("桂枝加芍药生姜各一两人参三两新加汤", "桂枝汤", "加芍药、生姜各一两，人参三两"),
    ("葛根加半夏汤", "葛根汤", "加半夏"),
    ("小承气汤", "大承气汤", "去芒硝，减厚朴枳实"),
    ("调胃承气汤", "大承气汤", "去厚朴枳实，加甘草"),
    ("通脉四逆加猪胆汁汤", "通脉四逆汤", "加猪胆汁"),
    "白通加猪胆汁汤", "白通汤", "加猪胆汁"),
    ("四逆加人参汤", "四逆汤", "加人参"),
    ("黄芩加半夏生姜汤", "黄芩汤", "加半夏、生姜"),
    "当归四逆加吴茱萸生姜汤", "当归四逆汤", "加吴茱萸、生姜"),
]

# 伤寒论条辨 → 方剂（条文引用关系）
# 格式：(条辨编号, 方剂名)
CLAUSE_FORMULA: list[tuple[int, str]] = [
    (12, "桂枝汤"),
    (13, "桂枝汤"),
    (15, "桂枝汤"),
    (20, "桂枝加附子汤"),
    (21, "桂枝去芍药汤"),
    (22, "桂枝去芍药加附子汤"),
    (23, "桂枝麻黄各半汤"),
    (25, "桂枝二麻黄一汤"),
    (27, "桂枝二越婢一汤"),
    (28, "桂枝去桂加茯苓白术汤"),
    (29, "甘草干姜汤"),
    (29, "芍药甘草汤"),
    (29, "调胃承气汤"),
    (29, "四逆汤"),
    (31, "葛根汤"),
    (33, "葛根加半夏汤"),
    (34, "葛根黄芩黄连汤"),
    (35, "麻黄汤"),
    (38, "大青龙汤"),
    (40, "小青龙汤"),
    (43, "桂枝加厚朴杏子汤"),
    (61, "干姜附子汤"),
    (62, "桂枝加芍药生姜各一两人参三两新加汤"),
    (63, "麻黄杏仁甘草石膏汤"),
    (64, "茯苓桂枝白术甘草汤"),
    (68, "芍药甘草附子汤"),
    (69, "茯苓四逆汤"),
    (71, "五苓散"),
    (73, "五苓散"),
    (102, "小建中汤"),
    (103, "大柴胡汤"),
    (107, "柴胡加龙骨牡蛎汤"),
    (135, "大陷胸汤"),
    (138, "小陷胸汤"),
    (149, "半夏泻心汤"),
    (154, "大黄黄连泻心汤"),
    (155, "附子泻心汤"),
    (157, "生姜泻心汤"),
    (158, "甘草泻心汤"),
    (161, "旋覆代赭汤"),
    (163, "桂枝人参汤"),
    (166, "瓜蒂散"),
    (172, "黄芩汤"),
    (172, "黄芩加半夏生姜汤"),
    (173, "黄连汤"),
    (174, "桂枝附子汤"),
    (174, "白术附子汤"),
    (175, "甘草附子汤"),
    (177, "炙甘草汤"),
    (208, "大承气汤"),
    (208, "小承气汤"),
    (213, "调胃承气汤"),
    (223, "白虎汤"),
    (26, "白虎加人参汤"),
    (319, "猪苓汤"),
    (236, "茵陈蒿汤"),
    (261, "栀子柏皮汤"),
    (262, "麻黄连翘赤小豆汤"),
    (279, "桂枝加芍药汤"),
    (279, "桂枝加大黄汤"),
    (357, "麻黄升麻汤"),
    (359, "干姜黄芩黄连人参汤"),
    (371, "白头翁汤"),
    (385, "四逆加人参汤"),
    (386, "理中丸"),
    (317, "通脉四逆汤"),
    (316, "真武汤"),
    (304, "附子汤"),
    (306, "桃花汤"),
    (310, "猪肤汤"),
    (311, "甘草汤"),
    (311, "桔梗汤"),
    (338, "乌梅丸"),
    (351, "当归四逆汤"),
    (352, "当归四逆加吴茱萸生姜汤"),
    (396, "竹叶石膏汤"),
    (393, "枳实栀子豉汤"),
    (395, "牡蛎泽泻散"),
    (96, "小柴胡汤"),
    (146, "柴胡桂枝汤"),
    (147, "柴胡桂枝干姜汤"),
    (106, "桃核承气汤"),
    (66, "厚朴生姜半夏甘草人参汤"),
    (76, "栀子豉汤"),
    (80, "栀子干姜汤"),
]
```

> **注意：** 上表中的方剂和条文映射基于伤寒论原文整理。执行者在实现时应对照伤寒论原文（维基文库版本已下载至 `data/raw/`）逐条校验。测试要求 `>= 113` 方剂，如果个别方剂有争议（如某些版本将白通汤单列），以宋本伤寒论 113 方为准。如有少量方剂数据需修正，在实现时直接修改 `formulas_data.py` 即可。

- [ ] **步骤 4：编写 kg_build.py**

`src/data/kg_build.py`:
```python
"""知识图谱构建模块

从伤寒论方剂数据构建 NetworkX 知识图谱，支持多跳查询和 JSON 持久化。
基于 Day 2 验证脚本 data/kg_test.py 的成熟逻辑重构，扩展至 113 方剂。
"""
import json
import networkx as nx

from src.data.formulas_data import (
    FORMULAS,
    SYNDROME_FORMULA,
    FORMULA_DERIVATIONS,
    CLAUSE_FORMULA,
)


def build_graph() -> nx.DiGraph:
    """构建伤寒论知识图谱"""
    G = nx.DiGraph()

    # 1. 添加方剂节点 + 药材节点 + CONTAINS 边
    seen_formulas = set()
    for formula, herbs in FORMULAS.items():
        if formula in seen_formulas:
            continue
        seen_formulas.add(formula)
        G.add_node(formula, type="formula", source="伤寒论")
        for herb in herbs:
            if herb not in G:
                G.add_node(herb, type="herb")
            G.add_edge(formula, herb, relation="contains")

    # 2. 添加证型节点 + TREATS 边
    for syndrome, formulas in SYNDROME_FORMULA.items():
        if syndrome not in G:
            G.add_node(syndrome, type="syndrome")
        for formula in formulas:
            if formula in G:
                G.add_edge(syndrome, formula, relation="treats")

    # 3. 添加方剂加减关系 DERIVED_FROM 边
    for derived, base, change in FORMULA_DERIVATIONS:
        if derived not in G:
            G.add_node(derived, type="formula", source="伤寒论")
        if base not in G:
            G.add_node(base, type="formula", source="伤寒论")
        G.add_edge(derived, base, relation="derived_from", change=change)

    # 4. 添加条文节点 + MENTIONS 边
    for clause_num, formula in CLAUSE_FORMULA:
        clause_id = f"第{clause_num}条"
        if clause_id not in G:
            G.add_node(clause_id, type="text", number=clause_num)
        if formula in G:
            G.add_edge(clause_id, formula, relation="mentions")

    return G


# ========== 查询函数 ==========

def query_herbs_in_formula(G: nx.DiGraph, formula_name: str) -> list[str]:
    """查询某方剂包含哪些药材"""
    if formula_name not in G:
        return []
    herbs = [n for n in G.successors(formula_name)
             if G.nodes[n].get("type") == "herb"]
    return sorted(herbs)


def query_formulas_containing_herb(G: nx.DiGraph, herb_name: str) -> list[str]:
    """查询哪些方剂含有某药材"""
    if herb_name not in G:
        return []
    formulas = [n for n in G.predecessors(herb_name)
                if G.nodes[n].get("type") == "formula"]
    return sorted(formulas)


def query_common_herbs(G: nx.DiGraph, f1: str, f2: str) -> list[str]:
    """查询两个方剂的共同药材"""
    herbs1 = set(query_herbs_in_formula(G, f1))
    herbs2 = set(query_herbs_in_formula(G, f2))
    return sorted(herbs1 & herbs2)


def query_herb_difference(G: nx.DiGraph, f1: str, f2: str) -> tuple[list[str], list[str]]:
    """查询两个方剂的药材差异"""
    herbs1 = set(query_herbs_in_formula(G, f1))
    herbs2 = set(query_herbs_in_formula(G, f2))
    return sorted(herbs1 - herbs2), sorted(herbs2 - herbs1)


def query_formulas_with_both_herbs(G: nx.DiGraph, h1: str, h2: str) -> list[str]:
    """查询同时含两种药材的方剂"""
    f1 = set(query_formulas_containing_herb(G, h1))
    f2 = set(query_formulas_containing_herb(G, h2))
    return sorted(f1 & f2)


def query_formulas_for_syndrome(G: nx.DiGraph, syndrome: str) -> list[str]:
    """查询某证型对应的方剂"""
    if syndrome not in G:
        return []
    formulas = [n for n in G.successors(syndrome)
                if G.nodes[n].get("type") == "formula"]
    return sorted(formulas)


def query_derivations(G: nx.DiGraph, formula: str) -> list[tuple[str, str]]:
    """查询某方剂的加减变化"""
    if formula not in G:
        return []
    result = []
    for n in G.predecessors(formula):
        if G.nodes[n].get("type") == "formula":
            edge = G.edges[n, formula]
            if edge.get("relation") == "derived_from":
                result.append((n, edge.get("change", "")))
    return result


# ========== 持久化 ==========

def save_graph(G: nx.DiGraph, path: str) -> None:
    """保存图谱为 JSON"""
    data = nx.node_link_data(G)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_graph(path: str) -> nx.DiGraph:
    """从 JSON 加载图谱"""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return nx.node_link_graph(data)


def graph_stats(G: nx.DiGraph) -> dict:
    """获取图谱统计信息"""
    types = {}
    for _, data in G.nodes(data=True):
        t = data.get("type", "unknown")
        types[t] = types.get(t, 0) + 1
    return {
        "total_nodes": G.number_of_nodes(),
        "total_edges": G.number_of_edges(),
        "formula_count": types.get("formula", 0),
        "herb_count": types.get("herb", 0),
        "syndrome_count": types.get("syndrome", 0),
        "text_count": types.get("text", 0),
    }
```

- [ ] **步骤 5：运行测试验证通过**

运行：`C:\Users\23919\.workbuddy\binaries\python\versions\3.13.12\python.exe -m pytest tests/unit/test_kg_build.py -v`
预期：全部 PASS。如果方剂数不足 113 或个别方剂药材不对，修正 `formulas_data.py` 后重跑。

- [ ] **步骤 6：Commit**

```bash
git add src/data/formulas_data.py src/data/kg_build.py tests/unit/test_kg_build.py
git commit -m "feat: 知识图谱构建 — 113 方剂 + 多跳查询 + 持久化"
```

---

## 任务 6：评测集构建

**文件：**
- 创建：`data/eval/eval_50.jsonl`
- 创建：`src/eval/dataset.py`
- 创建：`tests/unit/test_eval_dataset.py`

- [ ] **步骤 1：编写评测集数据文件**

`data/eval/eval_50.jsonl` — 50 题，每行一个 JSON 对象。

按 PRD §9 FR9 的 5 个类别：
- 经典原文检索（15 题）
- 方剂查询（10 题）
- 药材关联（10 题）
- 经典解释（10 题）
- 综合问答（5 题）

```jsonl
{"id":1,"category":"经典原文检索","question":"伤寒论第1条原文是什么？","expected_answer":"太阳之为病，脉浮，头项强痛而恶寒。","reference_clauses":[1]}
{"id":2,"category":"经典原文检索","question":"伤寒论第12条原文是什么？","expected_answer":"太阳中风，阳浮而阴弱，热自发，汗自出，啬啬恶寒，淅淅恶风，翕翕发热，鼻鸣干呕者，桂枝汤主之。","reference_clauses":[12]}
{"id":3,"category":"经典原文检索","question":"伤寒论第35条原文是什么？","expected_answer":"太阳病，头痛发热，身疼腰痛，骨节疼痛，恶风无汗而喘者，麻黄汤主之。","reference_clauses":[35]}
{"id":4,"category":"经典原文检索","question":"伤寒论第38条原文是什么？","expected_answer":"太阳中风，脉浮紧，发热恶寒，身疼痛，不汗出而烦躁者，大青龙汤主之。","reference_clauses":[38]}
{"id":5,"category":"经典原文检索","question":"伤寒论第40条原文是什么？","expected_answer":"伤寒表不解，心下有水气，干呕，发热而咳，或渴，或利，或噎，或小便不利、少腹满，或喘者，小青龙汤主之。","reference_clauses":[40]}
{"id":6,"category":"经典原文检索","question":"伤寒论第71条原文是什么？","expected_answer":"太阳病，发汗后，大汗出，胃中干，烦躁不得眠，欲得饮水者，少少与饮之，令胃气和则愈。若脉浮，小便不利，微热消渴者，五苓散主之。","reference_clauses":[71]}
{"id":7,"category":"经典原文检索","question":"伤寒论第96条原文是什么？","expected_answer":"伤寒五六日中风，往来寒热，胸胁苦满，嘿嘿不欲饮食，心烦喜呕，或胸中烦而不呕，或渴，或腹中痛，或胁下痞硬，或心下悸、小便不利，或不渴、身有微热，或咳者，小柴胡汤主之。","reference_clauses":[96]}
{"id":8,"category":"经典原文检索","question":"伤寒论第135条原文是什么？","expected_answer":"伤寒六七日，结胸热实，脉沉而紧，心下痛，按之石硬者，大陷胸汤主之。","reference_clauses":[135]}
{"id":9,"category":"经典原文检索","question":"伤寒论第149条原文是什么？","expected_answer":"伤寒五六日，呕而发热者，柴胡汤证具，而以他药下之，柴胡证仍在者，复与柴胡汤。此虽已下之，不为逆，必蒸蒸而振，却发热汗出而解。若心下满而硬痛者，此为结胸也，大陷胸汤主之。但满而不痛者，此为痞，柴胡不中与之，宜半夏泻心汤。","reference_clauses":[149]}
{"id":10,"category":"经典原文检索","question":"伤寒论第177条原文是什么？","expected_answer":"伤寒脉结代，心动悸，炙甘草汤主之。","reference_clauses":[177]}
{"id":11,"category":"经典原文检索","question":"伤寒论第208条原文是什么？","expected_answer":"阳明病，脉迟，虽汗出不恶寒者，其身必重，短气，腹满而喘，有潮热者，此外欲解，可攻里也。手足濈然汗出者，此大便已硬也，大承气汤主之。","reference_clauses":[208]}
{"id":12,"category":"经典原文检索","question":"伤寒论第316条原文是什么？","expected_answer":"少阴病，二三日不已，至四五日，腹痛，小便不利，四肢沉重疼痛，自下利者，此为有水气，其人或咳，或小便利，或下利，或呕者，真武汤主之。","reference_clauses":[316]}
{"id":13,"category":"经典原文检索","question":"伤寒论第317条原文是什么？","expected_answer":"少阴病，下利清谷，里寒外热，手足厥逆，脉微欲绝，身反不恶寒，其人面色赤，或腹痛，或干呕，或咽痛，或利止脉不出者，通脉四逆汤主之。","reference_clauses":[317]}
{"id":14,"category":"经典原文检索","question":"伤寒论第338条原文是什么？","expected_answer":"伤寒脉微而厥，至七八日肤冷，其人躁无暂安时者，此为脏厥，非蛔厥也。蛔厥者，其人当吐蛔。今病者静，而复时烦者，此为脏寒。蛔上入其膈，故烦，须臾复止，得食而呕，又烦者，蛔闻食臭出，其人常自吐蛔。蛔厥者，乌梅丸主之。又主久利。","reference_clauses":[338]}
{"id":15,"category":"经典原文检索","question":"伤寒论第351条原文是什么？","expected_answer":"手足厥寒，脉细欲绝者，当归四逆汤主之。","reference_clauses":[351]}
{"id":16,"category":"方剂查询","question":"桂枝汤的组成是什么？","expected_answer":"桂枝、芍药、甘草、生姜、大枣","reference_clauses":[12]}
{"id":17,"category":"方剂查询","question":"麻黄汤的组成是什么？","expected_answer":"麻黄、桂枝、甘草、杏仁","reference_clauses":[35]}
{"id":18,"category":"方剂查询","question":"小柴胡汤的组成是什么？","expected_answer":"柴胡、黄芩、人参、半夏、甘草、生姜、大枣","reference_clauses":[96]}
{"id":19,"category":"方剂查询","question":"大承气汤的组成是什么？","expected_answer":"大黄、厚朴、枳实、芒硝","reference_clauses":[208]}
{"id":20,"category":"方剂查询","question":"五苓散的组成是什么？","expected_answer":"猪苓、泽泻、白术、茯苓、桂枝","reference_clauses":[71]}
{"id":21,"category":"方剂查询","question":"真武汤的组成是什么？","expected_answer":"茯苓、芍药、白术、生姜、附子","reference_clauses":[316]}
{"id":22,"category":"方剂查询","question":"炙甘草汤的组成是什么？","expected_answer":"甘草、生姜、人参、生地黄、桂枝、阿胶、麦门冬、麻仁、大枣","reference_clauses":[177]}
{"id":23,"category":"方剂查询","question":"四逆汤的组成是什么？","expected_answer":"甘草、干姜、附子","reference_clauses":[29]}
{"id":24,"category":"方剂查询","question":"白虎汤的组成是什么？","expected_answer":"知母、石膏、甘草、粳米","reference_clauses":[223]}
{"id":25,"category":"方剂查询","question":"乌梅丸的组成是什么？","expected_answer":"乌梅、细辛、干姜、黄连、当归、附子、蜀椒、桂枝、人参、黄柏","reference_clauses":[338]}
{"id":26,"category":"药材关联","question":"同时含有桂枝和麻黄的方剂有哪些？","expected_answer":"麻黄汤、大青龙汤、小青龙汤、葛根汤、葛根加半夏汤、桂枝加葛根汤、桂枝麻黄各半汤、桂枝二麻黄一汤、桂枝二越婢一汤","reference_clauses":[]}
{"id":27,"category":"药材关联","question":"含有甘草的方剂有哪些？","expected_answer":"桂枝汤、麻黄汤、小柴胡汤、四逆汤、白虎汤、炙甘草汤等（甘草是伤寒论最高频药材）","reference_clauses":[]}
{"id":28,"category":"药材关联","question":"含有附子的方剂有哪些？","expected_answer":"桂枝加附子汤、桂枝去芍药加附子汤、四逆汤、干姜附子汤、附子汤、真武汤、甘草附子汤等","reference_clauses":[]}
{"id":29,"category":"药材关联","question":"含有大黄的方剂有哪些？","expected_answer":"大承气汤、小承气汤、调胃承气汤、桃核承气汤、抵当汤、大黄黄连泻心汤、附子泻心汤、桂枝加大黄汤、大柴胡汤等","reference_clauses":[]}
{"id":30,"category":"药材关联","question":"含有石膏的方剂有哪些？","expected_answer":"白虎汤、白虎加人参汤、大青龙汤、麻黄杏仁甘草石膏汤、桂枝二越婢一汤、竹叶石膏汤、麻黄升麻汤等","reference_clauses":[]}
{"id":31,"category":"药材关联","question":"含有半夏的方剂有哪些？","expected_answer":"小柴胡汤、大柴胡汤、半夏泻心汤、生姜泻心汤、甘草泻心汤、葛根加半夏汤、小陷胸汤、旋覆代赭汤、黄芩加半夏生姜汤、黄连汤等","reference_clauses":[]}
{"id":32,"category":"药材关联","question":"含有黄芩的方剂有哪些？","expected_answer":"小柴胡汤、大柴胡汤、半夏泻心汤、生姜泻心汤、甘草泻心汤、葛根黄芩黄连汤、黄芩汤、黄芩加半夏生姜汤、柴胡桂枝干姜汤、柴胡加龙骨牡蛎汤、附子泻心汤、麻黄升麻汤等","reference_clauses":[]}
{"id":33,"category":"药材关联","question":"含有人参的方剂有哪些？","expected_answer":"白虎加人参汤、桂枝新加汤、茯苓四逆汤、小柴胡汤、半夏泻心汤、生姜泻心汤、甘草泻心汤、旋覆代赭汤、桂枝人参汤、理中丸、附子汤、黄连汤、干姜黄芩黄连人参汤、炙甘草汤等","reference_clauses":[]}
{"id":34,"category":"药材关联","question":"同时含有黄连和黄芩的方剂有哪些？","expected_answer":"半夏泻心汤、生姜泻心汤、甘草泻心汤、葛根黄芩黄连汤、附子泻心汤、干姜黄芩黄连人参汤等","reference_clauses":[]}
{"id":35,"category":"药材关联","question":"含有干姜的方剂有哪些？","expected_answer":"甘草干姜汤、四逆汤、干姜附子汤、通脉四逆汤、茯苓四逆汤、半夏泻心汤、生姜泻心汤、甘草泻心汤、黄连汤、桂枝人参汤、理中丸、桃花汤、干姜黄芩黄连人参汤、麻黄升麻汤、乌梅丸等","reference_clauses":[]}
{"id":36,"category":"经典解释","question":"什么是太阳中风证？","expected_answer":"太阳中风证是外感风邪所致的表虚证，主要表现为发热、汗出、恶风、脉浮缓，以桂枝汤为主方。","reference_clauses":[12]}
{"id":37,"category":"经典解释","question":"什么是太阳伤寒证？","expected_answer":"太阳伤寒证是外感寒邪所致的表实证，主要表现为恶寒、无汗、身疼痛、脉浮紧，以麻黄汤为主方。","reference_clauses":[35]}
{"id":38,"category":"经典解释","question":"什么是阳明病？","expected_answer":"阳明病是外感病发展过程中，阳热亢盛、胃肠燥热的阶段，分为阳明经证（白虎汤证）和阳明腑证（承气汤证），主要表现为大热、大汗、大渴、脉洪大。","reference_clauses":[208]}
{"id":39,"category":"经典解释","question":"什么是少阳病？","expected_answer":"少阳病是邪犯少阳胆经，枢机不利所致，主要表现为往来寒热、胸胁苦满、默默不欲饮食、心烦喜呕，以小柴胡汤为主方。","reference_clauses":[96]}
{"id":40,"category":"经典解释","question":"什么是太阴病？","expected_answer":"太阴病是脾阳虚衰、寒湿内盛所致，主要表现为腹满而吐、食不下、自利益甚、时腹自痛，以理中丸、四逆汤类温中散寒。","reference_clauses":[]}
{"id":41,"category":"经典解释","question":"什么是少阴病？","expected_answer":"少阴病是心肾阳衰或阴虚火旺所致，分为寒化证（四逆汤证）和热化证（黄连阿胶汤证），主要表现为脉微细、但欲寐。","reference_clauses":[]}
{"id":42,"category":"经典解释","question":"什么是厥阴病？","expected_answer":"厥阴病是邪入厥阴、寒热错杂的阶段，主要表现为消渴、气上撞心、心中疼热、饥而不欲食、食则吐蛔，以乌梅丸为主方。","reference_clauses":[338]}
{"id":43,"category":"经典解释","question":"什么是蓄水证？","expected_answer":"蓄水证是太阳病邪传膀胱、气化不利、水停下焦所致，主要表现为小便不利、口渴、烦渴欲饮水、水入则吐，以五苓散为主方。","reference_clauses":[71]}
{"id":44,"category":"经典解释","question":"什么是蓄血证？","expected_answer":"蓄血证是邪热内传、瘀血结于下焦所致，主要表现为少腹急结或硬满、如狂或发狂、小便自利，以桃核承气汤或抵当汤为主方。","reference_clauses":[]}
{"id":45,"category":"经典解释","question":"什么是结胸证？","expected_answer":"结胸证是邪热与痰水结于心下所致，主要表现为心下硬满疼痛、按之石硬，分为大结胸（大陷胸汤）和小结胸（小陷胸汤）。","reference_clauses":[135]}
{"id":46,"category":"综合问答","question":"桂枝汤和麻黄汤在主治上的区别是什么？","expected_answer":"桂枝汤主治太阳中风证（表虚），表现为发热汗出恶风脉浮缓；麻黄汤主治太阳伤寒证（表实），表现为恶寒无汗身痛脉浮紧。桂枝汤走和法调和营卫，麻黄汤走汗法发汗解表。","reference_clauses":[12,35]}
{"id":47,"category":"综合问答","question":"大承气汤和小承气汤有什么区别？","expected_answer":"大承气汤含大黄、厚朴、枳实、芒硝，主治痞满燥实俱全的阳明腑实证；小承气汤去芒硝，减厚朴枳实用量，主治痞满实而不燥的轻证。","reference_clauses":[208]}
{"id":48,"category":"综合问答","question":"小柴胡汤和大柴胡汤有什么区别？","expected_answer":"小柴胡汤治少阳证，含柴胡、黄芩、人参等，和解少阳；大柴胡汤在小柴胡汤基础上去人参甘草，加芍药、枳实、大黄，主治少阳兼阳明里实证。","reference_clauses":[96,103]}
{"id":49,"category":"综合问答","question":"白虎汤和白虎加人参汤有什么区别？","expected_answer":"白虎汤含知母、石膏、甘草、粳米，主治阳明热证；白虎加人参汤加人参，主治阳明热证兼气津两伤，见大汗出、口大渴、背微恶寒者。","reference_clauses":[26,223]}
{"id":50,"category":"综合问答","question":"四逆汤和通脉四逆汤有什么区别？","expected_answer":"四逆汤含甘草、干姜、附子，主治少阴寒化证；通脉四逆汤在四逆汤基础上加大干姜和附子用量，主治阴盛格阳、里寒外热的重证。","reference_clauses":[29,317]}
```

- [ ] **步骤 2：编写失败的测试**

`tests/unit/test_eval_dataset.py`:
```python
"""评测集测试"""
from pathlib import Path

from src.eval.dataset import load_eval_set, EvalQuestion


EVAL_PATH = str(Path(__file__).parent.parent.parent / "data" / "eval" / "eval_50.jsonl")


class TestLoadEvalSet:
    def test_loads_50_questions(self):
        questions = load_eval_set(EVAL_PATH)
        assert len(questions) == 50

    def test_each_question_has_required_fields(self):
        questions = load_eval_set(EVAL_PATH)
        for q in questions:
            assert isinstance(q, EvalQuestion)
            assert q.id > 0
            assert len(q.category) > 0
            assert len(q.question) > 0
            assert len(q.expected_answer) > 0

    def test_category_distribution(self):
        questions = load_eval_set(EVAL_PATH)
        categories = {}
        for q in questions:
            categories[q.category] = categories.get(q.category, 0) + 1
        assert categories.get("经典原文检索") == 15
        assert categories.get("方剂查询") == 10
        assert categories.get("药材关联") == 10
        assert categories.get("经典解释") == 10
        assert categories.get("综合问答") == 5

    def test_ids_are_sequential(self):
        questions = load_eval_set(EVAL_PATH)
        ids = [q.id for q in questions]
        assert ids == list(range(1, 51))
```

- [ ] **步骤 3：运行测试验证失败**

运行：`C:\Users\23919\.workbuddy\binaries\python\versions\3.13.12\python.exe -m pytest tests/unit/test_eval_dataset.py -v`
预期：FAIL，`ModuleNotFoundError`

- [ ] **步骤 4：编写实现代码**

`src/eval/dataset.py`:
```python
"""评测集加载模块"""
import json
from dataclasses import dataclass


@dataclass
class EvalQuestion:
    """单条评测题目"""
    id: int
    category: str
    question: str
    expected_answer: str
    reference_clauses: list[int]


def load_eval_set(path: str) -> list[EvalQuestion]:
    """加载 JSONL 格式的评测集"""
    questions = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            questions.append(EvalQuestion(
                id=data["id"],
                category=data["category"],
                question=data["question"],
                expected_answer=data["expected_answer"],
                reference_clauses=data.get("reference_clauses", []),
            ))
    return questions
```

- [ ] **步骤 5：运行测试验证通过**

运行：`C:\Users\23919\.workbuddy\binaries\python\versions\3.13.12\python.exe -m pytest tests/unit/test_eval_dataset.py -v`
预期：全部 PASS

- [ ] **步骤 6：Commit**

```bash
git add data/eval/eval_50.jsonl src/eval/dataset.py tests/unit/test_eval_dataset.py
git commit -m "feat: 50 题评测集 — 5 类别分层 + 加载器"
```

---

## 任务 7：端到端集成验证

**文件：**
- 创建：`tests/unit/test_integration.py`

- [ ] **步骤 1：编写集成测试**

`tests/unit/test_integration.py`:
```python
"""端到端集成测试：验证各模块协同工作"""
from pathlib import Path

from src.data.extract import extract_pdf, generate_quality_report
from src.data.clean import clean_pages, segment_by_formula
from src.data.sft_format import format_sft_pairs, filter_and_dedup, save_jsonl
from src.data.kg_build import build_graph, query_herbs_in_formula, graph_stats, save_graph
from src.eval.dataset import load_eval_set


PROJECT_ROOT = Path(__file__).parent.parent.parent


class TestPipelineIntegration:
    """验证 PDF → 清洗 → SFT 流水线"""

    def test_pdf_to_sft(self, sample_pdf, tmp_path):
        # 1. 提取
        result = extract_pdf(sample_pdf)
        assert result.page_count > 0

        # 2. 清洗
        cleaned = clean_pages(result.pages)
        assert all(len(p.text) > 0 for p in cleaned)

        # 3. 合并文本
        all_text = "\n\n".join(p.text for p in cleaned)

        # 4. SFT 格式化
        samples = format_sft_pairs(all_text, min_length=10)
        assert len(samples) > 0

        # 5. 去重
        samples = filter_and_dedup(samples)

        # 6. 保存
        path = str(tmp_path / "train.jsonl")
        count = save_jsonl(samples, path)
        assert count > 0


class TestGraphIntegration:
    """验证知识图谱构建 + 查询 + 持久化"""

    def test_build_query_persist(self, tmp_path):
        # 1. 构建
        G = build_graph()
        stats = graph_stats(G)
        assert stats["formula_count"] >= 113

        # 2. 查询
        herbs = query_herbs_in_formula(G, "桂枝汤")
        assert "桂枝" in herbs

        # 3. 持久化
        path = str(tmp_path / "kg.json")
        save_graph(G, path)
        assert Path(path).exists()


class TestEvalSetIntegration:
    """验证评测集完整性"""

    def test_eval_set_loads(self):
        eval_path = str(PROJECT_ROOT / "data" / "eval" / "eval_50.jsonl")
        questions = load_eval_set(eval_path)
        assert len(questions) == 50

    def test_eval_set_covers_formulas(self):
        """评测集中的方剂查询题应该在图谱中能找到"""
        eval_path = str(PROJECT_ROOT / "data" / "eval" / "eval_50.jsonl")
        questions = load_eval_set(eval_path)
        G = build_graph()

        formula_questions = [q for q in questions if q.category == "方剂查询"]
        for q in formula_questions:
            # 提取问题中的方剂名
            for formula in G.nodes:
                if formula in q.question and G.nodes[formula].get("type") == "formula":
                    herbs = query_herbs_in_formula(G, formula)
                    assert len(herbs) > 0, f"图谱中 {formula} 没有药材数据"
                    break
```

- [ ] **步骤 2：运行全部测试**

运行：`C:\Users\23919\.workbuddy\binaries\python\versions\3.13.12\python.exe -m pytest -v`
预期：全部 PASS

- [ ] **步骤 3：更新 README.md**

在 README.md 的「项目状态」部分，将数据流水线标记为已完成，更新项目结构。

- [ ] **步骤 4：Commit**

```bash
git add tests/unit/test_integration.py README.md
git commit -m "test: 端到端集成测试 + 更新 README"
```

---

## 自检

### 1. 规格覆盖度

| PRD 需求 | 对应任务 | 状态 |
|---------|---------|------|
| FR1 PDF 提取与文本清洗 | 任务 2 + 任务 3 | ✅ 覆盖 |
| FR2 SFT 训练数据构建 | 任务 4 | ✅ 覆盖 |
| FR3 知识图谱构建 | 任务 5 | ✅ 覆盖 |
| FR9 评测体系（评测集部分） | 任务 6 | ✅ 覆盖 |
| §9 项目结构 | 任务 1 | ✅ 覆盖 |
| §11 M1 验收标准 | 任务 7 集成测试 | ✅ 覆盖 |

**未覆盖（属于后续里程碑，不在 M1 范围）：**
- FR4 LoRA 微调 → M3
- FR5 向量 RAG → M2
- FR6 GraphRAG 查询 → M4
- FR7 查询路由与生成 → M2/M4
- FR8 Web 服务 → M2
- FR9 评测指标计算与阶段对比 → M2-M4

### 2. 占位符扫描

- 无 "TODO"、"待定"、"后续实现" — ✅
- 所有代码步骤都有完整代码块 — ✅
- formulas_data.py 包含 100+ 方剂实际数据 — ✅（注意：部分方剂数据需在实现时对照原文校验修正）
- eval_50.jsonl 包含 50 题实际内容 — ✅

### 3. 类型一致性

| 类型/函数 | 定义位置 | 使用位置 | 一致性 |
|----------|---------|---------|--------|
| `Page` | extract.py | clean.py, test_extract.py, test_clean.py | ✅ |
| `ExtractionResult` | extract.py | test_extract.py | ✅ |
| `CleanedChapter` | clean.py | test_clean.py | ✅ |
| `SFTSample` | sft_format.py | test_sft_format.py | ✅ |
| `build_graph()` | kg_build.py | test_kg_build.py, test_integration.py | ✅ |
| `query_herbs_in_formula()` | kg_build.py | test_kg_build.py, test_integration.py | ✅ |
| `EvalQuestion` | dataset.py | test_eval_dataset.py | ✅ |
| `load_eval_set()` | dataset.py | test_eval_dataset.py, test_integration.py | ✅ |

### 4. 已知风险

1. **formulas_data.py 数据准确性**：计划中包含的方剂数据基于伤寒论原文整理，但个别方剂的药味组成在不同版本中有差异。实现时需对照宋本伤寒论原文逐条校验。测试用 `>= 113` 作为下限，允许实现时增补。

2. **SFT 规则法质量**：规则法生成的 instruction 可能不够自然（如"请解释{方剂名}汤的组成和主治。"）。PRD §15 已预留 GPT 生成作为 fallback，M1 先跑通流水线，M1 结束后人工抽检 50 条评估质量。

3. **segment_by_formula 切分粒度**：方剂名正则可能无法覆盖所有讲稿中的方剂提及（如口语化简称）。M1 阶段以段落切分为主要手段，方剂名切分作为辅助，实际效果需在真实 PDF 上验证。

---

## 执行交接

计划已完成并保存到 `docs/superpowers/plans/2026-08-04-m1-data-pipeline.md`。两种执行方式：

**1. 子代理驱动（推荐）** — 每个任务调度一个新的子代理，任务间进行审查，快速迭代

**2. 内联执行** — 在当前会话中使用 executing-plans 执行任务，批量执行并设有检查点

选哪种方式？
