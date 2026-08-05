# 中医经典知识检索与理解助手

> 基于 LoRA 微调 + GraphRAG 的中医经典知识检索系统，部署在 2 核 4G 服务器上。

## 项目简介

用倪海厦医师讲稿微调 Qwen2.5-1.5B 模型，使其回答带有中医讲解风格；用知识图谱实现方剂-药材-证型的多跳关联查询；用向量库实现经典原文语义检索。三者结合，构建一个能检索、能解释、能关联的中医经典知识助手。

**这不是问诊系统**，而是中医经典知识的检索与理解工具——检索原文、解释方剂、关联药证。

## 技术栈

| 维度 | 选型 | 说明 |
|------|------|------|
| 基座模型 | Qwen2.5-1.5B-Instruct | Q4 量化 ~1.2GB，2 核 4G 可跑 |
| 微调方法 | LoRA (rank=8) | 仅训练 0.14% 参数 |
| 推理方案 | Ollama / Transformers | 本地推理，零 API 成本 |
| 向量库 | ChromaDB | 嵌入式，bge-small-zh-v1.5 |
| 图谱引擎 | NetworkX | 内存图，150MB 够用 |
| Web 框架 | FastAPI + SSE | 流式输出 |
| 部署环境 | 2 核 4G 云服务器 | 内存预算 ~3.1GB |

## 项目状态

- [x] 可行性验证（PDF 提取 + 知识图谱查询）
- [x] PRD v2.1
- [x] P0 纯 RAG Baseline — recall@5=5.41%, keyword_acc=18.3%
- [ ] P1 LoRA + 向量 RAG（训练中）
- [ ] P2 GraphRAG 升级
- [ ] 公众号开源介绍文章

### P0 评测结果

| 指标 | 数值 | 说明 |
|------|------|------|
| recall@5 | 5.41% | 检索是最大瓶颈 |
| 关键词命中率 | 18.3% | 1.5B 幻觉严重 |
| 延迟 p50 | 2.51s | 远优于 25-40s 预估 |
| 延迟 p95 | 9.28s | — |

分类表现：方剂查询 71.5% > 综合问答 20% > 药材关联 12.3% > 经典解释 6.4% > 原文检索 1.2%

## 快速开始

### 环境要求

- Python 3.11+
- 6 核 CPU / 8GB RAM（训练）/ 2 核 4G（推理部署）
- 无需 GPU

### 安装

```bash
git clone https://github.com/APEX160230/model-distillation.git
cd model-distillation
python -m venv .venv
.venv/Scripts/activate    # Windows
# source .venv/bin/activate  # Linux/Mac

pip install -e ".[dev]"
```

### 下载模型

```bash
# Qwen2.5-1.5B-Instruct（约 2.9GB）
mkdir -p models/qwen25-15b-base
# 从 hf-mirror.com 下载到 models/qwen25-15b-base/
```

### 运行测试

```bash
pytest tests/ -v
```

### 启动服务

```bash
python -m src.serve.main
# 访问 http://localhost:8000
```

### 训练（CPU）

```bash
python scripts/train_lora_cpu.py
# 预计 5-8 小时，每 50 步自动保存 checkpoint
# 支持断点恢复：中断后重新运行会自动从最近 checkpoint 接着跑
```

## 项目结构

```
├── docs/                       # 文档
│   ├── PRD.md                  # 产品需求文档 v2.1
│   ├── feasibility-report.md   # 可行性验证报告
│   └── superpowers/plans/      # 实现计划
├── src/                        # 源代码
│   ├── data/                   # 数据处理
│   │   ├── extract.py          # PDF 批量提取 + 质量报告
│   │   ├── clean.py            # 文本清洗（去噪 + 分章）
│   │   ├── sft_format.py       # SFT 数据格式化
│   │   ├── sft_generate.py     # SFT 数据生成（1179 条）
│   │   ├── formulas_db.py      # 伤寒论 60+ 方剂数据库
│   │   └── kg_build.py         # 知识图谱构建 + 查询 + 持久化
│   ├── rag/                    # RAG 管道
│   │   ├── embed.py            # Embedding（SentenceTransformer 单例）
│   │   ├── retrieve.py         # 向量检索（ChromaDB）
│   │   ├── generate.py         # 生成器（Ollama 接口）
│   │   ├── local_generate.py   # 本地推理器（Transformers）
│   │   └── pipeline.py         # RAG 管道（检索→生成→SSE 流式）
│   ├── eval/                   # 评测
│   │   ├── metrics.py          # recall@5, 关键词命中率, 延迟
│   │   └── runner.py           # 评测运行器
│   └── serve/                  # Web 服务
│       ├── api.py              # FastAPI 路由（health/graph/chat）
│       ├── frontend.html       # 前端页面
│       └── main.py             # 服务入口
├── scripts/                    # 脚本
│   ├── train_lora_cpu.py       # CPU LoRA 训练（断点恢复）
│   ├── train_lora.py           # GPU/Colab 训练
│   ├── colab_train.ipynb       # Colab 一键训练 + 转换
│   ├── convert_to_gguf.py      # HF → GGUF 转换
│   ├── convert_to_ollama.py    # 端到端：merge → GGUF → Q4 → Ollama
│   └── run_p1_eval.py          # P1 评测 + P0 对比
├── data/                       # 数据
│   ├── eval/eval_50.jsonl      # 50 题评测集（5 类别）
│   ├── processed/              # 处理后数据
│   │   ├── classics/shanghan_clauses.jsonl  # 330 条伤寒论原文
│   │   ├── sft_train_p1.jsonl               # 1179 条 SFT 训练数据
│   │   └── p0_baseline_report.json          # P0 评测报告
│   ├── extract_content.py      # [历史] Day 1 PDF 提取验证
│   ├── extract_test.py         # [历史] Day 1 质量检测
│   ├── kg_test.py              # [历史] Day 2 知识图谱验证
│   └── raw/                    # 原始数据（gitignored）
├── tests/                      # 测试（68 个用例）
│   └── unit/                   # 单元测试 + 集成测试
├── pyproject.toml              # 依赖管理
├── .gitignore
└── README.md
```

## 三阶段渐进规划

| 阶段 | 做什么 | 产出 | 状态 |
|------|--------|------|------|
| P0 Baseline | 纯 RAG + 原始模型 | 基线成绩 | ✅ 完成 |
| P1 核心微调 | LoRA + 向量 RAG | 微调增量对比 | ⏳ 训练中 |
| P2 进阶 | GraphRAG 升级 | 图谱 vs 向量差异 | 待开始 |

## 文档

- [PRD v2.1](docs/PRD.md) — 产品需求文档：产品定位、技术选型、9 个功能需求、系统架构、评测体系、模型能力边界
- [可行性验证报告](docs/feasibility-report.md) — 两天 spike 验证结果

## 数据来源

| 来源 | 内容 | 获取方式 |
|------|------|---------|
| qhsxzh.com | 人纪系列 5 本 PDF（针灸/内经/本草/伤寒/金匮） | 直接下载 |
| 维基文库 | 伤寒论原文（公版） | WebFetch |
| nihaixia.org | 完整资料包 13 文件 | 百度网盘 |

> 原始数据文件（PDF、提取文本等）已 gitignore，仅本地保存。

## License

MIT
