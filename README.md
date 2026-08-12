# 中医经典知识检索与理解助手

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![CI/CD](https://github.com/APEX160230/model-distillation/actions/workflows/ci.yml/badge.svg)](https://github.com/APEX160230/model-distillation/actions/workflows/ci.yml)

基于 LoRA 微调 + GraphRAG 的中医经典知识检索系统，可部署在 2 核 4G 的端侧服务器上。

## 项目简介

用倪海厦医师讲稿微调 Qwen2.5-1.5B 模型，使其回答带有中医讲解风格；用知识图谱实现方剂-药材-证型的多跳关联查询；用向量库实现经典原文语义检索。三者结合，构建一个能检索、能解释、能关联的中医经典知识助手。

**这不是问诊系统**，而是中医经典知识的检索与理解工具——检索原文、解释方剂、关联药证。

## 特性

- **混合检索（Hybrid RAG）：** BM25 关键词 + 向量语义 + 知识图谱三路召回，路由查询类型后融合重排，recall@5 从 5.4% 提升至 100%
- **口语化风格（LoRA 微调）：** 基于倪海厦讲稿风格做 SFT，回答像老师讲课一样通俗易懂
- **多跳关联（GraphRAG）：** 方剂-药材-证型知识图谱（269 节点 / 518 边），支持「此方治何证」「此证用何药」等多跳查询
- **端侧部署：** 2 核 4G 云服务器全链路跑通，内存预算约 3.1 GB，零 API 成本
- **CI/CD：** push main 自动执行 74 个测试 + 脚本校验 + 前端冒烟 + 自动部署到服务器

## 技术栈

| 维度 | 选型 | 说明 |
|------|------|------|
| 基座模型 | Qwen2.5-1.5B-Instruct | Q4 量化 ~1.2 GB，2 核 4G 可跑 |
| 微调方法 | LoRA (rank=8) | 仅训练 0.14% 参数，云端 RTX 3090 约 30 分钟 |
| 推理方案 | Ollama | 本地推理，零 API 成本 |
| 向量库 | ChromaDB + bge-small-zh-v1.5 | 嵌入式，条文语义检索 |
| 关键词检索 | BM25 (rank_bm25) | 术语精确命中 |
| 图谱引擎 | NetworkX | 内存图，多跳关联查询 |
| Web 框架 | FastAPI + SSE | 流式输出 |
| 部署环境 | 2 核 4G 云服务器 | nginx + systemd + Docker |

## 项目状态

- [x] 可行性验证（PDF 提取 + 知识图谱查询）
- [x] PRD v2.2
- [x] P0 纯 RAG Baseline（recall@5 = 5.4%）
- [x] P1 LoRA 微调 + 向量 RAG
- [x] P2 混合检索（BM25 + 向量）+ GraphRAG
- [x] P2.1 ConceptGraph 概念映射升级
- [x] P3 Web 应用 + 云端部署（2 核 4G）
- [x] CI/CD 流水线（测试 + 自动部署）
- [ ] 公众号开源介绍文章

## 评测结果

### 各阶段对比（50 题评测集）

| 指标 | P0 基线 | P1 LoRA | P2 Hybrid | P2.1 ConceptGraph | 提升 |
|------|---------|---------|-----------|-------------------|------|
| recall@5 | 5.41% | 5.41% | 83.78% | **100%** | +1748% |
| 关键词命中率 | 18.3% | 23.2% | 50.79% | **72.06%** | +294% |

### 真实用户场景（10 题口语化问答）

| 指标 | 数值 |
|------|------|
| recall@5 | 90.5% |
| LLM judge 均分 | 3.56 / 5 |
| 查询路由准确率 | 100% |

> 完整报告见 [P1 评测报告](docs/p1_eval_report.html) 与 [P2 评测报告](docs/p2_eval_report.html)。

## 系统架构

```
用户浏览器
    │  https (nginx 443)
    ▼
FastAPI (8002, systemd)
    ├── QueryRouter     查询路由（方剂/症状/原文/概念）
    ├── ConceptMapper   概念映射（同义词归一化）
    ├── BM25            关键词检索
    ├── VectorRetriever 向量检索（ChromaDB）
    ├── GraphRAG        知识图谱多跳查询（269 节点 / 518 边）
    └── Generator ──► Ollama (127.0.0.1:11434) ──► tcm-model
```

## 快速开始

### 环境要求

- Python 3.11+
- 推理部署：2 核 CPU / 4 GB RAM（无需 GPU）
- 训练：云端 GPU（RTX 3090，约 30 分钟）或本地 CPU（较慢）

### 安装

```bash
git clone https://github.com/APEX160230/model-distillation.git
cd model-distillation
python -m venv .venv
.venv/Scripts/activate    # Windows
# source .venv/bin/activate  # Linux/Mac

pip install -e ".[rag,serve,dev]"
```

### 运行测试

```bash
pytest tests/unit tests/integration -q
```

### 启动服务

```bash
# 1. 构建向量库（首次或条文数据变更时）
python scripts/build_chroma.py

# 2. 启动服务（需要本机 Ollama 已加载 tcm-model）
python -m src.serve.main
# 访问 http://localhost:8000
```

## 部署

- [部署指南](docs/deployment.md) — 裸机（nginx + systemd）/ Docker 两种方式，含内存预算与故障排查
- CI/CD 自动部署：push main 后由 GitHub Actions 执行 [scripts/deploy.sh](scripts/deploy.sh)（测试通过 → 上传源码 → 幂等重建向量库 → 重启服务 → 健康检查）

## 训练

- [云端训练指南](docs/cloud_training_guide.html) — AutoDL RTX 3090 一键训练（约 30 分钟，自动合并 + 导出 GGUF）
- 脚本：`scripts/cloud_train.py`（bf16 LoRA）+ `scripts/cloud_run.sh`（一键执行）
- 本地 CPU 训练：`scripts/train_lora_cpu.py`（支持断点恢复）
- 云端 GGUF 修复：`scripts/fix_cloud_gguf.sh`（manifest 定位 blob）

## 项目结构

```
├── docs/                       # 文档
│   ├── PRD.md                  # 产品需求文档 v2.2
│   ├── deployment.md           # 部署指南
│   ├── cloud_training_guide.html # 云端训练指南
│   ├── feasibility-report.md   # 可行性验证报告
│   ├── p1_eval_report.html     # P1 评测报告
│   └── p2_eval_report.html     # P2 评测报告
├── src/                        # 源代码
│   ├── data/                   # 数据处理
│   │   ├── extract.py          # PDF 批量提取 + 质量报告
│   │   ├── clean.py            # 文本清洗（去噪 + 分章）
│   │   ├── sft_format.py       # SFT 数据格式化
│   │   ├── sft_generate.py     # SFT 数据生成
│   │   ├── formulas_db.py      # 伤寒论方剂数据库（62 味药 / 71 证型）
│   │   ├── formulas_data.py    # 方剂数据结构化数据
│   │   └── kg_build.py         # 知识图谱构建 + 查询 + 持久化
│   ├── rag/                    # 检索与生成
│   │   ├── embed.py            # Embedding（bge-small-zh-v1.5）
│   │   ├── retrieve.py         # 向量检索（ChromaDB）
│   │   ├── bm25.py             # BM25 关键词检索
│   │   ├── concept_mapper.py   # 概念映射（同义词归一化）
│   │   ├── query_router.py     # 查询路由
│   │   ├── hybrid_retriever.py # 混合检索（路由 + 融合重排）
│   │   ├── graph_rag.py        # GraphRAG 多跳查询
│   │   ├── generate.py         # Ollama 生成器（SSE 流式）
│   │   ├── local_generate.py   # 本地推理器（Transformers）
│   │   └── pipeline.py         # RAG 管道（检索→生成→SSE）
│   ├── eval/                   # 评测
│   │   ├── metrics.py          # recall@5, 关键词命中率, 延迟
│   │   ├── llm_judge.py        # LLM-as-judge 打分
│   │   └── runner.py           # 评测运行器
│   └── serve/                  # Web 服务
│       ├── api.py              # FastAPI 路由（health/graph/chat）
│       ├── frontend.html       # 前端页面
│       └── main.py             # 服务入口
├── scripts/                    # 训练/评测/部署脚本
│   ├── cloud_train.py          # 云端训练（AutoDL RTX 3090）
│   ├── cloud_run.sh            # 云端一键执行
│   ├── train_lora_cpu.py       # 本地 CPU 训练（断点恢复）
│   ├── build_chroma.py         # 向量库构建（幂等）
│   ├── deploy.sh               # CI/CD 部署脚本
│   ├── extract_sft_from_pdf.py # SFT 数据提取（4400 条）
│   ├── refine_sft_data.py      # SFT 数据精修
│   └── ...                     # 评测、转换、合并等
├── data/                       # 数据
│   ├── eval/                   # 评测集（50 题 + 10 题真实场景）
│   ├── processed/              # 处理后数据（条文 330 条、SFT 3960+440 条）
│   └── raw/                    # 原始数据（gitignored）
├── tests/                      # 测试（68 单元 + 6 集成）
├── pyproject.toml              # 依赖管理
├── Dockerfile / docker-compose.yml
├── Modelfile                   # Ollama 模型模板
└── README.md
```

## 三阶段渐进规划

| 阶段 | 做什么 | 产出 | 状态 |
|------|--------|------|------|
| P0 Baseline | 纯 RAG + 原始模型 | 基线成绩（recall@5 = 5.4%） | ✅ 完成 |
| P1 核心微调 | LoRA + 向量 RAG | 微调增量对比 | ✅ 完成 |
| P2 进阶 | 混合检索 + GraphRAG | 图谱 vs 向量差异（recall@5 = 100%） | ✅ 完成 |
| P2.1 | ConceptGraph 概念映射 | 真实用户场景评测 | ✅ 完成 |
| P3 上线 | Web 应用 + 云端部署 | 线上服务（2 核 4G） | ✅ 完成 |
| P0-3 CI/CD | 自动化流水线 | 测试 + 自动部署 | ✅ 完成 |

## 文档

- [PRD v2.2](docs/PRD.md) — 产品需求文档：产品定位、技术选型、9 个功能需求、系统架构、评测体系、模型能力边界
- [部署指南](docs/deployment.md) — 裸机 / Docker 部署、内存预算、故障排查
- [云端训练指南](docs/cloud_training_guide.html) — AutoDL 一键训练
- [可行性验证报告](docs/feasibility-report.md) — 两天 spike 验证结果
- [产品审计](docs/product-audit.md) — 产品化差距分析

## 数据来源

| 来源 | 内容 | 获取方式 |
|------|------|---------|
| qhsxzh.com | 人纪系列 5 本 PDF（针灸/内经/本草/伤寒/金匮） | 直接下载 |
| 维基文库 | 伤寒论原文（公版） | WebFetch |
| nihaixia.org | 完整资料包 13 文件 | 百度网盘 |

> 原始数据文件（PDF、提取文本等）已 gitignore，仅本地保存。

## License

[MIT](LICENSE)
