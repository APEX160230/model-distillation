# 中医经典知识检索与理解助手

基于 LoRA 微调 + GraphRAG 的中医经典知识检索系统，部署在 2 核 4G 服务器上。

## 项目简介

用倪海厦医师讲稿微调 Qwen2.5-1.5B 模型，使其回答带有中医讲解风格；用知识图谱实现方剂-药材-证型的多跳关联查询；用向量库实现经典原文语义检索。三者结合，构建一个能检索、能解释、能关联的中医经典知识助手。

## 技术栈

| 维度 | 选型 |
|------|------|
| 基座模型 | Qwen2.5-1.5B-Instruct（Q4 量化，~1.2GB） |
| 微调方法 | LoRA（低秩适配器） |
| 推理方案 | Ollama 本地推理 |
| 向量库 | ChromaDB（嵌入式） |
| 图谱引擎 | NetworkX（内存图） |
| Web 框架 | FastAPI + SSE 流式输出 |
| 部署环境 | 2 核 4G 云服务器 |

## 项目状态

- [x] 可行性验证（PDF 提取 + 知识图谱查询）
- [x] PRD v2.1
- [x] M1 数据流水线（PDF 提取 → 文本清洗 → SFT 构建 → 知识图谱 → 评测集）
- [ ] P0 纯 RAG Baseline
- [ ] P1 LoRA + 向量 RAG
- [ ] P2 GraphRAG 升级
- [ ] 公众号开源介绍文章

## 文档

- [PRD v2.1](docs/PRD.md) — 产品需求文档，覆盖产品定位、技术选型、9 个功能需求、系统架构、三阶段渐进规划、评测体系、模型能力边界分析
- [可行性验证报告](docs/feasibility-report.md) — 两天 spike 验证结果

## 数据来源

| 来源 | 内容 | 获取方式 |
|------|------|---------|
| qhsxzh.com | 人纪系列 5 本 PDF（针灸/内经/本草/伤寒/金匮） | 直接下载 |
| 维基文库 | 伤寒论原文（公版） | WebFetch |
| nihaixia.org | 完整资料包 13 文件 | 百度网盘 |

> 原始数据文件（PDF、提取文本等）已 gitignore，仅本地保存。

## 项目结构

```
├── docs/                  # PRD、可行性报告、实现计划
│   ├── PRD.md
│   ├── feasibility-report.md
│   └── superpowers/plans/
├── src/                   # 源代码
│   ├── data/              # 数据处理模块
│   │   ├── extract.py         # PDF 批量提取 + 质量报告
│   │   ├── clean.py           # 文本清洗（去噪 + 分章）
│   │   ├── sft_format.py      # SFT 数据格式化（规则法）
│   │   ├── kg_build.py        # 知识图谱构建 + 查询 + 持久化
│   │   └── formulas_data.py   # 伤寒论 113 方剂数据
│   └── eval/             # 评测模块
│       └── dataset.py         # 评测集加载器
├── data/                  # 数据文件
│   ├── eval/             # 评测集
│   │   └── eval_50.jsonl      # 50 题评测集（5 类别）
│   ├── extract_test.py   # Day 1 验证脚本
│   ├── kg_test.py        # Day 2 验证脚本
│   └── raw/              # 原始数据（gitignored）
├── tests/                 # 测试（58 个测试用例）
│   ├── conftest.py       # 公共 fixture
│   └── unit/             # 单元测试 + 集成测试
├── pyproject.toml         # 依赖管理
└── README.md
```

## License

MIT
