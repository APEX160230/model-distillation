#!/usr/bin/env python3
"""重建伤寒论向量检索库（幂等）

设计：data/chroma 为可重建资产，不进 git（见 .gitignore）。
CI/CD 部署时若条文数据变更，由本脚本在服务器端重建向量库。

用法:
    python scripts/build_chroma.py
        # 默认: data/processed/classics/shanghan_clauses.jsonl -> data/chroma
    python scripts/build_chroma.py --jsonl PATH --persist-dir PATH
"""
import argparse
import sys
from pathlib import Path

DEFAULT_JSONL = "data/processed/classics/shanghan_clauses.jsonl"
DEFAULT_PERSIST = "data/chroma"


def main() -> int:
    parser = argparse.ArgumentParser(description="重建伤寒论向量检索库")
    parser.add_argument("--jsonl", default=DEFAULT_JSONL, help="条文 JSONL 路径")
    parser.add_argument("--persist-dir", default=DEFAULT_PERSIST, help="ChromaDB 持久化目录")
    parser.add_argument("--model", default=None,
                        help="嵌入模型名或本地路径（默认 BAAI/bge-small-zh-v1.5，"
                             "服务器可传本地已下载的模型目录避免联网）")
    args = parser.parse_args()

    jsonl = Path(args.jsonl)
    if not jsonl.exists():
        print(f"ERROR: 条文文件不存在: {jsonl}", file=sys.stderr)
        return 1

    # 允许在项目子目录下执行时仍能找到 src 包
    root = Path(__file__).resolve().parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from src.rag.retrieve import VectorRetriever
    from src.rag.embed import Embedder

    embedder = Embedder(model_name=args.model) if args.model else None

    print(f"==> 重建向量库: {jsonl} -> {args.persist_dir}"
          + (f" (model={args.model})" if args.model else ""))
    retriever = VectorRetriever(
        persist_dir=args.persist_dir,
        embedder=embedder,
    )
    retriever.build_index(str(jsonl))
    print(f"==> 完成: {retriever.count()} 条条文已入库")

    # 自检：随机取一条条文查询应能命中
    with open(jsonl, "r", encoding="utf-8") as f:
        first = __import__("json").loads(f.readline())
    probe = first.get("original_text", "")[:12]
    hits = retriever.query(probe, top_k=1)
    if not hits:
        print("ERROR: 自检查询未命中任何条文", file=sys.stderr)
        return 1
    print(f"==> 自检通过: 查询 {probe!r} -> 命中条 {hits[0].clause_id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
