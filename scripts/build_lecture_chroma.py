#!/usr/bin/env python3
"""重建倪师讲稿向量库（PRD v3.0 §5 FR4，幂等）

讲稿素材 = SFT 训练数据的 output 字段（已清洗的倪师讲解原文），
供辨证链路第三层【讲解与调理】引用倪师原话。

用法:
    python scripts/build_lecture_chroma.py
        # 默认: data/processed/sft_train_final.jsonl -> data/chroma (collection: lectures)
    python scripts/build_lecture_chroma.py --jsonl PATH --persist-dir PATH --model PATH
"""
import argparse
import sys
from pathlib import Path

DEFAULT_JSONL = "data/processed/sft_train_final.jsonl"
DEFAULT_PERSIST = "data/chroma"


def main() -> int:
    parser = argparse.ArgumentParser(description="重建倪师讲稿向量库")
    parser.add_argument("--jsonl", default=DEFAULT_JSONL, help="讲稿 JSONL 路径")
    parser.add_argument("--persist-dir", default=DEFAULT_PERSIST, help="ChromaDB 持久化目录")
    parser.add_argument("--model", default=None,
                        help="嵌入模型名或本地路径（服务器可传本地已下载目录避免联网）")
    args = parser.parse_args()

    jsonl = Path(args.jsonl)
    if not jsonl.exists():
        print(f"ERROR: 讲稿文件不存在: {jsonl}", file=sys.stderr)
        return 1

    root = Path(__file__).resolve().parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from src.rag.retrieve import VectorRetriever
    from src.rag.embed import Embedder

    embedder = Embedder(model_name=args.model) if args.model else None

    print(f"==> 重建讲稿库: {jsonl} -> {args.persist_dir}/lectures"
          + (f" (model={args.model})" if args.model else ""))
    retriever = VectorRetriever(
        persist_dir=args.persist_dir,
        collection_name="lectures",
        embedder=embedder,
    )
    retriever.build_lecture_index(str(jsonl))
    print(f"==> 完成: {retriever.count()} 条讲稿片段已入库")

    # 自检：检索"太阳伤寒怕冷无汗"应命中伤寒讲解片段
    probe = "太阳伤寒，怕冷，无汗，身疼痛"
    hits = retriever.query(probe, top_k=1)
    if not hits:
        print("ERROR: 自检查询未命中任何讲稿片段", file=sys.stderr)
        return 1
    print(f"==> 自检通过: 命中 book={hits[0].metadata.get('book')} 片段 {len(hits[0].text)} 字")
    return 0


if __name__ == "__main__":
    sys.exit(main())
