"""向量检索模块

使用 ChromaDB 持久化向量库 + bge-small-zh-v1.5 嵌入模型。
支持从 JSONL 建库、top-k 检索、持久化重载。
"""
import json
import os
from dataclasses import dataclass, field
from pathlib import Path

# 国内环境 HuggingFace 直连超时，使用镜像
if "HF_ENDPOINT" not in os.environ:
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import chromadb
from chromadb.config import Settings

from src.rag.embed import Embedder


@dataclass
class RetrievalResult:
    """单条检索结果"""
    id: str
    text: str
    metadata: dict
    distance: float

    @property
    def clause_id(self) -> int:
        """条辨编号（便捷访问）"""
        return self.metadata.get("clause_id", 0)

    @property
    def chapter(self) -> str:
        """所属篇章"""
        return self.metadata.get("chapter", "")

    @property
    def score(self) -> float:
        """相似度分数（1 - distance，越高越相似）"""
        return 1.0 - self.distance


class VectorRetriever:
    """伤寒论向量检索器

    使用 ChromaDB 持久化存储 + bge-small-zh-v1.5 嵌入。
    嵌入向量由 Embedder 生成，ChromaDB 负责 HNSW 索引和余弦相似度检索。

    使用示例:
        r = VectorRetriever(persist_dir="./data/chroma")
        r.build_index("data/processed/classics/shanghan_clauses.jsonl")
        results = r.query("桂枝汤主治什么", top_k=5)
    """

    def __init__(
        self,
        persist_dir: str = "data/chroma",
        collection_name: str = "shanghan",
    ):
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name

        self._embedder = Embedder()
        self._client = chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=Settings(anonymized_telemetry=False),
        )
        self._collection = None

    @property
    def collection(self):
        """延迟获取或创建 collection"""
        if self._collection is None:
            self._collection = self._client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collection

    def build_index(self, jsonl_path: str) -> None:
        """从 JSONL 文件构建向量索引

        读取条辨数据，批量嵌入后写入 ChromaDB。
        如果 collection 已有数据，先清空再重建（幂等）。

        Args:
            jsonl_path: JSONL 文件路径，每行一个 JSON 对象，
                       必须包含 clause_id 和 original_text 字段
        """
        # 读取条辨数据
        clauses = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    clauses.append(json.loads(line))

        if not clauses:
            raise ValueError(f"No clauses found in {jsonl_path}")

        # 幂等: 先删除已有 collection 再重建
        try:
            self._client.delete_collection(self.collection_name)
        except Exception:
            pass  # collection 不存在时忽略
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        # 准备数据
        texts = [c["original_text"] for c in clauses]
        ids = [f"clause_{c['clause_id']}" for c in clauses]
        metadatas = []
        for c in clauses:
            meta = {
                "clause_id": c["clause_id"],
                "chapter": c.get("chapter", ""),
                "original_text": c["original_text"],
            }
            # ChromaDB metadata 值只能是 str/int/float/bool
            metadatas.append(meta)

        # 批量嵌入
        embeddings = self._embedder.encode_batch(texts)

        # 写入 ChromaDB
        self.collection.add(
            ids=ids,
            embeddings=embeddings.tolist(),
            documents=texts,
            metadatas=metadatas,
        )

    def query(self, text: str, top_k: int = 5) -> list[RetrievalResult]:
        """检索 top-k 最相似的条辨

        Args:
            text: 查询文本
            top_k: 返回条数

        Returns:
            RetrievalResult 列表，按相似度降序排列。

        Raises:
            ValueError: text 为空
        """
        if not text or not text.strip():
            raise ValueError("query text must not be empty")

        query_vec = self._embedder.encode(text)

        results = self.collection.query(
            query_embeddings=[query_vec.tolist()],
            n_results=min(top_k, self.count()),
            include=["metadatas", "distances", "documents"],
        )

        # 解析结果
        items = []
        for i in range(len(results["ids"][0])):
            items.append(RetrievalResult(
                id=results["ids"][0][i],
                text=results["documents"][0][i],
                metadata=results["metadatas"][0][i],
                distance=results["distances"][0][i],
            ))

        return items

    def count(self) -> int:
        """返回索引中的文档数"""
        return self.collection.count()
