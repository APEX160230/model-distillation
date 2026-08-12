"""Embedding 模块

使用 sentence-transformers 加载 bge-small-zh-v1.5 中文嵌入模型。
输出 L2 归一化的 512 维向量，用于 ChromaDB 向量检索。
"""
import os
import threading

# 国内环境 HuggingFace 直连超时，使用镜像
if "HF_ENDPOINT" not in os.environ:
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import numpy as np
from sentence_transformers import SentenceTransformer

# 模块级单例：SentenceTransformer 加载 torch，Windows 上重复加载会 segfault
# 生产环境中也只需要一份模型实例（线程安全，encode 无状态）
_model_cache: dict[str, SentenceTransformer] = {}
_model_lock = threading.Lock()


def _get_model(model_name: str) -> SentenceTransformer:
    """获取或创建模型单例（线程安全）"""
    if model_name not in _model_cache:
        with _model_lock:
            if model_name not in _model_cache:
                _model_cache[model_name] = SentenceTransformer(model_name)
    return _model_cache[model_name]


class Embedder:
    """中文文本嵌入器

    默认使用 BAAI/bge-small-zh-v1.5:
    - 512 维
    - ~100MB 模型大小
    - 中文检索效果优秀
    - 首次加载自动从 HuggingFace 下载

    模型实例全局共享（单例），避免重复加载 torch 导致内存泄漏或 segfault。

    使用示例:
        emb = Embedder()
        vec = emb.encode("太阳之为病，脉浮，头项强痛而恶寒。")
        vecs = emb.encode_batch(["桂枝汤主之", "麻黄汤主之"])
    """

    def __init__(self, model_name: str | None = None):
        """初始化嵌入器

        Args:
            model_name: 模型名或本地模型目录路径。为 None 时读取环境变量
                ``TCM_EMBED_MODEL``（生产环境用绝对路径指向本地模型目录，
                避免服务进程在线访问 HuggingFace），未设置则回退默认
                ``BAAI/bge-small-zh-v1.5``。
        """
        if model_name is None:
            model_name = os.environ.get("TCM_EMBED_MODEL", "BAAI/bge-small-zh-v1.5")
        self.model_name = model_name

    @property
    def model(self) -> SentenceTransformer:
        """延迟加载模型（首次使用时下载，全局单例共享）"""
        return _get_model(self.model_name)

    @property
    def dim(self) -> int:
        """嵌入维度"""
        return self.model.get_embedding_dimension()

    def encode(self, text: str) -> np.ndarray:
        """编码单条文本

        Args:
            text: 输入文本

        Returns:
            L2 归一化的嵌入向量, shape=(dim,)
        """
        vec = self.model.encode(text, normalize_embeddings=True)
        return np.asarray(vec, dtype=np.float32)

    def encode_batch(self, texts: list[str]) -> np.ndarray:
        """批量编码文本

        Args:
            texts: 文本列表

        Returns:
            L2 归一化的嵌入矩阵, shape=(len(texts), dim)

        Raises:
            ValueError: texts 为空
        """
        if not texts:
            raise ValueError("texts must not be empty")
        vecs = self.model.encode(
            texts,
            normalize_embeddings=True,
            batch_size=32,
            show_progress_bar=False,
        )
        return np.asarray(vecs, dtype=np.float32)
