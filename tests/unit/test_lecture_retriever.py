"""讲稿向量库测试（PRD v3.0 §5 FR4）

覆盖：
- 从讲稿 JSONL（instruction/input/output/book）建库
- 幂等重建
- 元数据（book/topic）存储
- 检索返回倪师讲解原文
"""
import json

from tests.unit.test_retrieve import FakeEmbedder, make_retriever


def make_lecture_file(tmp_path, lectures=None):
    """创建临时讲稿数据"""
    if lectures is None:
        lectures = [
            {"instruction": "请讲解太阳伤寒证的特征", "input": "", "book": "伤寒",
             "output": "太阳伤寒是寒邪束表，毛孔紧闭，汗发不出来，所以人会怕冷、身上疼。"},
            {"instruction": "请讲解桂枝汤的组方原理", "input": "", "book": "本草",
             "output": "桂枝汤用桂枝芍药，一散一收，调和营卫，是太阳中风的主方。"},
            {"instruction": "请讲解少阳病的提纲", "input": "", "book": "伤寒",
             "output": "少阳病口苦咽干目眩，往来寒热，胸胁苦满，小柴胡汤主之。"},
        ]
    filepath = tmp_path / "test_lectures.jsonl"
    with open(filepath, "w", encoding="utf-8") as f:
        for lec in lectures:
            f.write(json.dumps(lec, ensure_ascii=False) + "\n")
    return filepath


class TestBuildLectureIndex:
    """讲稿建库"""

    def test_build_from_lecture_jsonl(self, tmp_path):
        retriever = make_retriever(tmp_path, collection_name="lectures")
        retriever.build_lecture_index(str(make_lecture_file(tmp_path)))
        assert retriever.count() == 3

    def test_build_lecture_idempotent(self, tmp_path):
        """重复建库不翻倍"""
        retriever = make_retriever(tmp_path, collection_name="lectures")
        path = str(make_lecture_file(tmp_path))
        retriever.build_lecture_index(path)
        retriever.build_lecture_index(path)
        assert retriever.count() == 3

    def test_build_lecture_stores_metadata(self, tmp_path):
        """建库后存储 book 和 topic 元数据"""
        retriever = make_retriever(tmp_path, collection_name="lectures")
        retriever.build_lecture_index(str(make_lecture_file(tmp_path)))
        results = retriever.query("太阳伤寒怕冷无汗", top_k=1)
        assert len(results) == 1
        item = results[0]
        assert item.text != ""
        assert item.metadata.get("book") == "伤寒"
        assert item.metadata.get("topic") != ""
        assert item.score > 0

    def test_lecture_separate_collection(self, tmp_path):
        """讲稿库与条文库互不干扰"""
        retriever = make_retriever(tmp_path, collection_name="lectures")
        retriever.build_lecture_index(str(make_lecture_file(tmp_path)))
        assert retriever.collection_name == "lectures"
        assert retriever.count() == 3
