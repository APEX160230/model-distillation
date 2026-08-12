"""集成测试：mock Ollama HTTP 服务，验证 Generator 真实 HTTP 链路

在 CI 中不加载真实 1.5B 模型（2核4G 跑不动），而是起一个本地 HTTP server
模拟 Ollama /api/chat 端点，验证 Generator 的请求构造、JSON/流式解析、错误处理。
"""
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from src.rag.retrieve import RetrievalResult
from src.rag.generate import Generator

CHUNKS = [
    {"message": {"content": "这个方子"}},
    {"message": {"content": "由桂枝"}},
    {"message": {"content": "芍药组成。"}},
    {"message": {"content": ""}},
]

last_request: dict | None = None


class _FakeOllamaHandler(BaseHTTPRequestHandler):
    """模拟 Ollama /api/chat 端点"""

    def do_POST(self):
        global last_request
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")
        last_request = body

        if body.get("model", "").endswith("-error"):
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"error": "internal error"}')
            return

        stream = body.get("stream", False)
        if stream:
            # 流式响应：ndjson，每行一个 JSON
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson")
            self.end_headers()
            for chunk in CHUNKS:
                self.wfile.write(json.dumps(chunk, ensure_ascii=False).encode() + b"\n")
        else:
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            resp = {"message": {"content": "这是模拟的完整回答", "role": "assistant"}}
            self.wfile.write(json.dumps(resp, ensure_ascii=False).encode())

    def log_message(self, *args):  # 静默日志
        pass


@pytest.fixture(scope="module")
def fake_ollama_url():
    server = HTTPServer(("127.0.0.1", 0), _FakeOllamaHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()
    thread.join(timeout=5)


def _make_doc():
    return [
        RetrievalResult(
            id="clause_12",
            text="太阳中风，阳浮而阴弱……",
            metadata={"clause_id": 12, "chapter": "辨太阳病脉证并治"},
            distance=0.1,
        )
    ]


class TestFakeOllamaNonStream:
    def test_generate_returns_full_answer(self, fake_ollama_url, monkeypatch):
        monkeypatch.setenv("OLLAMA_HOST", fake_ollama_url)
        gen = Generator(model="tcm-model")
        answer = gen.generate("桂枝汤的组成", _make_doc())
        assert answer == "这是模拟的完整回答"

    def test_generate_constructs_correct_request(self, fake_ollama_url, monkeypatch):
        monkeypatch.setenv("OLLAMA_HOST", fake_ollama_url)
        gen = Generator(model="tcm-model")
        gen.generate("桂枝汤的组成", _make_doc(), temperature=0.5, max_tokens=256)
        assert last_request is not None
        assert last_request["model"] == "tcm-model"
        assert last_request["stream"] is False
        assert last_request["options"]["temperature"] == 0.5
        assert last_request["options"]["num_predict"] == 256
        messages = last_request["messages"]
        assert messages[0]["role"] == "system"
        assert messages[-1]["role"] == "user"
        assert "桂枝汤的组成" in messages[-1]["content"]
        assert "第12条" in messages[-1]["content"]  # 检索结果已拼入 prompt


class TestFakeOllamaStream:
    def test_stream_generate_yields_chunks(self, fake_ollama_url, monkeypatch):
        monkeypatch.setenv("OLLAMA_HOST", fake_ollama_url)
        gen = Generator(model="tcm-model")
        chunks = list(gen.stream_generate("桂枝汤的组成", _make_doc()))
        # 空 chunk 被过滤，且保留顺序
        assert chunks == ["这个方子", "由桂枝", "芍药组成。"]
        assert "".join(chunks) == "这个方子由桂枝芍药组成。"

    def test_stream_request_has_stream_true(self, fake_ollama_url, monkeypatch):
        monkeypatch.setenv("OLLAMA_HOST", fake_ollama_url)
        gen = Generator(model="tcm-model")
        list(gen.stream_generate("问题", _make_doc()))
        assert last_request is not None
        assert last_request["stream"] is True


class TestFakeOllamaErrors:
    def test_generate_raises_on_http_error(self, fake_ollama_url, monkeypatch):
        monkeypatch.setenv("OLLAMA_HOST", fake_ollama_url)
        gen = Generator(model="tcm-model-error")  # handler 对 -error 后缀返回 500
        with pytest.raises(Exception):
            gen.generate("问题", _make_doc())

    def test_stream_generate_raises_on_http_error(self, fake_ollama_url, monkeypatch):
        monkeypatch.setenv("OLLAMA_HOST", fake_ollama_url)
        gen = Generator(model="tcm-model-error")
        with pytest.raises(Exception):
            list(gen.stream_generate("问题", _make_doc()))
