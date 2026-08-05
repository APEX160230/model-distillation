"""本地模型生成器 — 使用 transformers 直接推理

用于 P1 评测：加载 LoRA 微调后的合并模型，用 transformers 进行推理。
与 Ollama Generator 接口完全一致，可直接替换。

优势：不需要 GGUF 转换，直接用 HF 格式的微调模型评测
劣势：bf16 推理比 Q4_K_M 慢，内存占用更大（~3GB vs ~1GB）
"""
from __future__ import annotations

from typing import Iterator
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.rag.retrieve import RetrievalResult
from src.rag.generate import SYSTEM_PROMPT, build_prompt


class LocalGenerator:
    """本地 transformers 生成器

    使用示例:
        gen = LocalGenerator("output_merged")
        answer = gen.generate("桂枝汤的组成", docs)
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cpu",
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        self._model_path = model_path
        self._device = device
        self._dtype = dtype
        self._model: AutoModelForCausalLM | None = None
        self._tokenizer: AutoTokenizer | None = None

    @property
    def model(self) -> AutoModelForCausalLM:
        """延迟加载模型"""
        if self._model is None:
            print(f"加载本地模型: {self._model_path} ({self._dtype})...")
            t0 = time.time()
            self._model = AutoModelForCausalLM.from_pretrained(
                self._model_path,
                torch_dtype=self._dtype,
                device_map=self._device,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )
            self._model.eval()
            load_time = time.time() - t0
            mem_mb = sum(
                p.nelement() * p.element_size() for p in self._model.parameters()
            ) / 1024 / 1024
            print(f"模型加载完成: {load_time:.1f}s, 内存: {mem_mb:.0f}MB")
        return self._model

    @property
    def tokenizer(self) -> AutoTokenizer:
        """延迟加载 tokenizer"""
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(
                self._model_path, trust_remote_code=True, padding_side="left"
            )
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token
        return self._tokenizer

    def generate(
        self,
        question: str,
        docs: list[RetrievalResult],
        temperature: float = 0.7,
        max_tokens: int = 512,
    ) -> str:
        """同步生成回答"""
        prompt = build_prompt(question, docs)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.tokenizer(text, return_tensors="pt").to(self._device)

        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=0.9,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        # 只取新生成的 token
        new_tokens = output[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    def stream_generate(
        self,
        question: str,
        docs: list[RetrievalResult],
        temperature: float = 0.7,
        max_tokens: int = 512,
    ) -> Iterator[str]:
        """流式生成回答（逐 token yield）"""
        prompt = build_prompt(question, docs)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.tokenizer(text, return_tensors="pt").to(self._device)

        with torch.no_grad():
            # 使用 TextIteratorStreamer 实现流式
            from transformers import TextIteratorStreamer
            from threading import Thread

            streamer = TextIteratorStreamer(
                self.tokenizer,
                skip_prompt=True,
                skip_special_tokens=True,
            )

            generation_kwargs = {
                **inputs,
                "max_new_tokens": max_tokens,
                "temperature": temperature,
                "top_p": 0.9,
                "do_sample": temperature > 0,
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
                "streamer": streamer,
            }

            thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
            thread.start()

            for text_chunk in streamer:
                if text_chunk:
                    yield text_chunk

            thread.join()
