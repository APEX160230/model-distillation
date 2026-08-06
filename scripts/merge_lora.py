"""合并 LoRA adapter 到 float16 基座模型

4-bit QLoRA 训练后无法直接 merge_and_unload()，
需要用 float16 重新加载基座模型，应用 LoRA adapter，然后合并保存。

用法：
    python scripts/merge_lora.py
"""
import os
import sys
import time
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

BASE_MODEL = "models/qwen25-15b-base"
LORA_PATH = "./output_lora"
MERGED_DIR = "./output_merged"


def main():
    print("=" * 60, flush=True)
    print("合并 LoRA adapter → float16 基座模型", flush=True)
    print("=" * 60, flush=True)

    # 检查路径
    if not Path(LORA_PATH).exists():
        print(f"ERROR: LoRA adapter 不存在: {LORA_PATH}", flush=True)
        sys.exit(1)
    if not Path(BASE_MODEL).exists():
        print(f"ERROR: 基座模型不存在: {BASE_MODEL}", flush=True)
        sys.exit(1)

    # 如果 output_merged 已存在且有效，跳过
    merged_config = Path(MERGED_DIR) / "config.json"
    if merged_config.exists():
        print(f"\n{MERGED_DIR} 已存在，跳过合并。", flush=True)
        return

    # 1. 加载 tokenizer
    print("\n[1/4] 加载 tokenizer...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    print(f"  Vocab: {tokenizer.vocab_size}", flush=True)

    # 2. 加载基座模型 (float16)
    # 优先 GPU，回退 CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.bfloat16
    print(f"\n[2/4] 加载基座模型 ({device}, {dtype})...", flush=True)
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=dtype,
        device_map=device,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    print(f"  加载耗时: {time.time()-t0:.1f}s", flush=True)
    if device == "cuda":
        vram = torch.cuda.mem_get_info()
        print(f"  VRAM: {(vram[1]-vram[0])/1024**3:.2f}GB used / {vram[1]/1024**3:.1f}GB total", flush=True)

    # 3. 加载 LoRA adapter
    print(f"\n[3/4] 加载 LoRA adapter: {LORA_PATH}...", flush=True)
    model = PeftModel.from_pretrained(model, LORA_PATH)
    print("  合并 adapter...", flush=True)
    model = model.merge_and_unload()
    print("  合并完成!", flush=True)

    # 4. 保存合并后的模型
    print(f"\n[4/4] 保存合并模型到 {MERGED_DIR}...", flush=True)
    model.save_pretrained(MERGED_DIR, safe_serialization=True, max_shard_size="2GB")
    tokenizer.save_pretrained(MERGED_DIR)
    print(f"  保存完成!", flush=True)

    # 验证
    saved_files = list(Path(MERGED_DIR).glob("*"))
    total_size = sum(f.stat().st_size for f in saved_files if f.is_file()) / 1024**3
    print(f"\n合并模型文件数: {len(saved_files)}", flush=True)
    print(f"总大小: {total_size:.2f}GB", flush=True)

    print("\n" + "=" * 60, flush=True)
    print("合并完成! 可以运行评测:", flush=True)
    print(f"  python scripts/run_p1_eval.py", flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()
