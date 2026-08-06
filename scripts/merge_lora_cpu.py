"""合并 LoRA adapter — CPU bfloat16 版本（避免 GPU 内存问题）"""
import os, sys, time, torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

BASE_MODEL = "models/qwen25-15b-base"
LORA_PATH = "./output_lora"
MERGED_DIR = "./output_merged"

print("=" * 60, flush=True)
print("合并 LoRA → bfloat16 基座 (CPU)", flush=True)
print("=" * 60, flush=True)

# 1. tokenizer
print("\n[1/4] tokenizer...", flush=True)
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)

# 2. 基座模型 (bfloat16, CPU, 内存映射)
print("\n[2/4] 加载基座模型 (bfloat16, CPU)...", flush=True)
t0 = time.time()
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.bfloat16,
    device_map="cpu",
    trust_remote_code=True,
    low_cpu_mem_usage=True,
)
print(f"  耗时: {time.time()-t0:.1f}s", flush=True)

# 3. LoRA adapter + merge
print(f"\n[3/4] 加载 + 合并 LoRA adapter...", flush=True)
model = PeftModel.from_pretrained(model, LORA_PATH)
model = model.merge_and_unload()
print("  合并完成!", flush=True)

# 4. 保存
print(f"\n[4/4] 保存到 {MERGED_DIR}...", flush=True)
model.save_pretrained(MERGED_DIR, safe_serialization=True, max_shard_size="2GB")
tokenizer.save_pretrained(MERGED_DIR)

# 验证
files = list(Path(MERGED_DIR).glob("*"))
total = sum(f.stat().st_size for f in files if f.is_file()) / 1024**3
print(f"\n文件数: {len(files)}, 总大小: {total:.2f}GB", flush=True)
print("\n完成! 可以运行评测: python scripts/run_p1_eval.py", flush=True)
