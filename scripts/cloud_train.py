"""云端 LoRA 微调训练脚本 — Qwen2.5-1.5B-Instruct + 中医 SFT

针对 AutoDL RTX 3090 (24GB VRAM) 优化：
- bf16 全精度 LoRA（非 4-bit QLoRA），24GB 足够 1.5B 模型
- MAX_LENGTH=512（本地 GTX 1650 被迫降到 384，云端恢复）
- batch_size=8, grad_accum=2（effective batch=16，PRD §FR4 要求）
- 训练中评估（24GB 可以承受 eval 时的额外显存）
- 训练后自动合并 LoRA → float16 基座 → 导出 GGUF Q4_K_M

与本地 train_lora_gpu.py 的差异：
  | 参数         | 本地 (GTX 1650)  | 云端 (RTX 3090) |
  |-------------|------------------|-----------------|
  | 量化         | 4-bit nf4 (QLoRA) | 无（bf16 LoRA）  |
  | MAX_LENGTH   | 384              | 512              |
  | batch_size   | 1                | 8                |
  | grad_accum   | 8                | 2                |
  | eval_strategy| no（OOM 风险）    | steps            |
  | dtype        | fp16             | bf16             |
  | save_steps   | 10（频繁防崩溃）   | 50               |
  | num_workers  | 0（Windows）      | 4                |
  | 合并方式      | 需单独 merge脚本  | 训练后自动合并    |

预计耗时: ~30 min (RTX 3090, 3960 samples, 3 epochs)
预计费用: ~5 元 (AutoDL RTX 3090, 1.5元/小时)

用法:
    python cloud_train.py                    # 训练 + 合并
    python cloud_train.py --skip-train       # 跳过训练，只做合并+GGUF
    python cloud_train.py --skip-merge       # 只训练，不合并
"""
import os
import sys
import json
import time
import shutil
import argparse
import subprocess
import torch
from pathlib import Path
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    TrainerCallback,
)
from peft import LoraConfig, get_peft_model, TaskType

# ==================== 配置 ====================
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

# AutoDL 预下载模型到 /root/autodl-tmp/models，避免每次从 HuggingFace 下载
def _find_local_model():
    """搜索本地已下载的模型路径（兼容 ModelScope/HuggingFace 缓存结构）"""
    base = Path("/root/autodl-tmp/models")
    patterns = [
        "**/Qwen--Qwen2.5-1.5B-Instruct/**/config.json",
        "**/Qwen2.5-1.5B-Instruct/**/config.json",
        "**/Qwen/Qwen2.5-1.5B-Instruct/**/config.json",
    ]
    for pattern in patterns:
        for candidate in base.glob(pattern):
            return str(candidate.parent)
    return None

_local_model_path = _find_local_model()
_LOCAL_MODEL = Path("models/qwen25-15b-base")
if _local_model_path:
    MODEL_NAME = _local_model_path
    print(f"[INFO] 使用本地模型: {MODEL_NAME}", flush=True)
elif _LOCAL_MODEL.exists():
    MODEL_NAME = str(_LOCAL_MODEL)
else:
    MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

DATA_PATH = "data/processed/sft_train_final.jsonl"
VAL_DATA_PATH = "data/processed/sft_val_final.jsonl"
OUTPUT_DIR = "./output_lora_v2"
MERGED_DIR = "./output_merged"
MAX_LENGTH = 512  # 云端恢复 512（本地 GTX 1650 被迫降到 384）

# LoRA 配置 (PRD §FR4 对齐)
LORA_RANK = 8
LORA_ALPHA = 32  # PRD 要求 32
LORA_DROPOUT = 0.05
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]

# 训练配置 (PRD §FR4 对齐)
LEARNING_RATE = 2e-4
NUM_EPOCHS = 3
BATCH_SIZE = 8          # PRD 要求 8（本地被迫用 1）
GRAD_ACCUM = 2          # effective batch = 16
WARMUP_RATIO = 0.03
LOGGING_STEPS = 5
EVAL_STEPS = 50
SAVE_STEPS = 50         # 云端稳定，不需要像本地那样每 10 步保存

SYSTEM_PROMPT = (
    "你是一位中医老师，擅长用通俗易懂的方式讲解中医经典知识。"
    "请用口语化的讲解风格，像老师讲课一样回答问题。"
    "引用经典原文时标注条文编号。解释方剂时列出完整组成。"
    "不提供具体诊疗建议。"
)


# ==================== 数据加载与格式化 ====================
def load_sft_data(path: str) -> Dataset:
    with open(path, "r", encoding="utf-8") as f:
        raw = [json.loads(line) for line in f]
    records = []
    for item in raw:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": item["instruction"]},
            {"role": "assistant", "content": item["output"]},
        ]
        records.append({"messages": messages})
    return Dataset.from_list(records)


def format_to_ids(example: dict, tokenizer) -> dict:
    messages = example["messages"]
    full_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    full_ids = tokenizer(
        full_text, truncation=True, max_length=MAX_LENGTH,
        padding=False, return_tensors=None,
    )["input_ids"]

    prefix_text = tokenizer.apply_chat_template(
        messages[:2], tokenize=False, add_generation_prompt=True
    )
    prefix_ids = tokenizer(
        prefix_text, truncation=True, max_length=MAX_LENGTH,
        padding=False, return_tensors=None,
    )["input_ids"]

    labels = [-100] * len(prefix_ids) + full_ids[len(prefix_ids):]
    labels = labels[:len(full_ids)]

    return {
        "input_ids": full_ids,
        "attention_mask": [1] * len(full_ids),
        "labels": labels,
    }


# ==================== 训练 ====================
def train():
    print("=" * 60, flush=True)
    print("云端 LoRA 微调: Qwen2.5-1.5B-Instruct + 中医 SFT", flush=True)
    print("=" * 60, flush=True)

    # 环境检查
    print(f"\nPyTorch: {torch.__version__}", flush=True)
    print(f"CUDA available: {torch.cuda.is_available()}", flush=True)
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available. 云端训练需要 GPU。", flush=True)
        sys.exit(1)
    gpu_name = torch.cuda.get_device_name(0)
    vram_free = torch.cuda.mem_get_info()[0] / 1024**3
    vram_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"GPU: {gpu_name}", flush=True)
    print(f"VRAM: {vram_total:.1f}GB total, {vram_free:.1f}GB free", flush=True)

    # bf16 支持检查
    if not torch.cuda.is_bf16_supported():
        print("WARNING: GPU 不支持 bf16，回退到 fp16", flush=True)
        use_bf16 = False
    else:
        print("bf16 支持: Yes", flush=True)
        use_bf16 = True

    # 数据检查
    if not Path(DATA_PATH).exists():
        print(f"ERROR: 训练数据不存在: {DATA_PATH}", flush=True)
        sys.exit(1)

    # 1. 加载 tokenizer
    print(f"\n[1/7] 加载 tokenizer: {MODEL_NAME}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME, trust_remote_code=True, padding_side="right", local_files_only=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"  Vocab size: {tokenizer.vocab_size}", flush=True)

    # 2. 加载模型 (bf16，不做 4-bit 量化)
    print(f"\n[2/7] 加载模型 (bf16, 无量化)...", flush=True)
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16 if use_bf16 else torch.float16,
        device_map="auto",
        trust_remote_code=True,
        local_files_only=True,
    )
    model.config.use_cache = False
    load_time = time.time() - t0
    vram_after = torch.cuda.mem_get_info()[0] / 1024**3
    print(f"  加载耗时: {load_time:.1f}s", flush=True)
    print(f"  VRAM 使用: {vram_total - vram_after:.2f}GB / {vram_total:.1f}GB", flush=True)
    print(f"  参数量: {model.num_parameters() / 1e6:.1f}M", flush=True)

    # 3. gradient checkpointing
    print("\n[3/7] 启用 gradient checkpointing...", flush=True)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()

    # 4. LoRA 配置
    print(f"\n[4/7] 配置 LoRA (rank={LORA_RANK}, alpha={LORA_ALPHA})...", flush=True)
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=TARGET_MODULES,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 5. 加载数据
    print(f"\n[5/7] 加载 SFT 数据...", flush=True)
    dataset = load_sft_data(DATA_PATH)
    print(f"  训练样本: {len(dataset)}", flush=True)

    val_dataset = None
    if Path(VAL_DATA_PATH).exists():
        val_dataset = load_sft_data(VAL_DATA_PATH)
        print(f"  验证样本: {len(val_dataset)}", flush=True)

    print("\n  Tokenize...", flush=True)
    dataset = dataset.map(
        lambda x: format_to_ids(x, tokenizer),
        remove_columns=dataset.column_names,
        desc="Tokenizing train",
    )
    dataset = dataset.filter(lambda x: len(x["input_ids"]) > 10)
    print(f"  预处理后训练样本: {len(dataset)}", flush=True)

    if val_dataset is not None:
        val_dataset = val_dataset.map(
            lambda x: format_to_ids(x, tokenizer),
            remove_columns=val_dataset.column_names,
            desc="Tokenizing val",
        )
        val_dataset = val_dataset.filter(lambda x: len(x["input_ids"]) > 10)
        print(f"  预处理后验证样本: {len(val_dataset)}", flush=True)

    # 长度统计
    lengths = [len(x["input_ids"]) for x in dataset]
    avg_len = sum(lengths) / len(lengths)
    over_max = sum(1 for l in lengths if l >= MAX_LENGTH)
    print(f"  Token 长度: avg={avg_len:.0f}, max={max(lengths)}, >={MAX_LENGTH}: {over_max} ({over_max/len(lengths)*100:.1f}%)", flush=True)

    # 6. 训练
    print(f"\n[6/7] 开始训练...", flush=True)
    total_steps = (len(dataset) // (BATCH_SIZE * GRAD_ACCUM)) * NUM_EPOCHS
    print(f"  Epochs: {NUM_EPOCHS}", flush=True)
    print(f"  Batch: {BATCH_SIZE} x {GRAD_ACCUM} = {BATCH_SIZE * GRAD_ACCUM} effective", flush=True)
    print(f"  LR: {LEARNING_RATE}", flush=True)
    print(f"  Max length: {MAX_LENGTH}", flush=True)
    print(f"  Estimated steps: ~{total_steps}", flush=True)
    print(f"  Eval every {EVAL_STEPS} steps", flush=True)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LEARNING_RATE,
        warmup_steps=int(WARMUP_RATIO * total_steps),
        logging_steps=LOGGING_STEPS,
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        save_total_limit=3,
        eval_strategy="steps",
        eval_steps=EVAL_STEPS,
        bf16=use_bf16,
        fp16=not use_bf16,
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        report_to="none",
        remove_unused_columns=False,
        dataloader_pin_memory=True,
        dataloader_num_workers=4,   # Linux 支持多进程加载
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        disable_tqdm=False,         # 云端可以显示进度条
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer, padding=True, return_tensors="pt"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )

    # 进度回调
    progress_file = os.path.join(OUTPUT_DIR, "progress.txt")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    class ProgressCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs is None:
                return
            loss = logs.get("loss", "?")
            lr = logs.get("learning_rate", "?")
            eval_loss = logs.get("eval_loss", "")
            step = state.global_step
            epoch = state.epoch
            pct = step / state.max_steps * 100 if state.max_steps else 0
            vram_free = torch.cuda.mem_get_info()[0] / 1024**3 if torch.cuda.is_available() else 0
            eval_str = f" eval_loss={eval_loss}" if eval_loss else ""
            line = f"[step {step}/{state.max_steps}] epoch={epoch:.2f} loss={loss}{eval_str} lr={lr} vram_free={vram_free:.1f}GB ({pct:.1f}%)"
            print(line, flush=True)
            with open(progress_file, "a", encoding="utf-8") as f:
                f.write(line + "\n")

    trainer.add_callback(ProgressCallback())

    # 自动从 checkpoint 恢复
    resume_ckpt = None
    ckpts = sorted(Path(OUTPUT_DIR).glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))
    if ckpts:
        resume_ckpt = str(ckpts[-1])
        print(f"  从 checkpoint 恢复: {resume_ckpt}", flush=True)

    t_start = time.time()
    train_result = trainer.train(resume_from_checkpoint=resume_ckpt)
    train_time = time.time() - t_start

    print(f"\n训练完成!", flush=True)
    print(f"  Final loss: {train_result.training_loss:.4f}", flush=True)
    print(f"  Total time: {train_time/60:.1f} min ({train_time:.0f}s)", flush=True)

    # 保存 LoRA adapter
    print("\n保存 LoRA adapter...", flush=True)
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"  LoRA adapter: {OUTPUT_DIR}/", flush=True)

    # 保存训练统计
    stats = {
        "model": MODEL_NAME,
        "method": "LoRA (bf16, no quantization)",
        "lora_rank": LORA_RANK,
        "lora_alpha": LORA_ALPHA,
        "target_modules": TARGET_MODULES,
        "epochs": NUM_EPOCHS,
        "batch_size": BATCH_SIZE,
        "grad_accum": GRAD_ACCUM,
        "effective_batch_size": BATCH_SIZE * GRAD_ACCUM,
        "lr": LEARNING_RATE,
        "max_length": MAX_LENGTH,
        "train_samples": len(dataset),
        "val_samples": len(val_dataset) if val_dataset else 0,
        "final_train_loss": train_result.training_loss,
        "train_time_seconds": round(train_time, 1),
        "train_time_minutes": round(train_time / 60, 1),
        "system_prompt": SYSTEM_PROMPT,
        "data_source": "PDF oral (sft_train_final.jsonl)",
        "device": f"GPU ({gpu_name})",
        "dtype": "bf16" if use_bf16 else "fp16",
        "gradient_checkpointing": True,
        "cloud": True,
    }
    with open(f"{OUTPUT_DIR}/training_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"  训练统计: {OUTPUT_DIR}/training_stats.json", flush=True)

    return model, tokenizer


# ==================== 合并 LoRA ====================
def merge_lora(model=None, tokenizer=None):
    print("\n" + "=" * 60, flush=True)
    print("合并 LoRA adapter -> float16 基座模型", flush=True)
    print("=" * 60, flush=True)

    # 如果已有 merged 模型，跳过
    if Path(MERGED_DIR).exists() and (Path(MERGED_DIR) / "config.json").exists():
        print(f"{MERGED_DIR} 已存在，跳过合并。", flush=True)
        return

    if model is None:
        # 单独调用合并（跳过训练时）
        print(f"\n加载基座模型: {MODEL_NAME}", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16,
            device_map="cuda",
            trust_remote_code=True,
            local_files_only=True,
            low_cpu_mem_usage=True,
        )
        print(f"加载 LoRA adapter: {OUTPUT_DIR}", flush=True)
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, OUTPUT_DIR)

    print("合并 adapter...", flush=True)
    merged_model = model.merge_and_unload()

    print(f"保存合并模型到 {MERGED_DIR}...", flush=True)
    merged_model.save_pretrained(MERGED_DIR, safe_serialization=True, max_shard_size="2GB")
    tokenizer.save_pretrained(MERGED_DIR)

    # 关键修复：从基座模型复制原始 tokenizer 文件，避免 save_pretrained 写坏 tokenizer.json
    _tokenizer_files = ["tokenizer.json", "tokenizer_config.json", "vocab.json",
                        "merges.txt", "special_tokens_map.json", "chat_template.jinja"]
    _base_dir = Path(MODEL_NAME)
    _copied = 0
    for fname in _tokenizer_files:
        _src = _base_dir / fname
        if _src.exists():
            shutil.copy2(_src, Path(MERGED_DIR) / fname)
            _copied += 1
    print(f"  从基座模型复制了 {_copied} 个 tokenizer 文件", flush=True)

    # 统计
    saved_files = list(Path(MERGED_DIR).glob("*"))
    total_size = sum(f.stat().st_size for f in saved_files if f.is_file()) / 1024**3
    print(f"合并模型: {len(saved_files)} 文件, {total_size:.2f}GB", flush=True)


# ==================== GGUF 转换 + 量化 ====================
def convert_to_gguf():
    print("\n" + "=" * 60, flush=True)
    print("GGUF 转换 + Q4_K_M 量化", flush=True)
    print("=" * 60, flush=True)

    if not Path(MERGED_DIR).exists():
        print(f"ERROR: 合并模型不存在: {MERGED_DIR}", flush=True)
        print("请先运行合并步骤。", flush=True)
        return False

    llama_cpp_dir = Path("./llama.cpp")
    gguf_f16 = "./qwen25-15b-tcm-f16.gguf"
    gguf_q4 = "./qwen25-15b-tcm-q4_k_m.gguf"

    # 1. 克隆 llama.cpp
    print("\n[1/4] 克隆 llama.cpp...", flush=True)
    if not llama_cpp_dir.exists():
        ret = os.system(f"git clone https://github.com/ggerganov/llama.cpp.git {llama_cpp_dir}")
        if ret != 0:
            print("ERROR: git clone llama.cpp 失败", flush=True)
            return False
    else:
        print("  llama.cpp 已存在，跳过", flush=True)

    # 2. 安装转换依赖
    print("\n[2/4] 安装转换依赖...", flush=True)
    req_file = llama_cpp_dir / "requirements" / "requirements-convert_hf_to_gguf.txt"
    if req_file.exists():
        os.system(f"pip install -r {req_file} -q")
    else:
        # 新版 llama.cpp 可能没有这个文件，手动安装
        os.system("pip install sentencepiece protobuf -q")

    # 3. 转换为 GGUF (F16)
    print("\n[3/4] 转换为 GGUF (F16)...", flush=True)
    convert_script = llama_cpp_dir / "convert_hf_to_gguf.py"
    if not convert_script.exists():
        # 兼容旧版路径
        convert_script = llama_cpp_dir / "convert.py"
    ret = os.system(f"python {convert_script} {MERGED_DIR} --outfile {gguf_f16} --outtype f16")
    if ret != 0 or not Path(gguf_f16).exists():
        print("ERROR: GGUF F16 转换失败", flush=True)
        return False
    f16_size = Path(gguf_f16).stat().st_size / 1024**3
    print(f"  F16 GGUF: {f16_size:.2f}GB", flush=True)

    # 4. 量化为 Q4_K_M
    print("\n[4/4] 量化为 Q4_K_M...", flush=True)

    # 编译 llama-quantize
    print("  编译 llama-quantize...", flush=True)
    ret = os.system(f"cd {llama_cpp_dir} && make llama-quantize -j$(nproc) 2>&1 | tail -3")
    if ret != 0:
        # 尝试 cmake
        print("  make 失败，尝试 cmake...", flush=True)
        build_dir = llama_cpp_dir / "build"
        os.system(f"mkdir -p {build_dir} && cd {build_dir} && cmake .. -DLLAMA_LLAMA_CPP=ON 2>&1 | tail -3")
        os.system(f"cd {build_dir} && make llama-quantize -j$(nproc) 2>&1 | tail -3")
        quant_bin = str(build_dir / "bin" / "llama-quantize")
    else:
        quant_bin = str(llama_cpp_dir / "llama-quantize")

    if not Path(quant_bin).exists():
        # 搜索可能的位置
        candidates = list(llama_cpp_dir.rglob("llama-quantize"))
        if candidates:
            quant_bin = str(candidates[0])
        else:
            print("ERROR: llama-quantize 编译失败", flush=True)
            return False

    ret = os.system(f"{quant_bin} {gguf_f16} {gguf_q4} q4_k_m")
    if ret != 0 or not Path(gguf_q4).exists():
        print("ERROR: Q4_K_M 量化失败", flush=True)
        return False

    q4_size = Path(gguf_q4).stat().st_size / 1024**3
    print(f"  Q4_K_M GGUF: {q4_size:.2f}GB", flush=True)

    # 5. 生成 Modelfile
    modelfile_content = '''FROM ./qwen25-15b-tcm-q4_k_m.gguf

TEMPLATE """{{ if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}{{ if .Prompt }}<|im_start|>user
{{ .Prompt }}<|im_end|>
{{ end }}<|im_start|>assistant
{{ .Response }}<|im_end|>
"""

SYSTEM """你是一位中医老师，擅长用通俗易懂的方式讲解中医经典知识。请用口语化的讲解风格，像老师讲课一样回答问题。引用经典原文时标注条文编号。解释方剂时列出完整组成。不提供具体诊疗建议。如果检索结果中没有相关信息，请如实说明。"""

PARAMETER stop "<|im_start|>"
PARAMETER stop "<|im_end|>"
PARAMETER temperature 0.7
PARAMETER top_p 0.8
PARAMETER top_k 20
PARAMETER num_predict 512
'''
    with open("Modelfile", "w", encoding="utf-8") as f:
        f.write(modelfile_content)
    print(f"\n  Modelfile 已生成", flush=True)

    # 6. 打包下载
    print("\n" + "=" * 60, flush=True)
    print("GGUF 转换完成!", flush=True)
    print(f"  GGUF (Q4_K_M): {gguf_q4} ({q4_size:.2f}GB)", flush=True)
    print(f"  Modelfile: ./Modelfile", flush=True)
    print("=" * 60, flush=True)

    # 打包
    print("\n打包下载文件...", flush=True)
    os.system(f"tar -czf tcm_model_package.tar.gz {gguf_q4} Modelfile output_lora_v2/training_stats.json")
    pkg_size = Path("tcm_model_package.tar.gz").stat().st_size / 1024**3
    print(f"  打包完成: tcm_model_package.tar.gz ({pkg_size:.2f}GB)", flush=True)
    print(f"  下载命令 (本地执行): scp -P <端口> root@<host>:/root/autodl-tmp/tcm_model_package.tar.gz .", flush=True)

    return True


# ==================== 主流程 ====================
def main():
    parser = argparse.ArgumentParser(description="云端 LoRA 微调训练")
    parser.add_argument("--skip-train", action="store_true", help="跳过训练，只做合并+GGUF")
    parser.add_argument("--skip-merge", action="store_true", help="只训练，不合并")
    parser.add_argument("--skip-gguf", action="store_true", help="跳过 GGUF 转换")
    args = parser.parse_args()

    model = None
    tokenizer = None

    if not args.skip_train:
        model, tokenizer = train()

    if not args.skip_merge:
        merge_lora(model, tokenizer)

    if not args.skip_gguf:
        convert_to_gguf()

    print("\n" + "=" * 60, flush=True)
    print("全流程完成!", flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()
