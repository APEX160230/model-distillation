"""GPU QLoRA 微调训练脚本 — Qwen2.5-1.5B-Instruct + 伤寒论 SFT

针对 GTX 1650 (4GB VRAM) 优化：
- 4-bit 量化 (bitsandbytes)：模型 0.8GB vs bfloat16 3GB
- LoRA rank=8, 4 target modules
- gradient checkpointing + use_reentrant=False
- batch_size=1, gradient_accumulation=8

预计耗时: 10-30 分钟 (GPU 比 CPU 快 10-50x)
"""
import os
import sys
import json
import time
import torch
from pathlib import Path
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    BitsAndBytesConfig,
    TrainerCallback,
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training

# ==================== 配置 ====================
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

_LOCAL_MODEL = Path("models/qwen25-15b-base")
MODEL_NAME = str(_LOCAL_MODEL) if _LOCAL_MODEL.exists() else "Qwen/Qwen2.5-1.5B-Instruct"
DATA_PATH = "data/processed/sft_train_final.jsonl"
VAL_DATA_PATH = "data/processed/sft_val_final.jsonl"
OUTPUT_DIR = "./output_lora"
MERGED_DIR = "./output_merged"
MAX_LENGTH = 1024  # PRD §4.3 要求，覆盖 95%+ 训练数据

# QLoRA 配置 (PRD §FR4 对齐)
LORA_RANK = 8
LORA_ALPHA = 32  # PRD 要求 32
LORA_DROPOUT = 0.05
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]

# 训练配置 (PRD §FR4 对齐)
LEARNING_RATE = 2e-4
NUM_EPOCHS = 3  # PRD 要求 3 epochs
BATCH_SIZE = 1
GRAD_ACCUM = 8
WARMUP_RATIO = 0.03  # PRD 要求 0.03
LOGGING_STEPS = 5
EVAL_STEPS = 50  # 每 50 步评估一次验证集

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
def main():
    print("=" * 60, flush=True)
    print("GPU QLoRA 微调: Qwen2.5-1.5B-Instruct + 伤寒论 SFT", flush=True)
    print("=" * 60, flush=True)

    # 检查环境
    print(f"\nPyTorch: {torch.__version__}", flush=True)
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available. Use train_lora_cpu.py instead.", flush=True)
        sys.exit(1)
    gpu_name = torch.cuda.get_device_name(0)
    vram_free = torch.cuda.mem_get_info()[0] / 1024**3
    vram_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"GPU: {gpu_name}", flush=True)
    print(f"VRAM: {vram_total:.1f}GB total, {vram_free:.1f}GB free", flush=True)

    # 检查数据文件
    if not Path(DATA_PATH).exists():
        print(f"ERROR: 训练数据不存在: {DATA_PATH}", flush=True)
        sys.exit(1)

    # 1. 加载 tokenizer
    print("\n[1/7] 加载 tokenizer...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME, trust_remote_code=True, padding_side="right"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"Tokenizer loaded. Vocab size: {tokenizer.vocab_size}", flush=True)

    # 2. 配置 4-bit 量化
    print("\n[2/7] 配置 4-bit 量化 (QLoRA)...", flush=True)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    # 3. 加载模型 (4-bit 量化)
    print("\n[3/7] 加载模型 (4-bit)...", flush=True)
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False
    load_time = time.time() - t0
    vram_after_load = torch.cuda.mem_get_info()[0] / 1024**3
    print(f"模型加载完成: {load_time:.1f}s", flush=True)
    print(f"VRAM 使用: {vram_total - vram_after_load:.2f}GB / {vram_total:.1f}GB", flush=True)
    print(f"模型参数量: {model.num_parameters() / 1e6:.1f}M", flush=True)

    # 4. 准备 k-bit 训练
    print("\n[4/7] 准备 k-bit 训练...", flush=True)
    model = prepare_model_for_kbit_training(model)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    # 5. 配置 LoRA
    print("\n[5/7] 配置 LoRA...", flush=True)
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

    # 6. 加载和预处理数据
    print("\n[6/7] 加载 SFT 训练数据...", flush=True)
    dataset = load_sft_data(DATA_PATH)
    print(f"训练样本数: {len(dataset)}", flush=True)

    # 加载验证集
    val_dataset = None
    if Path(VAL_DATA_PATH).exists():
        val_dataset = load_sft_data(VAL_DATA_PATH)
        print(f"验证样本数: {len(val_dataset)}", flush=True)
    else:
        print(f"WARNING: 验证集不存在: {VAL_DATA_PATH}", flush=True)

    print("\n预处理 (tokenize)...", flush=True)
    dataset = dataset.map(
        lambda x: format_to_ids(x, tokenizer),
        remove_columns=dataset.column_names,
        desc="Tokenizing train",
    )
    dataset = dataset.filter(lambda x: len(x["input_ids"]) > 10)
    print(f"预处理后训练样本数: {len(dataset)}", flush=True)

    if val_dataset is not None:
        val_dataset = val_dataset.map(
            lambda x: format_to_ids(x, tokenizer),
            remove_columns=val_dataset.column_names,
            desc="Tokenizing val",
        )
        val_dataset = val_dataset.filter(lambda x: len(x["input_ids"]) > 10)
        print(f"预处理后验证样本数: {len(val_dataset)}", flush=True)

    # 长度分布统计
    lengths = [len(x["input_ids"]) for x in dataset]
    avg_len = sum(lengths) / len(lengths)
    over_max = sum(1 for l in lengths if l >= MAX_LENGTH)
    print(f"Token 长度: avg={avg_len:.0f}, max={max(lengths)}, >={MAX_LENGTH}: {over_max} ({over_max/len(lengths)*100:.1f}%)", flush=True)

    # 7. 训练
    print("\n[7/7] 开始 GPU 训练...", flush=True)
    print(f"  Epochs: {NUM_EPOCHS}", flush=True)
    print(f"  Batch size: {BATCH_SIZE} x {GRAD_ACCUM} (effective: {BATCH_SIZE * GRAD_ACCUM})", flush=True)
    print(f"  Learning rate: {LEARNING_RATE}", flush=True)
    print(f"  Max length: {MAX_LENGTH}", flush=True)
    total_steps = (len(dataset) // (BATCH_SIZE * GRAD_ACCUM)) * NUM_EPOCHS
    print(f"  Estimated steps: ~{total_steps}", flush=True)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LEARNING_RATE,
        warmup_ratio=WARMUP_RATIO,
        logging_steps=LOGGING_STEPS,
        save_strategy="steps",
        save_steps=50,
        save_total_limit=3,
        evaluation_strategy="steps" if val_dataset else "no",
        eval_steps=EVAL_STEPS if val_dataset else None,
        fp16=True,           # GPU 用 fp16
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        report_to="none",
        remove_unused_columns=False,
        dataloader_pin_memory=False,
        dataloader_num_workers=0,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        disable_tqdm=True,
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

    # 自定义进度回调
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

        def on_step_end(self, args, state, control, **kwargs):
            with open(progress_file, "a", encoding="utf-8") as f:
                f.write(f"heartbeat: step={state.global_step}/{state.max_steps} t={time.time():.0f}\n")

    trainer.add_callback(ProgressCallback())

    t_start = time.time()
    resume_ckpt = None
    ckpts = sorted(Path(OUTPUT_DIR).glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))
    if ckpts:
        resume_ckpt = str(ckpts[-1])
        print(f"  从 checkpoint 恢复: {resume_ckpt}", flush=True)
    train_result = trainer.train(resume_from_checkpoint=resume_ckpt)
    train_time = time.time() - t_start

    print(f"\n训练完成!", flush=True)
    print(f"  Final loss: {train_result.training_loss:.4f}", flush=True)
    print(f"  Total time: {train_time/60:.1f} min ({train_time:.0f}s)", flush=True)

    # 保存
    print("\n保存模型...", flush=True)
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"LoRA adapter: {OUTPUT_DIR}/", flush=True)

    # 合并 adapter 到基座模型
    print("合并 adapter 到基座模型 (float16)...", flush=True)
    try:
        merged_model = model.merge_and_unload()
        merged_model.save_pretrained(MERGED_DIR, safe_serialization=True, max_shard_size="2GB")
        tokenizer.save_pretrained(MERGED_DIR)
        print(f"合并模型: {MERGED_DIR}/", flush=True)
    except Exception as e:
        print(f"WARNING: 合并失败 (4-bit 模型不支持直接合并): {e}", flush=True)
        print("LoRA adapter 已保存，可在推理时动态加载。", flush=True)

    # 保存训练统计
    stats = {
        "model": MODEL_NAME,
        "method": "QLoRA (4-bit nf4)",
        "lora_rank": LORA_RANK,
        "lora_alpha": LORA_ALPHA,
        "target_modules": TARGET_MODULES,
        "epochs": NUM_EPOCHS,
        "batch_size": BATCH_SIZE,
        "grad_accum": GRAD_ACCUM,
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
        "dtype": "float16 (compute) + nf4 (weights)",
        "gradient_checkpointing": True,
    }
    with open(f"{OUTPUT_DIR}/training_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"训练统计: {OUTPUT_DIR}/training_stats.json", flush=True)

    print("\n" + "=" * 60, flush=True)
    print("全部完成!", flush=True)
    print(f"  LoRA adapter: {OUTPUT_DIR}/", flush=True)
    print(f"  合并模型: {MERGED_DIR}/", flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()
