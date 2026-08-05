"""LoRA CPU 微调训练脚本 — Qwen2.5-1.5B-Instruct + 伤寒论 SFT 数据

针对无 GPU 环境优化：
- bfloat16 (CPU 原生支持，比 fp16 更稳定)
- gradient checkpointing (省内存，代价是 ~30% 速度)
- batch_size=1 + gradient_accumulation=8 (有效 batch=8)
- 4 个 target modules (比 7 个省 40% 训练参数)
- max_length=384 (中医 Q&A 通常 <300 token)

预计耗时: 2-4 小时 (6 核 CPU, 8GB RAM)
"""
import os
import sys
import json
import time
import torch
import gc
from pathlib import Path
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model, TaskType

# ==================== 配置 ====================
# HuggingFace 镜像 + 禁用 xet (国内镜像不支持 xet 协议)
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

# 优先使用本地模型，避免网络下载
_LOCAL_MODEL = Path("models/qwen25-15b-base")
MODEL_NAME = str(_LOCAL_MODEL) if _LOCAL_MODEL.exists() else "Qwen/Qwen2.5-1.5B-Instruct"
DATA_PATH = "data/processed/sft_train_p1.jsonl"
OUTPUT_DIR = "./output_lora"
MERGED_DIR = "./output_merged"
MAX_LENGTH = 384

# LoRA 配置 — CPU 优化
LORA_RANK = 8          # rank=8 比 16 省一半训练参数，1.5B 模型够用
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]  # 4 个而非 7 个

# 训练配置 — CPU 优化
LEARNING_RATE = 2e-4
NUM_EPOCHS = 2          # 2 轮够了，3 轮容易过拟合
BATCH_SIZE = 1          # CPU 内存有限
GRAD_ACCUM = 8          # 有效 batch = 1*8 = 8
WARMUP_RATIO = 0.05
LOGGING_STEPS = 5       # 频繁日志方便监控进度

SYSTEM_PROMPT = (
    "你是一位中医老师，擅长用通俗易懂的方式讲解中医经典知识。"
    "引用经典原文时标注条文编号。解释方剂时列出完整组成。"
    "不提供具体诊疗建议。"
)


# ==================== 数据加载与格式化 ====================
def load_sft_data(path: str) -> Dataset:
    """加载 SFT JSONL 数据并转为 Qwen 对话格式"""
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
    """将对话转为 token ids，构建 labels（只对 assistant 部分计算 loss）"""
    messages = example["messages"]
    # 完整对话
    full_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    full_ids = tokenizer(
        full_text, truncation=True, max_length=MAX_LENGTH,
        padding=False, return_tensors=None,
    )["input_ids"]

    # prefix (system + user，不含 assistant 回答)
    prefix_text = tokenizer.apply_chat_template(
        messages[:2], tokenize=False, add_generation_prompt=True
    )
    prefix_ids = tokenizer(
        prefix_text, truncation=True, max_length=MAX_LENGTH,
        padding=False, return_tensors=None,
    )["input_ids"]

    # labels: prefix 部分 -100，assistant 部分保留
    labels = [-100] * len(prefix_ids) + full_ids[len(prefix_ids):]
    labels = labels[:len(full_ids)]

    return {
        "input_ids": full_ids,
        "attention_mask": [1] * len(full_ids),
        "labels": labels,
    }


# ==================== 训练 ====================
def main():
    print("=" * 60)
    print("LoRA CPU 微调: Qwen2.5-1.5B-Instruct + 伤寒论 SFT")
    print("=" * 60)

    # 检查环境
    print(f"\nPyTorch: {torch.__version__}")
    print(f"CPU threads: {torch.get_num_threads()}")
    torch.set_num_threads(6)
    print(f"Set threads to: 6")

    if torch.cuda.is_available():
        print(f"GPU detected: {torch.cuda.get_device_name(0)}")
        print("WARNING: This script is optimized for CPU. Using GPU may work but is not tested.")
    else:
        print("Mode: CPU-only training (bfloat16)")

    # 检查数据文件
    if not Path(DATA_PATH).exists():
        print(f"ERROR: 训练数据不存在: {DATA_PATH}")
        sys.exit(1)

    # 1. 加载 tokenizer
    print("\n[1/6] 加载 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME, trust_remote_code=True, padding_side="right"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"Tokenizer loaded. Vocab size: {tokenizer.vocab_size}")

    # 2. 加载模型 (bfloat16 节省内存)
    print("\n[2/6] 加载模型 (bfloat16)...")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False  # 训练时关闭 KV cache
    # 启用 gradient checkpointing
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    load_time = time.time() - t0
    mem_mb = sum(
        p.nelement() * p.element_size() for p in model.parameters()
    ) / 1024 / 1024
    print(f"模型加载完成: {load_time:.1f}s, 内存占用: {mem_mb:.0f}MB")
    print(f"模型参数量: {model.num_parameters() / 1e6:.1f}M")

    # 3. 加载和预处理数据
    print("\n[3/6] 加载 SFT 训练数据...")
    dataset = load_sft_data(DATA_PATH)
    print(f"训练样本数: {len(dataset)}")

    with open(DATA_PATH, "r", encoding="utf-8") as f:
        raw = [json.loads(l) for l in f]
    from collections import Counter
    cats = Counter(r.get("category", "unknown") for r in raw)
    print("类别分布:")
    for cat, cnt in cats.most_common():
        print(f"  {cat}: {cnt} ({cnt/len(raw)*100:.1f}%)")

    print("\n预处理 (tokenize)...")
    dataset = dataset.map(
        lambda x: format_to_ids(x, tokenizer),
        remove_columns=dataset.column_names,
        desc="Tokenizing",
    )
    # 过滤掉太短的样本
    dataset = dataset.filter(lambda x: len(x["input_ids"]) > 10)
    print(f"预处理后样本数: {len(dataset)}")

    # 4. 配置 LoRA
    print("\n[4/6] 配置 LoRA...")
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

    # 5. 训练
    print("\n[5/6] 开始 CPU 训练...")
    print(f"  Epochs: {NUM_EPOCHS}")
    print(f"  Batch size: {BATCH_SIZE} x {GRAD_ACCUM} (effective: {BATCH_SIZE * GRAD_ACCUM})")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  Max length: {MAX_LENGTH}")
    total_steps = (len(dataset) // (BATCH_SIZE * GRAD_ACCUM)) * NUM_EPOCHS
    print(f"  Estimated steps: ~{total_steps}")

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LEARNING_RATE,
        warmup_ratio=WARMUP_RATIO,
        logging_steps=LOGGING_STEPS,
        save_strategy="epoch",
        save_total_limit=2,
        bf16=True,           # CPU bfloat16
        fp16=False,
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        report_to="none",
        remove_unused_columns=False,
        dataloader_pin_memory=False,  # CPU 不需要 pin_memory
        dataloader_num_workers=0,     # 单线程加载，避免内存竞争
        gradient_checkpointing=True,
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer, padding=True, return_tensors="pt"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )

    t_start = time.time()
    train_result = trainer.train()
    train_time = time.time() - t_start

    print(f"\n训练完成!")
    print(f"  Final loss: {train_result.training_loss:.4f}")
    print(f"  Total time: {train_time/60:.1f} min ({train_time:.0f}s)")

    # 6. 保存
    print("\n[6/6] 保存模型...")
    # 保存 LoRA adapter
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"LoRA adapter: {OUTPUT_DIR}/")

    # 合并 adapter 到基座模型
    print("合并 adapter 到基座模型...")
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(MERGED_DIR, safe_serialization=True, max_shard_size="2GB")
    tokenizer.save_pretrained(MERGED_DIR)
    print(f"合并模型: {MERGED_DIR}/")

    # 保存训练统计
    stats = {
        "model": MODEL_NAME,
        "lora_rank": LORA_RANK,
        "lora_alpha": LORA_ALPHA,
        "target_modules": TARGET_MODULES,
        "epochs": NUM_EPOCHS,
        "batch_size": BATCH_SIZE,
        "grad_accum": GRAD_ACCUM,
        "lr": LEARNING_RATE,
        "max_length": MAX_LENGTH,
        "train_samples": len(dataset),
        "final_loss": train_result.training_loss,
        "train_time_seconds": round(train_time, 1),
        "train_time_minutes": round(train_time / 60, 1),
        "category_distribution": dict(cats),
        "device": "CPU",
        "dtype": "bfloat16",
        "gradient_checkpointing": True,
    }
    with open(f"{OUTPUT_DIR}/training_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"训练统计: {OUTPUT_DIR}/training_stats.json")

    print("\n" + "=" * 60)
    print("全部完成!")
    print(f"  LoRA adapter: {OUTPUT_DIR}/")
    print(f"  合并模型: {MERGED_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
