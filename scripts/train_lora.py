"""LoRA 微调训练脚本 — Qwen2.5-1.5B-Instruct + 伤寒论 SFT 数据

使用方法（Google Colab T4 GPU）：
1. 上传 sft_train_p1.jsonl 到 Colab
2. 运行此脚本
3. 训练完成后下载 adapter 和 merged model

依赖：
    pip install transformers peft datasets accelerate bitsandbytes

关键参数：
- 基座模型: Qwen2.5-1.5B-Instruct (fp16, ~3GB VRAM)
- LoRA rank: 16, alpha: 32
- 目标模块: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
- 学习率: 2e-4, warmup: 3%
- Epochs: 3, batch: 4, grad_accum: 4 (effective batch 16)
- Max length: 512
"""
import os
import json
import torch
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
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
DATA_PATH = "sft_train_p1.jsonl"
OUTPUT_DIR = "./output_lora"
MERGED_DIR = "./output_merged"
MAX_LENGTH = 512
LORA_RANK = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LEARNING_RATE = 2e-4
NUM_EPOCHS = 3
BATCH_SIZE = 4
GRAD_ACCUM = 4
WARMUP_RATIO = 0.03

SYSTEM_PROMPT = (
    "你是一位中医老师，擅长用通俗易懂的方式讲解中医经典知识。"
    "请根据提供的经典原文回答问题。"
    "引用经典原文时标注条文编号。解释方剂时列出完整组成。"
    "不提供具体诊疗建议。如果检索结果中没有相关信息，请如实说明。"
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
    """将对话转为 token ids，构建 labels"""
    messages = example["messages"]
    # 使用 tokenizer 的 chat template
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    # 编码
    full_ids = tokenizer(
        text,
        truncation=True,
        max_length=MAX_LENGTH,
        padding=False,
        return_tensors=None,
    )["input_ids"]

    # 构建 labels：只对 assistant 部分计算 loss
    # 先编码 system + user 部分（不包含 assistant 回答）
    prefix_messages = messages[:2]  # system + user
    prefix_text = tokenizer.apply_chat_template(
        prefix_messages,
        tokenize=False,
        add_generation_prompt=True,  # 加上 assistant 开头标记
    )
    prefix_ids = tokenizer(
        prefix_text,
        truncation=True,
        max_length=MAX_LENGTH,
        padding=False,
        return_tensors=None,
    )["input_ids"]

    # labels: prefix 部分设为 -100（不计算 loss），assistant 部分保留
    labels = [-100] * len(prefix_ids) + full_ids[len(prefix_ids):]
    # 截断到相同长度
    labels = labels[:len(full_ids)]

    return {
        "input_ids": full_ids,
        "attention_mask": [1] * len(full_ids),
        "labels": labels,
    }


# ==================== 训练 ====================
def main():
    print("=" * 60)
    print("LoRA 微调训练: Qwen2.5-1.5B-Instruct + 伤寒论 SFT")
    print("=" * 60)

    # 检查 GPU
    if not torch.cuda.is_available():
        raise RuntimeError("需要 GPU 环境！请在 Google Colab 中设置为 GPU 运行时。")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB")

    # 1. 加载 tokenizer 和 model
    print("\n[1/5] 加载模型和 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False
    print(f"模型参数量: {model.num_parameters() / 1e6:.1f}M")

    # 2. 加载和预处理数据
    print("\n[2/5] 加载 SFT 训练数据...")
    dataset = load_sft_data(DATA_PATH)
    print(f"训练样本数: {len(dataset)}")

    # 统计类别分布
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        raw = [json.loads(l) for l in f]
    from collections import Counter
    cats = Counter(r.get("category", "unknown") for r in raw)
    print("类别分布:")
    for cat, cnt in cats.most_common():
        print(f"  {cat}: {cnt}")

    print("\n预处理数据...")
    dataset = dataset.map(
        lambda x: format_to_ids(x, tokenizer),
        remove_columns=dataset.column_names,
        desc="Tokenizing",
    )
    print(f"预处理后样本数: {len(dataset)}")

    # 3. 配置 LoRA
    print("\n[3/5] 配置 LoRA...")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 4. 训练
    print("\n[4/5] 开始训练...")
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LEARNING_RATE,
        warmup_ratio=WARMUP_RATIO,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=3,
        fp16=True,
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        report_to="none",
        remove_unused_columns=False,
        dataloader_pin_memory=True,
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        return_tensors="pt",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )

    train_result = trainer.train()
    print(f"\n训练完成！loss: {train_result.training_loss:.4f}")

    # 5. 保存
    print("\n[5/5] 保存模型...")
    # 保存 LoRA adapter
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"LoRA adapter 已保存到: {OUTPUT_DIR}")

    # 合并 adapter 到基座模型
    print("合并 adapter...")
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(MERGED_DIR, safe_serialization=True, max_shard_size="2GB")
    tokenizer.save_pretrained(MERGED_DIR)
    print(f"合并后的完整模型已保存到: {MERGED_DIR}")

    # 保存训练统计
    stats = {
        "model": MODEL_NAME,
        "lora_rank": LORA_RANK,
        "lora_alpha": LORA_ALPHA,
        "epochs": NUM_EPOCHS,
        "batch_size": BATCH_SIZE,
        "grad_accum": GRAD_ACCUM,
        "lr": LEARNING_RATE,
        "max_length": MAX_LENGTH,
        "train_samples": len(dataset),
        "final_loss": train_result.training_loss,
        "category_distribution": dict(cats),
    }
    with open(f"{OUTPUT_DIR}/training_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print("训练完成！")
    print(f"  LoRA adapter: {OUTPUT_DIR}/")
    print(f"  合并模型: {MERGED_DIR}/")
    print(f"  训练统计: {OUTPUT_DIR}/training_stats.json")
    print("=" * 60)
    print("\n下一步：将合并后的模型转为 GGUF 格式并导入 Ollama")
    print("参考: convert_to_gguf.py")


if __name__ == "__main__":
    main()
