"""诊断 OOM: 测试不同配置下的前向+反向"""
import os, time, torch, json, traceback
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
import psutil

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

MODEL_NAME = "models/qwen25-15b-base"
DATA_PATH = "data/processed/sft_train_p1.jsonl"
proc = psutil.Process()

def mem_info(label=""):
    m = proc.memory_info().rss / 1024**2
    sys_mem = psutil.virtual_memory()
    avail = sys_mem.available / 1024**3
    print(f"  {label} RSS={m:.0f}MB, system_avail={avail:.1f}GB", flush=True)
    return m

print(f"PyTorch: {torch.__version__}", flush=True)
torch.set_num_threads(4)  # 减少线程数，降低内存开销
mem_info("初始")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True, padding_side="right")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 加载一条短数据
with open(DATA_PATH, "r", encoding="utf-8") as f:
    item = json.loads(f.readline())
messages = [
    {"role": "system", "content": "你是一位中医老师。"},
    {"role": "user", "content": item["instruction"]},
    {"role": "assistant", "content": item["output"]},
]
full_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, r=8, lora_alpha=16, lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], bias="none",
)

# 测试不同 max_length
for max_len in [128, 256, 384]:
    print(f"\n{'='*50}")
    print(f"测试 max_length={max_len}")
    print(f"{'='*50}", flush=True)

    enc = tokenizer(full_text, truncation=True, max_length=max_len, return_tensors="pt")
    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]
    labels = input_ids.clone()
    print(f"  实际 token 长度: {input_ids.shape[1]}", flush=True)

    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.bfloat16, device_map="cpu",
        trust_remote_code=True, low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    # 用新的 non-reentrant gradient checkpointing
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model = get_peft_model(model, lora_config)
    model.train()
    mem_info("模型加载后")

    try:
        t0 = time.time()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        print(f"  前向: {time.time()-t0:.1f}s, loss={loss.item():.4f}", flush=True)
        mem_info("前向后")

        t0 = time.time()
        loss.backward()
        print(f"  反向: {time.time()-t0:.1f}s", flush=True)
        mem_info("反向后")
        print(f"  ✅ max_length={max_len} 成功!", flush=True)

        # 清理
        del model, outputs, loss
        import gc; gc.collect()
        time.sleep(2)
        mem_info("清理后")
    except Exception as e:
        print(f"  ❌ 失败: {e}", flush=True)
        traceback.print_exc()
        del model
        import gc; gc.collect()
        time.sleep(2)

print("\n=== 诊断完成 ===", flush=True)
