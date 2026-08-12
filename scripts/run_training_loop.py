"""训练自动重启 wrapper
崩溃后自动从 checkpoint 恢复，不依赖 WorkBuddy 后台任务。
日志写入 output_lora_v2/training_log.txt
"""
import subprocess
import sys
import time
import os
from pathlib import Path

PYTHON = sys.executable  # 使用当前解释器，避免硬编码路径
SCRIPT_DIR = Path(__file__).parent.parent
TRAIN_SCRIPT = str(SCRIPT_DIR / "scripts" / "train_lora_gpu.py")
LOG_FILE = str(SCRIPT_DIR / "output_lora_v2" / "training_log.txt")
MAX_RETRIES = 100

os.makedirs(str(SCRIPT_DIR / "output_lora_v2"), exist_ok=True)

def is_training_done():
    """检查训练是否已完成（final adapter 已保存）"""
    done_marker = SCRIPT_DIR / "output_lora_v2" / "adapter_model.safetensors"
    return done_marker.exists()

for attempt in range(MAX_RETRIES):
    if is_training_done():
        msg = "检测到训练已完成 (adapter_model.safetensors 存在)，退出。"
        print(msg, flush=True)
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(msg + "\n")
        break

    msg = f"\n{'='*60}\n训练尝试 {attempt+1}/{MAX_RETRIES}\n{'='*60}\n"
    print(msg, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(msg)

    # 清理超长环境变量（accelerate 的 clear_environment 会崩溃）
    clean_env = {k: v for k, v in os.environ.items() if len(v) < 32000}

    t0 = time.time()
    result = subprocess.run(
        [PYTHON, TRAIN_SCRIPT],
        cwd=str(SCRIPT_DIR),
        env=clean_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    elapsed = time.time() - t0

    # 写日志
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(result.stdout)
        f.write(f"\n--- 进程退出 returncode={result.returncode} elapsed={elapsed:.0f}s ---\n")

    if result.returncode == 0:
        msg = "训练进程正常退出。"
        print(msg, flush=True)
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(msg + "\n")
        if is_training_done():
            break

    msg = f"训练崩溃 (returncode={result.returncode})，{15} 秒后自动从 checkpoint 恢复..."
    print(msg, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(msg + "\n")
    time.sleep(15)

msg = "Wrapper 退出。"
print(msg, flush=True)
with open(LOG_FILE, "a", encoding="utf-8") as f:
    f.write(msg + "\n")
