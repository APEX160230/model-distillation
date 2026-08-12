#!/bin/bash
# ============================================================
# AutoDL 一键云端训练脚本
# 用法: 在 AutoDL 实例上执行 bash cloud_run.sh
# ============================================================
set -e

echo "============================================================"
echo "  AutoDL 云端训练 — Qwen2.5-1.5B LoRA + 中医 SFT"
echo "============================================================"
echo ""

# ==================== 配置 ====================
WORK_DIR="/root/autodl-tmp/model-distillation"
MODEL_DIR="/root/autodl-tmp/models"
MODEL_NAME="Qwen/Qwen2.5-1.5B-Instruct"
LOCAL_MODEL_DIR="${MODEL_DIR}/Qwen2.5-1.5B-Instruct"

# ==================== Step 0: 环境检查 ====================
echo "[Step 0] 环境检查..."
echo "  Python: $(python --version 2>&1)"
echo "  PyTorch: $(python -c 'import torch; print(torch.__version__)' 2>&1)"
echo "  CUDA: $(python -c 'import torch; print(torch.version.cuda)' 2>&1)"
if ! python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    echo "ERROR: CUDA 不可用，请确认选择了 GPU 实例"
    exit 1
fi
GPU_NAME=$(python -c "import torch; print(torch.cuda.get_device_name(0))")
echo "  GPU: ${GPU_NAME}"

# ==================== Step 1: 安装依赖 ====================
echo ""
echo "[Step 1] 安装 Python 依赖..."
pip install -q transformers peft datasets accelerate sentencepiece protobuf
# bitsandbytes 云端不需要（用 bf16 LoRA 而非 4-bit QLoRA），但装上以备不时
pip install -q bitsandbytes 2>/dev/null || echo "  bitsandbytes 安装跳过（不影响训练）"
echo "  依赖安装完成"

# ==================== Step 2: 准备工作目录 ====================
echo ""
echo "[Step 2] 准备工作目录..."
mkdir -p "${WORK_DIR}/data/processed"
mkdir -p "${MODEL_DIR}"
cd "${WORK_DIR}"
echo "  工作目录: ${WORK_DIR}"

# ==================== Step 3: 下载基座模型 ====================
echo ""
echo "[Step 3] 下载基座模型 Qwen2.5-1.5B-Instruct..."
if [ -d "${LOCAL_MODEL_DIR}" ] && [ -f "${LOCAL_MODEL_DIR}/config.json" ]; then
    echo "  模型已存在，跳过下载"
else
    # 优先用 modelscope（国内更快）
    pip install -q modelscope 2>/dev/null
    python -c "
import os
os.environ['MODELSCOPE_CACHE'] = '${MODEL_DIR}'
from modelscope import snapshot_download
model_dir = snapshot_download('Qwen/Qwen2.5-1.5B-Instruct', cache_dir='${MODEL_DIR}')
print(f'模型下载到: {model_dir}')
" 2>/dev/null || {
        echo "  modelscope 失败，尝试 HuggingFace mirror..."
        pip install -q huggingface_hub
        python -c "
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from huggingface_hub import snapshot_download
model_dir = snapshot_download('${MODEL_NAME}', cache_dir='${MODEL_DIR}', local_dir='${LOCAL_MODEL_DIR}')
print(f'模型下载到: {model_dir}')
"
    }
    echo "  模型下载完成"
fi

# ==================== Step 4: 检查 SFT 数据 ====================
echo ""
echo "[Step 4] 检查 SFT 数据..."
if [ ! -f "data/processed/sft_train_final.jsonl" ]; then
    echo "ERROR: 训练数据不存在: data/processed/sft_train_final.jsonl"
    echo "  请先上传数据文件到 ${WORK_DIR}/data/processed/"
    echo "  需要的文件:"
    echo "    - sft_train_final.jsonl (3960 条, ~6MB)"
    echo "    - sft_val_final.jsonl (440 条, ~0.7MB)"
    exit 1
fi
TRAIN_COUNT=$(wc -l < data/processed/sft_train_final.jsonl)
VAL_COUNT=$(wc -l < data/processed/sft_val_final.jsonl 2>/dev/null || echo "0")
echo "  训练集: ${TRAIN_COUNT} 条"
echo "  验证集: ${VAL_COUNT} 条"

# ==================== Step 5: 检查训练脚本 ====================
echo ""
echo "[Step 5] 检查训练脚本..."
if [ ! -f "scripts/cloud_train.py" ]; then
    echo "ERROR: 训练脚本不存在: scripts/cloud_train.py"
    echo "  请先上传 scripts/ 目录"
    exit 1
fi
echo "  训练脚本就绪"

# ==================== Step 6: 开始训练 ====================
echo ""
echo "============================================================"
echo "  开始训练 (预计 ~30 min on RTX 3090)"
echo "============================================================"
START_TIME=$(date +%s)

python scripts/cloud_train.py
TRAIN_EXIT=$?

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
echo ""
echo "训练耗时: $((DURATION / 60)) min $((DURATION % 60)) s"

if [ ${TRAIN_EXIT} -ne 0 ]; then
    echo "ERROR: 训练失败 (exit code: ${TRAIN_EXIT})"
    echo "  查看日志: cat ${WORK_DIR}/output_lora_v2/progress.txt"
    exit ${TRAIN_EXIT}
fi

# ==================== Step 7: 验证产出 ====================
echo ""
echo "[Step 7] 验证产出..."
echo "  --- training_stats.json ---"
cat output_lora_v2/training_stats.json 2>/dev/null || echo "  (not found)"
echo ""
echo "  --- 文件列表 ---"
ls -lh output_lora_v2/adapter_config.json output_lora_v2/adapter_model.safetensors 2>/dev/null
ls -lh output_merged/config.json output_merged/model-*.safetensors 2>/dev/null
ls -lh qwen25-15b-tcm-q4_k_m.gguf Modelfile tcm_model_package.tar.gz 2>/dev/null

echo ""
echo "============================================================"
echo "  全流程完成!"
echo "============================================================"
echo ""
echo "下载到本地 (在本地终端执行):"
echo "  scp -P <AutoDL端口> root@<AutoDL地址>:${WORK_DIR}/tcm_model_package.tar.gz ."
echo ""
echo "本地部署:"
echo "  tar -xzf tcm_model_package.tar.gz"
echo "  ollama create qwen25-15b-tcm -f Modelfile"
echo "  ollama run qwen25-15b-tcm '什么是太阳病'"
