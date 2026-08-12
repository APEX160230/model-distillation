#!/bin/bash
# ============================================================
# 本地打包脚本 — 将云端训练所需文件打包成 tar.gz
# 用法: 在项目根目录执行 bash scripts/pack_for_cloud.sh
# 产出: cloud_training_package.tar.gz (~7MB)
# ============================================================

set -e
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${PROJECT_ROOT}"

OUTPUT="cloud_training_package.tar.gz"

echo "============================================================"
echo "  打包云端训练文件"
echo "============================================================"
echo "  项目根目录: ${PROJECT_ROOT}"
echo ""

# 需要上传的文件清单
FILES=(
    "scripts/cloud_train.py"
    "scripts/cloud_run.sh"
    "data/processed/sft_train_final.jsonl"
    "data/processed/sft_val_final.jsonl"
)

# 验证文件存在
echo "[1/2] 检查文件..."
ALL_OK=true
for f in "${FILES[@]}"; do
    if [ -f "$f" ]; then
        SIZE=$(du -h "$f" | cut -f1)
        echo "  OK  ${f} (${SIZE})"
    else
        echo "  MISSING  ${f}"
        ALL_OK=false
    fi
done

if [ "${ALL_OK}" = "false" ]; then
    echo ""
    echo "ERROR: 有文件缺失，请检查"
    exit 1
fi

# 打包
echo ""
echo "[2/2] 打包..."
tar -czf "${OUTPUT}" "${FILES[@]}"
PKG_SIZE=$(du -h "${OUTPUT}" | cut -f1)
echo "  打包完成: ${OUTPUT} (${PKG_SIZE})"
echo ""
echo "上传到 AutoDL (在本地终端执行):"
echo "  scp -P <AutoDL端口> ${OUTPUT} root@<AutoDL地址>:/root/autodl-tmp/"
echo ""
echo "在 AutoDL 上解压并运行:"
echo "  cd /root/autodl-tmp"
echo "  tar -xzf ${OUTPUT}"
echo "  mkdir -p model-distillation && mv scripts data model-distillation/"
echo "  cd model-distillation"
echo "  bash scripts/cloud_run.sh"
