#!/usr/bin/env bash
set -euo pipefail

# 修复云端 GGUF 提取：从 output_merged 重新走 Ollama 转换，
# 通过 manifest 定位真正的 weights blob，验证 GGUF magic 后打包。

cd /root/autodl-tmp/model-distillation

echo "=== 1. 启动 Ollama ==="
if ! pgrep -x "ollama" > /dev/null; then
    nohup ollama serve > /tmp/ollama.log 2>&1 &
    for i in {1..10}; do
        sleep 1
        if ollama list >/dev/null 2>&1; then
            echo "Ollama ready"
            break
        fi
        echo "Waiting for Ollama... $i"
    done
else
    echo "Ollama already running"
fi

echo "=== 2. 检查 output_merged ==="
if [[ ! -d output_merged ]]; then
    echo "ERROR: output_merged not found"
    exit 1
fi
ls -lh output_merged/ | head -20

echo "=== 3. 准备 Modelfile（FROM 用绝对路径）==="
FROM_PATH="/root/autodl-tmp/model-distillation/output_merged"

if [[ -f Modelfile ]]; then
    sed "s|^FROM.*|FROM $FROM_PATH|" Modelfile > /tmp/Modelfile
else
    cat > /tmp/Modelfile <<EOF
FROM $FROM_PATH
TEMPLATE """{{ if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}{{ if .Prompt }}<|im_start|>user
{{ .Prompt }}<|im_end|>
{{ end }}<|im_start|>assistant
{{ .Response }}<|im_end|>"""
PARAMETER stop <|im_start|>
PARAMETER stop <|im_end|>
PARAMETER temperature 0.7
EOF
fi

echo "Modelfile content:"
cat /tmp/Modelfile

echo "=== 4. 重新创建 Ollama 模型 ==="
ollama rm tcm-model 2>/dev/null || true
ollama create tcm-model -f /tmp/Modelfile

echo "=== 5. 查找 manifest ==="
MANIFEST=$(find ~/.ollama/models/manifests -path "*tcm-model*" -name "latest" 2>/dev/null | head -1)
if [[ -z "$MANIFEST" ]]; then
    echo "ERROR: manifest not found"
    exit 1
fi
echo "Manifest: $MANIFEST"
cat "$MANIFEST" | python3 -m json.tool

echo "=== 6. 从 manifest 定位 weights blob ==="
MODEL_DIGEST=$(python3 - <<'PY'
import json, os
manifest = os.path.expanduser('~/.ollama/models/manifests/registry.ollama.ai/library/tcm-model/latest')
with open(manifest) as f:
    data = json.load(f)
for layer in data.get('layers', []):
    mt = layer.get('mediaType', '')
    if 'model' in mt:
        print(layer['digest'])
        break
PY
)

if [[ -z "$MODEL_DIGEST" ]]; then
    echo "ERROR: model digest not found in manifest"
    exit 1
fi

echo "Model digest: $MODEL_DIGEST"
BLOB_PATH="$HOME/.ollama/models/blobs/${MODEL_DIGEST/:/-}"
if [[ ! -f "$BLOB_PATH" ]]; then
    echo "ERROR: blob not found at $BLOB_PATH"
    exit 1
fi
ls -lh "$BLOB_PATH"

echo "=== 7. 验证并复制为有效 GGUF ==="
cp -f "$BLOB_PATH" /root/autodl-tmp/model-distillation/qwen25-15b-tcm-valid.gguf

python3 - <<'PY'
import sys
path = '/root/autodl-tmp/model-distillation/qwen25-15b-tcm-valid.gguf'
size = __import__('os').path.getsize(path)
with open(path, 'rb') as f:
    magic = f.read(4)
print(f'File size: {size / 1024 / 1024:.1f} MB')
print(f'Magic bytes: {magic}')
if magic != b'GGUF':
    print('ERROR: invalid GGUF magic bytes')
    sys.exit(1)
print('OK: valid GGUF')
PY

echo "=== 8. 修正 Modelfile 指向 GGUF ==="
sed -i 's|^FROM.*|FROM ./qwen25-15b-tcm-valid.gguf|' Modelfile
echo "Modelfile 前 3 行："
head -3 Modelfile

echo "=== 9. 重新打包 ==="
rm -f tcm_model_package.tar.gz
tar czf tcm_model_package.tar.gz Modelfile qwen25-15b-tcm-valid.gguf
ls -lh tcm_model_package.tar.gz

echo "=== 完成 ==="
echo "下载这个文件：/root/autodl-tmp/model-distillation/tcm_model_package.tar.gz"
