#!/usr/bin/env bash
# CD 部署脚本：push main 后由 GitHub Actions 调用，部署到 zzy1n.cc
#
# 需要环境变量（GitHub Actions secrets）：
#   DEPLOY_HOST  服务器地址（如 zzy1n.cc）
#   DEPLOY_USER  服务器用户（默认 ubuntu）
#   DEPLOY_KEY   SSH 私钥（部署专用，仅授权部署目录）
#
# 也可本地手动执行：DEPLOY_HOST=zzy1n.cc DEPLOY_KEY="$(cat ~/.ssh/deploy_key)" bash scripts/deploy.sh
set -euo pipefail

HOST="${DEPLOY_HOST:?缺少 DEPLOY_HOST}"
USER="${DEPLOY_USER:-ubuntu}"
SSH_KEY="${DEPLOY_KEY:?缺少 DEPLOY_KEY}"

# 写入部署私钥（不覆盖用户已有 ~/.ssh 配置）
mkdir -p ~/.ssh
printf '%s\n' "$SSH_KEY" > ~/.ssh/tcm_deploy_key
chmod 600 ~/.ssh/tcm_deploy_key

SSH="ssh -i ~/.ssh/tcm_deploy_key -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=15 -o ServerAliveInterval=30 -o ServerAliveCountMax=20"
SCP="scp -i ~/.ssh/tcm_deploy_key -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=15 -o ServerAliveInterval=30 -o ServerAliveCountMax=20"

echo "==> [1/6] 上传源码 src/"
$SCP -r src "$USER@$HOST:/home/ubuntu/tcm/backend/"

echo "==> [2/6] 上传数据资产（条文数据 + 重建脚本；chroma 向量库为可重建资产，服务器端重建）"
$SCP -r data/processed "$USER@$HOST:/home/ubuntu/tcm/backend/data/"
$SSH "$USER@$HOST" "mkdir -p /home/ubuntu/tcm/backend/scripts"
$SCP scripts/build_chroma.py scripts/build_lecture_chroma.py "$USER@$HOST:/home/ubuntu/tcm/backend/scripts/"

echo "==> [3/6] 上传 Modelfile"
$SCP Modelfile "$USER@$HOST:/home/ubuntu/tcm/"

echo "==> [4/6] 条文变更时重建向量库（幂等，未变更则跳过）"
SRC_MD5=$(md5sum data/processed/classics/shanghan_clauses.jsonl | awk '{print $1}')
NEED_REBUILD=$($SSH "$USER@$HOST" "mark=/home/ubuntu/tcm/backend/data/chroma/.source_md5; if [ -f \"\$mark\" ] && [ \"\$(cat \"\$mark\")\" = \"$SRC_MD5\" ]; then echo 0; else echo 1; fi")
if [ "$NEED_REBUILD" = "1" ]; then
    echo "条文数据已变更，重建 chroma（约 1-2 分钟）..."
    # 优先用服务器本地 bge 模型（hf-mirror 的 HEAD 请求有 bug，huggingface_hub 下载不可靠）
    $SSH "$USER@$HOST" "cd /home/ubuntu/tcm/backend && if [ -d /home/ubuntu/tcm/models/bge-small-zh-v1.5 ]; then /usr/bin/python3 scripts/build_chroma.py --model /home/ubuntu/tcm/models/bge-small-zh-v1.5; else /usr/bin/python3 scripts/build_chroma.py; fi && echo $SRC_MD5 > data/chroma/.source_md5"
else
    echo "条文数据未变更，跳过重建"
fi

echo "==> [4b/6] 讲稿数据变更时重建讲稿库（FR4，幂等）"
LEC_MD5=$(md5sum data/processed/sft_train_final.jsonl | awk '{print $1}')
LEC_NEED=$($SSH "$USER@$HOST" "mark=/home/ubuntu/tcm/backend/data/chroma/.lecture_source_md5; if [ -f \"\$mark\" ] && [ \"\$(cat \"\$mark\")\" = \"$LEC_MD5\" ]; then echo 0; else echo 1; fi")
if [ "$LEC_NEED" = "1" ]; then
    echo "讲稿数据已变更，重建讲稿库（2核CPU约2-5分钟）..."
    # 讲稿库为增强资产：失败不阻塞部署（可降级），失败时不写 md5，下轮部署重试
    $SSH "$USER@$HOST" "cd /home/ubuntu/tcm/backend && if [ -d /home/ubuntu/tcm/models/bge-small-zh-v1.5 ]; then /usr/bin/python3 scripts/build_lecture_chroma.py --model /home/ubuntu/tcm/models/bge-small-zh-v1.5; else /usr/bin/python3 scripts/build_lecture_chroma.py; fi && echo $LEC_MD5 > data/chroma/.lecture_source_md5" \
        || echo "⚠️ 讲稿库重建失败，本次部署跳过（服务可正常降级运行）"
else
    echo "讲稿数据未变更，跳过重建"
fi

echo "==> [5/6] 重启 tcm-backend 服务"
$SSH "$USER@$HOST" "sudo systemctl restart tcm-backend"

echo "==> [6/6] 健康检查"
sleep 12
echo "--- 内网 health ---"
$SSH "$USER@$HOST" "curl -s -m 5 http://127.0.0.1:8002/api/health" | head -c 300
echo
echo "--- 公网 https 验证 ---"
curl -s -m 10 "https://$HOST/api/health" | head -c 300
echo
echo "✅ 部署完成"
