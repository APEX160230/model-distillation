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

SSH="ssh -i ~/.ssh/tcm_deploy_key -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=15"
SCP="scp -i ~/.ssh/tcm_deploy_key -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=15"

echo "==> [1/5] 上传源码 src/"
$SCP -r src "$USER@$HOST:/home/ubuntu/tcm/backend/"

echo "==> [2/5] 上传数据资产（chroma 向量库 + 条文数据）"
$SCP -r data/chroma data/processed "$USER@$HOST:/home/ubuntu/tcm/backend/data/"

echo "==> [3/5] 上传 Modelfile"
$SCP Modelfile "$USER@$HOST:/home/ubuntu/tcm/"

echo "==> [4/5] 重启 tcm-backend 服务"
$SSH "$USER@$HOST" "sudo systemctl restart tcm-backend"

echo "==> [5/5] 健康检查"
sleep 12
echo "--- 内网 health ---"
$SSH "$USER@$HOST" "curl -s -m 5 http://127.0.0.1:8002/api/health" | head -c 300
echo
echo "--- 公网 https 验证 ---"
curl -s -m 10 "https://$HOST/api/health" | head -c 300
echo
echo "✅ 部署完成"
