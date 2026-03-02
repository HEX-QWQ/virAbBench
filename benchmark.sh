#!/usr/bin/env bash

set -euo pipefail

# 在这里维护要测试的 min_diff_k 值
MIN_DIFF_K_VALUES=(1 3 5 10)

# 训练脚本与固定参数
TRAIN_SCRIPT="train.py"
MODEL_NAME="LucaBCR-opt"
BASE_LOG_PATH="./logs/LucaBCR-opt/train.log"

for k in "${MIN_DIFF_K_VALUES[@]}"; do
  log_dir="$(dirname "$BASE_LOG_PATH")"
  log_file="$(basename "$BASE_LOG_PATH")"
  run_log_path="${log_dir}/min_diff_${k}_${log_file}"

  echo "[INFO] Running with --min_diff_k=${k}, --log_path=${run_log_path}"

  CUDA_VISIBLE_DEVICES=2 python "$TRAIN_SCRIPT" \
    --model_name "$MODEL_NAME" \
    --min_diff_k "$k" \
    --log_path "$run_log_path"
done

echo "[INFO] All benchmark runs completed."
