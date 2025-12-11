#!/bin/bash

# 批处理入口，调用 Python 版本以避免重复加载模型
# 用法：
#   ./foraug_preprocess_batch.sh [RGB_DIR] [MASK_DIR] [FG_DIR] [BG_DIR]


conda activate DGInStyle

BASE_DIR="/home/yufan/project/code/DGForAug"
PYTHON_SCRIPT="${BASE_DIR}/foraug_preprocess_batch.py"

RGB_DIR="${1:-${BASE_DIR}/vicos_data/synthetic_data_for_controlnet_real/rgb}"
MASK_DIR="${2:-${BASE_DIR}/vicos_data/synthetic_data_for_controlnet_real/mask}"
FOREGROUND_DIR="${3:-${BASE_DIR}/vicos_data/synthetic_data_for_controlnet_real/foreground}"
BACKGROUND_DIR="${4:-${BASE_DIR}/vicos_data/synthetic_data_for_controlnet_real/background}"

INPAINT_MODEL=${INPAINT_MODEL:-"attentive_eraser"}
BACKGROUND_FORMAT=${BACKGROUND_FORMAT:-"jpg"}

echo "[Info] 批处理开始"
echo "[Info] RGB_DIR          = ${RGB_DIR}"
echo "[Info] MASK_DIR         = ${MASK_DIR}"
echo "[Info] FOREGROUND_DIR   = ${FOREGROUND_DIR}"
echo "[Info] BACKGROUND_DIR   = ${BACKGROUND_DIR}"
echo "[Info] INPAINT_MODEL    = ${INPAINT_MODEL}"
echo "[Info] BACKGROUND_FMT   = ${BACKGROUND_FORMAT}"

python "${PYTHON_SCRIPT}" \
  --input-rgb-dir "${RGB_DIR}" \
  --input-mask-dir "${MASK_DIR}" \
  --foreground-dir "${FOREGROUND_DIR}" \
  --background-dir "${BACKGROUND_DIR}" \
  --inpaint-model "${INPAINT_MODEL}" \
  --background-format "${BACKGROUND_FORMAT}"

echo "[Info] 批处理完成"
