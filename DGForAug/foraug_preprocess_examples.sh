#!/bin/bash

# 单张示例：提取前景 + 使用 ForAug inpainting 重建背景
# 用法：
#   ./foraug_preprocess_examples.sh [RGB_PATH] [MASK_PATH] [OUTPUT_DIR]
# 若不提供参数，则使用示例数据。可通过环境变量覆盖 INPAINT_MODEL、BACKGROUND_FORMAT、OUTPUT_PREFIX。


conda activate DGInStyle

BASE_DIR="/home/yufan/project/code/DGForAug"
PYTHON_SCRIPT="${BASE_DIR}/foraug_preprocess_examples.py"

RGB_DEFAULT="${BASE_DIR}/vicos_data/examples/rgb/image_0000_view10_ls4_camera0.jpg"
MASK_DEFAULT="${BASE_DIR}/vicos_data/examples/mask/image_0000_view10_ls4_camera0.png"
OUTPUT_DIR_DEFAULT="${BASE_DIR}/vicos_output/preprocessed_examples_single"

INPUT_RGB="${1:-${RGB_DEFAULT}}"
INPUT_MASK="${2:-${MASK_DEFAULT}}"
OUTPUT_DIR="${3:-${OUTPUT_DIR_DEFAULT}}"

INPAINT_MODEL=${INPAINT_MODEL:-"attentive_eraser"}  # 备选: lama
BACKGROUND_FORMAT=${BACKGROUND_FORMAT:-"jpg"}        # jpg 或 png

if [[ -n "${OUTPUT_PREFIX:-}" ]]; then
  PREFIX_VALUE="${OUTPUT_PREFIX}"
else
  BASENAME="$(basename "${INPUT_RGB}")"
  PREFIX_VALUE="${BASENAME%.*}"
fi

echo "[Info] BASE_DIR          = ${BASE_DIR}"
echo "[Info] PYTHON_SCRIPT     = ${PYTHON_SCRIPT}"
echo "[Info] INPUT_RGB         = ${INPUT_RGB}"
echo "[Info] INPUT_MASK        = ${INPUT_MASK}"
echo "[Info] OUTPUT_DIR        = ${OUTPUT_DIR}"
echo "[Info] OUTPUT_PREFIX     = ${PREFIX_VALUE}"
echo "[Info] INPAINT_MODEL     = ${INPAINT_MODEL}"
echo "[Info] BACKGROUND_FORMAT = ${BACKGROUND_FORMAT}"

mkdir -p "${OUTPUT_DIR}"

CMD=(
  python "${PYTHON_SCRIPT}"
  --input-rgb "${INPUT_RGB}"
  --input-mask "${INPUT_MASK}"
  --output-dir "${OUTPUT_DIR}"
  --output-prefix "${PREFIX_VALUE}"
  --inpaint-model "${INPAINT_MODEL}"
  --background-format "${BACKGROUND_FORMAT}"
)

echo "[Info] Running command:"
printf '  %q' "${CMD[@]}"
echo

"${CMD[@]}"

echo "[Info] Done. Foreground/Background saved under: ${OUTPUT_DIR}"

