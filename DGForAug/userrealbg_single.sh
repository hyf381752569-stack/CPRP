#!/bin/bash

# 单图片版运行脚本：参考 userrealbg.sh，调用 single_image_vicos_realbg_viz.py
# 用法：
#   ./userrealbg_single.sh \ 
#     /abs/path/to/rgb.jpg /abs/path/to/mask.png /abs/path/to/clean_bg.jpg [/abs/path/to/output_dir]
#
# 若未提供位置参数，则使用下方的默认变量。

# --- 1) 基本路径与模型路径 ---
conda activate DGInStyle
BASE_DIR="/home/yufan/project/code/DGForAug"

BASE_MODEL="${BASE_DIR}/downloaded_models/realistic-vision-v6-b1"
CONTROLNET_TILE="${BASE_DIR}/downloaded_models/control_v11f1e_sd15_tile"
VAE_MODEL="${BASE_DIR}/downloaded_models/sd-vae-ft-mse"

# --- 2) 输入/输出（单张图）---
# 固定输入输出路径
INPUT_RGB="/home/yufan/project/code/DGForAug/vicos_data/examples/rgb/image_0000_view10_ls4_camera0.jpg"
INPUT_FOREGROUND="/home/yufan/project/code/DGForAug/vicos_data/examples/foreground/image_0000_view10_ls4_camera0_foreground.png"
INPUT_CLEAN_BG="/home/yufan/project/code/DGForAug/vicos_data/examples/background/image_0000_view10_ls4_camera0_background.jpg"
INPUT_MASK="/home/yufan/project/code/DGForAug/vicos_data/examples/mask/image_0000_view10_ls4_camera0.png"
OUTPUT_DIR="${BASE_DIR}/vicos_output/single_image_out"

# --- 3) 推理/控制参数（与批处理脚本风格一致）---
CONTROLNET_SCALE=0.6
STRENGTH=0.9
STEPS=35
EDGE_FEATHER=5
SEED=500
DEVICE=cuda

NEGATIVE_PROMPT="ugly, deformed, disfigured, poor details, bad anatomy, worst quality, low quality, blurry, noisy, text, watermark, signature, cloth, fabric, towel, wrinkles"
PROMPT=""

# --- 4) 参数检查与打印 ---
# for p in "$BASE_MODEL" "$CONTROLNET_TILE" "$VAE_MODEL" "$INPUT_RGB" "$INPUT_MASK" "$INPUT_CLEAN_BG"; do
#   if [[ ! -e "$p" ]]; then
#     echo "[Error] 路径不存在: $p" >&2
#     exit 1
#   fi
# done

mkdir -p "$OUTPUT_DIR"

echo "[Info] BASE_MODEL          = $BASE_MODEL"
echo "[Info] CONTROLNET_TILE     = $CONTROLNET_TILE"
echo "[Info] VAE_MODEL           = $VAE_MODEL"
echo "[Info] INPUT_RGB           = $INPUT_RGB"
echo "[Info] INPUT_FOREGROUND    = $INPUT_FOREGROUND"
echo "[Info] INPUT_MASK          = $INPUT_MASK"
echo "[Info] INPUT_CLEAN_BG      = $INPUT_CLEAN_BG"
echo "[Info] OUTPUT_DIR          = $OUTPUT_DIR"
echo "[Info] CONTROLNET_SCALE    = $CONTROLNET_SCALE"
echo "[Info] STRENGTH            = $STRENGTH"
echo "[Info] STEPS               = $STEPS"
echo "[Info] EDGE_FEATHER        = $EDGE_FEATHER"
echo "[Info] SEED                = ${SEED:-<none>}"
echo "[Info] DEVICE              = $DEVICE"
[[ -n "$PROMPT" ]] && echo "[Info] PROMPT              = $PROMPT"

# --- 5) 执行 Python 脚本 ---
PYTHON_SCRIPT="${BASE_DIR}/single_image_vicos_realbg_viz.py"

CMD=(
  python "$PYTHON_SCRIPT"
  --base_model_path "$BASE_MODEL"
  --controlnet_tile_path "$CONTROLNET_TILE"
  --vae_path "$VAE_MODEL"
  --input_rgb "$INPUT_RGB"
  --input_foreground "$INPUT_FOREGROUND"
  --input_mask "$INPUT_MASK"
  --input_clean_bg "$INPUT_CLEAN_BG"
  --output_dir "$OUTPUT_DIR"
  --controlnet_scale "$CONTROLNET_SCALE"
  --strength "$STRENGTH"
  --steps "$STEPS"
  --edge_feather_pixels "$EDGE_FEATHER"
  --device "$DEVICE"
  --negative_prompt "$NEGATIVE_PROMPT"
)

CMD+=(--seed "$SEED")
if [[ -n "$PROMPT" ]]; then
  CMD+=(--prompt "$PROMPT")
fi

echo "[Info] Running: ${CMD[*]}"
"${CMD[@]}"

echo "[Info] Done. See outputs under: $OUTPUT_DIR"
