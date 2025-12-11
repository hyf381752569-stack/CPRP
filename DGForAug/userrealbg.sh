#!/bin/bash

# --- 1. 定义路径变量 ---
conda activate DGInStyle
BASE_DIR="/home/yufan/project/code/DGForAug"

# 模型路径
BASE_MODEL="${BASE_DIR}/downloaded_models/realistic-vision-v6-b1"
CONTROLNET_TILE="${BASE_DIR}/downloaded_models/control_v11f1e_sd15_tile"
VAE_MODEL="${BASE_DIR}/downloaded_models/sd-vae-ft-mse"

# 输入/输出路径
INPUT_RGB="${BASE_DIR}/vicos_data/synthetic_data_for_controlnet_real/rgb"
INPUT_MASK="${BASE_DIR}/vicos_data/synthetic_data_for_controlnet_real/mask"
INPUT_FOREGROUND="${BASE_DIR}/vicos_data/synthetic_data_for_controlnet_real/foreground"
INPUT_CLEAN_BG="${BASE_DIR}/vicos_data/synthetic_data_for_controlnet_real/background"
OUTPUT_DIR="${BASE_DIR}/vicos_output/output"

# --- 2. 定义核心参数 ---
CONTROLNET_SCALE=0.8      # ControlNet的权重，为了避免布料幽灵，我们将其降低到0.85
STRENGTH=0.8               # 风格化强度，1.0意味着彻底重绘背景
STEPS=35                   # 推理步数，略微增加以提高质量
FEATHER=5                 # 边缘羽化像素
SEED=42                    # 批处理种子 

# 通用负向提示词
NEGATIVE_PROMPT="ugly, deformed, disfigured, poor details, bad anatomy, worst quality, low quality, blurry, noisy, text, watermark, signature, cloth, fabric, towel, wrinkles"


# --- 3. 执行 Python 脚本 ---
python "${BASE_DIR}/generate_vicos_realbg.py" \
    --base_model_path "${BASE_MODEL}" \
    --controlnet_tile_path "${CONTROLNET_TILE}" \
    --vae_path "${VAE_MODEL}" \
    --input_rgb_dir "${INPUT_RGB}" \
    --input_mask_dir "${INPUT_MASK}" \
    --input_foreground_dir "${INPUT_FOREGROUND}" \
    --input_clean_bg_dir "${INPUT_CLEAN_BG}" \
    --output_dir "${OUTPUT_DIR}" \
    \
    --controlnet_scale "${CONTROLNET_SCALE}" \
    --strength "${STRENGTH}" \
    --edge_feather_pixels "${FEATHER}" \
    --steps "${STEPS}" \
    --batch_seed "${SEED}" \
    --negative_prompt "${NEGATIVE_PROMPT}"
