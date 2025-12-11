#!/bin/bash
cd /home/yufan/project/code/CeDiRNet/src || exit 1
conda activate CeDiRNet-py3.8

# ------------------------- 公共环境变量 -------------------------
export OUTPUT_DIR="/data/hyf/dataset/exp/VICOSGMM-2B"
export USE_DEPTH=True
export PRETRAINED_CENTER_MODEL="/home/yufan/project/code/CeDiRNet/model/localization_checkpoint.pth"
export MODEL_VARIANT='gaussian_mixture' # "deterministic", "single_gaussian", "gaussian_mixture"
export MODEL_NUM_MIXTURES=2
# =================================================================
#                       1. 合成数据预训练 (MuJoCo)
# =================================================================
echo "开始合成数据预训练..."
export DATASET=mujoco
export MUJOCO_DIR="/home/yufan/project/data/Vicos/dataset/MuJoCo"

python train.py --config model.kwargs.backbone=tu-convnext_base n_epochs=10 train_dataset.batch_size=4 train_dataset.workers=16 pretrained_center_model_path="$PRETRAINED_CENTER_MODEL" skip_if_exists=True display=False save_interval=5

# =================================================================
#                  2. 真实数据训练 (ViCoSTowel)
# =================================================================
echo "开始真实数据训练..."
export DATASET=vicos_towel
export VICOS_TOWEL_DATASET_DIR="/data/hyf/dataset/ViCoSTowelDataset"
export TRAIN_SIZE=768
export TEST_SIZE=768

# 从预训练模型加载权重
PRETRAINED_MODEL="/data/hyf/dataset/exp/VICOSGMM-2B/mujoco/backbone=tu-convnext_base/modeling=gaussian_mixture/mixtures=2/num_train_epoch=10/depth=True/multitask_weight=off/checkpoint.pth"

python train.py --config model.kwargs.backbone=tu-convnext_base n_epochs=10 train_dataset.batch_size=4 skip_if_exists=True display=False save_interval=1 pretrained_center_model_path="$PRETRAINED_CENTER_MODEL" pretrained_model_path="$PRETRAINED_MODEL"

# =================================================================
#                       3. 最终模型评估
# =================================================================
echo "开始模型评估..."
python test.py --config model.kwargs.backbone=tu-convnext_base eval_epoch="" train_settings.n_epochs=10 display=False center_checkpoint_path="$PRETRAINED_CENTER_MODEL" center_checkpoint_name_list=None 
# python infer.py --config model.kwargs.backbone=tu-convnext_base eval_epoch="" train_settings.n_epochs=20 display=False center_checkpoint_path="$PRETRAINED_CENTER_MODEL" center_checkpoint_name_list=None 



echo "所有任务执行完毕！"
