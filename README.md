# CPRP: Comprehensive Perception & Rational Prediction

This repository contains the official implementation of **CPRP**, a deformable-object keypoint localization framework integrating:

- **GenerativeAug** (diffusion-based data augmentation)
- **ProbPredictor** (Gaussian Mixture Model based probabilistic prediction)
- Fully compatible with the CeDiRNet-3DoF baseline.

---

## 📁 Dataset Preparation

### 1. aRTF-K Dataset (Annotated Clothes Dataset)
We extend the original aRTF Clothes Dataset with ViCoS-style keypoint annotations.

Download link:
🔗 **aRTF-K Dataset**  


Google Drive(https://drive.google.com/file/d/1q1Z_lGAwAHK0vbdZiFJeZbp4tIMkbB7F/view?usp=drive_link)



Unzip and place it under:
```

datasets/aRTF-K/

```

### 2. ViCoS Towel Dataset
Official dataset link:  
https://vicos.si/resources/towel-dataset/](https://github.com/vicoslab/CeDiRNet-3DoF

Place as:
```

datasets/ViCoS/

```

---

## 🧩 Pretrained Weights

We provide two pretrained GMM-based probabilistic CPRP models:

### 🔹 ARF-2B Model (trained on aRTF-K)
```

models/ARF_2B.pth

```
Download:  
🔗 (https://drive.google.com/file/d/1GHy-alsSHlCWydtV1eClVEsyMYXiJiUT/view?usp=drive_link)


### 🔹 VICOS-2B Model (trained on ViCoS)
```

models/VICOS_2B.pth

```
Download:  
🔗 (https://drive.google.com/file/d/1_bzicdKTJ60834rg8C8dYN5vPDkfGqAi/view?usp=drive_link)

Make sure the final structure is:
```

models/
ARF_2B.pth
VICOS_2B.pth

````

---

## ⚙️ Environment Setup

```
conda env create -f environment.yml
conda activate CPRP   
````



# 🌈 Data Generation (GenerativeAug)

CPRP integrates a powerful **GenerativeAug** pipeline, which performs:

1. Foreground–background separation
2. Seamless background inpainting
3. Stable Diffusion background replacement
4. Lossless foreground compositing

To generate new training data:

---

## ▶ Step 1 — Foreground–background separation

(Using Attentive Eraser)

```bash
bash DGForAug/foraug_preprocess_batch.sh
```

This script performs:

* Attentive Eraser clean background reconstruction
* File organization into `fg/`, `bg/`, `mask/` folders

Outputs will be stored in:

```
DGForAug/output_preprocess/
```

---

## ▶ Step 2 — Background generation with Stable Diffusion

```bash
bash DGForAug/userrealbg.sh
```

This script performs:

* Random prompt generation
* ControlNet-based structural conditioning
* Full-mask global inpainting
* Multi-style background synthesis

Generated training data will appear in:

```
DGForAug/output_synthesis/
```

---




## 🚀 Running Experiments

We provide reproducible scripts for both datasets.

---

### ▶ Run experiments on **aRTF-K Dataset**

```bash
bash scripts/EXPERIMENTS_ARF.sh
```

This script will:

* Train CPRP with GenerativeAug
* Train ProbPredictor with k=2 Gaussian mixture
* Evaluate on the unseen-background test set

Logs and results will be saved in:

```
results/aRTF-K/
```

---

### ▶ Run experiments on **ViCoS Dataset**

```bash
bash scripts/EXPERIMENTS_VICOS.sh
```

This script will:

* Train on ViCoS training split
* Test on the official ViCoS unseen backgrounds & towel types
* Generate visual comparisons and F1 reports

Outputs in:

```
results/ViCoS/
```

---
# 📊 Project Structure

```
CPRP/
│
├── models/
│   ├── ARF_2B.pth
│   └── VICOS_2B.pth
│
├── DGForAug/   # GenerativeAug pipeline
│   ├── foraug_preprocess_batch.sh
│   ├── userrealbg.sh
│   └── ...
│
├── scripts/
│   ├── EXPERIMENTS_ARF.sh
│   └── EXPERIMENTS_VICOS.sh
│
└── datasets/
    ├── aRTF-K/
    └── ViCoS/
```

---
