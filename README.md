# CPRP: Comprehensive Perception & Rational Prediction

This repository contains the official implementation of **CPRP**, a deformable-object keypoint localization framework integrating:

- **GenerativeAug** (diffusion-based data augmentation)
- **ProbPredictor** (Gaussian Mixture Model based probabilistic prediction)
- Fully compatible with the CeDiRNet-3DoF baseline.

---

## 📁 Dataset Preparation

### 1. aRTF-K Dataset (Annotated Clothes Dataset)
We extend the original aRTF Clothes Dataset with ViCoS-style keypoint annotations.

Download link (replace with your own):
🔗 **aRTF-K Dataset**  
```

[Google Drive] [https://drive.google.com/xxxxxx](https://drive.google.com/xxxxxx)

```

Unzip and place it under:
```

datasets/aRTF-K/

```

### 2. ViCoS Towel Dataset
Official dataset link:  
https://vicos.si/resources/towel-dataset/

Place as:
```

datasets/ViCoS/

```

---

## 🧩 Pretrained Weights

We provide two pretrained GMM-based probabilistic CPRP models:

### 🔹 ARF-2B Model (trained on aRTF-K)
```

models/CPRP_ARF_2B.pth

```
Download:  
🔗 https://drive.google.com/xxxxx  


### 🔹 VICOS-2B Model (trained on ViCoS)
```

models/CPRP_VICOS_2B.pth

```
Download:  
🔗 https://drive.google.com/xxxxx  

Make sure the final structure is:
```

models/
CPRP_ARF_2B.pth
CPRP_VICOS_2B.pth

````

---

## ⚙️ Environment Setup

```bash
conda create -n CPRP python=3.8
conda activate CPRP

pip install -r requirements.txt
````

(If using the exact environment as CeDiRNet-3DoF, no further changes needed.)

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

