# CPRP

This repository contains the official implementation of the method proposed in the paper:

**Comprehensive Perception, Rational Prediction: Achieving More Robust Deformable Object Keypoint Localization Using Generative Data Augmentation and Probabilistic Prediction Models**

The code is provided for research and review purposes.


### Dependency

* Python >= 3.8
* PyTorch >= 1.9
* `segmentation_models_pytorch`
* `timm`
* `opencv-python`
* `numpy`, `scipy`, `scikit_image`, `scikit_learn`

Please refer to `requirements.txt` for the complete list of dependencies.

### Recommended Installation with Conda

```bash
conda create -n=CPRP-py3.8 python=3.8
conda activate CPRP-py3.8

# install PyTorch according to your CUDA version (example for CUDA 11.7)
pip install torch==2.0.0+cu117 torchvision==0.15.1+cu117  \
     -f https://download.pytorch.org/whl/torch_stable.html

pip install -r requirements.txt
```

---

##  Pre-trained Models

For convenience, we provide pre-trained model weights for CPRP.

Please download the checkpoint from the following link and place it into the `models/` directory:

🔗 **Pre-trained model (Google Drive):**

```
https://drive.google.com/file/d/191FUTuleq9indHuw43spSvxbK1gc5cih/view?usp=drive_link
```

After downloading, organize your directory as follows:

```
models/
 └─ ARF-2B.pth
```

---



