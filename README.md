# ZEROSUM-AI-ML-Project
# Offroad Semantic Segmentation – Hackathon Submission

## 📌 Overview

This project implements semantic segmentation for offroad scenes using a pretrained DINOv2 (ViT-S/14) backbone and a lightweight ConvNeXt-style segmentation head.

The objective was to maximize Mean Intersection over Union (IoU) through controlled optimization and training improvements.

Final Validation Performance:
- Mean IoU: 0.3575
- Dice Score: 0.5169
- Pixel Accuracy: 0.7110

---

# 🧰 Environment & Dependencies

## Required Environment

- Python 3.10+
- Conda (recommended)
- CUDA-enabled GPU (tested on RTX 4050)
- PyTorch with CUDA (cu118 recommended)

## Required Python Packages

- torch
- torchvision
- numpy
- opencv-python
- pillow
- matplotlib
- tqdm

If needed, install using:

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install numpy opencv-python pillow matplotlib tqdm


---

# ⚙️ Step-by-Step Setup

## 1️⃣ Activate Environment

If using conda:

conda activate EDU

If creating manually:
conda create -n EDU python=3.10
conda activate EDU
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

pip install numpy opencv-python pillow matplotlib tqdm

---

# 🚀 Training the Model

Run from the `Offroad_Segmentation_Scripts` directory:
python train_segmentation.py

What happens during training:

- DINOv2 backbone is loaded and frozen
- Segmentation head is trained for 20 epochs
- Weighted CrossEntropy loss is applied
- AdamW optimizer is used
- Training curves are saved in `train_stats/`
- Final model saved as:
  segmentation_head.pth

---

# 🧪 Testing / Evaluation

To evaluate on validation dataset:
python test_segmentation.py --model_path segmentation_head.pth --data_dir <validation_dataset_path>

Example:
python test_segmentation.py --model_path segmentation_head.pth --data_dir ../Offroad_Segmentation_Training_Dataset/val

---

# 📊 Reproducing Final Results

To reproduce the reported IoU:

1. Ensure:
   - Loss = Weighted CrossEntropy
   - Optimizer = AdamW
   - Learning Rate = 3e-4
   - Epochs = 20
   - Backbone frozen

2. Run:
python train_segmentation.py

3. Then evaluate:
python test_segmentation.py --model_path segmentation_head.pth --data_dir ../Offroad_Segmentation_Training_Dataset/val


Expected Output:

- Mean IoU ≈ 0.3575
- Dice Score ≈ 0.51
- Pixel Accuracy ≈ 0.71

(Note: Small variations may occur depending on hardware or randomness.)

---

# 📁 Expected Outputs

After evaluation, a `predictions/` folder will be created containing:

## predictions/
- `masks/` → Raw predicted class ID masks
- `masks_color/` → Color-coded segmentation outputs
- `comparisons/` → Side-by-side input, ground truth, and prediction
- `evaluation_metrics.txt` → Numerical metrics summary
- `per_class_metrics.png` → Bar chart of per-class IoU

---

# 📈 Interpreting Outputs

- **Mean IoU**: Primary evaluation metric. Higher is better.
- **Per-class IoU**: Indicates performance on individual classes.
- **Dice Score**: F1-style metric for segmentation.
- **Pixel Accuracy**: Overall pixel-wise classification accuracy.

Strong classes: Sky and Landscape  
Challenging classes: Small objects (Logs, Rocks, Clutter)

---

# 🧠 Model Summary

- Backbone: DINOv2 (ViT-S/14)
- Segmentation Head: ConvNeXt-style
- Loss: Weighted CrossEntropy
- Optimizer: AdamW
- Training Epochs: 20

Backbone remains frozen to ensure stable and efficient training.

---

# 🔮 Future Improvements

- Multi-scale training
- Hybrid Dice + CrossEntropy loss
- Backbone fine-tuning
- Stronger augmentation strategies


