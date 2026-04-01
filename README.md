# 🎥 Video Action Classification with VideoMAE

This project implements a **video-based human action recognition pipeline** using a **pretrained VideoMAE transformer backbone** from HuggingFace.  
It supports multi-frame input, strong data augmentation, train/validation split, early stopping, and **test-time augmentation (TTA)** with multi-clip averaging.

Designed for **Kaggle-style competitions** with CSV submission output.

---

## ✨ Features

- 🎞️ Multi-frame video input
- 🤗 Pretrained VideoMAE backbone (HuggingFace)
- 🧠 Temporal aggregation
- 🔀 Strong data augmentation
- 🧪 Train / validation split
- ⏹️ Early stopping on best validation accuracy
- 🔁 Test-time augmentation (multi-clip averaging)
- ⚡ Mixed precision training (AMP)
- 📄 Automatic CSV submission generation

---

## 📁 Project Structure

```text
Video-Action-Classification/
├── hmdb51_data/        # Dataset directory
├── weights/           # Saved models
├── LSVIT-HMDB51.ipynb # Training & inference notebook
├── VideoMAE.ipynb     # Model experiments
├── README.md
└── .gitignore
---
```

## 🧠 Model

This project uses:

- **VideoMAE** (Masked Autoencoder for Video Transformers)
- Pretrained on large-scale video datasets
- Fine-tuned for action classification

Backbone source:
> https://huggingface.co/MCG-NJU/videomae-base

---

## 📦 Installation

```bash
pip install torch torchvision torchaudio
pip install transformers accelerate tqdm opencv-python
```

## 📂 Dataset Format

Expected structure (train):
```text
data_train/
├── class_1/
│   └── video_001/
│       ├── 0001.jpg
│       ├── 0002.jpg
│       └── ...
├── class_2/
│   └── ...

data_test/
├── 00001/
│   ├── 0001.jpg
│   ├── 0002.jpg
│   └── ...
├── 00002/
│   └── ...
```
## ⚙️ Configuration

Key parameters:

```python
NUM_FRAMES = 16
FRAME_STRIDE = 2
IMG_SIZE = 224

BATCH_SIZE = 8
EPOCHS = 20
BASE_LR = 2e-5
WEIGHT_DECAY = 0.05
GRAD_ACCUM_STEPS = 8
```
## 🏋️ Training

Training includes:

- Mixed precision (AMP)
- Gradient accumulation
- Cosine LR schedule
- Early stopping

### To train

Run the training cells inside the notebook:

```python
train_one_epoch(...)
```

Best model is automatically saved as:

```python
best_videomae.pt
```
## 🛑 Early Stopping

The best model is saved based on validation accuracy:
```python
if val_acc > best_acc:
    torch.save(...)
```
## 🔍 Inference

Load best model:
```python
ckpt = torch.load("best_videomae.pt")
model.load_state_dict(ckpt["model"])
model.eval()
```
## 🔁 Test-Time Augmentation (TTA)

TTA improves performance by averaging predictions across multiple temporal offsets.

Offsets example:
```python
offsets = (0, 4, 8)
```
Final prediction = mean of logits from all offsets.

## 🧾 Submission Format

The script generates:
submission.csv
Format:
```python
id,class
1,Walking
2,Running
3,Jumping
```
## 📊 Results
| Metric              | Score                     |
| ------------------- | ------------------------- |
| Train Accuracy      | ~0.99                     |
| Validation Accuracy | ~0.78                     |
| Public Test Score   | ~0.61 → Improved with TTA |

---

## ⚠️ Limitation
- Temporal information is not fully captured due to frame aggregation strategy
- Mean pooling may lose fine-grained motion dynamics between frames

---

## 🚀 Future Improvements

- Grouped train/val split by video ID
- Larger `NUM_FRAMES` (e.g., 32)
- More temporal offsets for TTA
- Ensemble models
- Audio fusion

---

## 📚 References

- **VideoMAE**: https://arxiv.org/abs/2203.12602 ↗
- **HuggingFace Transformers**: https://huggingface.co/docs/transformers ↗

