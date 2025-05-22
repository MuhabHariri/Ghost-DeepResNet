# 🧠 Ghost-DeepResNet — An ultra-lightweight classification model with very low FLOPs and parameter count for identifying strawberry diseases and disorders

This repository accompanies the paper **"Ghost-DeepResNet: A High-Performance Ultra-Lightweight Model for Strawberry Disease and Disorder Detection"**

The project is structured for both **researchers** and **practitioners**, offering a clean, modular, and reproducible codebase.

> 🏗️ This implementation reflects the **base** version of Ghost-DeepResNet. 

---

## 📦 Key Features

- ✅ **Ghost-DeepResNet Block** — Efficient residual units with attention  
- ✅ **Ghost-DeepResNet Model** — The classification model  
- ✅ Flexible config system for architecture variants  
- ✅ Multi-GPU distributed training (via `MirroredStrategy`)

---

## 🚀 Getting Started
### 1. Clone the Repository

```bash
git clone https://github.com/MuhabHariri/Ghost-DeepResNet.git
```
```bash
cd Ghost-DeepResNet
```


---

### 2. Install Requirements

```bash
pip install -r requirements.txt
```



---

### 3. Prepare Your Dataset
```bash
Dataset/
├── Train/
│   ├── class_1/
│   ├── class_2/
│   └── ...
└── Val/
│   ├── class_1/
│   ├── class_2/
│   └── ...
│
└── Test/
    ├── class_1/
    ├── class_2/
    └── ...
```
Update paths in src/config.py: 
```bash
TRAIN_DIR = "\Dataset\Train"
VAL_DIR   = "\Dataset\Val"
Test_DIR  = "\Dataset\Test"
```

---


### 4. Train the Model 
```bash
python train.py
```
---
