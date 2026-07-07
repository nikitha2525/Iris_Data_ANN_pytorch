# 🌸 Day 55 — ANN for Iris Classification: Multiclass Classification in PyTorch

<div align="center">

![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-Dataset-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-Tensors-013243?style=flat-square&logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Viz-11557C?style=flat-square)
![Challenge](https://img.shields.io/badge/100%20Days%20AI%2FML-Day%2055-blueviolet?style=flat-square)

**Achieving high accuracy on Iris is expected — it's small and clean. The real challenge is applying the same concepts to larger, noisier, real-world datasets.**

</div>

---

## 📌 Overview

The Iris dataset is the "Hello World" of classification — small, balanced, and clean. But building it **from scratch in PyTorch** (not scikit-learn) is where the real learning happens. Day 55 implements a full ANN for **multiclass classification**: 4 flower measurements → 3 species, using a custom `nn.Module`, CrossEntropy loss, and a complete training + evaluation loop.

> **Hard truth learned today:** Achieving high accuracy on the Iris dataset is expected because it is small and relatively clean. The real challenge is applying the same concepts to larger, noisier, and more complex real-world datasets.

---

## 🌺 Dataset — Iris

```
                 Sepal             Petal
              ┌──────────┐      ┌──────────┐
              │  Length  │      │  Length  │
              │  Width   │      │  Width   │
              └────┬─────┘      └────┬─────┘
                   │                 │
                   └────────┬────────┘
                            │
                       4 Features
                            │
                            ▼
                 ┌──────────────────────┐
                 │        ANN           │
                 └──────────┬───────────┘
                            │
              ┌─────────────┼─────────────┐
              ▼             ▼             ▼
           Setosa      Versicolor      Virginica
           (Class 0)   (Class 1)      (Class 2)
```

| Property | Detail |
|---|---|
| Source | `sklearn.datasets.load_iris()` |
| Samples | 150 (50 per class) |
| Features | 4 (sepal length, sepal width, petal length, petal width) |
| Classes | 3 (Setosa, Versicolor, Virginica) |
| Class balance | Perfectly balanced (50/50/50) |
| Task | Multiclass classification |

---

## 🏗️ Model Architecture

```
Input Layer        Hidden Layer 1     Hidden Layer 2     Output Layer
(4 neurons)        (16 neurons)       (8 neurons)        (3 neurons)

  x₁ ──────┐       ┌─── h₁₁ ───┐    ┌─── h₂₁ ───┐    ┌─── logit₀ ─► P(Setosa)
  x₂ ──────┼──────►│    h₁₂    ├───►│    h₂₂    ├───►│   logit₁ ─► P(Versicolor)
  x₃ ──────┤       │    h₁₃    │    │    h₂₃    │    │   logit₂ ─► P(Virginica)
  x₄ ──────┘       │    ...    │    └───────────┘    └────────────────────────────
                    └───────────┘
  4 inputs         Linear + ReLU      Linear + ReLU    Linear (no activation)
                                                        ↑
                                              CrossEntropyLoss applies
                                              Softmax internally
```

**Why no activation on the output layer?**
`nn.CrossEntropyLoss` in PyTorch combines `LogSoftmax + NLLLoss` internally — applying Softmax manually before it would cause double-softmax errors.

---

## 📐 Multiclass Classification — Key Concepts

### CrossEntropy Loss

$$\mathcal{L} = -\frac{1}{N} \sum_{i=1}^{N} \log\left(\frac{e^{z_{y_i}}}{\sum_{j} e^{z_j}}\right)$$

Where:
- $z_{y_i}$ = logit for the correct class of sample $i$
- $\sum_j e^{z_j}$ = sum of all class logits (the Softmax denominator)

**In plain terms:** Maximize the predicted probability for the true class. If the model outputs `[2.1, 0.3, 0.8]` for a Setosa sample (class 0), CrossEntropy penalizes based on how far 2.1 is from dominating the distribution.

### Softmax (applied at inference)

$$P(\text{class } k) = \frac{e^{z_k}}{\sum_{j=1}^{3} e^{z_j}}$$

Converts raw logits into probabilities that sum to 1. The class with the highest probability is the prediction.

```
Raw logits:    [2.1,  0.3,  0.8]
After Softmax: [0.73, 0.12, 0.15]   ← sums to 1.0
Prediction:    Class 0 (Setosa) ✅
```

---

## 🔬 What I Implemented

### 1. Data Preparation

```python
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt

# ─── Load Dataset ────────────────────────────────────────────────────────────────
iris = load_iris()
X, y = iris.data, iris.target        # X: (150, 4)  y: (150,) with values 0/1/2

print("Classes:", iris.target_names)  # ['setosa' 'versicolor' 'virginica']
print("Features:", iris.feature_names)

# ─── Scale Features ──────────────────────────────────────────────────────────────
scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ─── Train/Test Split ────────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# ─── Convert to Tensors ──────────────────────────────────────────────────────────
X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.long)     # CrossEntropy needs LongTensor
X_test_t  = torch.tensor(X_test,  dtype=torch.float32)
y_test_t  = torch.tensor(y_test,  dtype=torch.long)

print(f"Train: {X_train_t.shape}  |  Test: {X_test_t.shape}")
```

> **Important:** Labels must be `torch.long` (int64) for `nn.CrossEntropyLoss` — not `float32`. This is a very common PyTorch beginner mistake.

### 2. ANN Model

```python
# ─── Model ───────────────────────────────────────────────────────────────────────
class IrisANN(nn.Module):
    def __init__(self):
        super(IrisANN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(4, 16),           # Input → Hidden 1
            nn.ReLU(),
            nn.Linear(16, 8),           # Hidden 1 → Hidden 2
            nn.ReLU(),
            nn.Linear(8, 3)             # Hidden 2 → Output (3 classes, no activation)
        )

    def forward(self, x):
        return self.network(x)

model = IrisANN()
print(model)
```

### 3. Training Loop

```python
# ─── Loss & Optimizer ────────────────────────────────────────────────────────────
criterion = nn.CrossEntropyLoss()                # Softmax + NLLLoss combined
optimizer = optim.Adam(model.parameters(), lr=0.01)

# ─── Train ───────────────────────────────────────────────────────────────────────
EPOCHS = 150
train_losses = []
train_accuracies = []

for epoch in range(EPOCHS):
    model.train()
    optimizer.zero_grad()                        # ① Clear gradients

    outputs = model(X_train_t)                   # ② Forward pass → logits (120, 3)
    loss    = criterion(outputs, y_train_t)      # ③ CrossEntropy loss

    loss.backward()                              # ④ Backpropagation
    optimizer.step()                             # ⑤ Update weights

    # ─── Track accuracy ──────────────────────────────────────────────────────────
    with torch.no_grad():
        preds = torch.argmax(outputs, dim=1)     # class with highest logit
        acc   = (preds == y_train_t).float().mean().item()

    train_losses.append(loss.item())
    train_accuracies.append(acc)

    if (epoch + 1) % 15 == 0:
        print(f"Epoch [{epoch+1}/{EPOCHS}] | Loss: {loss.item():.4f} | Acc: {acc:.4f}")
```

### 4. Evaluation

```python
# ─── Evaluate ────────────────────────────────────────────────────────────────────
model.eval()
with torch.no_grad():
    test_logits  = model(X_test_t)
    test_preds   = torch.argmax(test_logits, dim=1)
    test_acc     = (test_preds == y_test_t).float().mean().item()

print(f"\nTest Accuracy: {test_acc:.4f} ({test_acc*100:.1f}%)")

# ─── Per-class breakdown ─────────────────────────────────────────────────────────
from sklearn.metrics import classification_report
print("\nClassification Report:")
print(classification_report(
    y_test_t.numpy(),
    test_preds.numpy(),
    target_names=iris.target_names
))

# ─── Softmax probabilities for one sample ────────────────────────────────────────
sample = X_test_t[0].unsqueeze(0)                 # shape: (1, 4)
logits = model(sample)
probs  = torch.softmax(logits, dim=1)
print(f"\nSample probabilities: {probs.numpy()}")
print(f"Predicted class: {iris.target_names[torch.argmax(probs).item()]}")
print(f"Actual class   : {iris.target_names[y_test[0]]}")
```

### 5. Training Curves

```python
# ─── Plot ────────────────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(train_losses, color='tomato', linewidth=1.5)
ax1.set_title('Training Loss (CrossEntropy)')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')

ax2.plot(train_accuracies, color='steelblue', linewidth=1.5)
ax2.axhline(y=1.0, color='green', linestyle='--', alpha=0.5, label='Perfect')
ax2.set_title('Training Accuracy')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy')
ax2.legend()

plt.tight_layout()
plt.savefig('outputs/training_curves.png', dpi=150)
plt.show()
```

---

## 🔄 Binary vs Multiclass — Key Differences

| Aspect | Binary (Day 51/52) | Multiclass (Day 55) |
|---|---|---|
| Output neurons | 1 | N (one per class) |
| Output activation | Sigmoid | None (logits) |
| Loss function | BCELoss | CrossEntropyLoss |
| Label dtype | `torch.float32` | `torch.long` |
| Prediction | `(output ≥ 0.5)` | `torch.argmax(output)` |
| Probability | `sigmoid(logit)` | `softmax(logits)` |

---

## 💡 Key Learnings

- **ANN handles multiclass classification** by adding one output neuron per class — the rest of the architecture stays the same
- **PyTorch makes building custom networks easy** — `nn.Sequential` chains layers cleanly without boilerplate
- **Output layer has no activation** — `CrossEntropyLoss` expects raw logits, not probabilities
- **Labels must be `torch.long`** — using `float32` for class indices causes a silent dimension mismatch error
- **`torch.argmax(dim=1)` converts logits to predicted classes** — the class with the highest logit wins

---

## ⚠️ Common Mistakes in PyTorch Multiclass

```python
# ❌ WRONG — applying Softmax before CrossEntropyLoss (double-softmax)
outputs = torch.softmax(model(X), dim=1)
loss = nn.CrossEntropyLoss()(outputs, y)

# ✅ CORRECT — pass raw logits directly
outputs = model(X)
loss = nn.CrossEntropyLoss()(outputs, y)

# ❌ WRONG — float labels for CrossEntropy
y = torch.tensor(labels, dtype=torch.float32)

# ✅ CORRECT — long (int64) labels
y = torch.tensor(labels, dtype=torch.long)
```

---

## 🗂️ Project Structure

```
day-55-pytorch-iris-ann/
├── iris_ann.py                  # Full training + evaluation pipeline
├── outputs/
│   ├── training_curves.png      # Loss + accuracy over epochs
│   └── classification_report.txt
└── README.md
```

---

## 🚀 Quick Start

```bash
git clone https://github.com/your-username/day-55-pytorch-iris-ann
cd day-55-pytorch-iris-ann
pip install -r requirements.txt
python iris_ann.py
```

**Requirements:**
```
torch
scikit-learn
numpy
matplotlib
```

---

## 🔗 Part of the 100 Days AI/ML Engineer Challenge

> Day 55 of 100 — PyTorch ANN for Multiclass Classification

| ← Previous | Current | Next → |
|---|---|---|
| [Day 53 — Fake News Detection](#) | **Day 55 — Iris ANN** | [Day 56 — Linear Regression](#) |


---

<div align="center">
<sub>Built with curiosity · Part of #100DaysOfAIML · #PyTorch #ANN #IrisDataset #MulticlassClassification #DeepLearning #CrossEntropy</sub>
</div>
