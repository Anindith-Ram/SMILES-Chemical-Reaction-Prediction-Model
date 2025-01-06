# **Chemical Reaction Prediction using SMILES**

This repository contains the code and resources for training and evaluating a transformer-based model for chemical reaction prediction using SMILES (Simplified Molecular Input Line Entry System) notation. The project incorporates RoBERTa, a transformer decoder, and additional features like max pooling for efficient representation learning.

---

## **Table of Contents**
- [Features](#features)
- [Model Architecture](#model-architecture)
- [Setup](#setup)
- [Usage](#usage)
- [Dataset](#dataset)
- [Pretrained Model](#pretrained-model)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

---

## **Features**
- **Data Preprocessing**: Canonicalizes and splits SMILES reactions into reactants and products.
- **Model Architecture**:
  - RoBERTa encoder for learning chemical representations.
  - Transformer decoder with attention for product prediction.
  - Max pooling layer to reduce dimensionality and enhance performance.
- **Training and Evaluation**:
  - AMP (Automatic Mixed Precision) for faster training.
  - Early stopping to prevent overfitting.
  - Metrics: Loss, Accuracy, F1-Score, Precision, Recall.

---

## **Model Architecture**
- **Encoder**: Pretrained `RobertaModel` (ChemBERTa).
- **Decoder**: Transformer decoder with multi-head attention.
- **Pooling**: Adaptive max pooling to refine feature representations.

---

## **Setup**
### **Requirements**
Install the necessary dependencies:
```bash
pip install -r requirements.txt
