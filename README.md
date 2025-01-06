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

## **Repository Structure**
src/
├── __init__.py               # Makes src a package
├── model.py                  # Model architecture
├── data_processing.py        # Data preprocessing utilities
├── train.py                  # Training script
├── utils.py                  # Additional helper functions
models/                       # Folder to store trained models
README.md                     # Project documentation
requirements.txt              # Python dependencies

---

### **Usage**
1. **Preprocess Data**
   Run the `data_processing.py` script to preprocess the SMILES dataset: python src/data_processing.py

2. **Train the Model**
Use the `train.py` script to train the model: python src/train.py


3. **Evaluate the Model**
Evaluate the trained model using the metrics provided in the script.

---

### **Dataset**
The project uses the **USPTO-50K** dataset for reaction prediction. A sample of this dataset is located in `data/`.
Download the full dataset from [Papers with Code](https://paperswithcode.com/dataset/uspto-50k).
Place the dataset in the `data/` directory.

---

### **Pretrained Model**
Download the pretrained model:
- [Google Drive Link](drive.google.com/file/d/1GJmqMhXb4y5wTRmntxN7XrhWJU__06Aw/view?usp=sharing)
Place the downloaded model in the `models/` directory in the root of the repository:
models/
└── best_model.pth

---

## **Results**
| Metric        | Training | Validation |
|---------------|----------|------------|
| Loss          | 0.037    | 0.001      |
| Accuracy      | 99.54%   | 99.99%     |
| F1-Score      | 90.64%   | 99.31%     |

---

## **License**
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## **Acknowledgments**
- [RDKit](https://www.rdkit.org/) for chemical informatics tools.
- [ChemBERTa](https://github.com/seyonechithrananda/bert-loves-chemistry) for pre-trained embeddings.
- USPTO-50K dataset.
