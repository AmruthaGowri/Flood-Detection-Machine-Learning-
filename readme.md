# Flood Detection Using Deep Learning
### CNN-Based Image Classification on FloodIMG Dataset

---

## Overview

Floods are among the most frequent and destructive natural disasters, causing significant human and economic loss. With the increasing availability of real-time imagery from social media and drones, there is an opportunity to automate flood detection.

This project implements a deep learning-based image classification system to identify whether a given image represents a flooded or non-flooded scene. The system is designed to support disaster response by enabling faster and more scalable analysis of visual data.

---

## Objectives

- Develop a binary image classification model for flood detection
- Compare a custom Convolutional Neural Network (CNN) with a transfer learning model (ResNet-50)
- Analyze the impact of extreme class imbalance
- Evaluate models using robust metrics such as ROC-AUC and F1-score instead of accuracy alone

---

## Project Structure

```
Flood-Detection/
│
├── Flood_Detection_Full_Notebook_Final.ipynb
├── Flood_Detection_Report.pdf
├── ML Final PPT.pptx
├── dataset/ (not included)
└── README.md
```

---

## Dataset

- Source: FloodIMG dataset (Kaggle)
- Total Images: 7,526  
  - Flooded: 7,492  
  - Non-Flooded: 34  

The dataset is highly imbalanced, with approximately 99.55% flooded images and only 0.45% non-flooded images.

### Preprocessing

- Images resized to 224 × 224
- Normalization using ImageNet statistics
- Train / Validation / Test split: 70 / 20 / 10 (stratified)

---

## Key Challenge: Class Imbalance

Due to the extreme imbalance, a model that always predicts "flooded" can achieve very high accuracy (~99.7%) without learning meaningful patterns.

To address this, the project emphasizes:
- ROC-AUC for ranking performance
- Precision, Recall, and F1-score for minority class evaluation
- Confusion matrix analysis

---

## Models

### 1. Baseline CNN

A compact CNN architecture implemented in PyTorch:

- Conv2D (32 filters) → ReLU → MaxPool  
- Conv2D (64 filters) → ReLU → MaxPool  
- Fully Connected Layer (128 units)  
- Output Layer (1 neuron)

Training Configuration:
- Loss Function: BCEWithLogitsLoss
- Optimizer: Adam
- Learning Rate: 0.001
- Batch Size: 32
- Epochs: 5

---

### 2. ResNet-50 (Transfer Learning)

- Pretrained on ImageNet
- Final layer modified for binary classification
- Initial layers frozen, followed by partial fine-tuning

---

## Results

| Model               | Accuracy | ROC-AUC | Minority F1 |
|--------------------|---------|--------|------------|
| Majority Baseline  | 0.997   | 0.50   | 0.00       |
| Baseline CNN       | 0.997   | 0.83   | 0.15       |
| ResNet-50 (TL)     | 0.997   | 0.40   | 0.00       |

### Observations

- Accuracy is misleading due to class imbalance
- The baseline CNN achieves the best performance in terms of ROC-AUC
- ResNet-50 overfits to the majority class and fails to detect minority samples
- Simpler models can outperform complex architectures under extreme imbalance conditions

---

## Technologies Used

- Python
- PyTorch
- Google Colab
- Kaggle API

---

## Challenges

- Handling extreme class imbalance
- Managing corrupted image files
- Kaggle API authentication issues
- Limited GPU resources in Google Colab
- Instability in minority class predictions

---

## Future Work

- Implement focal loss or class-weighted loss functions
- Apply advanced data augmentation and oversampling techniques
- Explore architectures such as EfficientNet and Vision Transformers
- Extend from classification to segmentation (e.g., U-Net)
- Integrate explainability methods such as Grad-CAM
- Develop a real-time deployment pipeline

---

## Dataset Download

The dataset is large (approximately 12 GB) and is not included in this repository.

Download it from the following link:
https://drive.google.com/file/d/1UOqv2C0kBV2t6BC2ZDxGK2OKtsaLy-n5/view?usp=sharing

After downloading, extract the dataset and place it in the following structure:

```
Flood-Detection/
└── dataset/
    ├── flooded/
    └── non-flooded/
```

---

## Conclusion

This project demonstrates that evaluation strategy is critical when working with imbalanced datasets. While multiple models achieve similar accuracy, ROC-AUC and minority class metrics reveal that the baseline CNN provides more reliable performance than a deeper transfer learning model.

---

## Author

Amrutha Gowri Jayasimha Hanumesh  
Master’s Student, Computer Science and Engineering  
Texas A&M University–San Antonio
