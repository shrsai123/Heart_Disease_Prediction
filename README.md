# Foundations of AI: Final Project
Heart Disease Prediction

Group: Amanda Pang, Shreyas Raman, Devansh Thakkar


## Overview
This project builds a machine learning pipeline that predicts the likelihood of heart disease occurrence using multiple models, including Random Forest, Logistic Regression, K-Nearest Neighbors (KNN), XGBoost, and a Neural Network (PyTorch).  
It features automated class balancing (SMOTE), model interpretability (SHAP feature importance), and professional performance visualizations.  
Both the Framingham Heart Study dataset and UCI Cleveland Heart Disease dataset are supported.

---

## Features

### Heart Disease Prediction
- Supports prediction using four classifiers:
  - Random Forest
  - Logistic Regression
  - K-Nearest Neighbors (KNN)
  - XGBoost Classifier
  - Neural Network (PyTorch)
- Hyperparameter tuning for each model using GridSearchCV
- Balances datasets automatically using SMOTE
- Evaluates models using Accuracy, Precision, Recall metrics
- Visualizes results with confusion matrices and metric comparison plots

### SHAP-Based Model Interpretability
- Uses SHAP (SHapley Additive exPlanations) to explain feature importance
- Generates feature importance bar plots for XGBoost model

### Professional Visualizations
- Class distribution plots before and after SMOTE
- Confusion matrices for each model
- Classifier performance comparison bar plots
- SHAP summary plots for feature importance

---

## Document Overview

| File/Folder | Purpose |
|:---|:---|
| `main.py` | Main application driver file; trains, evaluates, and visualizes models |
| `src/data_loader.py` | Loads and cleans Framingham and UCI Heart Disease datasets |
| `src/preprocessing.py` | Balances datasets (SMOTE), splits, and scales data |
| `src/model.py` | Defines PyTorch neural network |
| `src/train.py` | Contains training loops for ML and Neural Network models |
| `src/evaluate.py` | Evaluation utilities: metrics, confusion matrices |
| `src/hyperparameter_tuning.py` | Hyperparameter tuning using GridSearchCV |
| `src/visualize.py` | Plots classifier comparison metrics |
| `src/interpret.py` | Generates SHAP feature importance plots |
| `raw_data/` | Folder containing `framingham.csv` and `processed.cleveland.data` |
| `requirements.txt` | Python dependencies |
---


## Getting Started

### Prerequisites
Ensure you have the following installed to run the pipeline:

- Python 3.8+
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- imbalanced-learn
- xgboost
- torch
- shap

## Setup Instructions

1. Clone the Repository
   ```bash
   git clone https://github.com/shrsai123/heart-disease-prediction.git
   cd Heart_Disease_Prediction

2. Install Dependencies
   ```bash
   pip install -r requirements.txt

3. Run the main pipeline
   ```bash
   python main.py

---

### Model and Results
1) Random Forest, Logistic Regression, KNN, XGBoost, and Neural Network models are trained and evaluated.
2) Confusion Matrices, Class Distribution plots, and SHAP Feature Importance plots are automatically generated.
3) Results are visualized via Matplotlib and Seaborn.
4) Accuracy, Precision, and Recall scores for all models are compared via a unified bar plot.

### Sample Visualizations:

1) Classifier Comparison Bar Chart
2) SHAP Feature Importance (for XGBoost)
3) Class Distribution (Before vs After Balancing)

### Acknowledgements

This project uses the following libraries and resources:

[Scikit-learn](https://scikit-learn.org/stable/) for machine learning models

[PyTorch](https://pytorch.org/tutorials/beginner/basics/intro.html) for building Neural Networks

[XGBoost](https://xgboost.readthedocs.io/en/release_3.0.0/get_started.html) for gradient boosting models

[SHAP](https://shap.readthedocs.io/en/latest/index.html) for model interpretation


### Resources

[Framingham Heart Study](https://www.kaggle.com/datasets/sciencely/framingham-heart-study)

[UCI Machine Learning Repository - Heart Disease Dataset](https://archive.ics.uci.edu/ml/datasets/heart+Disease)

