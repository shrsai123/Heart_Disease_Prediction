from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from collections import Counter
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import torch

def balance_data(X, y):
    """Apply SMOTE and undersampling to balance the dataset."""
    num_before = dict(Counter(y))
    over = SMOTE(sampling_strategy='auto')
    under = RandomUnderSampler(sampling_strategy='auto')
    steps = [("o", over), ("u", under)]
    pipeline = Pipeline(steps=steps)
    X_smote, y_smote = pipeline.fit_resample(X, y)
    num_after = dict(Counter(y_smote))
    labels = ["Negative Cases", "Positive Cases"]
    plt.figure(figsize=(15,6))

    plt.subplot(1,2,1)
    sns.barplot(x=labels, y=list(num_before.values()))
    plt.title("Class Distribution Before Balancing", fontsize=14)
    plt.ylabel("Count")
    plt.xlabel("Classes")

    plt.subplot(1,2,2)
    sns.barplot(x=labels, y=list(num_after.values()))
    plt.title("Class Distribution After Balancing", fontsize=14)
    plt.ylabel("Count")
    plt.xlabel("Classes")

    plt.tight_layout()
    plt.show()
    return X_smote, y_smote

def split_and_scale(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    X_train = pd.DataFrame(X_train_scaled, columns=X.columns)
    X_test = pd.DataFrame(X_test_scaled, columns=X.columns)
    return X_train, X_test, y_train, y_test

def prepare_tensors(X_train, X_test, y_train, y_test, device):
    """Convert to PyTorch tensors."""
    X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32).to(device)
    X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1,1).to(device)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1,1).to(device)
    return X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor