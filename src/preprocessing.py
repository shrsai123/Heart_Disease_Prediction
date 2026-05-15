from collections import Counter
 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
 
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
 
 
# ---------------------------------------------------------------------------
# Splitting and scaling
# ---------------------------------------------------------------------------
 
def split_data(X, y, test_size=0.2, random_state=42, stratify=True):
    """Stratified train/test split. MUST be called before any resampling."""
    stratify_arg = y if stratify else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_arg,
    )
    return X_train, X_test, y_train, y_test
 
 
def scale_train_test(X_train, X_test):
    """Fit StandardScaler on X_train only, then transform X_test.
 
    Returns DataFrames so column names are preserved for downstream SHAP.
    Used only for the PyTorch model, which is trained outside the
    imblearn Pipeline. Pipeline-based models embed their own scaler.
    """
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index,
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index,
    )
    return X_train_scaled, X_test_scaled, scaler
 
 
# ---------------------------------------------------------------------------
# Resampling — train set only
# ---------------------------------------------------------------------------
 
def resample_train_only(X_train, y_train, random_state=42, plot=False,
                        labels=("Negative Cases", "Positive Cases")):
    """Apply SMOTE + RandomUnderSampler to the TRAINING set only.
 
    Use this for the PyTorch neural network, which does not live inside
    a sklearn/imblearn pipeline. The test set is never resampled.
    """
    num_before = dict(Counter(y_train))
 
    over = SMOTE(sampling_strategy="auto", random_state=random_state)
    under = RandomUnderSampler(sampling_strategy="auto", random_state=random_state)
    pipeline = ImbPipeline(steps=[("o", over), ("u", under)])
    X_res, y_res = pipeline.fit_resample(X_train, y_train)
 
    num_after = dict(Counter(y_res))
 
    if plot:
        plt.figure(figsize=(15, 6))
        plt.subplot(1, 2, 1)
        sns.barplot(x=list(labels), y=list(num_before.values()))
        plt.title("Training Set Before Balancing", fontsize=14)
        plt.ylabel("Count"); plt.xlabel("Classes")
 
        plt.subplot(1, 2, 2)
        sns.barplot(x=list(labels), y=list(num_after.values()))
        plt.title("Training Set After Balancing", fontsize=14)
        plt.ylabel("Count"); plt.xlabel("Classes")
        plt.tight_layout()
        plt.show()
 
    return X_res, y_res
 
 
# ---------------------------------------------------------------------------
# Leakage-safe pipeline factory for sklearn estimators
# ---------------------------------------------------------------------------
 
def make_resampling_pipeline(estimator, random_state=42):
    """Build an imblearn Pipeline that is safe to pass to GridSearchCV.
 
    Order:
        StandardScaler -> SMOTE -> RandomUnderSampler -> estimator
 
    Scaling is performed first so that SMOTE's nearest-neighbour search
    operates in a standardised feature space (otherwise large-magnitude
    features such as cholesterol dominate the distance metric).
 
    Crucially, when this pipeline is passed to GridSearchCV(cv=k):
      * Each fold's TRAINING portion is scaled and resampled.
      * Each fold's VALIDATION portion is only scaled (imblearn skips
        fit_resample at predict-time by design).
      * The test set passed to .predict() afterwards is also only scaled,
        never resampled.
    """
    return ImbPipeline(steps=[
        ("scaler", StandardScaler()),
        ("smote", SMOTE(sampling_strategy="auto", random_state=random_state)),
        ("under", RandomUnderSampler(sampling_strategy="auto",
                                     random_state=random_state)),
        ("clf", estimator),
    ])
 
 
# ---------------------------------------------------------------------------
# PyTorch tensor conversion
# ---------------------------------------------------------------------------
 
def prepare_tensors(X_train, X_test, y_train, y_test, device):
    """Convert features/labels to torch tensors on `device`.
 
    Accepts DataFrames, ndarrays, or Series.
    """
    def _to_array(x):
        return x.values if hasattr(x, "values") else np.asarray(x)
 
    X_train_tensor = torch.tensor(_to_array(X_train), dtype=torch.float32).to(device)
    X_test_tensor = torch.tensor(_to_array(X_test), dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(_to_array(y_train), dtype=torch.float32).view(-1, 1).to(device)
    y_test_tensor = torch.tensor(_to_array(y_test), dtype=torch.float32).view(-1, 1).to(device)
    return X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor