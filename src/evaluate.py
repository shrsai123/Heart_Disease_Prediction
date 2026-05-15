import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
 
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
 
 
# ---------------------------------------------------------------------------
# Probability extraction (works for sklearn estimators and imblearn pipelines)
# ---------------------------------------------------------------------------
 
def _get_positive_class_probs(model, X_test):
    """Return P(y=1 | X_test) if available, else a decision-function score
    in [0, 1], else None.
 
    Note: Brier score and calibration curves only make sense for true
    probabilities. We therefore prefer predict_proba and skip calibration
    metrics for models that only expose decision_function.
    """
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X_test)[:, 1], True   # is_probability=True
    if hasattr(model, "decision_function"):
        return model.decision_function(X_test), False    # NOT a probability
    return None, False
 
 
# ---------------------------------------------------------------------------
# sklearn / imblearn-pipeline models
# ---------------------------------------------------------------------------
 
def evaluate_ml_model(model, X_test, y_test, model_name="Model",
                      class_labels=("No CHD", "CHD")):
    y_pred = model.predict(X_test)
    y_score, is_probability = _get_positive_class_probs(model, X_test)
 
    auc_roc = roc_auc_score(y_test, y_score) if y_score is not None else None
    brier = brier_score_loss(y_test, y_score) if is_probability else None
 
    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="binary", zero_division=0)
    recall = recall_score(y_test, y_pred, average="binary", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="binary", zero_division=0)
    mcc = matthews_corrcoef(y_test, y_pred)
 
    print(f"===== {model_name} Evaluation =====")
    print(f"Accuracy:    {acc:.4f}")
    print(f"Precision:   {precision:.4f}")
    print(f"Recall:      {recall:.4f}")
    print(f"F1-score:    {f1:.4f}")
    print(f"MCC:         {mcc:.4f}")
    if auc_roc is not None:
        print(f"AUROC:       {auc_roc:.4f}")
    if brier is not None:
        print(f"Brier score: {brier:.4f}    (lower is better, 0 = perfect)")
 
    print("\nClassification Report:\n",
          classification_report(y_test, y_pred, zero_division=0))
 
    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cm, model_name, list(class_labels))
 
    return {
        "model": model_name,
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "mcc": mcc,
        "auc_roc": auc_roc,
        "brier": brier,
        "y_score": y_score,
        "is_probability": is_probability,
        "y_test": np.asarray(y_test),
    }
 
 
# ---------------------------------------------------------------------------
# PyTorch model
# ---------------------------------------------------------------------------
 
def evaluate_pytorch_model(model, X_test_tensor, y_test_tensor,
                           model_name="Neural Network",
                           class_labels=("No CHD", "CHD")):
    model.eval()
    with torch.no_grad():
        y_score = model(X_test_tensor).detach().cpu().numpy().ravel()
        y_pred_np = (y_score > 0.5).astype(int)
 
    y_test_np = y_test_tensor.detach().cpu().numpy().ravel().astype(int)
 
    acc = accuracy_score(y_test_np, y_pred_np)
    precision = precision_score(y_test_np, y_pred_np, average="binary", zero_division=0)
    recall = recall_score(y_test_np, y_pred_np, average="binary", zero_division=0)
    f1 = f1_score(y_test_np, y_pred_np, average="binary", zero_division=0)
    mcc = matthews_corrcoef(y_test_np, y_pred_np)
    auc_roc = roc_auc_score(y_test_np, y_score)
    brier = brier_score_loss(y_test_np, y_score)
 
    print(f"===== {model_name} Evaluation =====")
    print(f"Accuracy:    {acc:.4f}")
    print(f"Precision:   {precision:.4f}")
    print(f"Recall:      {recall:.4f}")
    print(f"F1-score:    {f1:.4f}")
    print(f"MCC:         {mcc:.4f}")
    print(f"AUROC:       {auc_roc:.4f}")
    print(f"Brier score: {brier:.4f}")
    print("\nClassification Report:\n",
          classification_report(y_test_np, y_pred_np, zero_division=0))
 
    cm = confusion_matrix(y_test_np, y_pred_np)
    plot_confusion_matrix(cm, model_name, list(class_labels))
 
    return {
        "model": model_name,
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "mcc": mcc,
        "auc_roc": auc_roc,
        "brier": brier,
        "y_score": y_score,
        "is_probability": True,
        "y_test": y_test_np,
    }
 
 
# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
 
def plot_confusion_matrix(cm, model_name, class_labels, save_path=None):
    sns.set_style("white")
    fig, ax = plt.subplots(figsize=(8, 8))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="rocket", linewidths=2,
        xticklabels=class_labels, yticklabels=class_labels,
        cbar=False, square=True, ax=ax,
    )
    ax.set_title(f"{model_name} Confusion Matrix",
                 fontsize=18, weight="bold", pad=16)
    ax.set_xlabel("Predicted Label", fontsize=14)
    ax.set_ylabel("Actual Label", fontsize=14)
    ax.tick_params(axis="both", labelsize=12)
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
 
 
def plot_calibration_curves(results, dataset_label, n_bins=10,
                            strategy="quantile", save_path=None):
    """Plot reliability curves for every model that produced calibrated
    probabilities (skips models with only decision_function output).
 
    Parameters
    ----------
    results : list of dicts returned by evaluate_ml_model /
              evaluate_pytorch_model.
    """
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(8, 8))
 
    # Perfect calibration reference
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray",
            label="Perfectly calibrated")
 
    for r in results:
        if not r.get("is_probability"):
            continue
        y_true = r["y_test"]
        y_prob = r["y_score"]
        prob_true, prob_pred = calibration_curve(
            y_true, y_prob, n_bins=n_bins, strategy=strategy
        )
        brier = r.get("brier")
        label = (f"{r['model']} (Brier={brier:.3f})"
                 if brier is not None else r["model"])
        ax.plot(prob_pred, prob_true, marker="o", linewidth=2, label=label)
 
    ax.set_title(f"Calibration Curves — {dataset_label}",
                 fontsize=16, weight="bold", pad=14)
    ax.set_xlabel("Mean Predicted Probability", fontsize=13)
    ax.set_ylabel("Fraction of Positives (Empirical)", fontsize=13)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.legend(loc="upper left", fontsize=10)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()