import torch
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report,
    roc_auc_score
)


def evaluate_ml_model(model, X_test, y_test, model_name="Model", class_labels=["No CHD", "CHD"]):
    y_pred = model.predict(X_test)

    # probability / score for AUROC
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)[:, 1]
        auc_roc = roc_auc_score(y_test, y_score)
    elif hasattr(model, "decision_function"):
        y_score = model.decision_function(X_test)
        auc_roc = roc_auc_score(y_test, y_score)
    else:
        y_score = None
        auc_roc = None

    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="binary", zero_division=0)
    recall = recall_score(y_test, y_pred, average="binary", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="binary", zero_division=0)
    mcc = matthews_corrcoef(y_test, y_pred)

    print(f"===== {model_name} Evaluation =====")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print(f"MCC:       {mcc:.4f}")
    if auc_roc is not None:
        print(f"AUROC:     {auc_roc:.4f}")

    print("\nClassification Report:\n", classification_report(y_test, y_pred, zero_division=0))

    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cm, model_name, class_labels)

    return {
        "model": model_name,
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "mcc": mcc,
        "auc_roc": auc_roc
    }


def evaluate_pytorch_model(model, X_test_tensor, y_test_tensor, model_name="Neural Network", class_labels=["No CHD", "CHD"]):
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

    print(f"===== {model_name} Evaluation =====")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print(f"MCC:       {mcc:.4f}")
    print(f"AUROC:     {auc_roc:.4f}")
    print("\nClassification Report:\n", classification_report(y_test_np, y_pred_np, zero_division=0))

    cm = confusion_matrix(y_test_np, y_pred_np)
    plot_confusion_matrix(cm, model_name, class_labels)

    return {
        "model": model_name,
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "mcc": mcc,
        "auc_roc": auc_roc
    }


def plot_confusion_matrix(cm, model_name, class_labels, save_path=None):
    sns.set_style("white")
    fig, ax = plt.subplots(figsize=(8, 8))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="rocket",
        linewidths=2,
        xticklabels=class_labels,
        yticklabels=class_labels,
        cbar=False,
        square=True,
        ax=ax
    )

    ax.set_title(f"{model_name} Confusion Matrix", fontsize=18, weight="bold", pad=16)
    ax.set_xlabel("Predicted Label", fontsize=14)
    ax.set_ylabel("Actual Label", fontsize=14)
    ax.tick_params(axis="both", labelsize=12)

    for spine in ax.spines.values():
        spine.set_linewidth(1.5)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()