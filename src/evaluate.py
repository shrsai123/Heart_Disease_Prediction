import torch
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns


def evaluate_ml_model(model, X_test, y_test, model_name="Model", class_labels=["No CHD", "CHD"]):
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted")
    recall = recall_score(y_test, y_pred, average="weighted")

    print(f"===== {model_name} Evaluation =====")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    # Confusion Matrix Plot
    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cm, model_name, class_labels)

    return {
        "model": model_name,
        "accuracy": acc,
        "precision": precision,
        "recall": recall
    }


def evaluate_pytorch_model(model, X_test_tensor, y_test_tensor, model_name="Neural Network", class_labels=["No CHD", "CHD"]):
    model.eval()
    with torch.no_grad():
        y_pred_test = model(X_test_tensor)
        y_pred_labels = (y_pred_test > 0.5).float()

    y_pred_np = y_pred_labels.cpu().numpy()
    y_test_np = y_test_tensor.cpu().numpy()

    acc = accuracy_score(y_test_np, y_pred_np)
    precision = precision_score(y_test_np, y_pred_np,average="weighted")
    recall = recall_score(y_test_np, y_pred_np,average="weighted")

    print(f"===== {model_name} Evaluation =====")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print("\nClassification Report:\n", classification_report(y_test_np, y_pred_np))

    # Confusion Matrix Plot
    cm = confusion_matrix(y_test_np, y_pred_np)
    plot_confusion_matrix(cm, model_name, class_labels)

    return {
        "model": model_name,
        "accuracy": acc,
        "precision": precision,
        "recall": recall
    }


def plot_confusion_matrix(cm, model_name, class_labels):
    with sns.plotting_context("notebook", font_scale=1.5), sns.axes_style("white"):
        fig, ax = plt.subplots(figsize=(8,8))

        sns.heatmap(cm, annot=True, fmt="d", cmap='rocket', linewidths=2,
                    xticklabels=class_labels, yticklabels=class_labels, ax=ax)

        ax.set_title(f'{model_name} Confusion Matrix', fontsize=16, weight='bold', pad=20)
        ax.set_xlabel(f"Predicted {class_labels[1]}", fontsize=14)
        ax.set_ylabel(f"Actual {class_labels[1]}", fontsize=14)
        ax.axhline(y=0, color='black', linewidth=3)
        ax.axhline(y=cm.shape[0], color='black', linewidth=3)
        ax.axvline(x=0, color='black', linewidth=3)
        ax.axvline(x=cm.shape[1], color='black', linewidth=3)

        plt.tight_layout()
        plt.show()