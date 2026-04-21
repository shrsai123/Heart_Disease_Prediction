import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import roc_curve, roc_auc_score


MODEL_COLORS = {
    "Logistic Regression": "#1f77b4",
    "Random Forest": "#ff7f0e",
    "XGBoost": "#2ca02c",
    "KNN": "#d62728",
    "Neural Network": "#9467bd"
}


def plot_classifier_comparison(results, save_path=None):
    """
    Plots publication-quality bar chart comparing model metrics.
    Expects each result dict to include:
    model, accuracy, precision, recall, f1_score, mcc, auc_roc
    """
    all_scores = pd.DataFrame(results)

    if "model" in all_scores.columns:
        all_scores = all_scores.rename(columns={"model": "Classifier"})

    metric_cols = [c for c in ["accuracy", "precision", "recall", "f1_score", "mcc", "auc_roc"] if c in all_scores.columns]
    all_scores = all_scores[["Classifier"] + metric_cols]

    melted = pd.melt(
        all_scores,
        id_vars="Classifier",
        var_name="Metric",
        value_name="Score"
    )

    sns.set_style("whitegrid")
    plt.figure(figsize=(14, 8))

    ax = sns.barplot(
        data=melted,
        x="Classifier",
        y="Score",
        hue="Metric"
    )

    ax.set_title("Classifier Performance Comparison", fontsize=18, weight="bold", pad=16)
    ax.set_xlabel("Classifier", fontsize=14)
    ax.set_ylabel("Metric Score", fontsize=14)
    ax.set_ylim(0.40, 1.05)

    ax.tick_params(axis="x", labelsize=12, rotation=25)
    ax.tick_params(axis="y", labelsize=12)

    legend = ax.legend(
        title="Metric",
        bbox_to_anchor=(1.02, 0.5),
        loc="center left",
        frameon=True,
        fontsize=11,
        title_fontsize=12
    )
    legend.get_frame().set_linewidth(1.2)

    for spine in ax.spines.values():
        spine.set_linewidth(1.5)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def _get_model_scores(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        return model.decision_function(X)
    raise ValueError(f"Model {type(model).__name__} does not provide probability/score output for ROC.")


def _get_nn_scores(nn_model, X_tensor):
    nn_model.eval()
    with torch.no_grad():
        y_scores = nn_model(X_tensor).detach().cpu().numpy().ravel()
    return y_scores


def plot_roc_curves_with_nn(models, nn_model, X_test, y_test, X_test_tensor, dataset_name, save_path=None):
    """
    Plots high-resolution ROC curves with AUROC values in legend.

    models: dict of trained sklearn models
    nn_model: trained pytorch model
    X_test, y_test: sklearn-style test data
    X_test_tensor: pytorch test tensor
    dataset_name: str
    save_path: output file path
    """
    sns.set_style("white")
    plt.figure(figsize=(10, 8))

    # sklearn models
    for model_name, model in models.items():
        y_scores = _get_model_scores(model, X_test)
        fpr, tpr, _ = roc_curve(y_test, y_scores)
        auc_val = roc_auc_score(y_test, y_scores)

        plt.plot(
            fpr,
            tpr,
            linewidth=2.8,
            color=MODEL_COLORS.get(model_name, None),
            label=f"{model_name} (AUROC = {auc_val:.3f})"
        )

    # neural network
    y_scores_nn = _get_nn_scores(nn_model, X_test_tensor)
    fpr_nn, tpr_nn, _ = roc_curve(y_test, y_scores_nn)
    auc_nn = roc_auc_score(y_test, y_scores_nn)

    plt.plot(
        fpr_nn,
        tpr_nn,
        linewidth=2.8,
        color=MODEL_COLORS["Neural Network"],
        label=f"Neural Network (AUROC = {auc_nn:.3f})"
    )

    # chance line
    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        linewidth=1.8,
        color="gray",
        label="Chance"
    )

    plt.title(f"ROC Curves for {dataset_name}", fontsize=18, weight="bold", pad=16)
    plt.xlabel("False Positive Rate", fontsize=14)
    plt.ylabel("True Positive Rate", fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlim(0, 1)
    plt.ylim(0, 1.02)

    legend = plt.legend(
        loc="lower right",
        frameon=True,
        fontsize=11,
        title="Models",
        title_fontsize=12
    )
    legend.get_frame().set_linewidth(1.2)

    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()