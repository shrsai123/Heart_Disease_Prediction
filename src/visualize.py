import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_classifier_comparison(results):
    """Plot Accuracy, Average Precision, and Average Recall for all classifiers."""
    all_scores = pd.DataFrame(results)
    if 'model' in all_scores.columns:
        all_scores.rename(columns={'model': 'Classifier'}, inplace=True)
    all_scores = pd.melt(all_scores, id_vars='Classifier')
    all_scores.rename(columns={'variable': 'Metric'}, inplace=True)

    with sns.plotting_context("notebook", font_scale=1.5, rc={"legend.fontsize":12, "legend.title_fontsize":14}), sns.axes_style("ticks"):
        fig, axs = plt.subplots(figsize=(12,8))
        sns.barplot(data=all_scores, x='Classifier', y='value', hue='Metric', ax=axs)
        axs.set_xticklabels(axs.get_xticklabels(), rotation=30, horizontalalignment='right', rotation_mode='anchor')
        axs.set_ylabel('Metric Score')
        axs.set_title('Comparing Scoring Metrics for Classifiers')
        axs.set_ylim(0.40, 1.05)
        for spine in ['left', 'top', 'right', 'bottom']:
            axs.spines[spine].set_linewidth(3)
        plt.legend(bbox_to_anchor=[1.01, 0.5], loc='center left')
        plt.tight_layout()
        plt.show()