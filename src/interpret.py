import shap
import matplotlib.pyplot as plt

def shap_feature_importance(xgboost_model, X_test, save_prefix=None):
    explainer = shap.TreeExplainer(xgboost_model)
    shap_values = explainer.shap_values(X_test)

    # 1) Bar plot
    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
    plt.title("SHAP Feature Importance (Bar Plot)", fontsize=14)
    plt.tight_layout()
    if save_prefix:
        plt.gcf().savefig(f"{save_prefix}_shap_bar.png", dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()

    # 2) Beeswarm plot
    shap.summary_plot(shap_values, X_test, show=False)
    plt.title("SHAP Summary Plot (Beeswarm)", fontsize=14)
    plt.tight_layout()
    if save_prefix:
        plt.gcf().savefig(f"{save_prefix}_shap_beeswarm.png", dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()