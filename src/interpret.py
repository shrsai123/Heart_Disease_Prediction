import shap
import matplotlib.pyplot as plt

def shap_feature_importance(xgboost_model, X_test):
    """Generate SHAP feature importance plot for XGBoost model."""
    explainer = shap.TreeExplainer(xgboost_model)
    shap_values = explainer.shap_values(X_test)

    shap.summary_plot(shap_values, X_test, plot_type="bar")