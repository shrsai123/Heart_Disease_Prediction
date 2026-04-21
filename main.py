import torch
from src.data_loader import load_data, clean_data, load_uci_heart_data
from src.preprocessing import balance_data, split_and_scale, prepare_tensors
from src.model import NeuralNetwork
from src.train import train_model, train_pytorch_model
from src.evaluate import evaluate_ml_model, evaluate_pytorch_model
from src.hyperparameter_tuning import (
    logistic_regression_hyperparam, knn_hyperparam, xgboost_hyperparam, random_forest_hyperparam
)
from src.visualize import plot_classifier_comparison, plot_roc_curves_with_nn
from src.interpret import shap_feature_importance

def process_dataset(dataset_name):
    # Dataset paths
    dataset_paths = {
        'framingham': 'raw_data/raw_Framingham/framingham.csv',
        'uci_heart': 'raw_data/UCI/heart_disease/processed.cleveland.data'
    }

    print("\n" + "=" * 90)
    print(f"Processing Dataset: {dataset_name.upper()}")
    print("=" * 90 + "\n")

    # Load and clean dataset
    if 'uci_heart' in dataset_name.lower():
        df_clean = load_uci_heart_data(dataset_paths['uci_heart'])
        dataset_type = 'uci_heart'
    elif 'framingham' in dataset_name.lower():
        df = load_data(dataset_paths['framingham'])
        df_clean = clean_data(df)
        dataset_type = 'framingham'
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")

    # Feature/target selection
    if dataset_type == 'framingham':
        X = df_clean.drop(columns=['TenYearCHD'])
        y = df_clean['TenYearCHD']
        class_labels = ["No CHD", "CHD"]
        dataset_label = "Framingham Dataset"
    else:
        X = df_clean.drop(columns=['num', 'source'])
        y = df_clean['num'].apply(lambda x: 1 if x > 0 else 0)
        class_labels = ["No HD", "HD"]
        dataset_label = "Cleveland Subset of the UCI Heart Disease Dataset"

    # Balance, Split, and Scale
    X_bal, y_bal = balance_data(X, y)
    X_train, X_test, y_train, y_test = split_and_scale(X_bal, y_bal)

    # Hyperparameter tuning
    rf_params, rf_model = random_forest_hyperparam(X_train, y_train)
    lr_params, lr_model = logistic_regression_hyperparam(X_train, y_train)
    knn_params, knn_model = knn_hyperparam(X_train, y_train)
    xgb_params, xgb_model = xgboost_hyperparam(X_train, y_train)

    models = {
        'Random Forest': (rf_params, rf_model),
        'Logistic Regression': (lr_params, lr_model),
        'KNN': (knn_params, knn_model),
        'XGBoost': (xgb_params, xgb_model)
    }

    trained_models_for_roc = {}
    model_scores = []
    best_xgb_model = None

    print("\n================= Training and Evaluating Traditional ML Models =================\n")

    for model_name, (params, model) in models.items():
        print(f"\nTraining {model_name} with Best Params: {params}")
        model = train_model(model, X_train, y_train)

        results = evaluate_ml_model(
            model,
            X_test,
            y_test,
            model_name=model_name,
            class_labels=class_labels
        )
        model_scores.append(results)
        trained_models_for_roc[model_name] = model

        if model_name == 'XGBoost':
            best_xgb_model = model

    print("\n================= Training and Evaluating Neural Network Model =================\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor = prepare_tensors(
        X_train, X_test, y_train, y_test, device
    )

    nn_model = NeuralNetwork(X_train.shape[1]).to(device)
    nn_model = train_pytorch_model(nn_model, X_train_tensor, y_train_tensor)

    nn_results = evaluate_pytorch_model(
        nn_model,
        X_test_tensor,
        y_test_tensor,
        model_name="Neural Network",
        class_labels=class_labels
    )
    model_scores.append(nn_results)

    print("\n================= Plotting Classifier Comparison =================\n")
    plot_classifier_comparison(
        model_scores,
        save_path=f"{dataset_name}_classifier_comparison.png"
    )

    print("\n================= Plotting ROC Curves =================\n")
    plot_roc_curves_with_nn(
        models=trained_models_for_roc,
        nn_model=nn_model,
        X_test=X_test,
        y_test=y_test,
        X_test_tensor=X_test_tensor,
        dataset_name=dataset_label,
        save_path=f"{dataset_name}_roc_curve.png"
    )

    if best_xgb_model:
        print("\n================= SHAP Feature Importance (XGBoost) =================\n")
        shap_feature_importance(best_xgb_model, X_test)
def main():
    datasets = ['framingham', 'uci_heart']
    for dataset in datasets:
        process_dataset(dataset)

if __name__ == "__main__":
    main()
