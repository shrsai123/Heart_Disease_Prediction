import torch

from src.data_loader import load_data, clean_data, load_uci_heart_data
from src.preprocessing import (
    split_data,
    scale_train_test,
    resample_train_only,
    prepare_tensors,
)
from src.model import NeuralNetwork
from src.train import train_model, train_pytorch_model
from src.evaluate import (
    evaluate_ml_model,
    evaluate_pytorch_model,
    plot_calibration_curves,
)
from src.hyperparameter_tuning import (
    logistic_regression_hyperparam,
    knn_hyperparam,
    xgboost_hyperparam,
    random_forest_hyperparam,
)
from src.visualize import plot_classifier_comparison, plot_roc_curves_with_nn
from src.interpret import shap_feature_importance


def process_dataset(dataset_name):
    dataset_paths = {
        "framingham": "raw_data/raw_Framingham/framingham.csv",
        "uci_heart": "raw_data/UCI/heart_disease/processed.cleveland.data",
    }

    print("\n" + "=" * 90)
    print(f"Processing Dataset: {dataset_name.upper()}")
    print("=" * 90 + "\n")

    # -------- Load and clean --------
    if "uci_heart" in dataset_name.lower():
        df_clean = load_uci_heart_data(dataset_paths["uci_heart"])
        dataset_type = "uci_heart"
    elif "framingham" in dataset_name.lower():
        df = load_data(dataset_paths["framingham"])
        df_clean = clean_data(df)
        dataset_type = "framingham"
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")

    if dataset_type == "framingham":
        X = df_clean.drop(columns=["TenYearCHD"])
        y = df_clean["TenYearCHD"]
        class_labels = ["No CHD", "CHD"]
        dataset_label = "Framingham Dataset"
    else:
        X = df_clean.drop(columns=["num", "source"])
        y = df_clean["num"].apply(lambda x: 1 if x > 0 else 0)
        class_labels = ["No HD", "HD"]
        dataset_label = "Cleveland Subset of the UCI Heart Disease Dataset"

    # -------- Split FIRST (no SMOTE, no scaling yet) --------
    X_train, X_test, y_train, y_test = split_data(
        X, y, test_size=0.2, random_state=42, stratify=True
    )
    print(f"Train shape: {X_train.shape} | Test shape: {X_test.shape}")
    print(f"Train positive rate: {y_train.mean():.3f} | "
          f"Test positive rate: {y_test.mean():.3f}\n")

    # -------- Hyperparameter tuning (leakage-safe) --------
    # Each function returns (best_params_dict, fitted_pipeline). The
    # pipeline embeds Scaler -> SMOTE -> Under -> Classifier; we pass the
    # RAW X_train/y_train into it.
    rf_params, rf_pipe = random_forest_hyperparam(X_train, y_train)
    lr_params, lr_pipe = logistic_regression_hyperparam(X_train, y_train)
    knn_params, knn_pipe = knn_hyperparam(X_train, y_train)
    xgb_params, xgb_pipe = xgboost_hyperparam(X_train, y_train)

    models = {
        "Random Forest": (rf_params, rf_pipe),
        "Logistic Regression": (lr_params, lr_pipe),
        "KNN": (knn_params, knn_pipe),
        "XGBoost": (xgb_params, xgb_pipe),
    }

    trained_pipelines_for_roc = {}
    model_results = []
    best_xgb_pipeline = None

    print("\n========== Evaluating Traditional ML Models on Raw Test Set ==========\n")
    for name, (params, pipe) in models.items():
        print(f"\n--- {name} ---")
        print(f"Best params: {params}")
        # The pipeline is already fitted by GridSearchCV(refit=True); just evaluate.
        results = evaluate_ml_model(
            pipe, X_test, y_test,
            model_name=name, class_labels=class_labels,
        )
        model_results.append(results)
        trained_pipelines_for_roc[name] = pipe

        if name == "XGBoost":
            best_xgb_pipeline = pipe

    # -------- Neural network (manual SMOTE + scaling, train only) --------
    print("\n========== Training and Evaluating Neural Network ==========\n")

    # Step 1: scale (fit on X_train only)
    X_train_scaled, X_test_scaled, _ = scale_train_test(X_train, X_test)

    # Step 2: resample TRAINING set only (test is never resampled)
    X_train_bal, y_train_bal = resample_train_only(
        X_train_scaled, y_train, plot=True,
        labels=class_labels,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_tr_t, X_te_t, y_tr_t, y_te_t = prepare_tensors(
        X_train_bal, X_test_scaled, y_train_bal, y_test, device
    )

    nn_model = NeuralNetwork(X_train_bal.shape[1]).to(device)
    nn_model = train_pytorch_model(nn_model, X_tr_t, y_tr_t)

    nn_results = evaluate_pytorch_model(
        nn_model, X_te_t, y_te_t,
        model_name="Neural Network", class_labels=class_labels,
    )
    model_results.append(nn_results)

    # -------- Comparison plots --------
    print("\n========== Classifier Comparison ==========\n")
    plot_classifier_comparison(
        model_results,
        save_path=f"{dataset_name}_classifier_comparison.png",
    )

    print("\n========== ROC Curves ==========\n")
    plot_roc_curves_with_nn(
        models=trained_pipelines_for_roc,
        nn_model=nn_model,
        X_test=X_test,                 # raw; pipelines scale internally
        y_test=y_test,
        X_test_tensor=X_te_t,          # scaled tensor for the NN
        dataset_name=dataset_label,
        save_path=f"{dataset_name}_roc_curve.png",
    )

    # -------- Calibration curves (NEW — addresses reviewer comment) --------
    print("\n========== Calibration Curves ==========\n")
    plot_calibration_curves(
        model_results,
        dataset_label=dataset_label,
        n_bins=10,
        strategy="quantile",
        save_path=f"{dataset_name}_calibration_curve.png",
    )

    # -------- SHAP --------
    if best_xgb_pipeline is not None:
        print("\n========== SHAP Feature Importance (XGBoost) ==========\n")
        # SHAP needs the bare booster + scaled features, not the whole
        # pipeline. Pull them out of the fitted pipeline.
        xgb_clf = best_xgb_pipeline.named_steps["clf"]
        scaler = best_xgb_pipeline.named_steps["scaler"]
        import pandas as pd
        X_test_for_shap = pd.DataFrame(
            scaler.transform(X_test), columns=X_test.columns, index=X_test.index
        )
        shap_feature_importance(
    xgb_clf,
    X_test_for_shap,
    save_prefix=f"{dataset_name}_xgboost"
)


def main():
    for dataset in ["framingham", "uci_heart"]:
        process_dataset(dataset)


if __name__ == "__main__":
    main()