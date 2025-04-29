import torch
from src.data_loader import load_data, clean_data, load_uci_heart_data
from src.preprocessing import balance_data, split_and_scale, prepare_tensors
from src.model import NeuralNetwork
from src.train import train_model, train_pytorch_model
from src.evaluate import evaluate_ml_model, evaluate_pytorch_model
from src.hyperparameter_tuning import (
    logistic_regression_hyperparam, knn_hyperparam, xgboost_hyperparam,random_forest_hyperparam
)
from src.visualize import plot_classifier_comparison
from src.interpret import shap_feature_importance

def process_dataset(dataset_name):
    # Dataset paths
    dataset_paths = {
        'framingham': 'raw_data/raw_Framingham/framingham.csv',
        'uci_heart': 'raw_data/UCI/heart_disease/processed.cleveland.data'
    }

    print("\n" + "="*90)
    print(f"Processing Dataset: {dataset_name.upper()}")
    print("="*90 + "\n")

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
    elif dataset_type=='uci_heart': 
        X = df_clean.drop(columns=['num', 'source'])
        y = df_clean['num']
        y = y.apply(lambda x: 1 if x > 0 else 0)  # Binarize target
        class_labels = ["No HD", "HD"]

    # Balance, Split, and Scale
    X_bal, y_bal = balance_data(X, y)
    X_train, X_test, y_train, y_test = split_and_scale(X_bal, y_bal)

    # ML Models: Random Forest, Logistic Regression, KNN, XGBoost
    models = {
        'Random Forest': random_forest_hyperparam(X_train,y_train),
        'Logistic Regression': logistic_regression_hyperparam(X_train, y_train),
        'KNN': knn_hyperparam(X_train, y_train),
        'XGBoost': xgboost_hyperparam(X_train, y_train)
    }
    model_scores=[]
    best_xgb_model = None
    print("\n================= Training and Evaluating Traditional ML Models =================\n")
    
    for model_name, (params, model) in models.items():
        print(f"\nTraining {model_name} with Best Params: {params}")
        model = train_model(model, X_train, y_train)
        results=evaluate_ml_model(model, X_test, y_test, model_name=model_name, class_labels=class_labels)
        model_scores.append(results)
        if model_name == 'XGBoost':
            best_xgb_model = model  # Save best XGBoost model

    # Neural Network Model
    print("\n================= Training and Evaluating Neural Network Model =================\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor = prepare_tensors(X_train, X_test, y_train, y_test, device)

    nn_model = NeuralNetwork(X_train.shape[1]).to(device)
    nn_model = train_pytorch_model(nn_model, X_train_tensor, y_train_tensor)
    results=evaluate_pytorch_model(nn_model, X_test_tensor, y_test_tensor, model_name="Neural Network", class_labels=class_labels)
    model_scores.append(results)
    # === Plotting Metrics Comparison ===
    print("\n================= Plotting Classifier Comparison =================\n")
    plot_classifier_comparison(model_scores)

    # === SHAP for XGBoost ===
    if best_xgb_model:
        print("\n================= SHAP Feature Importance (XGBoost) =================\n")
        shap_feature_importance(best_xgb_model, X_test)

def main():
    datasets = ['framingham', 'uci_heart']
    for dataset in datasets:
        process_dataset(dataset)

if __name__ == "__main__":
    main()
