import pandas as pd

def load_data(filepath):
    df = pd.read_csv(filepath)
    return df

def clean_data(df):
    full_df = df.dropna()
    full_df = full_df.drop_duplicates()
    return full_df

def load_uci_heart_data(filepath, save_cleaned=True):
    columns = [
        "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
        "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num"
    ]
    print(f"\nLoading UCI Heart Disease Dataset: {filepath}")

    df = pd.read_csv(filepath, names=columns, na_values="?")
    df["source"] = "cleveland"

    print(f"Original shape: {df.shape}")
    df_clean = clean_data(df)

    return df_clean