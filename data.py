"""Data loading and exploratory analysis."""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from config import DATA_PATH, OUTPUT_DIR


def load_data(path=DATA_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find {path}. Download creditcard.csv from "
            "https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud "
            "and place it in the project root."
        )
    df = pd.read_csv(path)
    print(f"Loaded {len(df):,} rows, {df.shape[1]} columns")
    return df


def run_eda(df: pd.DataFrame) -> None:
    fraud_rate = df["Class"].mean()
    print(f"Fraud rate: {fraud_rate:.4%} ({df['Class'].sum():,} of {len(df):,} transactions)")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    df["Class"].value_counts().plot(kind="bar", ax=axes[0], color=["#4C72B0", "#C44E52"])
    axes[0].set_title("Class Distribution (0=Legit, 1=Fraud)")
    axes[0].set_ylabel("Count")

    sns.histplot(df["Amount"], bins=50, ax=axes[1], log_scale=(False, True))
    axes[1].set_title("Transaction Amount Distribution")

    plt.tight_layout()
    out_path = OUTPUT_DIR / "eda_class_and_amount.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved EDA plot -> {out_path}")
