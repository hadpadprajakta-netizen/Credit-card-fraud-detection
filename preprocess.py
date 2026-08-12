import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from config import RANDOM_STATE


def scale_amount_and_time(X: pd.DataFrame, scaler: StandardScaler = None):
    
    X = X.copy()
    if scaler is None:
        scaler = StandardScaler()
        X[["Amount", "Time"]] = scaler.fit_transform(X[["Amount", "Time"]])
    else:
        X[["Amount", "Time"]] = scaler.transform(X[["Amount", "Time"]])
    return X, scaler


def preprocess_and_split(df: pd.DataFrame):
    X = df.drop(columns=["Class"])
    y = df["Class"]

    X, scaler = scale_amount_and_time(X)

    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    print(
        f"Train: {len(X_train):,} rows ({y_train.mean():.4%} fraud) | "
        f"Test: {len(X_test):,} rows ({y_test.mean():.4%} fraud)"
    )
    return X_train, X_test, y_train, y_test, scaler
