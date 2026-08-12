"""Model configurations, training, and cross-model comparison."""

import time

import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from xgboost import XGBClassifier

from config import OUTPUT_DIR, RANDOM_STATE


def get_model_configs():
    """Returns {name: (model_or_None, needs_smote)}.
    XGBoost's class-weighted variant is built dynamically in
    train_and_evaluate() once the train-set imbalance ratio is known.
    """
    return {
        "LogReg (class_weight)": (
            LogisticRegression(class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE),
            False,
        ),
        "RandomForest (class_weight)": (
            RandomForestClassifier(
                n_estimators=200, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1
            ),
            False,
        ),
        "XGBoost (scale_pos_weight)": (None, False),
        "LogReg (SMOTE)": (
            LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
            True,
        ),
        "RandomForest (SMOTE)": (
            RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1),
            True,
        ),
        "XGBoost (SMOTE)": (
            XGBClassifier(eval_metric="logloss", random_state=RANDOM_STATE, n_jobs=-1),
            True,
        ),
    }


def train_and_evaluate(X_train, X_test, y_train, y_test):
    results = []
    fitted_models = {}

    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    smote = SMOTE(random_state=RANDOM_STATE)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    print(f"SMOTE-resampled train set: {len(X_train_smote):,} rows (balanced)")

    configs = get_model_configs()

    for name, (model, use_smote) in configs.items():
        t0 = time.time()

        if model is None:  # XGBoost class-weighted variant
            model = XGBClassifier(
                eval_metric="logloss",
                scale_pos_weight=scale_pos_weight,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )

        Xtr, ytr = (X_train_smote, y_train_smote) if use_smote else (X_train, y_train)
        model.fit(Xtr, ytr)

        y_proba = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)

        pr_auc = average_precision_score(y_test, y_proba)
        roc_auc = roc_auc_score(y_test, y_proba)
        f1 = f1_score(y_test, y_pred)
        elapsed = time.time() - t0

        results.append(
            {
                "model": name,
                "pr_auc": pr_auc,
                "roc_auc": roc_auc,
                "f1_default_threshold": f1,
                "train_time_s": elapsed,
            }
        )
        fitted_models[name] = model
        print(
            f"{name:32s} | PR-AUC: {pr_auc:.4f} | ROC-AUC: {roc_auc:.4f} "
            f"| F1@0.5: {f1:.4f} | {elapsed:.1f}s"
        )

    results_df = pd.DataFrame(results).sort_values("pr_auc", ascending=False).reset_index(drop=True)
    out_path = OUTPUT_DIR / "model_comparison.csv"
    results_df.to_csv(out_path, index=False)
    print(f"\nSaved comparison table -> {out_path}")
    return results_df, fitted_models
