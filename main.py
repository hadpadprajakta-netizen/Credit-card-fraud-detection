import json

import joblib

from config import OUTPUT_DIR
from data import load_data, run_eda
from evaluate import tune_threshold
from models import train_and_evaluate
from preprocess import preprocess_and_split


def main():
    df = load_data()
    run_eda(df)

    X_train, X_test, y_train, y_test, scaler = preprocess_and_split(df)
    results_df, fitted_models = train_and_evaluate(X_train, X_test, y_train, y_test)

    best_name = results_df.iloc[0]["model"]
    best_model = fitted_models[best_name]
    print(f"\nBest model by PR-AUC: {best_name}")

    best_threshold, report = tune_threshold(best_model, X_test, y_test, best_name)

    # Persist everything needed to reuse or later wrap in an API
    joblib.dump(best_model, OUTPUT_DIR / "best_model.joblib")
    joblib.dump(scaler, OUTPUT_DIR / "scaler.joblib")
    with open(OUTPUT_DIR / "run_summary.json", "w") as f:
        json.dump(
            {
                "best_model": best_name,
                "best_threshold": float(best_threshold),
                "pr_auc": float(results_df.iloc[0]["pr_auc"]),
                "roc_auc": float(results_df.iloc[0]["roc_auc"]),
            },
            f,
            indent=2,
        )
    print(f"\nSaved model -> {OUTPUT_DIR / 'best_model.joblib'}")
    print(f"Saved scaler -> {OUTPUT_DIR / 'scaler.joblib'}")
    print(f"Saved run summary -> {OUTPUT_DIR / 'run_summary.json'}")

    print(
        "\nProduction next steps (write these up in your README/portfolio):\n"
        "  - Monitor prediction score distribution weekly for drift\n"
        "  - Retrain on a rolling window (e.g. monthly) as fraud patterns shift\n"
        "  - Add a human review queue for predictions near the decision threshold\n"
        "  - Log false negatives caught downstream (chargebacks) to recalibrate cost ratio"
    )


if __name__ == "__main__":
    main()
