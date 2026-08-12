# Credit Card Fraud Detection

A portfolio project built to handle severe class imbalance, comparing models with the right metrics, tuning a
decision threshold against a real business tradeoff, and structuring code like a
real (small) codebase rather than one notebook.

## Project structure

```
fraud_detection/
├── config.py             # shared constants (paths, random seed, cost ratio)
├── data.py                # load_data(), run_eda()
├── preprocess.py          # scale_amount_and_time(), preprocess_and_split()
├── models.py               # model configs + train_and_evaluate()
├── evaluate.py              # tune_threshold() against business cost tradeoff
├── main.py                  # entry point — run this
├── tests/
│   └── test_preprocess.py    # pytest unit tests for preprocessing
├── outputs/                   # generated plots, model, metrics (created on run)
├── requirements.txt
└── README.md
```

This separation matters, not just for style: `preprocess.py` is import-friendly
and unit-tested in isolation, and the same module logic can be reused directly
inside a future API service (e.g. FastAPI) without touching training or plotting code.

## Problem framing

Credit card fraud is rare (~0.17% of transactions in this dataset) but expensive
to miss. This project assumes:

- **Missing a fraud case (false negative)** costs ~10x more than
- **Blocking a legitimate transaction (false positive)**

This ratio drives the final threshold-tuning step in `evaluate.py`, instead of
using the default 0.5 cutoff.

## Dataset

[Kaggle: Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
— 284,807 transactions, 492 fraudulent. Features `V1`-`V28` are PCA-transformed
for privacy; `Amount` and `Time` are raw.

Download `creditcard.csv` and place it in the project root before running.

## Setup and run

```bash
pip install -r requirements.txt
python main.py
```

Outputs land in `outputs/`:
- `eda_class_and_amount.png` — class imbalance + amount distribution
- `model_comparison.csv` — PR-AUC / ROC-AUC / F1 for all 6 model configs
- `precision_recall_curve.png` — PR curve for the best model
- `confusion_matrix_tuned.png` — confusion matrix at the cost-optimal threshold
- `best_model.joblib`, `scaler.joblib` — ready to load into an API
- `run_summary.json` — best model name, threshold, headline metrics

## Run the tests

```bash
pytest tests/ -v
```

Tests cover: scaler fit/reuse behavior (fit only on train, reuse on test —
a common leakage bug), that `V1`-`V28` are left untouched, that the split is
properly stratified, correct split sizes, and that the target column doesn't
leak into features.

## What's compared

| Imbalance strategy | Models |
|---|---|
| `class_weight='balanced'` / `scale_pos_weight` | Logistic Regression, Random Forest, XGBoost |
| SMOTE oversampling | Logistic Regression, Random Forest, XGBoost |

Six configurations total, compared by **PR-AUC** — not accuracy or ROC-AUC,
both of which look misleadingly good under 0.17% positive class rate (a
model that predicts "not fraud" always scores 99.83% accuracy).

## Talking about this project in interviews

Be ready to explain, in your own words:
1. Why stratified splitting matters here
2. Why you compared class-weighting vs. SMOTE rather than picking one blindly
3. Why the final threshold isn't 0.5, and how you derived it
4. Why the scaler is fit only on train data and reused (not refit) on test —
   this is tested explicitly in `test_preprocess.py`
5. What you'd monitor if this were in production

## Possible extensions

- Wrap `outputs/best_model.joblib` in a FastAPI `/predict` endpoint
- Add SHAP values to explain individual fraud predictions
- Add MLflow to track all 6 runs instead of a flat CSV
- Add tests for `models.py` and `evaluate.py` too
