"""Threshold tuning against the stated business cost tradeoff, plus
final confusion matrix / classification report."""

import matplotlib.pyplot as plt
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
)

from config import COST_FALSE_NEGATIVE, COST_FALSE_POSITIVE, OUTPUT_DIR


def tune_threshold(model, X_test, y_test, best_name: str):
    y_proba = model.predict_proba(X_test)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_test, y_proba)

    # Evaluate expected cost at every threshold and keep the minimum.
    # cost = FN * cost_fn + FP * cost_fp
    best_threshold, best_cost = 0.5, float("inf")
    for t in thresholds:
        y_pred_t = (y_proba >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred_t).ravel()
        cost = fn * COST_FALSE_NEGATIVE + fp * COST_FALSE_POSITIVE
        if cost < best_cost:
            best_cost, best_threshold = cost, t

    y_pred_final = (y_proba >= best_threshold).astype(int)

    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, label=best_name)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve (Best Model)")
    plt.legend()
    plt.tight_layout()
    pr_path = OUTPUT_DIR / "precision_recall_curve.png"
    plt.savefig(pr_path, dpi=150)
    plt.close()

    cm = confusion_matrix(y_test, y_pred_final)
    disp = ConfusionMatrixDisplay(cm, display_labels=["Legit", "Fraud"])
    disp.plot(cmap="Blues", values_format="d")
    plt.title(f"Confusion Matrix @ threshold={best_threshold:.3f}")
    plt.tight_layout()
    cm_path = OUTPUT_DIR / "confusion_matrix_tuned.png"
    plt.savefig(cm_path, dpi=150)
    plt.close()

    report = classification_report(y_test, y_pred_final, target_names=["Legit", "Fraud"])
    print(
        f"\nBest threshold (min expected cost, FN:FP = "
        f"{COST_FALSE_NEGATIVE}:{COST_FALSE_POSITIVE}) = {best_threshold:.4f}"
    )
    print(report)
    print(f"Saved PR curve -> {pr_path}")
    print(f"Saved confusion matrix -> {cm_path}")

    return best_threshold, report
