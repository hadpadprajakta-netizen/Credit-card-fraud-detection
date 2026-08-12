"""Unit tests for preprocess.py.

Run with: pytest tests/

These don't need the real Kaggle dataset — they build small synthetic
DataFrames shaped like it, which is exactly what makes preprocessing logic
testable in isolation from data availability.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Allow running `pytest` from the project root without packaging setup.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from preprocess import preprocess_and_split, scale_amount_and_time


def make_fake_df(n=200, fraud_rate=0.1, seed=0):
    rng = np.random.default_rng(seed)
    n_fraud = int(n * fraud_rate)
    df = pd.DataFrame(
        {
            "Time": rng.uniform(0, 100000, n),
            "Amount": rng.exponential(50, n),
            **{f"V{i}": rng.normal(0, 1, n) for i in range(1, 5)},  # small subset of V1-V28
            "Class": [1] * n_fraud + [0] * (n - n_fraud),
        }
    )
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)


class TestScaleAmountAndTime:
    def test_fit_produces_zero_mean(self):
        df = make_fake_df()
        X = df.drop(columns=["Class"])
        X_scaled, scaler = scale_amount_and_time(X)
        assert np.isclose(X_scaled["Amount"].mean(), 0, atol=1e-8)
        assert np.isclose(X_scaled["Time"].mean(), 0, atol=1e-8)

    def test_reuses_fitted_scaler_without_refitting(self):
        df = make_fake_df(seed=1)
        X = df.drop(columns=["Class"])
        _, fitted_scaler = scale_amount_and_time(X)

        # A different dataset, scaled with the *already-fitted* scaler,
        # should not re-center to zero mean — it must reuse train-set stats.
        df2 = make_fake_df(seed=2)
        X2 = df2.drop(columns=["Class"])
        X2_scaled, scaler_out = scale_amount_and_time(X2, scaler=fitted_scaler)

        assert scaler_out is fitted_scaler
        assert not np.isclose(X2_scaled["Amount"].mean(), 0, atol=1e-8)

    def test_v_columns_untouched(self):
        df = make_fake_df()
        X = df.drop(columns=["Class"])
        X_scaled, _ = scale_amount_and_time(X)
        pd.testing.assert_series_equal(X_scaled["V1"], X["V1"])


class TestPreprocessAndSplit:
    def test_split_is_stratified(self):
        df = make_fake_df(n=1000, fraud_rate=0.1)
        X_train, X_test, y_train, y_test, _ = preprocess_and_split(df)

        train_rate = y_train.mean()
        test_rate = y_test.mean()
        # Stratified split should keep fraud rate within a small tolerance
        assert abs(train_rate - test_rate) < 0.02

    def test_split_sizes(self):
        df = make_fake_df(n=1000)
        X_train, X_test, y_train, y_test, _ = preprocess_and_split(df)
        assert len(X_train) == 800
        assert len(X_test) == 200

    def test_no_class_column_leaks_into_features(self):
        df = make_fake_df()
        X_train, X_test, _, _, _ = preprocess_and_split(df)
        assert "Class" not in X_train.columns
        assert "Class" not in X_test.columns

    def test_missing_class_column_raises(self):
        df = make_fake_df().drop(columns=["Class"])
        with pytest.raises(KeyError):
            preprocess_and_split(df)
