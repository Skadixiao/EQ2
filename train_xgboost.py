"""Train a leakage-conscious XGBoost for DE-LU day-ahead hourly prices.

Run from your EQ2 project after placing this file in the project root:
    pip install pandas numpy scikit-learn xgboost
    python train_xgboost.py

Expected input: EQ2/data/processed/model_dataset.csv
If that path is absent, a model_dataset.csv next to this script is used.

Forecast scenario: at 10:30 Europe/Berlin on day D, predict every
hour of the delivery day D+1. All features must have been available then.
"""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from xgboost import XGBRegressor


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "processed" / "model_dataset.csv"
if not DATA_PATH.exists():
    DATA_PATH = ROOT / "model_dataset.csv"
OUTPUT_DIR = ROOT / "outputs" / "xgboost"

TIME_COL = "mtu_start_utc"
TARGET = "day_ahead_price_eur_mwh"
MARKET_TZ = "Europe/Berlin"
TRAIN_FRACTION = 0.70
VALIDATION_FRACTION = 0.15  # Final 15% of dates is untouched test data.
N_SPLITS = int(os.getenv("XGB_CV_SPLITS", "3"))
N_ITER = int(os.getenv("XGB_N_ITER", "10"))
SEED = 42

# Explicit allow-list: newly added columns will NEVER silently enter the model.
# Only add weather/forecast data after verifying its publication time (as of 10:30 D).
APPROVED_FEATURES = [
    "hour", "day_of_week", "month", "is_weekend", "is_holiday", "is_workday",
    "price_lag_24h", "price_lag_48h", "price_lag_72h", "price_lag_168h",
    "price_rolling_mean_48h", "price_rolling_std_48h",
    "wind_total_forecast_lag_48h", "solar_forecast_lag_48h",
    "actual_load_lag_48h", "actual_load_lag_72h", "actual_load_lag_168h",
    "actual_load_rolling_mean_48h", "actual_load_rolling_std_48h",
    "biomass_generation_lag_48h",
    "lignite_generation_lag_48h", "lignite_generation_lag_168h",
    "gas_generation_lag_48h", "gas_generation_lag_168h",
    "hard_coal_generation_lag_48h", "hard_coal_generation_lag_168h",
    "oil_generation_lag_48h",
    "pumped_storage_generation_lag_48h", "pumped_storage_generation_lag_168h",
]

# Do not include these without verifying *historical forecast publication times*.
# All use tomorrow's day-ahead load forecast; the CSV has no as-of timestamps.
UNVERIFIED_FEATURES = [
    "day_ahead_load_forecast_mw", "load_ramp_1h", "load_daily_max",
    "load_daily_mean", "load_vs_daily_mean", "load_change_24h",
]


def metrics(y_true, y_pred):
    return {
        "MAE_eur_mwh": round(float(mean_absolute_error(y_true, y_pred)), 3),
        "RMSE_eur_mwh": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 3),
        "R2": round(float(r2_score(y_true, y_pred)), 4),
    }


def make_date_cv(train_dates, date_series, n_splits):
    """Expanding-window folds, with entire Berlin delivery days kept together."""
    folds = []
    for tr_day_idx, va_day_idx in TimeSeriesSplit(n_splits=n_splits).split(train_dates):
        tr_days = set(train_dates[tr_day_idx])
        va_days = set(train_dates[va_day_idx])
        tr_rows = np.flatnonzero(date_series.isin(tr_days).to_numpy())
        va_rows = np.flatnonzero(date_series.isin(va_days).to_numpy())
        assert len(tr_rows) > 0 and len(va_rows) > 0
        assert tr_rows.max() < va_rows.min(), "CV fold is not chronological."
        folds.append((tr_rows, va_rows))
        print(f"  CV {len(folds)}: train through {train_dates[tr_day_idx[-1]]}, "
              f"validate {train_dates[va_day_idx[0]]} to {train_dates[va_day_idx[-1]]}")
    return folds


def main():
    if not DATA_PATH.is_file():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")
    if not (0 < TRAIN_FRACTION < TRAIN_FRACTION + VALIDATION_FRACTION < 1):
        raise ValueError("Invalid train/validation fractions.")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_PATH)
    required = [TIME_COL, TARGET] + APPROVED_FEATURES
    missing_cols = sorted(set(required) - set(df.columns))
    if missing_cols:
        raise ValueError(f"Required columns missing: {missing_cols}")

    df[TIME_COL] = pd.to_datetime(df[TIME_COL], utc=True, errors="raise")
    if df[TIME_COL].isna().any() or df[TIME_COL].duplicated().any():
        raise ValueError("Invalid/duplicate delivery timestamps detected.")
    df = df.sort_values(TIME_COL).reset_index(drop=True)
    if df[TARGET].isna().any():
        raise ValueError("Missing targets: check data rather than drop test rows silently.")
    if not np.isfinite(df[TARGET].to_numpy(dtype=float)).all():
        raise ValueError("Non-finite target values detected.")

    # Training across DST changes: each LOCAL delivery date belongs to one split.
    df["delivery_date"] = df[TIME_COL].dt.tz_convert(MARKET_TZ).dt.date
    dates = np.array(df["delivery_date"].drop_duplicates().tolist())
    n_train_days = int(len(dates) * TRAIN_FRACTION)
    n_val_days = int(len(dates) * VALIDATION_FRACTION)
    if n_train_days < (N_SPLITS + 1) * 14 or n_val_days < 14:
        raise ValueError("Not enough data for these chronological CV splits.")

    train_days = dates[:n_train_days]
    val_days = dates[n_train_days:n_train_days + n_val_days]
    test_days = dates[n_train_days + n_val_days:]

    train_df = df.loc[df["delivery_date"].isin(train_days)].copy()
    val_df = df.loc[df["delivery_date"].isin(val_days)].copy()
    test_df = df.loc[df["delivery_date"].isin(test_days)].copy()
    assert train_df[TIME_COL].max() < val_df[TIME_COL].min() < test_df[TIME_COL].min()

    print(f"Dataset: {len(df):,} rows, {len(APPROVED_FEATURES)} approved features")
    print(f"Train: {train_days[0]} to {train_days[-1]} ({len(train_df):,} rows)")
    print(f"Val:   {val_days[0]} to {val_days[-1]} ({len(val_df):,} rows)")
    print(f"Test:  {test_days[0]} to {test_days[-1]} ({len(test_df):,} rows)")
    print("Excluded unverified forecast columns:", ", ".join(UNVERIFIED_FEATURES))

    # XGBoost handles NaNs. No scaling or imputation fitted on the full dataset.
    for feature in APPROVED_FEATURES:
        df_dtype = df[feature].dtype
        if not pd.api.types.is_numeric_dtype(df_dtype):
            raise TypeError(f"Feature {feature} must be numeric, not {df_dtype}.")
    X_train, y_train = train_df[APPROVED_FEATURES], train_df[TARGET]
    X_val, y_val = val_df[APPROVED_FEATURES], val_df[TARGET]
    X_test, y_test = test_df[APPROVED_FEATURES], test_df[TARGET]

    cv = make_date_cv(train_days, train_df["delivery_date"], N_SPLITS)

    # Hyperparameter search ONLY uses training dates (never val/test).
    base = XGBRegressor(
        objective="reg:squarederror", tree_method="hist", eval_metric="mae",
        random_state=SEED, n_jobs=4,
    )
    parameters = {
        "n_estimators": [250, 400, 600, 800],
        "learning_rate": [0.03, 0.05, 0.08, 0.12],
        "max_depth": [3, 4, 5, 6],
        "min_child_weight": [1, 3, 7],
        "subsample": [0.75, 0.9, 1.0],
        "colsample_bytree": [0.75, 0.9, 1.0],
        "reg_lambda": [1, 5, 10],
        "reg_alpha": [0, 0.1, 1],
    }
    search = RandomizedSearchCV(
        estimator=base, param_distributions=parameters, n_iter=N_ITER,
        scoring="neg_mean_absolute_error", cv=cv,
        n_jobs=1, random_state=SEED, refit=True, verbose=1,
        return_train_score=False,
    )
    print("\nTuning hyperparameters on TRAIN only ...")
    search.fit(X_train, y_train)
    print("Best parameters:", search.best_params_)
    print(f"Mean CV MAE: {-search.best_score_:.3f} EUR/MWh")

    # One genuinely out-of-sample validation check (not used to tune parameters).
    val_prediction = search.best_estimator_.predict(X_val)
    val_metrics = metrics(y_val, val_prediction)
    print("Validation:", val_metrics)

    # Refit on train + validation, then score the test set ONCE.
    train_val_df = pd.concat([train_df, val_df], ignore_index=True)
    final_model = XGBRegressor(
        objective="reg:squarederror", tree_method="hist", eval_metric="mae",
        random_state=SEED, n_jobs=4, **search.best_params_,
    )
    final_model.fit(train_val_df[APPROVED_FEATURES], train_val_df[TARGET])
    test_prediction = final_model.predict(X_test)
    test_metrics = metrics(y_test, test_prediction)
    print("TEST (held out until now):", test_metrics)

    # Last-week price is a useful leakage-safe naive benchmark.
    naive_metrics = metrics(y_test, test_df["price_lag_168h"])
    print("TEST naive (same hour last week):", naive_metrics)

    # XGBoost's native JSON model and named feature schema are SHAP-friendly.
    final_model.save_model(OUTPUT_DIR / "xgboost_model.json")
    (OUTPUT_DIR / "feature_columns.json").write_text(
        json.dumps(APPROVED_FEATURES, indent=2), encoding="utf-8"
    )
    test_export = test_df[[TIME_COL, "delivery_date", TARGET]].copy()
    test_export["predicted_price_eur_mwh"] = test_prediction
    test_export["absolute_error_eur_mwh"] = np.abs(y_test.to_numpy() - test_prediction)
    test_export.to_csv(OUTPUT_DIR / "test_predictions.csv", index=False)

    # Exact feature values for later TreeExplainer / SHAP, with stable row order.
    pd.concat([test_df[[TIME_COL]].reset_index(drop=True),
               X_test.reset_index(drop=True)], axis=1).to_csv(
        OUTPUT_DIR / "test_features_for_shap.csv", index=False
    )
    pd.DataFrame(search.cv_results_).sort_values("rank_test_score").to_csv(
        OUTPUT_DIR / "tuning_results.csv", index=False
    )
    report = {
        "forecast_definition": "At 10:30 Europe/Berlin on D, predict all prices for D+1",
        "data_path": str(DATA_PATH),
        "n_rows": len(df), "n_features": len(APPROVED_FEATURES),
        "split_by": "Europe/Berlin local delivery date",
        "train_dates": [str(train_days[0]), str(train_days[-1])],
        "validation_dates": [str(val_days[0]), str(val_days[-1])],
        "test_dates": [str(test_days[0]), str(test_days[-1])],
        "training_rows": len(train_df), "validation_rows": len(val_df),
        "test_rows": len(test_df),
        "cv_folds": N_SPLITS, "search_iterations": N_ITER,
        "best_cv_mae_eur_mwh": round(float(-search.best_score_), 3),
        "best_params": search.best_params_,
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
        "test_last_week_naive_metrics": naive_metrics,
        "excluded_unverified_features": UNVERIFIED_FEATURES,
        "publication_timestamp_warning": (
            "Raw CSVs are not point-in-time snapshots. Verify historical availability "
            "and revisions before claiming a fully leakage-free live backtest."
        ),
        "xgboost_version": xgboost.__version__,
        "sklearn_version": sklearn.__version__,
    }
    (OUTPUT_DIR / "metrics.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nSaved model, test predictions, SHAP input and metrics to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
