"""Train a DE-LU day-ahead price XGBoost model using Optuna (TPE).

Forecasting scenario: at 10:30 Europe/Berlin on day D, forecast every
hour of the NEXT delivery day (D+1). Only use features known at that time.

Project layout:
    EQ2/
      train_xgboost_optuna.py
      data/processed/model_dataset.csv

Install: pip install pandas numpy scikit-learn xgboost optuna
Run:     python train_xgboost_optuna.py

Optional quick run (2 trials):
    OPTUNA_N_TRIALS=2 python train_xgboost_optuna.py

Important: This code prevents time-split leakage, but cannot certify the
historical publication timestamps/revisions of external data. Audit them
before interpreting test metrics as a genuine real-time trading backtest.
"""

import json
import os
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBRegressor


# ------------------------------ Settings ----------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "model_dataset.csv"
if not DATA_PATH.exists():
    DATA_PATH = ROOT / "model_dataset.csv"
OUTPUT_DIR = Path(os.getenv("MODEL_OUTPUT_DIR", str(ROOT / "outputs" / "xgboost_optuna")))

TIME_COL = "mtu_start_utc"
TARGET = "day_ahead_price_eur_mwh"
MARKET_TZ = "Europe/Berlin"
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
N_FOLDS = 3
N_TRIALS = int(os.getenv("OPTUNA_N_TRIALS", "30"))
SEED = 42
XGB_N_JOBS = 4

# Explicit allow-list prevents unreviewed columns from entering the model.
# Forecasts and actuals must be known at 10:30 local on D for delivery D+1.
FEATURES = [
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

# These variables depend on tomorrow's day-ahead load forecast (or derivatives
# of it). Excluded pending as-of publication-time/version checks.
UNVERIFIED = [
    "day_ahead_load_forecast_mw", "load_ramp_1h", "load_daily_max",
    "load_daily_mean", "load_vs_daily_mean", "load_change_24h",
]


def evaluate(y_true, y_pred):
    return {
        "MAE_eur_mwh": round(float(mean_absolute_error(y_true, y_pred)), 3),
        "RMSE_eur_mwh": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 3),
        "R2": round(float(r2_score(y_true, y_pred)), 4),
    }


def read_data():
    if not DATA_PATH.is_file():
        raise FileNotFoundError(f"Cannot find dataset: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    required = {TIME_COL, TARGET, *FEATURES}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df[TIME_COL] = pd.to_datetime(df[TIME_COL], utc=True, errors="raise")
    if df[TIME_COL].isna().any() or df[TIME_COL].duplicated().any():
        raise ValueError("Null or duplicate delivery timestamps.")
    df = df.sort_values(TIME_COL).reset_index(drop=True)
    if not np.isfinite(df[TARGET].to_numpy(dtype=float)).all():
        raise ValueError("Targets must be finite and non-null.")

    for col in FEATURES:
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise TypeError(f"Feature {col} is not numeric.")
        if np.isinf(df[col].to_numpy(dtype=float)).any():
            raise ValueError(f"Infinite values in feature: {col}")
    # XGBoost accepts NaNs in features; no full-data imputation/scaling needed.

    # Sanity-check price lags against historical prices at exact UTC offsets.
    # This catches accidentally using current/future target values as lags.
    hist = df.set_index(TIME_COL)[TARGET]
    for hours in (24, 48, 72, 168):
        col = f"price_lag_{hours}h"
        expected = hist.reindex(df[TIME_COL] - pd.Timedelta(hours=hours)).to_numpy()
        supplied = df[col].to_numpy(dtype=float)
        comparable = np.isfinite(expected) & np.isfinite(supplied)
        if not np.allclose(expected[comparable], supplied[comparable], atol=1e-5):
            raise ValueError(f"Potential leakage/misalignment in {col}.")

    # Price rolling features must end BEFORE the local delivery-day midnight.
    # Using tomorrow's full-day price in these statistics would be leakage.
    local_midnights_utc = df[TIME_COL].dt.tz_convert(MARKET_TZ).dt.normalize().dt.tz_convert("UTC")
    for name, series in (
        ("price_rolling_mean_48h", hist.rolling(48, min_periods=48).mean().shift(1)),
        ("price_rolling_std_48h", hist.rolling(48, min_periods=48).std().shift(1)),
    ):
        expected = series.reindex(local_midnights_utc).to_numpy()
        supplied = df[name].to_numpy(dtype=float)
        comparable = np.isfinite(expected) & np.isfinite(supplied)
        # Source CSV rounds rolling statistics to four decimals.
        if not np.allclose(expected[comparable], supplied[comparable], atol=1e-4):
            raise ValueError(f"Potential leakage/misalignment in {name}.")

    # Local market delivery date is essential across CET/CEST transitions.
    df["delivery_date"] = df[TIME_COL].dt.tz_convert(MARKET_TZ).dt.date
    return df


def split_by_delivery_day(df):
    dates = np.array(df["delivery_date"].drop_duplicates().tolist())
    n_train = int(len(dates) * TRAIN_FRAC)
    n_val = int(len(dates) * VAL_FRAC)
    train_days = dates[:n_train]
    val_days = dates[n_train:n_train + n_val]
    test_days = dates[n_train + n_val:]
    if len(train_days) < 60 or len(val_days) < 14 or len(test_days) < 14:
        raise ValueError("Not enough dates for train/validation/test split.")

    train = df[df["delivery_date"].isin(train_days)].copy()
    val = df[df["delivery_date"].isin(val_days)].copy()
    test = df[df["delivery_date"].isin(test_days)].copy()
    assert train[TIME_COL].max() < val[TIME_COL].min() < test[TIME_COL].min()
    return train, val, test


def get_expanding_folds(train):
    """Split unique Berlin delivery dates, NOT randomly shuffled rows."""
    dates = np.array(train["delivery_date"].drop_duplicates().tolist())
    folds = []
    splitter = TimeSeriesSplit(n_splits=N_FOLDS)
    for d_tr, d_va in splitter.split(dates):
        tr_idx = np.flatnonzero(train["delivery_date"].isin(dates[d_tr]).to_numpy())
        va_idx = np.flatnonzero(train["delivery_date"].isin(dates[d_va]).to_numpy())
        if tr_idx.max() >= va_idx.min():
            raise ValueError("Nonchronological CV fold detected.")
        folds.append((tr_idx, va_idx))
    return folds


def model_params(params):
    return {
        "objective": "reg:squarederror",
        "tree_method": "hist",  # CPU-friendly
        "eval_metric": "mae",
        "random_state": SEED,
        "n_jobs": XGB_N_JOBS,
        **params,
    }


def main():
    if N_TRIALS < 1:
        raise ValueError("OPTUNA_N_TRIALS must be at least 1.")
    df = read_data()
    train, val, test = split_by_delivery_day(df)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for label, subset in (("Train", train), ("Validation", val), ("Test", test)):
        print(f"{label:10}: {subset['delivery_date'].min()} -> "
              f"{subset['delivery_date'].max()} ({len(subset):,} rows)")
    print(f"Features: {len(FEATURES)} | Optuna trials: {N_TRIALS} | CV folds: {N_FOLDS}")
    print("Excluded unverified day-ahead load columns:", ", ".join(UNVERIFIED))

    X_train, y_train = train[FEATURES], train[TARGET]
    X_val, y_val = val[FEATURES], val[TARGET]
    X_test, y_test = test[FEATURES], test[TARGET]
    cv_folds = get_expanding_folds(train)

    def objective(trial):
        # Optuna's TPE uses previous trial results to guide its proposals.
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 800, step=100),
            "max_depth": trial.suggest_int("max_depth", 3, 7),
            "learning_rate": trial.suggest_float("learning_rate", 0.015, 0.15, log=True),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "subsample": trial.suggest_float("subsample", 0.65, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.65, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 5.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 30.0, log=True),
        }
        cv_maes = []
        for train_idx, valid_idx in cv_folds:
            model = XGBRegressor(**model_params(params))
            model.fit(X_train.iloc[train_idx], y_train.iloc[train_idx])
            preds = model.predict(X_train.iloc[valid_idx])
            cv_maes.append(mean_absolute_error(y_train.iloc[valid_idx], preds))

        # No access to the held-out validation or test set in this objective.
        trial.set_user_attr("fold_maes", [round(float(x), 4) for x in cv_maes])
        return float(np.mean(cv_maes))

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="minimize", sampler=optuna.samplers.TPESampler(seed=SEED)
    )

    def log_trial(study, trial):
        print(f"Trial {trial.number + 1}/{N_TRIALS}: CV MAE = {trial.value:.3f} "
              f"| best = {study.best_value:.3f} EUR/MWh")

    print("\nTuning with Optuna (training data ONLY) ...")
    study.optimize(objective, n_trials=N_TRIALS, callbacks=[log_trial])
    best_params = study.best_params
    print("\nBest hyperparameters:", best_params)
    print(f"Best mean CV MAE: {study.best_value:.3f} EUR/MWh")

    # Hold-out validation: one assessment AFTER Optuna finishes.
    validation_model = XGBRegressor(**model_params(best_params))
    validation_model.fit(X_train, y_train)
    validation_pred = validation_model.predict(X_val)
    val_metrics = evaluate(y_val, validation_pred)
    print("Validation:", val_metrics)

    # Final model: refit on train + validation. Test remains untouched until now.
    train_val = pd.concat([train, val], ignore_index=True)
    final_model = XGBRegressor(**model_params(best_params))
    final_model.fit(train_val[FEATURES], train_val[TARGET])
    test_pred = final_model.predict(X_test)
    test_metrics = evaluate(y_test, test_pred)
    baseline_metrics = evaluate(y_test, test["price_lag_168h"])
    print("Final test:", test_metrics)
    print("Naive benchmark (last-week price):", baseline_metrics)

    # Save model and aligned test features for later SHAP TreeExplainer.
    final_model.save_model(OUTPUT_DIR / "xgboost_model.json")
    (OUTPUT_DIR / "feature_columns.json").write_text(
        json.dumps(FEATURES, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    prediction_out = test[[TIME_COL, "delivery_date", TARGET]].copy()
    prediction_out["predicted_price_eur_mwh"] = test_pred
    prediction_out["absolute_error_eur_mwh"] = abs(y_test.to_numpy() - test_pred)
    prediction_out.to_csv(OUTPUT_DIR / "test_predictions.csv", index=False)
    pd.concat([test[[TIME_COL]].reset_index(drop=True),
               X_test.reset_index(drop=True)], axis=1).to_csv(
        OUTPUT_DIR / "test_features_for_shap.csv", index=False
    )
    study.trials_dataframe().to_csv(OUTPUT_DIR / "optuna_trials.csv", index=False)

    report = {
        "forecast_scenario": "At 10:30 Europe/Berlin on D, predict D+1 delivery prices",
        "market_timezone": MARKET_TZ,
        "n_rows": len(df),
        "n_features": len(FEATURES),
        "train_dates": [str(train['delivery_date'].min()), str(train['delivery_date'].max())],
        "validation_dates": [str(val['delivery_date'].min()), str(val['delivery_date'].max())],
        "test_dates": [str(test['delivery_date'].min()), str(test['delivery_date'].max())],
        "train_rows": len(train), "validation_rows": len(val), "test_rows": len(test),
        "optuna_trials": N_TRIALS, "cv_folds": N_FOLDS,
        "best_cv_mae_eur_mwh": round(float(study.best_value), 3),
        "best_params": best_params,
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
        "test_last_week_naive_metrics": baseline_metrics,
        "unverified_excluded_features": UNVERIFIED,
        "data_availability_caveat": (
            "Dataset lacks historical publication/as-of timestamps for external forecasts "
            "and actuals; time splitting alone does not prove absence of leakage. "
            "Recheck rolling feature construction and publication revisions."
        ),
    }
    (OUTPUT_DIR / "metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nSaved model, tuning results, predictions and SHAP input to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
