"""Merge EQ2 hourly data and create day-ahead electricity-price features.

Run from anywhere: python merge.py
Dependency: pip install pandas holidays

Assumptions:
- merge.py is stored in the EQ2 project root.
- All four input files contain hourly UTC timestamps in mtu_start_utc.
- The default market is Germany (Europe/Berlin calendar).
- Renewable forecast vintage has NOT been verified as pre-auction.
"""
from pathlib import Path

import pandas as pd
import holidays


DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
MARKET_TZ = "Europe/Berlin"  # Change to your bidding zone if needed
HOLIDAY_COUNTRY = "DE"       # Change to the corresponding country code
USE_RENEWABLE_IN_MODEL = False  # True ONLY for forecasts available before auction


def read_csv(name):
    path = DATA_DIR / f"{name}.csv"
    df = pd.read_csv(path, parse_dates=["mtu_start_utc"])
    df["mtu_start_utc"] = pd.to_datetime(df["mtu_start_utc"], utc=True)
    if df["mtu_start_utc"].duplicated().any():
        raise ValueError(f"Duplicate timestamps in {path.name}")
    return df.set_index("mtu_start_utc").sort_index()


def add_features(df):
    model_features = ["day_ahead_load_forecast_mw"]
    renewable_features = []

    local = df.index.tz_convert(MARKET_TZ)
    local_day = pd.to_datetime(local.date)  # Local market delivery date, no timezone
    midnight = pd.DatetimeIndex(local_day).tz_localize(MARKET_TZ).tz_convert("UTC")
    prev_midnight = pd.DatetimeIndex(local_day - pd.Timedelta(days=1)).tz_localize(MARKET_TZ).tz_convert("UTC")

    # Calendar features are based on LOCAL delivery time, not UTC time.
    df["hour"] = local.hour
    df["day_of_week"] = local.dayofweek  # Monday=0, Sunday=6
    df["month"] = local.month
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    public_holidays = holidays.country_holidays(HOLIDAY_COUNTRY, years=sorted(set(local.year)))
    df["is_holiday"] = [int(d in public_holidays) for d in local.date]
    df["is_workday"] = ((df["is_weekend"] == 0) & (df["is_holiday"] == 0)).astype(int)
    model_features += ["hour", "day_of_week", "month", "is_weekend", "is_holiday", "is_workday"]

    # Historical prices. All lag values must be known at the previous day's auction.
    price = df["day_ahead_price_eur_mwh"]
    for lag in (24, 48, 72, 168):
        col = f"price_lag_{lag}h"
        df[col] = price.shift(lag)
        # DST autumn switch: t-24 can lie in the TARGET delivery day, so mask it.
        if lag == 24:
            df.loc[(df.index - pd.Timedelta(hours=24)) >= midnight, col] = float("nan")
        model_features.append(col)

    # For every hour of local day D, use the SAME last 48 hourly prices
    # strictly before D starts; never use any price from the target day.
    price_history = price.shift(1)
    df["price_rolling_mean_48h"] = price_history.rolling(48, min_periods=48).mean().reindex(midnight).to_numpy()
    df["price_rolling_std_48h"] = price_history.rolling(48, min_periods=48).std().reindex(midnight).to_numpy()
    model_features += ["price_rolling_mean_48h", "price_rolling_std_48h"]

    # Forecast load for the delivery day (requires pre-auction forecast vintage).
    load = df["day_ahead_load_forecast_mw"]
    df["load_ramp_1h"] = load.diff()
    df["load_daily_max"] = load.groupby(local_day).transform("max")
    df["load_daily_mean"] = load.groupby(local_day).transform("mean")
    df["load_vs_daily_mean"] = load / df["load_daily_mean"].replace(0, float("nan"))
    df["load_change_24h"] = load - load.shift(24)
    model_features += ["load_ramp_1h", "load_daily_max", "load_daily_mean", "load_vs_daily_mean", "load_change_24h"]

    # Renewable forecasts: computed for analysis, NOT used in model by default.
    df["wind_total"] = df["offshore_wind_forecast_mw"] + df["onshore_wind_forecast_mw"]
    df["renewable_forecast"] = df["wind_total"] + df["solar_forecast_mw"]
    df["renewable_share"] = df["renewable_forecast"] / load.replace(0, float("nan"))
    df["residual_load"] = load - df["renewable_forecast"]
    df["residual_load_ramp"] = df["residual_load"].diff()
    df["residual_load_daily_max"] = df["residual_load"].groupby(local_day).transform("max")
    # Even the 18:00 forecast vintage for D-2 is known before D-1's noon auction.
    # These lagged forecasts can safely provide a weaker historical weather signal.
    df["wind_total_forecast_lag_48h"] = df["wind_total"].shift(48)
    df["solar_forecast_lag_48h"] = df["solar_forecast_mw"].shift(48)
    model_features += ["wind_total_forecast_lag_48h", "solar_forecast_lag_48h"]
    renewable_features += [
        "offshore_wind_forecast_mw", "onshore_wind_forecast_mw", "solar_forecast_mw",
        "wind_total", "renewable_forecast", "renewable_share",
        "residual_load", "residual_load_ramp", "residual_load_daily_max",
    ]

    # Historical actual load: 48h/72h/168h lags only (never actual load at t).
    actual_load = df["actual_total_load_mw"]
    for lag in (48, 72, 168):
        col = f"actual_load_lag_{lag}h"
        df[col] = actual_load.shift(lag)
        model_features.append(col)

    # Last 48 actual-load hours before D-1 starts, avoiding data from the
    # auction morning; this is shared by all delivery hours of day D.
    actual_history = actual_load.shift(1)
    df["actual_load_rolling_mean_48h"] = actual_history.rolling(48, min_periods=48).mean().reindex(prev_midnight).to_numpy()
    df["actual_load_rolling_std_48h"] = actual_history.rolling(48, min_periods=48).std().reindex(prev_midnight).to_numpy()
    model_features += ["actual_load_rolling_mean_48h", "actual_load_rolling_std_48h"]

    # Actual generation per type, using historical lags only.
    generation = ["biomass", "lignite", "gas", "hard_coal", "oil", "pumped_storage"]
    for fuel in generation:
        source = f"{fuel}_generation_mw"
        col = f"{fuel}_generation_lag_48h"
        df[col] = df[source].shift(48)
        model_features.append(col)
        if fuel in ("gas", "lignite", "hard_coal", "pumped_storage"):
            col = f"{fuel}_generation_lag_168h"
            df[col] = df[source].shift(168)
            model_features.append(col)

    return df, model_features, renewable_features


def main():
    # Price is the target, so keep its timeline and LEFT JOIN other datasets.
    df = read_csv("day_ahead_price")
    for name in ("load_forecast", "renewable_forecast", "actual_generation_per_type"):
        df = df.join(read_csv(name), how="left", validate="one_to_one")

    # Make shifts mean actual clock-hours even if price timestamps are missing.
    full_hours = pd.date_range(df.index.min(), df.index.max(), freq="h", tz="UTC")
    if len(full_hours) != len(df):
        print(f"Warning: {len(full_hours) - len(df)} missing target timestamps inserted as NaN")
        df = df.reindex(full_hours)
        df.index.name = "mtu_start_utc"

    df, features, renewable = add_features(df)
    # ALL features, including leaky raw actuals and unverified renewable forecasts:
    df.reset_index().round(4).to_csv(DATA_DIR / "merged_features.csv", index=False)

    # Safer training dataset: excludes contemporaneous actuals and unverified
    # renewables (set USE_RENEWABLE_IN_MODEL only after checking vintage).
    model_columns = ["day_ahead_price_eur_mwh"] + features
    if USE_RENEWABLE_IN_MODEL:
        model_columns += renewable
    model = df[model_columns].dropna(subset=["day_ahead_price_eur_mwh", "price_lag_168h"])
    model.reset_index().round(4).to_csv(DATA_DIR / "model_dataset.csv", index=False)

    print(f"Merged: {len(df):,} hourly rows, {len(df.columns)} columns")
    print(f"Model:  {len(model):,} rows, {len(model.columns) - 1} features + 1 target (+ timestamp)")
    print(f"Saved to: {DATA_DIR}")
    if not USE_RENEWABLE_IN_MODEL:
        print("Renewable forecasts excluded from model_dataset.csv: verify pre-auction vintage first.")
    missing = model.isna().sum()
    print("Missing model values:", missing[missing > 0].to_dict())


if __name__ == "__main__":
    main()
