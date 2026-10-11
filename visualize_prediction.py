from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 1. Configuration
# ============================================================

INPUT_FILE = Path("outputs/xgboost_optuna/test_predictions.csv")

OUTPUT_DIR = Path("outputs/xgboost_optuna/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MARKET_TZ = "Europe/Berlin"

# Leave as None for automatic column detection.
# Set manually if your CSV uses different column names.
TIME_COL = "mtu_start_utc"
ACTUAL_COL = "day_ahead_price_eur_mwh"
PRED_COL = "predicted_price_eur_mwh"


# ============================================================
# 2. Load predictions
# ============================================================

df = pd.read_csv(INPUT_FILE)


def detect_column(df, specified, candidates):
    if specified is not None:
        if specified not in df.columns:
            raise ValueError(f"Column not found: {specified}")
        return specified

    lookup = {c.lower().strip(): c for c in df.columns}

    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]

    raise ValueError(
        f"Could not detect column from {candidates}.\n"
        f"Available columns: {df.columns.tolist()}"
    )


time_col = detect_column(
    df,
    TIME_COL,
    [
        "mtu_start_utc",
        "timestamp",
        "datetime",
        "datetime_utc",
        "delivery_time_utc",
        "time",
        "date",
    ],
)

actual_col = detect_column(
    df,
    ACTUAL_COL,
    [
        "actual",
        "actual_price",
        "y_true",
        "y_test",
        "price_actual",
        "day_ahead_price",
        "price_eur_mwh",
        "target",
        "price",
    ],
)

pred_col = detect_column(
    df,
    PRED_COL,
    [
        "predicted",
        "predicted_price",
        "prediction",
        "y_pred",
        "price_predicted",
        "price_prediction",
        "forecast",
    ],
)

df = df.rename(
    columns={
        time_col: "timestamp",
        actual_col: "actual",
        pred_col: "predicted",
    }
)

df["timestamp"] = pd.to_datetime(
    df["timestamp"],
    utc=True,
    errors="coerce",
)

df["actual"] = pd.to_numeric(df["actual"], errors="coerce")
df["predicted"] = pd.to_numeric(df["predicted"], errors="coerce")

df = (
    df.dropna(subset=["timestamp", "actual", "predicted"])
      .sort_values("timestamp")
      .reset_index(drop=True)
)

if df.empty:
    raise ValueError("No valid prediction rows found.")

if df["timestamp"].duplicated().any():
    raise ValueError("Duplicate prediction timestamps detected.")

# Error calculations
df["error"] = df["predicted"] - df["actual"]
df["abs_error"] = df["error"].abs()
df["squared_error"] = df["error"] ** 2

# German local delivery time
local_time = df["timestamp"].dt.tz_convert(MARKET_TZ)

df["local_date"] = local_time.dt.strftime("%Y-%m-%d")
df["local_hour"] = local_time.dt.hour


# ============================================================
# 3. Define market regimes
# ============================================================

# Negative electricity prices
df["negative_price"] = df["actual"] < 0

# Extreme high prices: top 5%
HIGH_PRICE_THRESHOLD = df["actual"].quantile(0.95)

df["extreme_high_price"] = (
    df["actual"] >= HIGH_PRICE_THRESHOLD
)

# High volatility:
# Daily realised price range = daily maximum - daily minimum

daily_range = (
    df.groupby("local_date")["actual"]
      .agg(lambda x: x.max() - x.min())
      .rename("daily_price_range")
)

VOLATILITY_THRESHOLD = daily_range.quantile(0.90)

high_vol_days = daily_range[
    daily_range >= VOLATILITY_THRESHOLD
].index

df["high_volatility"] = df["local_date"].isin(high_vol_days)


# ============================================================
# 4. Evaluation metrics
# ============================================================

def calculate_metrics(data):
    if data.empty:
        return {
            "Count": 0,
            "MAE": np.nan,
            "RMSE": np.nan,
            "Bias": np.nan,
        }

    return {
        "Count": len(data),
        "MAE": data["abs_error"].mean(),
        "RMSE": np.sqrt(data["squared_error"].mean()),
        "Bias": data["error"].mean(),
    }


regimes = {
    "All test hours": df,
    "Negative prices": df[df["negative_price"]],
    "Extreme high prices": df[df["extreme_high_price"]],
    "High volatility": df[df["high_volatility"]],
    "Other hours": df[
        ~(
            df["negative_price"]
            | df["extreme_high_price"]
            | df["high_volatility"]
        )
    ],
}

results = []

for name, subset in regimes.items():
    metrics = calculate_metrics(subset)
    metrics["Regime"] = name
    results.append(metrics)

metrics_df = pd.DataFrame(results)

metrics_df = metrics_df[
    ["Regime", "Count", "MAE", "RMSE", "Bias"]
]

metrics_df.to_csv(
    OUTPUT_DIR / "regime_metrics.csv",
    index=False,
)

print("\n=== Market Regime Performance ===")
print(metrics_df.round(2).to_string(index=False))

print(f"\nHigh price threshold: {HIGH_PRICE_THRESHOLD:.2f} €/MWh")
print(f"High volatility threshold: {VOLATILITY_THRESHOLD:.2f} €/MWh daily range")


# ============================================================
# 5. Plot helper
# ============================================================

def save_plot(filename):
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / filename,
        dpi=180,
        bbox_inches="tight",
    )
    plt.close()


# ============================================================
# 6. Overall actual vs predicted
# ============================================================

fig, ax = plt.subplots(figsize=(15, 5))

ax.plot(
    df["timestamp"],
    df["actual"],
    label="Actual",
    linewidth=1.3,
)

ax.plot(
    df["timestamp"],
    df["predicted"],
    label="Predicted",
    linewidth=1.1,
    alpha=0.8,
)

ax.axhline(0, linestyle="--", linewidth=0.8)

ax.set_title("Actual vs Predicted Day-Ahead Prices — Test Set")
ax.set_xlabel("Delivery Time (UTC)")
ax.set_ylabel("Electricity Price (€/MWh)")
ax.legend()
ax.grid(alpha=0.25)

save_plot("01_actual_vs_predicted.png")


# ============================================================
# 7. Worst 7-day prediction period
# ============================================================

# Rolling 7-day MAE, used only for retrospective evaluation.

rolling_mae = (
    df.set_index("timestamp")["abs_error"]
      .rolling(
          "7D",
          min_periods=min(72, len(df)),
      )
      .mean()
)

worst_end = rolling_mae.idxmax()
worst_start = worst_end - pd.Timedelta(days=7)

worst_period = df[
    (df["timestamp"] >= worst_start)
    & (df["timestamp"] <= worst_end)
]

fig, ax = plt.subplots(figsize=(15, 5))

ax.plot(
    worst_period["timestamp"],
    worst_period["actual"],
    label="Actual",
)

ax.plot(
    worst_period["timestamp"],
    worst_period["predicted"],
    label="Predicted",
)

ax.axhline(0, linestyle="--", linewidth=0.8)

ax.set_title(
    f"Worst 7-Day Prediction Period "
    f"(MAE = {worst_period['abs_error'].mean():.2f} €/MWh)"
)

ax.set_xlabel("Delivery Time (UTC)")
ax.set_ylabel("Electricity Price (€/MWh)")
ax.legend()
ax.grid(alpha=0.25)

save_plot("02_worst_7_days.png")


# ============================================================
# 8. Actual vs predicted scatter
# ============================================================

fig, ax = plt.subplots(figsize=(7, 7))

normal = df[
    ~(df["negative_price"] | df["extreme_high_price"])
]

negative = df[df["negative_price"]]
extreme = df[df["extreme_high_price"]]

ax.scatter(
    normal["actual"],
    normal["predicted"],
    alpha=0.35,
    s=15,
    label="Normal prices",
)

ax.scatter(
    negative["actual"],
    negative["predicted"],
    alpha=0.8,
    s=25,
    label="Negative prices",
)

ax.scatter(
    extreme["actual"],
    extreme["predicted"],
    alpha=0.8,
    s=25,
    label="Extreme high prices",
)

min_price = min(df["actual"].min(), df["predicted"].min())
max_price = max(df["actual"].max(), df["predicted"].max())

ax.plot(
    [min_price, max_price],
    [min_price, max_price],
    linestyle="--",
    label="Perfect prediction",
)

ax.set_title("Actual vs Predicted Price — Scatter")
ax.set_xlabel("Actual Price (€/MWh)")
ax.set_ylabel("Predicted Price (€/MWh)")
ax.legend()
ax.grid(alpha=0.25)

save_plot("03_prediction_scatter.png")


# ============================================================
# 9. Prediction error by hour
# ============================================================

hourly_mae = (
    df.groupby("local_hour")["abs_error"]
      .mean()
      .reindex(range(24))
)

fig, ax = plt.subplots(figsize=(12, 5))

ax.bar(
    hourly_mae.index,
    hourly_mae.values,
)

ax.set_title("Prediction MAE by Delivery Hour")
ax.set_xlabel("Hour of Day — Europe/Berlin")
ax.set_ylabel("MAE (€/MWh)")
ax.set_xticks(range(24))
ax.grid(axis="y", alpha=0.25)

save_plot("04_mae_by_hour.png")


# ============================================================
# 10. Daily prediction error
# ============================================================

daily_metrics = (
    df.groupby("local_date")
      .agg(
          MAE=("abs_error", "mean"),
          RMSE=("squared_error", lambda x: np.sqrt(x.mean())),
          Bias=("error", "mean"),
          Count=("actual", "size"),
      )
      .join(daily_range)
      .reset_index()
)

daily_metrics.to_csv(
    OUTPUT_DIR / "daily_metrics.csv",
    index=False,
)

fig, ax = plt.subplots(figsize=(15, 5))

ax.plot(
    daily_metrics["local_date"],
    daily_metrics["MAE"],
    linewidth=1.5,
)

ax.axhline(
    df["abs_error"].mean(),
    linestyle="--",
    label="Overall MAE",
)

ax.set_title("Daily Prediction Error")
ax.set_xlabel("Delivery Date — Europe/Berlin")
ax.set_ylabel("Daily MAE (€/MWh)")

# Show a manageable number of date labels
step = max(1, len(daily_metrics) // 12)

ax.set_xticks(
    range(0, len(daily_metrics), step)
)

ax.set_xticklabels(
    daily_metrics["local_date"].iloc[::step],
    rotation=45,
)

ax.legend()
ax.grid(alpha=0.25)

save_plot("05_daily_mae.png")


# ============================================================
# 11. MAE by market regime
# ============================================================

plot_metrics = metrics_df.dropna(subset=["MAE"])

fig, ax = plt.subplots(figsize=(10, 5))

bars = ax.bar(
    plot_metrics["Regime"],
    plot_metrics["MAE"],
)

for bar, count in zip(bars, plot_metrics["Count"]):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        f"n={count}",
        ha="center",
        va="bottom",
    )

ax.set_title("XGBoost Performance Across Market Regimes")
ax.set_ylabel("MAE (€/MWh)")
ax.tick_params(axis="x", rotation=20)
ax.grid(axis="y", alpha=0.25)

save_plot("06_mae_by_market_regime.png")


# ============================================================
# 12. Save detailed prediction errors
# ============================================================

df.to_csv(
    OUTPUT_DIR / "prediction_error_analysis.csv",
    index=False,
)

worst_errors = (
    df.sort_values("abs_error", ascending=False)
      .head(20)
)

worst_errors.to_csv(
    OUTPUT_DIR / "top_20_prediction_errors.csv",
    index=False,
)


# ============================================================
# 13. Summary
# ============================================================

print("\n=== Overall Test Performance ===")

overall = calculate_metrics(df)

for metric, value in overall.items():
    print(f"{metric}: {value:.2f}")


print("\n=== Negative Price Detection ===")

actual_negative = df["actual"] < 0
predicted_negative = df["predicted"] < 0

true_negative_predictions = (
    actual_negative & predicted_negative
).sum()

negative_recall = (
    true_negative_predictions / actual_negative.sum()
    if actual_negative.sum() > 0
    else np.nan
)

negative_precision = (
    true_negative_predictions / predicted_negative.sum()
    if predicted_negative.sum() > 0
    else np.nan
)

print(f"Actual negative hours: {actual_negative.sum()}")
print(f"Predicted negative hours: {predicted_negative.sum()}")
print(f"Negative price recall: {negative_recall:.2%}")
print(f"Negative price precision: {negative_precision:.2%}")


print("\n=== Worst 5 Days ===")

print(
    daily_metrics
    .sort_values("MAE", ascending=False)
    .head(5)
    .round(2)
    .to_string(index=False)
)

print(f"\nAll outputs saved to: {OUTPUT_DIR.resolve()}")