from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
import shap


# ============================================================
# 1. CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent

MODEL_DIR = ROOT / "outputs" / "xgboost_optuna"

MODEL_PATH = MODEL_DIR / "xgboost_model.json"
FEATURE_PATH = MODEL_DIR / "feature_columns.json"
TEST_PATH = MODEL_DIR / "test_features_for_shap.csv"

OUTPUT_DIR = MODEL_DIR / "shap"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. LOAD MODEL AND TEST DATA
# ============================================================

print("Loading trained XGBoost model...")

model = xgb.Booster()
model.load_model(str(MODEL_PATH))

with open(FEATURE_PATH) as f:
    feature_config = json.load(f)

# Supports either a list or {"feature_columns": [...]}
if isinstance(feature_config, list):
    feature_cols = feature_config
else:
    feature_cols = feature_config["feature_columns"]

test_df = pd.read_csv(TEST_PATH)

missing_cols = [
    col for col in feature_cols
    if col not in test_df.columns
]

if missing_cols:
    raise ValueError(f"Missing features: {missing_cols}")

# Preserve exact feature order used during training
X_test = test_df[feature_cols].copy()

X_test = X_test.apply(pd.to_numeric, errors="raise")

print(f"Test samples: {len(X_test)}")
print(f"Features: {len(feature_cols)}")


# ============================================================
# 3. CALCULATE SHAP VALUES
# ============================================================

print("Calculating SHAP values...")

dmatrix = xgb.DMatrix(
    X_test,
    feature_names=feature_cols
)

# Last column contains the expected/base prediction
contributions = model.predict(
    dmatrix,
    pred_contribs=True
)

shap_values = contributions[:, :-1]
base_values = contributions[:, -1]

predictions = model.predict(dmatrix)

# SHAP contributions must reconstruct model predictions
np.testing.assert_allclose(
    shap_values.sum(axis=1) + base_values,
    predictions,
    rtol=1e-5,
    atol=1e-3
)

explanation = shap.Explanation(
    values=shap_values,
    base_values=base_values,
    data=X_test.to_numpy(),
    feature_names=feature_cols
)

# Save raw SHAP contributions
shap_df = pd.DataFrame(
    shap_values,
    columns=feature_cols
)

shap_df["predicted_price_eur_mwh"] = predictions

shap_df.to_csv(
    OUTPUT_DIR / "shap_values_test.csv",
    index=False
)


# ============================================================
# 4. CLASSIFY FEATURES INTO MARKET DRIVER GROUPS
# ============================================================

def classify_feature(feature):

    name = feature.lower()

    # Historical electricity prices
    if (
        name.startswith("price_")
        or "day_ahead_price" in name
    ):
        return "Price History"

    # Calendar variables
    if (
        name in [
            "hour",
            "day_of_week",
            "dayofweek",
            "month",
            "day",
            "is_weekend",
            "is_holiday",
            "week",
            "week_of_year"
        ]
        or "calendar" in name
        or "weekend" in name
        or "holiday" in name
        or "hour_sin" in name
        or "hour_cos" in name
    ):
        return "Calendar"

    # Residual demand
    if "residual_load" in name:
        return "Residual Demand"

    # Load / demand
    if "load" in name or "demand" in name:
        return "Demand"

    # Renewable generation
    if any(
        word in name
        for word in [
            "wind",
            "solar",
            "renewable",
            "photovoltaic"
        ]
    ):
        return "Renewables"

    # Conventional generation
    if any(
        word in name
        for word in [
            "generation",
            "gas",
            "coal",
            "lignite",
            "nuclear",
            "hydro",
            "biomass",
            "oil",
            "fossil"
        ]
    ):
        return "Generation Mix"

    # Weather
    if any(
        word in name
        for word in [
            "temperature",
            "weather",
            "precip",
            "radiation",
            "dewpoint",
            "swvl",
            "t2m"
        ]
    ):
        return "Weather"

    return "Other"


# ============================================================
# 5. GLOBAL FEATURE IMPORTANCE
# ============================================================

importance = pd.DataFrame({
    "feature": feature_cols,
    "mean_abs_shap_eur_mwh": np.abs(
        shap_values
    ).mean(axis=0)
})

importance["group"] = importance["feature"].apply(
    classify_feature
)

importance = importance.sort_values(
    "mean_abs_shap_eur_mwh",
    ascending=False
).reset_index(drop=True)

importance["importance_pct"] = (
    importance["mean_abs_shap_eur_mwh"]
    / importance["mean_abs_shap_eur_mwh"].sum()
    * 100
)

importance.to_csv(
    OUTPUT_DIR / "feature_importance.csv",
    index=False
)

print("\nTop 15 Features:")
print(importance.head(15).to_string(index=False))


# ============================================================
# 6. VISUALIZATION 1: FEATURE IMPORTANCE
# ============================================================

shap.plots.bar(
    explanation,
    max_display=20,
    show=False
)

plt.title("XGBoost - Global SHAP Feature Importance")

plt.savefig(
    OUTPUT_DIR / "01_feature_importance.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close("all")


# ============================================================
# 7. VISUALIZATION 2: SHAP BEESWARM
# ============================================================

shap.plots.beeswarm(
    explanation,
    max_display=20,
    show=False
)

plt.title("SHAP Summary - Day-Ahead Price Forecast")

plt.savefig(
    OUTPUT_DIR / "02_beeswarm.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close("all")


# ============================================================
# 8. VISUALIZATION 3: MARKET DRIVER GROUPS
# ============================================================

group_importance = (
    importance
    .groupby("group", as_index=False)[
        "mean_abs_shap_eur_mwh"
    ]
    .sum()
    .sort_values(
        "mean_abs_shap_eur_mwh",
        ascending=False
    )
)

group_importance["importance_pct"] = (
    group_importance["mean_abs_shap_eur_mwh"]
    / group_importance["mean_abs_shap_eur_mwh"].sum()
    * 100
)

group_importance.to_csv(
    OUTPUT_DIR / "market_driver_importance.csv",
    index=False
)

fig, ax = plt.subplots(figsize=(10, 6))

plot_data = group_importance.sort_values(
    "importance_pct"
)

ax.barh(
    plot_data["group"],
    plot_data["importance_pct"]
)

ax.set_xlabel("Share of Total Absolute SHAP (%)")
ax.set_title("Market Driver Importance")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "03_market_drivers.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 9. VISUALIZATION 4: NON-PRICE MARKET DRIVERS
# ============================================================

market_features = importance[
    ~importance["group"].isin([
        "Price History",
        "Calendar"
    ])
].copy()

top_market = market_features.head(15)

fig, ax = plt.subplots(figsize=(10, 7))

plot_data = top_market.iloc[::-1]

ax.barh(
    plot_data["feature"],
    plot_data["mean_abs_shap_eur_mwh"]
)

ax.set_xlabel("Mean |SHAP Value| (EUR/MWh)")
ax.set_title(
    "Non-Price Market Drivers of Day-Ahead Electricity Prices"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "04_non_price_drivers.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 10. VISUALIZATION 5: SHAP DEPENDENCE PLOTS
# ============================================================

# Select the three most important non-price features
top_features = market_features["feature"].head(3).tolist()

for i, feature in enumerate(top_features, start=1):

    shap.plots.scatter(
        explanation[:, feature],
        show=False
    )

    plt.title(f"SHAP Dependence - {feature}")

    plt.savefig(
        OUTPUT_DIR / f"05_dependence_{i}.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close("all")


# ============================================================
# 11. VISUALIZATION 6: WATERFALL
# ============================================================

# Explain the hour with the highest predicted price
sample_idx = int(np.argmax(predictions))

shap.plots.waterfall(
    explanation[sample_idx],
    max_display=15,
    show=False
)

plt.title(
    f"Highest Predicted Price: "
    f"{predictions[sample_idx]:.2f} EUR/MWh"
)

plt.savefig(
    OUTPUT_DIR / "06_waterfall.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close("all")


# ============================================================
# 12. SUMMARY
# ============================================================

print("\nMarket Driver Importance:")
print(group_importance.to_string(index=False))

print("\nTop Non-Price Market Drivers:")
print(market_features.head(10).to_string(index=False))

print(f"\nSHAP analysis completed.")
print(f"Results saved to: {OUTPUT_DIR}")