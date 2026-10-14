"""Energy Quant | XGBoost forecasting, SHAP explainability, BESS, backtesting.

From EQ2 root:
    pip install -r dashboard_requirements.txt
    streamlit run dashboard.py

Expects files produced by train_xgboost_optuna.py and bess/backtest.py:
    outputs/xgboost_optuna/test_predictions.csv
    bess/results/hourly_dispatch.csv
    bess/results/daily_pnl.csv

Also contains an explicitly labelled synthetic example mode.

Optional: interactive SHAP via XGBoost native Tree SHAP (no shap package needed),
using outputs/xgboost_optuna/xgboost_model.json + test_features_for_shap.csv.
The dashboard READS existing outputs; it never runs the optimiser or refits ML.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st


ROOT = Path(__file__).resolve().parent
MARKET_TZ = "Europe/Berlin"
TIME = "mtu_start_utc"
ACTUAL = "day_ahead_price_eur_mwh"
FORECAST = "predicted_price_eur_mwh"

COLORS = {
    "xgb": "#20C6B7",
    "rule": "#E7AC54",
    "oracle": "#A696F0",
    "no_trade": "#8191A8",
    "actual": "#F3F6FA",
    "predicted": "#20C6B7",
    "negative": "#F27D87",
}
NAMES = {
    "xgb": "XGBoost strategy",
    "rule": "Fixed-time rule",
    "oracle": "Perfect foresight · hindsight",
    "no_trade": "No trading",
}
STRATEGIES = ["xgb", "rule", "oracle", "no_trade"]

st.set_page_config(
    page_title="Energy Quant | Price → BESS → P&L",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(
    """
    <style>
      .block-container {max-width: 1470px; padding-top: 1.6rem; padding-bottom: 2rem;}
      h1, h2, h3 {letter-spacing: -0.025em;}
      div[data-testid="stMetric"] {background:rgba(121,146,180,.08);
          padding: 14px 16px; border: 1px solid rgba(121,146,180,.17);
          border-radius: 12px; min-height: 115px;}
      div[data-testid="stMetricLabel"] {font-weight: 550;}
      .subline {color:#95a4b7; font-size:.93rem; padding:.1rem 0 .8rem;}
      .flow {border:1px solid rgba(121,146,180,.18); border-radius:13px;
          padding:18px 20px; background:rgba(121,146,180,.06); min-height:102px;}
      .flow strong {color:#20C6B7; font-size:.8rem; text-transform:uppercase; letter-spacing:.08em;}
      .flow p {margin:.4rem 0 0; font-size:1rem; font-weight:600;}
      .note {font-size:.86rem;color:#9AABB8;}
      div[data-testid="stDataFrame"] {border-radius: 10px;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def load_csv(file_path: str, mtime: float) -> pd.DataFrame:
    """mtime in key ensures rerunning after the backtest refreshes the UI."""
    return pd.read_csv(file_path)


def read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(str(path))
    return load_csv(str(path), path.stat().st_mtime).copy()


def first_existing(*paths: Path) -> Path:
    for candidate in paths:
        if candidate.is_file():
            return candidate
    return paths[0]


def validate_prices(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    required = {TIME, ACTUAL, FORECAST}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{name}: missing columns {sorted(missing)}")
    data = frame.copy()
    data[TIME] = pd.to_datetime(data[TIME], errors="raise", utc=True)
    for col in (ACTUAL, FORECAST):
        data[col] = pd.to_numeric(data[col], errors="raise")
    if data[[TIME, ACTUAL, FORECAST]].isna().any().any():
        raise ValueError(f"{name}: null timestamp/price values")
    if not np.isfinite(data[[ACTUAL, FORECAST]].to_numpy()).all():
        raise ValueError(f"{name}: non-finite prices")
    if data[TIME].duplicated().any():
        raise ValueError(f"{name}: duplicate UTC timestamps")
    data["delivery_date"] = data[TIME].dt.tz_convert(MARKET_TZ).dt.date
    return data.sort_values(TIME).reset_index(drop=True)


def validate_daily(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"delivery_date"} | {f"{s}_pnl_eur" for s in STRATEGIES}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"daily_pnl.csv: missing columns {sorted(missing)}")
    data = frame.copy()
    data["delivery_date"] = pd.to_datetime(data["delivery_date"], errors="raise").dt.date
    if data["delivery_date"].isna().any() or data["delivery_date"].duplicated().any():
        raise ValueError("daily_pnl.csv has missing or duplicate delivery dates")
    for strat in STRATEGIES:
        key = f"{strat}_pnl_eur"
        data[key] = pd.to_numeric(data[key], errors="raise")
        if not np.isfinite(data[key]).all():
            raise ValueError(f"daily_pnl.csv: invalid P&L in {key}")
    return data.sort_values("delivery_date").reset_index(drop=True)


def validate_dispatch(frame: pd.DataFrame) -> pd.DataFrame:
    required = {TIME, ACTUAL, FORECAST}
    for strat in ("xgb", "rule", "oracle"):
        required |= {f"{strat}_{s}" for s in ("charge_mw", "discharge_mw", "soc_mwh", "pnl_eur")}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"hourly_dispatch.csv: missing columns {sorted(missing)}")
    data = validate_prices(frame, "hourly_dispatch.csv")
    for col in required - {TIME, ACTUAL, FORECAST}:
        data[col] = pd.to_numeric(data[col], errors="raise")
        if not np.isfinite(data[col]).all():
            raise ValueError(f"hourly_dispatch.csv: non-finite numbers in {col}")
    return data


def format_eur(value: float) -> str:
    return f"€{value:,.0f}" if value >= 0 else f"−€{abs(value):,.0f}"


def series_drawdown(values: pd.Series) -> float:
    curve = np.r_[0.0, values.to_numpy(dtype=float).cumsum()]
    return float((curve - np.maximum.accumulate(curve)).min())


def style_figure(fig: go.Figure, height: int = 400) -> go.Figure:
    fig.update_layout(
        template="plotly_dark",
        height=height,
        margin=dict(l=10, r=12, t=35, b=15),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#B5C1CF", size=12),
        legend=dict(orientation="h", y=1.13, x=0, bgcolor="rgba(0,0,0,0)"),
        hovermode="x unified",
    )
    fig.update_xaxes(gridcolor="rgba(147,167,185,.10)", zeroline=False)
    fig.update_yaxes(gridcolor="rgba(147,167,185,.12)", zeroline=False)
    return fig


def day_axis(frame: pd.DataFrame) -> tuple[list[int], list[str], list[str]]:
    """Categorical index avoids hiding the repeated CET hour on DST Sundays."""
    berlin = frame[TIME].dt.tz_convert(MARKET_TZ)
    positions = list(range(len(frame)))
    ticks = [t.strftime("%H:%M") for t in berlin]
    labels = [t.strftime("%d %b %Y %H:%M %Z") for t in berlin]
    return positions, ticks, labels


def forecast_stats(frame: pd.DataFrame) -> dict[str, float]:
    error = frame[FORECAST].to_numpy() - frame[ACTUAL].to_numpy()
    return {
        "MAE": float(np.mean(np.abs(error))),
        "RMSE": float(np.sqrt(np.mean(error**2))),
        "Bias": float(np.mean(error)),
    }


def figure_cumulative(daily: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    x = daily["delivery_date"].astype(str)
    for strat in STRATEGIES:
        fig.add_trace(go.Scatter(
            x=x,
            y=daily[f"{strat}_pnl_eur"].cumsum(),
            mode="lines",
            name=NAMES[strat],
            line=dict(color=COLORS[strat], width=3 if strat == "xgb" else 2,
                      dash="dash" if strat == "oracle" else "solid"),
            hovertemplate="%{x}<br>€%{y:,.2f}<extra>%{fullData.name}</extra>",
        ))
    fig.update_yaxes(title_text="Cumulative P&L (€)", tickprefix="€")
    fig.update_xaxes(title_text="Berlin delivery date", type="date")
    fig.add_hline(y=0, line_color="rgba(190,200,215,.4)", line_width=1)
    return style_figure(fig, 425)


def figure_total(daily: pd.DataFrame) -> go.Figure:
    totals = [daily[f"{s}_pnl_eur"].sum() for s in STRATEGIES]
    fig = go.Figure(go.Bar(
        x=[NAMES[s] for s in STRATEGIES], y=totals,
        marker_color=[COLORS[s] for s in STRATEGIES],
        text=[format_eur(float(x)) for x in totals],
        textposition="outside", cliponaxis=False,
        hovertemplate="%{x}<br>%{y:,.2f} €<extra></extra>",
    ))
    fig.update_yaxes(title_text="Total P&L (€)")
    return style_figure(fig, 345)


def figure_daily_price(frame: pd.DataFrame) -> go.Figure:
    positions, ticks, labels = day_axis(frame)
    fig = go.Figure()
    for col, name, color in ((ACTUAL, "Actual market price", COLORS["actual"]),
                             (FORECAST, "XGBoost forecast", COLORS["predicted"])):
        fig.add_trace(go.Scatter(
            x=positions, y=frame[col], name=name, mode="lines+markers",
            line=dict(color=color, width=3 if col == ACTUAL else 2),
            marker=dict(size=5), customdata=labels,
            hovertemplate="%{customdata}<br>€%{y:.2f}/MWh<extra>%{fullData.name}</extra>",
        ))
    fig.update_xaxes(tickmode="array", tickvals=positions[::3],
                     ticktext=ticks[::3], title="Berlin local delivery hour")
    fig.update_yaxes(title_text="Price (€/MWh)")
    return style_figure(fig, 360)


def figure_scatter(frame: pd.DataFrame) -> go.Figure:
    if len(frame) > 12000:
        frame = frame.sample(12000, random_state=42)
    lower = float(min(frame[ACTUAL].min(), frame[FORECAST].min()))
    upper = float(max(frame[ACTUAL].max(), frame[FORECAST].max()))
    fig = go.Figure()
    fig.add_trace(go.Scattergl(
        x=frame[ACTUAL], y=frame[FORECAST], mode="markers",
        name="Hourly observations",
        marker=dict(size=4, opacity=.38, color=COLORS["xgb"]),
        hovertemplate="Actual €%{x:.1f}<br>Forecast €%{y:.1f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(x=[lower, upper], y=[lower, upper],
                             mode="lines", name="Perfect prediction",
                             line=dict(color=COLORS["actual"], dash="dash", width=1)))
    fig.update_xaxes(title="Actual (€/MWh)")
    fig.update_yaxes(title="Predicted (€/MWh)")
    return style_figure(fig, 365)


def figure_errors(frame: pd.DataFrame) -> go.Figure:
    residuals = frame[FORECAST] - frame[ACTUAL]
    fig = go.Figure(go.Histogram(x=residuals, nbinsx=45,
                                 marker_color=COLORS["xgb"], opacity=.8,
                                 hovertemplate="Forecast − actual: %{x:.1f}<br>Hours: %{y}<extra></extra>"))
    fig.add_vline(x=0, line_color=COLORS["actual"], line_dash="dash")
    fig.update_xaxes(title="Prediction error: forecast − actual (€/MWh)")
    fig.update_yaxes(title="Number of hours")
    return style_figure(fig, 365)


def figure_dispatch(day: pd.DataFrame, strategy: str) -> go.Figure:
    positions, ticks, labels = day_axis(day)
    charge = day[f"{strategy}_charge_mw"].to_numpy()
    discharge = day[f"{strategy}_discharge_mw"].to_numpy()
    soc = day[f"{strategy}_soc_mwh"].to_numpy()
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True, vertical_spacing=.08,
        row_heights=[.43, .30, .27],
        subplot_titles=("Market prices", "Battery dispatch", "State of charge"),
    )
    for col, name, color in ((ACTUAL, "Actual", COLORS["actual"]),
                             (FORECAST, "Forecast", COLORS["predicted"])):
        fig.add_trace(go.Scatter(
            x=positions, y=day[col], name=name, mode="lines+markers",
            line=dict(color=color, width=2.5), marker=dict(size=4),
            customdata=labels,
            hovertemplate="%{customdata}<br>€%{y:.2f}/MWh<extra>%{fullData.name}</extra>",
        ), row=1, col=1)
    fig.add_trace(go.Bar(x=positions, y=-charge, name="Charging (−MW)",
                         marker_color=COLORS["rule"], customdata=labels,
                         hovertemplate="%{customdata}<br>Charging: %{y:.2f} MW<extra></extra>"), row=2, col=1)
    fig.add_trace(go.Bar(x=positions, y=discharge, name="Discharging (+MW)",
                         marker_color=COLORS["xgb"], customdata=labels,
                         hovertemplate="%{customdata}<br>Discharging: %{y:.2f} MW<extra></extra>"), row=2, col=1)
    fig.add_trace(go.Scatter(x=positions, y=soc, name="SOC (MWh)",
                             line=dict(color=COLORS["oracle"], width=2),
                             fill="tozeroy", customdata=labels,
                             hovertemplate="%{customdata}<br>SOC: %{y:.2f} MWh<extra></extra>"), row=3, col=1)
    fig.update_layout(barmode="relative")
    fig.update_yaxes(title_text="€/MWh", row=1, col=1)
    fig.update_yaxes(title_text="MW", row=2, col=1)
    fig.update_yaxes(title_text="MWh", row=3, col=1, rangemode="tozero")
    fig.update_xaxes(tickmode="array", tickvals=positions[::2],
                     ticktext=ticks[::2], title_text="Berlin delivery hour", row=3, col=1)
    return style_figure(fig, 690)


def figure_daily_pnl(daily: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for strat in ("xgb", "rule"):
        fig.add_trace(go.Bar(
            x=daily["delivery_date"].astype(str), y=daily[f"{strat}_pnl_eur"],
            name=NAMES[strat], marker_color=COLORS[strat], opacity=.90,
            hovertemplate="%{x}<br>€%{y:,.2f}<extra>%{fullData.name}</extra>",
        ))
    fig.update_layout(barmode="group")
    fig.update_yaxes(title_text="Daily P&L (€)")
    fig.update_xaxes(title_text="Berlin delivery date", type="date")
    return style_figure(fig, 380)


def figure_drawdown(daily: pd.DataFrame) -> go.Figure:
    cum = np.r_[0., daily["xgb_pnl_eur"].to_numpy(dtype=float).cumsum()]
    dd = cum - np.maximum.accumulate(cum)
    fig = go.Figure(go.Scatter(
        x=daily["delivery_date"].astype(str), y=dd[1:],
        name="XGBoost drawdown", mode="lines", fill="tozeroy",
        line=dict(color=COLORS["negative"], width=2),
        hovertemplate="%{x}<br>€%{y:,.2f}<extra></extra>",
    ))
    fig.update_xaxes(title_text="Berlin delivery date", type="date")
    fig.update_yaxes(title_text="Drawdown from running P&L peak (€)")
    return style_figure(fig, 310)


def summary_table(daily: pd.DataFrame) -> pd.DataFrame:
    out = []
    for s in STRATEGIES:
        pnl = daily[f"{s}_pnl_eur"]
        out.append({
            "Strategy": NAMES[s],
            "Total P&L (€)": pnl.sum(),
            "Avg/day (€)": pnl.mean(),
            "Profitable days (%)": 100.0 * (pnl > 0).mean(),
            "Max drawdown (€)": series_drawdown(pnl),
        })
    return pd.DataFrame(out)


# ---------------------- Forecast diagnostics & interpretability ----------------------
@st.cache_data(show_spinner=False)
def load_json(path_text: str, modified: float) -> dict:
    with open(path_text, encoding="utf-8") as file:
        return json.load(file)


def optional_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        value = load_json(str(path), path.stat().st_mtime)
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError, TypeError):
        return {}


def figure_hourly_mae(frame: pd.DataFrame) -> go.Figure:
    tmp = frame.copy()
    tmp["hour_berlin"] = tmp[TIME].dt.tz_convert(MARKET_TZ).dt.hour
    tmp["absolute_error"] = (tmp[FORECAST] - tmp[ACTUAL]).abs()
    results = tmp.groupby("hour_berlin", as_index=False)["absolute_error"].mean()
    fig = go.Figure(go.Bar(
        x=results["hour_berlin"], y=results["absolute_error"],
        marker_color=COLORS["xgb"], name="Hourly MAE",
        hovertemplate="Berlin hour %{x:02d}:00<br>MAE %{y:.2f} €/MWh<extra></extra>",
    ))
    fig.update_xaxes(title_text="Delivery hour (Europe/Berlin)", dtick=2)
    fig.update_yaxes(title_text="MAE (€/MWh)")
    return style_figure(fig, 340)


def figure_daily_mae(frame: pd.DataFrame) -> go.Figure:
    tmp = frame.copy()
    tmp["absolute_error"] = (tmp[FORECAST] - tmp[ACTUAL]).abs()
    results = tmp.groupby("delivery_date", as_index=False)["absolute_error"].mean()
    fig = go.Figure(go.Scatter(
        x=results["delivery_date"].astype(str), y=results["absolute_error"],
        name="Daily MAE", mode="lines", fill="tozeroy",
        line=dict(color=COLORS["xgb"], width=2.5),
        hovertemplate="%{x}<br>MAE %{y:.2f} €/MWh<extra></extra>",
    ))
    fig.update_xaxes(type="date", title_text="Berlin delivery date")
    fig.update_yaxes(title_text="MAE (€/MWh)")
    return style_figure(fig, 340)


def figure_accuracy_comparison(metadata: dict) -> go.Figure | None:
    rows = []
    for key, label in (("validation_metrics", "Validation"),
                       ("test_metrics", "Test · XGBoost"),
                       ("test_last_week_naive_metrics", "Test · last-week price")):
        item = metadata.get(key)
        if not isinstance(item, dict):
            continue
        value = next((float(v) for k, v in item.items()
                      if k.casefold() in ("mae", "mae_eur_mwh") and v is not None), None)
        if value is not None and np.isfinite(value):
            rows.append((label, value))
    if not rows:
        return None
    fig = go.Figure(go.Bar(
        x=[r[0] for r in rows], y=[r[1] for r in rows],
        marker_color=[COLORS["oracle"], COLORS["xgb"], COLORS["rule"]][:len(rows)],
        text=[f"{r[1]:.2f}" for r in rows], textposition="outside",
    ))
    fig.update_yaxes(title_text="MAE (€/MWh)")
    return style_figure(fig, 325)


def figure_optuna_trials(trials: pd.DataFrame) -> go.Figure | None:
    if not {"number", "value"}.issubset(trials.columns):
        return None
    tmp = trials.copy()
    tmp["number"] = pd.to_numeric(tmp["number"], errors="coerce")
    tmp["value"] = pd.to_numeric(tmp["value"], errors="coerce")
    if "state" in tmp.columns:
        tmp = tmp[tmp["state"].astype(str).str.upper() == "COMPLETE"]
    tmp = tmp.dropna(subset=["number", "value"]).sort_values("number")
    if tmp.empty:
        return None
    tmp["best"] = tmp["value"].cummin()
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=tmp["number"] + 1, y=tmp["value"], mode="markers", name="Trial CV MAE",
        marker=dict(color=COLORS["rule"], size=7),
    ))
    fig.add_trace(go.Scatter(
        x=tmp["number"] + 1, y=tmp["best"], mode="lines", name="Best so far",
        line=dict(color=COLORS["xgb"], width=3),
    ))
    fig.update_xaxes(title_text="Optuna trial", dtick=max(1, len(tmp) // 15))
    fig.update_yaxes(title_text="Training CV MAE (€/MWh)")
    return style_figure(fig, 355)


def png_gallery_paths(base: Path) -> list[Path]:
    """Find arbitrary SHAP filenames nested inside the project's shap folder."""
    choices = []
    for sub in (base / "shap", base / "figures"):
        if sub.is_dir():
            choices.extend(sub.rglob("*.png"))
            choices.extend(sub.rglob("*.jpg"))
    # Handle e.g. outputs/xgboost_optuna/shap_beeswarm.png.
    choices.extend(base.glob("*shap*.png"))
    return sorted(set(choices), key=lambda p: str(p).casefold())


@st.cache_data(show_spinner="Loading Tree SHAP values …")
def compute_xgb_shap(model_file: str, model_mtime: float,
                     feature_file: str, feature_mtime: float,
                     schema_file: str, schema_mtime: float,
                     max_rows: int = 4000) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Explain saved holdout examples via XGBoost's native exact Tree SHAP.

    No recomputation of forecast predictions/fit. Sampling spans the full test
    interval, deterministic with a hard row cap for responsive portfolio use.
    """
    import xgboost as xgb  # optional runtime package

    with open(schema_file, encoding="utf-8") as file:
        features = json.load(file)
    if not isinstance(features, list) or not features or not all(isinstance(c, str) for c in features):
        raise ValueError("feature_columns.json must be a list of feature names")
    original = pd.read_csv(feature_file)
    absent = [c for c in features if c not in original.columns]
    if absent:
        raise ValueError(f"SHAP features missing: {absent[:8]}")
    if TIME not in original:
        raise ValueError(f"{feature_file} needs a {TIME} column")
    original[TIME] = pd.to_datetime(original[TIME], utc=True, errors="raise")
    if original[TIME].duplicated().any():
        raise ValueError("Duplicate UTC times in test_features_for_shap.csv")
    original = original.sort_values(TIME).reset_index(drop=True)
    if original.empty:
        raise ValueError("test_features_for_shap.csv has no rows")
    if len(original) > max_rows:
        take = np.linspace(0, len(original) - 1, max_rows, dtype=int)
        original = original.iloc[take].reset_index(drop=True)
    x = original[features].apply(pd.to_numeric, errors="raise")
    booster = xgb.Booster()
    booster.load_model(model_file)
    dmatrix = xgb.DMatrix(x, feature_names=features)
    values = np.asarray(booster.predict(dmatrix, pred_contribs=True), dtype=float)
    if values.ndim != 2 or values.shape[1] != len(features) + 1:
        raise ValueError("Unexpected XGBoost SHAP contribution shape")
    contributions = pd.DataFrame(values[:, :-1], columns=features)
    contributions.insert(0, TIME, original[TIME].to_numpy())
    contributions["delivery_date"] = original[TIME].dt.tz_convert(MARKET_TZ).dt.date
    contributions["base_value_eur_mwh"] = values[:, -1]
    contributions["model_prediction_eur_mwh"] = values.sum(axis=1)
    summary = pd.DataFrame({
        "Feature": features,
        "Mean |SHAP| (€/MWh)": np.abs(values[:, :-1]).mean(axis=0),
        "Mean SHAP (€/MWh)": values[:, :-1].mean(axis=0),
    }).sort_values("Mean |SHAP| (€/MWh)", ascending=False).reset_index(drop=True)
    return contributions, summary, {
        "sample_rows": len(original), "available_rows": len(pd.read_csv(feature_file, usecols=[TIME])),
        "feature_names": features,
    }


def figure_shap_importance(importance: pd.DataFrame, top_n: int = 15) -> go.Figure:
    values = importance.head(top_n).iloc[::-1]
    fig = go.Figure(go.Bar(
        x=values["Mean |SHAP| (€/MWh)"], y=values["Feature"],
        orientation="h", marker_color=COLORS["xgb"],
        hovertemplate="%{y}<br>Mean |SHAP| %{x:.3f} €/MWh<extra></extra>",
    ))
    fig.update_xaxes(title_text="Mean absolute contribution (€/MWh)")
    fig.update_yaxes(title_text="")
    return style_figure(fig, max(360, 28 * len(values) + 90))


def figure_shap_dependence(contrib: pd.DataFrame, feature: str,
                           features: pd.DataFrame) -> go.Figure:
    xdata = features[[TIME, feature]].copy()
    xdata[TIME] = pd.to_datetime(xdata[TIME], utc=True)
    subset = contrib[[TIME, feature]].merge(xdata, on=TIME, suffixes=("_shap", "_value"))
    fig = go.Figure(go.Scattergl(
        x=subset[f"{feature}_value"], y=subset[f"{feature}_shap"],
        mode="markers", marker=dict(color=COLORS["xgb"], opacity=.5, size=6),
        hovertemplate="Feature value: %{x:.3f}<br>Contribution: %{y:+.3f} €/MWh<extra></extra>",
    ))
    fig.add_hline(y=0, line_dash="dash", line_color=COLORS["actual"])
    fig.update_xaxes(title_text=f"{feature} (original feature units)")
    fig.update_yaxes(title_text="SHAP contribution to forecast (€/MWh)")
    return style_figure(fig, 380)


def figure_shap_one_hour(row: pd.Series, feature_names: list[str], limit: int = 12) -> go.Figure:
    values = pd.Series({f: float(row[f]) for f in feature_names}).sort_values(key=abs, ascending=False)
    shown = values.iloc[:limit]
    if len(values) > limit:
        shown = pd.concat([shown, pd.Series({"Other features": values.iloc[limit:].sum()})])
    shown = shown.iloc[::-1]
    fig = go.Figure(go.Bar(
        y=shown.index.tolist(), x=shown.to_numpy(), orientation="h",
        marker_color=[COLORS["xgb"] if v >= 0 else COLORS["negative"] for v in shown],
        hovertemplate="%{y}<br>%{x:+.3f} €/MWh<extra></extra>",
    ))
    fig.add_vline(x=0, line_color=COLORS["actual"], line_width=1)
    fig.update_xaxes(title_text="Contribution relative to model base value (€/MWh)")
    return style_figure(fig, max(390, 27 * len(shown) + 90))


# ---------------------- Data sources and date range ----------------------
project_prediction = first_existing(
    ROOT / "outputs" / "xgboost_optuna" / "test_predictions.csv",
    ROOT / "outputs" / "xgboost" / "test_predictions.csv",
)
project_model_dir = project_prediction.parent
project_hourly = ROOT / "bess" / "results" / "hourly_dispatch.csv"
project_daily = ROOT / "bess" / "results" / "daily_pnl.csv"
project_ready = project_hourly.is_file() and project_daily.is_file()
demo_dir = ROOT / "example_data"
demo_ready = all((demo_dir / x).is_file() for x in
                 ("test_predictions.csv", "hourly_dispatch.csv", "daily_pnl.csv"))

with st.sidebar:
    st.markdown("## ⚡ ENERGY QUANT")
    st.caption("DE-LU · Day-ahead BESS arbitrage")
    choices = ["Project files"] + (["Synthetic example"] if demo_ready else [])
    initial = 0 if project_ready else (1 if demo_ready else 0)
    source = st.radio("Data source", choices, index=initial,
                      help="Project files are the real outputs from your pipeline. "
                           "Synthetic example contains fabricated prices for UI testing only.")
    st.divider()

if source == "Project files":
    pred_path, hourly_path, daily_path = project_prediction, project_hourly, project_daily
else:
    pred_path = demo_dir / "test_predictions.csv"
    hourly_path = demo_dir / "hourly_dispatch.csv"
    daily_path = demo_dir / "daily_pnl.csv"

try:
    raw_hourly = read_csv(hourly_path)
    raw_daily = read_csv(daily_path)
    hourly = validate_dispatch(raw_hourly)
    daily = validate_daily(raw_daily)
    # A prediction file is optional: dispatch already contains actual + predicted.
    if pred_path.is_file():
        prediction = validate_prices(read_csv(pred_path), "test_predictions.csv")
    else:
        prediction = hourly[[TIME, ACTUAL, FORECAST]].copy()
        prediction["delivery_date"] = hourly["delivery_date"]
except (OSError, ValueError, KeyError) as exc:
    st.error(f"Cannot load dashboard data: {exc}")
    st.markdown(
        "Run `python bess/backtest.py` first, from the EQ2 root. Expected files: "
        "`bess/results/hourly_dispatch.csv` and `bess/results/daily_pnl.csv`. "
        "For the forecast tab, add `outputs/xgboost_optuna/test_predictions.csv` "
        "(otherwise hourly_dispatch is used)."
    )
    st.stop()

# Only present dates for which the backtest generated schedules.
days = set(daily["delivery_date"])
hourly_days = set(hourly["delivery_date"])
days = sorted(days.intersection(hourly_days))
if not days:
    st.error("The daily and hourly backtest files have no matching Berlin delivery dates.")
    st.stop()

with st.sidebar:
    dates = st.date_input("Backtest delivery date range",
                          value=(days[0], days[-1]),
                          min_value=days[0], max_value=days[-1],
                          help="All P&L and forecast KPIs use this selected window.")
    if not isinstance(dates, (list, tuple)) or len(dates) != 2:
        st.info("Select the start and end dates to continue.")
        st.stop()
    start, end = dates
    if start > end:
        st.error("The start date must be on/before the end date.")
        st.stop()
    st.caption(f"Market clock: {MARKET_TZ} (CET / CEST)")
    st.caption("Prices: EUR/MWh · Battery: MW / MWh · P&L: EUR")

view_daily = daily[(daily["delivery_date"] >= start) &
                   (daily["delivery_date"] <= end)].copy()
view_hourly = hourly[(hourly["delivery_date"] >= start) &
                     (hourly["delivery_date"] <= end)].copy()
view_prediction = prediction[(prediction["delivery_date"] >= start) &
                             (prediction["delivery_date"] <= end)].copy()

# Remove dates outside the joint hourly/daily backtest window (e.g., gaps in inputs).
common_dates = set(view_daily["delivery_date"]).intersection(view_hourly["delivery_date"])
view_daily = view_daily[view_daily["delivery_date"].isin(common_dates)].copy()
view_hourly = view_hourly[view_hourly["delivery_date"].isin(common_dates)].copy()
view_prediction = view_prediction[view_prediction["delivery_date"].isin(common_dates)].copy()
if view_daily.empty or view_prediction.empty:
    st.warning("No matching observations for this date range. Try expanding the range.")
    st.stop()

xgb_total = float(view_daily["xgb_pnl_eur"].sum())
rule_total = float(view_daily["rule_pnl_eur"].sum())
oracle_total = float(view_daily["oracle_pnl_eur"].sum())
uplift = xgb_total - rule_total
capture = (xgb_total / oracle_total * 100.0) if oracle_total > 0 else float("nan")
forecast_m = forecast_stats(view_prediction)

st.markdown("# Forecast → Explain → Optimise → Backtest")
st.markdown(
    "<div class='subline'>Germany–Luxembourg (DE-LU) day-ahead market &nbsp;·&nbsp; "
    "Hourly holdout &nbsp;·&nbsp; Forecast-based MILP dispatch &nbsp;·&nbsp; "
    "Historical-price settlement simulation</div>",
    unsafe_allow_html=True,
)
if source == "Synthetic example":
    st.warning("**SYNTHETIC DEMO DATA.** These are fabricated market prices and "
               "schedules to test the dashboard. Do not present these numbers as real trading results.")
else:
    st.info("**Historical simulation, not live trading.** Execution, trading fees, "
            "liquidity and point-in-time data availability are not fully modelled.", icon="ℹ️")

flow1, flow2, flow3, flow4 = st.columns(4, gap="small")
with flow1:
    st.markdown("<div class='flow'><strong>01 / Forecast</strong>"
                "<p>XGBoost day-ahead €/MWh</p></div>", unsafe_allow_html=True)
with flow2:
    st.markdown("<div class='flow'><strong>02 / Explain</strong>"
                "<p>Tree SHAP market drivers</p></div>", unsafe_allow_html=True)
with flow3:
    st.markdown("<div class='flow'><strong>03 / Optimise</strong>"
                "<p>MILP battery scheduling</p></div>", unsafe_allow_html=True)
with flow4:
    st.markdown("<div class='flow'><strong>04 / Backtest</strong>"
                "<p>Real prices → P&L / risk</p></div>", unsafe_allow_html=True)

st.write("")
k1, k2, k3, k4 = st.columns(4, gap="small")
k1.metric("XGBoost · realised P&L", format_eur(xgb_total),
          help="Settled at actual market prices; includes assumed discharge degradation cost.")
k2.metric("Value over fixed-time rule", format_eur(uplift),
          delta=format_eur(uplift) + " versus rule", delta_color="normal")
k3.metric("Theoretical profit captured", f"{capture:.1f}%" if np.isfinite(capture) else "n/a",
          help="XGBoost realised P&L divided by hindsight optimum. Not a trading benchmark.")
k4.metric("Forecast MAE", f"{forecast_m['MAE']:.1f} €/MWh",
          help="On aligned hourly prediction observations within the selected date window.")

st.caption(f"Selected: **{len(view_daily)} complete delivery days** · "
           f"**{len(view_hourly):,} hourly dispatch rows** · "
           f"**{len(view_prediction):,} forecast rows**. "
           "All P&L shown is backtested simulation, not executed trades.")

overview, forecasting, shap_tab, training_tab, dispatch, pnl_tab, methods = st.tabs([
    "01  Overview", "02  Forecast performance", "03  SHAP drivers",
    "04  Model training", "05  Battery dispatch", "06  P&L & risk",
    "07  Methodology",
])

with overview:
    st.subheader("Cumulative simulated trading performance")
    st.plotly_chart(figure_cumulative(view_daily), use_container_width=True)
    left, right = st.columns([1.05, .95], gap="large")
    with left:
        st.subheader("Total P&L by strategy")
        st.plotly_chart(figure_total(view_daily), use_container_width=True)
    with right:
        st.subheader("What this comparison means")
        st.markdown(
            "**XGBoost:** schedules based on predicted prices, settled at actual prices.\n\n"
            "**Fixed-time rule:** overnight charging and evening discharging; "
            "no look-ahead price signal.\n\n"
            "**Perfect foresight:** uses actual prices for scheduling. "
            "This is an **untradable hindsight upper bound**.\n\n"
            "**No trading:** zero P&L baseline."
        )
        st.caption("All active strategies use the same battery constraints "
                   "and daily SOC reset from the accompanying backtest.")

with forecasting:
    st.subheader("Out-of-sample price prediction")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("MAE", f"{forecast_m['MAE']:.2f} €/MWh")
    m2.metric("RMSE", f"{forecast_m['RMSE']:.2f} €/MWh")
    m3.metric("Mean forecast bias", f"{forecast_m['Bias']:+.2f} €/MWh")
    m4.metric("Observed hours", f"{len(view_prediction):,}")

    days_with_pred = sorted(set(view_prediction["delivery_date"]))
    picked = st.selectbox("Explore one German delivery day", days_with_pred,
                          index=len(days_with_pred) - 1, key="forecast_day",
                          format_func=lambda x: x.strftime("%d %b %Y"))
    pred_day = view_prediction[view_prediction["delivery_date"] == picked]
    st.plotly_chart(figure_daily_price(pred_day), use_container_width=True)

    left, right = st.columns(2, gap="large")
    with left:
        st.markdown("#### Forecast calibration")
        st.plotly_chart(figure_scatter(view_prediction), use_container_width=True)
    with right:
        st.markdown("#### Residual distribution")
        st.plotly_chart(figure_errors(view_prediction), use_container_width=True)

    q_low, q_high = view_prediction[ACTUAL].quantile([.05, .95])
    tails = view_prediction[(view_prediction[ACTUAL] <= q_low) |
                            (view_prediction[ACTUAL] >= q_high)]
    tail_mae = forecast_stats(tails)["MAE"] if len(tails) else float("nan")
    st.markdown("#### Extreme-price diagnostic")
    c1, c2, c3 = st.columns(3)
    c1.metric("Bottom 5% actual price", f"{q_low:.1f} €/MWh")
    c2.metric("Top 5% actual price", f"{q_high:.1f} €/MWh")
    c3.metric("MAE in both tails", f"{tail_mae:.1f} €/MWh")
    st.caption("Tail thresholds are computed **after the fact** from the selected sample. "
               "They diagnose performance, but are never used to choose a trading schedule.")

    metrics_path = project_model_dir / "metrics.json"
    with st.expander("Optional model training metadata"):
        if source == "Project files" and metrics_path.is_file():
            try:
                model_meta = json.loads(metrics_path.read_text(encoding="utf-8"))
                for k in ("train_dates", "validation_dates", "test_dates", "n_features",
                          "optuna_trials", "best_cv_mae_eur_mwh"):
                    if k in model_meta:
                        st.write(f"**{k.replace('_', ' ').title()}:** {model_meta[k]}")
                if "test_metrics" in model_meta:
                    st.json(model_meta["test_metrics"], expanded=False)
                st.caption("From saved training run; might cover a different window "
                           "than the date filter selected above.")
            except (OSError, ValueError) as exc:
                st.warning(f"Could not read model metrics: {exc}")
        else:
            st.caption("Optional: place `metrics.json` next to your XGBoost predictions.")

    st.markdown("#### Forecast errors through time")
    f_left, f_right = st.columns(2, gap="large")
    with f_left:
        st.plotly_chart(figure_daily_mae(view_prediction), use_container_width=True)
    with f_right:
        st.plotly_chart(figure_hourly_mae(view_prediction), use_container_width=True)

    st.markdown("#### Largest forecast misses in the selected backtest window")
    biggest = view_prediction[[TIME, "delivery_date", ACTUAL, FORECAST]].copy()
    biggest["Error (€/MWh)"] = biggest[FORECAST] - biggest[ACTUAL]
    biggest["Absolute error (€/MWh)"] = biggest["Error (€/MWh)"].abs()
    biggest["Berlin time"] = biggest[TIME].dt.tz_convert(MARKET_TZ).dt.strftime("%Y-%m-%d %H:%M %Z")
    biggest = biggest.sort_values("Absolute error (€/MWh)", ascending=False).head(20)
    st.dataframe(biggest[["Berlin time", ACTUAL, FORECAST, "Error (€/MWh)",
                          "Absolute error (€/MWh)"]], hide_index=True, use_container_width=True)

    if source == "Project files":
        fig_files = sorted((project_model_dir / "figures").glob("*.png")) if (project_model_dir / "figures").is_dir() else []
        if fig_files:
            with st.expander(f"Open saved forecast analysis figures ({len(fig_files)})"):
                chosen_forecast_fig = st.selectbox("Figure", fig_files, format_func=lambda p: p.stem.replace("_", " "))
                st.image(str(chosen_forecast_fig), caption=chosen_forecast_fig.name, use_container_width=True)

with shap_tab:
    st.subheader("What drives the XGBoost price forecast?")
    st.caption("SHAP decomposes the model's *predicted price* into a model baseline "
               "and contributions from input features. It does not establish "
               "causal drivers of the actual electricity market.")

    if source == "Project files":
        model_file = project_model_dir / "xgboost_model.json"
        schema_file = project_model_dir / "feature_columns.json"
        feature_file = project_model_dir / "test_features_for_shap.csv"
        images = [p for p in png_gallery_paths(project_model_dir)
                  if ("shap" in str(p.relative_to(project_model_dir)).lower())]
        ready_for_shap = all(p.is_file() for p in (model_file, schema_file, feature_file))
        if ready_for_shap:
            try:
                shap_rows, shap_importance, shap_info = compute_xgb_shap(
                    str(model_file), model_file.stat().st_mtime,
                    str(feature_file), feature_file.stat().st_mtime,
                    str(schema_file), schema_file.stat().st_mtime,
                )
                shap_rows = shap_rows[(shap_rows["delivery_date"] >= start) &
                                      (shap_rows["delivery_date"] <= end)].copy()
                # Compute importance for only the selected time window.
                feature_names = shap_info["feature_names"]
                if not shap_rows.empty:
                    matched = shap_rows[[TIME, "model_prediction_eur_mwh"]].merge(
                        prediction[[TIME, FORECAST]], on=TIME, how="inner")
                    if not matched.empty:
                        forecast_difference = (
                            matched["model_prediction_eur_mwh"] - matched[FORECAST]
                        ).abs()
                        if float(forecast_difference.max()) > 0.10:
                            st.warning(
                                f"SHAP model differs from test_predictions.csv "
                                f"(maximum difference: {forecast_difference.max():.2f} €/MWh). "
                                "Feature explanations below belong to the saved model, "
                                "which may not match the model used by the backtest."
                            )
                    scoped_imp = pd.DataFrame({
                        "Feature": feature_names,
                        "Mean |SHAP| (€/MWh)": shap_rows[feature_names].abs().mean().values,
                        "Mean SHAP (€/MWh)": shap_rows[feature_names].mean().values,
                    }).sort_values("Mean |SHAP| (€/MWh)", ascending=False)
                    st.caption(f"Computed with the **saved XGBoost model** and held-out features: "
                               f"{len(shap_rows):,} observations in selected window "
                               f"({shap_info['sample_rows']:,} sampled out of "
                               f"{shap_info['available_rows']:,} held-out rows overall).")
                    sc1, sc2 = st.columns([1.1, .9], gap="large")
                    with sc1:
                        st.markdown("#### Global SHAP importance")
                        top_k = st.slider("Top features", 1, min(25, len(scoped_imp)),
                                          min(15, len(scoped_imp)), key="shap_feature_count")
                        st.plotly_chart(figure_shap_importance(scoped_imp, top_k), use_container_width=True)
                    with sc2:
                        st.markdown("#### How to interpret")
                        st.markdown(
                            "**Mean |SHAP|:** average magnitude of impact on the model "
                            "prediction in €/MWh.\n\n"
                            "**Positive SHAP:** pushes the predicted price up.\n\n"
                            "**Negative SHAP:** pushes it down.\n\n"
                            "Correlated features complicate interpretation; "
                            "SHAP is not evidence of market causality."
                        )
                        st.dataframe(scoped_imp.head(12).round(3), hide_index=True,
                                     use_container_width=True)
                        st.download_button("Download SHAP importance",
                            scoped_imp.to_csv(index=False).encode("utf-8"),
                            "shap_importance_selected_window.csv", mime="text/csv")

                    st.markdown("#### Individual feature response")
                    feature_choice = st.selectbox("Feature to explore", scoped_imp["Feature"].tolist(), key="shap_feature")
                    raw_features = read_csv(feature_file)
                    st.plotly_chart(figure_shap_dependence(shap_rows, feature_choice, raw_features),
                                    use_container_width=True)
                    st.caption("X: observed feature value in held-out sample. Y: contribution "
                               "to the **forecast**, not to the actual market price.")

                    st.markdown("#### Explain an individual hourly forecast")
                    shap_rows = shap_rows.sort_values(TIME)
                    explain_timestamps = shap_rows[TIME].tolist()
                    target_hour = st.selectbox(
                        "Hour (UTC)", explain_timestamps,
                        index=len(explain_timestamps) - 1,
                        format_func=lambda t: pd.Timestamp(t).tz_convert(MARKET_TZ).strftime("%d %b %Y %H:%M %Z"),
                        key="shap_timestamp")
                    example = shap_rows[shap_rows[TIME] == target_hour].iloc[0]
                    matching = prediction.loc[prediction[TIME] == target_hour, FORECAST]
                    baseline = float(example["base_value_eur_mwh"])
                    pred_native = float(example["model_prediction_eur_mwh"])
                    cl1, cl2, cl3 = st.columns(3)
                    cl1.metric("SHAP model baseline", f"{baseline:.2f} €/MWh")
                    cl2.metric("Reconstructed prediction", f"{pred_native:.2f} €/MWh")
                    cl3.metric("Saved forecast", f"{float(matching.iloc[0]):.2f} €/MWh" if len(matching) else "n/a")
                    if len(matching) and abs(pred_native - float(matching.iloc[0])) > 0.10:
                        st.warning("Saved forecasts differ from current model predictions "
                                   "(>€0.10/MWh). Check that the XGBoost model and "
                                   "test_features_for_shap.csv correspond to test_predictions.csv.")
                    st.plotly_chart(figure_shap_one_hour(example, feature_names), use_container_width=True)
                    st.caption("Base value + sum of ALL SHAP contributions equals the "
                               "reconstructed model prediction (up to numerical precision). "
                               "Small features are aggregated under Other features.")
                else:
                    st.info("No SHAP feature observations within the selected date range.")
            except (OSError, ValueError, ImportError, RuntimeError, KeyError, TypeError) as exc:
                st.warning(f"Interactive SHAP unavailable: {exc}")
                st.caption("You can still view saved SHAP PNG figures below.")
        else:
            missing = [p.name for p in (model_file, schema_file, feature_file) if not p.is_file()]
            st.info("Interactive SHAP needs: " + ", ".join(missing) +
                    ". Saved SHAP figures can still be displayed below.")
        if images:
            st.markdown("#### Saved SHAP analysis figures")
            options = sorted(images, key=lambda p: p.name.casefold())
            chosen = st.selectbox("Saved SHAP figure", options,
                                  format_func=lambda p: p.relative_to(project_model_dir).as_posix(),
                                  key="saved_shap_image")
            st.image(str(chosen), caption=chosen.relative_to(project_model_dir).as_posix(),
                     use_container_width=True)
            with st.expander(f"View all {len(images)} saved SHAP plots"):
                group = st.columns(2)
                for index, img in enumerate(options[:24]):
                    with group[index % 2]:
                        st.image(str(img), caption=img.name, use_container_width=True)
        elif not ready_for_shap:
            st.caption("No SHAP PNGs found in `outputs/xgboost_optuna/shap/`. "
                       "Once generated, they will be discovered automatically.")
    else:
        st.info("Synthetic example does not include a fitted XGBoost model or SHAP values. "
                "Choose Project files to see genuine held-out feature contributions.")

with training_tab:
    st.subheader("XGBoost model training and validation")
    if source != "Project files":
        st.info("Training metadata is not part of the synthetic demo.")
    else:
        metadata = optional_json(project_model_dir / "metrics.json")
        if metadata:
            training_cols = st.columns(3)
            training_cols[0].metric("Input features", str(metadata.get("n_features", "n/a")))
            training_cols[1].metric("Optuna trials", str(metadata.get("optuna_trials", "n/a")))
            best_cv = metadata.get("best_cv_mae_eur_mwh")
            training_cols[2].metric("Best training CV MAE",
                f"{float(best_cv):.2f} €/MWh" if best_cv is not None else "n/a")
            for key, heading in (("train_dates", "Train"), ("validation_dates", "Validation"),
                                 ("test_dates", "Test")):
                if key in metadata:
                    st.write(f"**{heading}:** {metadata[key]}")
            result = figure_accuracy_comparison(metadata)
            if result:
                st.markdown("#### Validation and held-out test accuracy")
                st.plotly_chart(result, use_container_width=True)
                st.caption("The last-week price is a naïve FORECAST baseline "
                           "(different from the BESS fixed-time dispatch benchmark). "
                           "These saved metrics cover the full model evaluation windows.")
            for k in ("validation_metrics", "test_metrics", "test_last_week_naive_metrics"):
                if k in metadata:
                    with st.expander(k.replace("_", " ").title()):
                        st.json(metadata[k], expanded=True)
            with st.expander("Saved XGBoost best hyperparameters"):
                if "best_params" in metadata:
                    st.json(metadata["best_params"], expanded=True)
                else:
                    st.caption("No best_params entry in metrics.json")
        else:
            st.info("No metrics.json found in the selected model folder.")
        trials_path = project_model_dir / "optuna_trials.csv"
        if trials_path.is_file():
            st.markdown("#### Hyperparameter tuning with Optuna")
            try:
                trials_frame = read_csv(trials_path)
                trials_figure = figure_optuna_trials(trials_frame)
                if trials_figure:
                    st.plotly_chart(trials_figure, use_container_width=True)
                else:
                    st.caption("Could not chart Optuna trials: need `number`, `value` columns.")
                with st.expander("Raw Optuna trial results"):
                    st.dataframe(trials_frame, hide_index=True, use_container_width=True)
            except (OSError, ValueError) as exc:
                st.warning(f"Could not load Optuna file: {exc}")
        st.markdown("#### Data leakage controls / caveat")
        st.markdown("Chronological train/validation/test partitions and "
                    "pre-auction feature availability are separate requirements. "
                    "The model script reports temporal splits, but historical "
                    "as-of publication timestamps and revised data still need "
                    "verification before claiming a fully point-in-time backtest.")

with dispatch:
    st.subheader("Hourly optimisation decisions and settlement")
    day_choices = sorted(set(view_hourly["delivery_date"]))
    row1, row2 = st.columns([1, 1], gap="large")
    with row1:
        picked_day = st.selectbox("Delivery day", day_choices,
                                  index=len(day_choices) - 1, key="dispatch_day",
                                  format_func=lambda x: x.strftime("%d %b %Y"))
    with row2:
        picked_label = st.selectbox("Schedule", ["xgb", "rule", "oracle"],
                                    format_func=lambda x: NAMES[x])
    day = view_hourly[view_hourly["delivery_date"] == picked_day].sort_values(TIME)
    charge_mwh = float(day[f"{picked_label}_charge_mw"].sum())  # 1-hour MTUs
    discharge_mwh = float(day[f"{picked_label}_discharge_mw"].sum())
    pnl_eur = float(day[f"{picked_label}_pnl_eur"].sum())
    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Realised day P&L", format_eur(pnl_eur))
    d2.metric("Energy bought", f"{charge_mwh:.2f} MWh")
    d3.metric("Energy sold", f"{discharge_mwh:.2f} MWh")
    d4.metric("Peak stored energy", f"{day[f'{picked_label}_soc_mwh'].max():.2f} MWh")
    st.plotly_chart(figure_dispatch(day, picked_label), use_container_width=True)
    st.caption("Charging is shown as negative grid power. SOC is energy stored "
               "after each hour. The plot uses Berlin-local timestamps and preserves "
               "the duplicate 02:00 hour during autumn clock changes.")
    if picked_label == "xgb" and "xgb_expected_pnl_eur" in view_daily.columns:
        expected_rows = view_daily.loc[
            view_daily["delivery_date"] == picked_day, "xgb_expected_pnl_eur"
        ]
        if not expected_rows.empty:
            st.caption(f"Forecast-implied day margin: {format_eur(float(expected_rows.iloc[0]))} "
                       "(estimated before delivery). Compare with realised day P&L above; "
                       "both include the backtest's assumed degradation charge.")
    st.download_button("Download this day's hourly dispatch (CSV)",
                       data=day.to_csv(index=False).encode("utf-8"),
                       file_name=f"bess_dispatch_{picked_day}.csv", mime="text/csv")

with pnl_tab:
    st.subheader("Realised-price P&L, performance and risk")
    xgb = view_daily["xgb_pnl_eur"].astype(float)
    mm1, mm2, mm3, mm4 = st.columns(4)
    mm1.metric("Average daily P&L", format_eur(float(xgb.mean())))
    mm2.metric("Profitable days", f"{100 * (xgb > 0).mean():.1f}%")
    mm3.metric("Maximum drawdown", format_eur(series_drawdown(xgb)))
    mm4.metric("Days outperforming rule",
               f"{100 * (view_daily['xgb_pnl_eur'] > view_daily['rule_pnl_eur']).mean():.1f}%")
    st.markdown("#### Strategy scorecard")
    table = summary_table(view_daily)
    st.dataframe(table.style.format({
        "Total P&L (€)": "{:,.2f}", "Avg/day (€)": "{:,.2f}",
        "Profitable days (%)": "{:.1f}%", "Max drawdown (€)": "{:,.2f}",
    }), use_container_width=True, hide_index=True)
    st.markdown("#### Daily earnings comparison")
    st.plotly_chart(figure_daily_pnl(view_daily), use_container_width=True)
    st.markdown("#### Drawdown of XGBoost strategy")
    st.plotly_chart(figure_drawdown(view_daily), use_container_width=True)
    st.caption("Drawdown is calculated on the cumulative daily profit curve and "
               "starts from €0 for the selected window. This is absolute EUR "
               "drawdown, not portfolio-return drawdown or value-at-risk.")
    st.download_button("Download selected daily P&L (CSV)",
                       data=view_daily.to_csv(index=False).encode("utf-8"),
                       file_name="bess_daily_pnl_filtered.csv", mime="text/csv")

with methods:
    st.subheader("Model → optimisation → settlement")
    st.markdown(
        r"""
**1 · Price forecasting (at D 10:30 Europe/Berlin).** The XGBoost model estimates
hourly prices for the next German delivery day, D+1. Its input features must
have been available before the auction. The dashboard uses saved holdout forecasts;
it **does not retrain** the model.

**2 · Dispatch optimisation.** A daily mixed-integer linear programme (MILP)
chooses charging, discharging and state of charge from the **predicted prices**.
It obeys charge/discharge power, battery capacity, efficiency and no-simultaneous-
charge/discharge constraints. Starting and ending SOC are held equal per day.

**3 · Historical-price P&L.** Exactly the schedule chosen in step 2 is priced
using **actual** day-ahead clearing prices, plus an assumed degradation charge:

$$
\mathrm{P\&L}_t = P_t^{\mathrm{actual}}\,(E_t^{\mathrm{discharge}} - E_t^{\mathrm{charge}})
- c_{\mathrm{deg}}E_t^{\mathrm{discharge}}
$$

The backtest uses one-hour intervals, so energy in MWh equals power in MW multiplied
by one hour. Round-trip losses affect the SOC constraints; they are not subtracted
a second time in the P&L formula.

**4 · Benchmarks.** The fixed-time rule charges in local hours 00:00–05:59
and discharges in 17:00–20:59, subject to the same constraints. Perfect foresight
uses *actual future prices* to optimise retrospectively and is only a theoretical
upper bound; it is **not** a leakage-free investable strategy. No trading earns €0.
"""
    )
    st.markdown("#### Assumptions & limitations")
    st.markdown(
        """
- This is a **historical market-price simulation**, not actual trading P&L.
- Historical exchange order fills, transaction costs, grid/network charges,
  financing and bid restrictions are **not** modelled.
- Battery degradation is a simplified linear penalty; the original backtest
  uses **€5 per MWh discharged**, 2 MWh capacity, 1 MW power and 90% round-trip efficiency.
  If you change `backtest.py`, update these written assumptions accordingly.
- Energy **cannot be carried between days** in the current optimiser; each
  day starts and ends at the same SOC.
- The dashboard supports **hourly** products; check the time-resolution change
  before using data after September 2025.
- A chronological model split by itself does **not** prove point-in-time validity.
  Historical publication times and later ENTSO-E data revisions should be audited
  before claiming a leakage-free, live-replicable trading strategy.
- The oracle capture ratio can be negative if XGBoost loses money.
"""
    )
    st.markdown("#### Model explainability")
    st.caption("See the SHAP drivers tab for interactive Tree SHAP contribution "
               "charts and saved SHAP plots from `outputs/xgboost_optuna/shap/`.")

st.divider()
st.caption("Energy Quant Portfolio · DE-LU day-ahead power market · "
           "XGBoost forecast → MILP battery arbitrage → simulated P&L vs benchmarks")
