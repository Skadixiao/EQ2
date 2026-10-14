"""Backtest an hourly DE-LU day-ahead BESS arbitrage strategy.

Place this script at EQ2/bess/backtest.py, then run from EQ2:
    python bess/backtest.py

Input (from XGBoost holdout): outputs/xgboost_optuna/test_predictions.csv
Required columns:
    mtu_start_utc, day_ahead_price_eur_mwh, predicted_price_eur_mwh

Compare:
    1. XGBoost forecast -> schedule -> settled at actual prices
    2. Fixed night-charge / evening-discharge rule (no future price input)
    3. Perfect foresight (actual prices -> schedule; hindsight upper bound)
    4. No trading (€0)

This is an illustrative day-ahead gross-margin simulation, NOT proof of
actual exchange fills or a fully point-in-time/fees-inclusive trading result.
"""

from datetime import timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix


# ---------------------- Settings ----------------------
ROOT = Path(__file__).resolve().parents[1]  # EQ2/
INPUT_CSV = ROOT / "outputs" / "xgboost_optuna" / "test_predictions.csv"
OUTPUT_DIR = ROOT / "bess" / "results"

MARKET_TZ = "Europe/Berlin"
CAPACITY_MWH = 2.0
POWER_MW = 1.0
ROUND_TRIP_EFFICIENCY = 0.90
DEGRADATION_EUR_PER_MWH_DISCHARGED = 5.0
INITIAL_SOC_MWH = 0.0

TIME_COL = "mtu_start_utc"
ACTUAL_COL = "day_ahead_price_eur_mwh"
PRED_COL = "predicted_price_eur_mwh"


def optimize_day(price_signal, charge_allowed=None, discharge_allowed=None):
    """MILP: choose hourly charge/discharge using ONLY price_signal.

    Charge/discharge in MW, SOC in MWh. Charging and discharging cannot
    happen simultaneously. SOC begins AND ends at the same level each day.

    Here each input row is exactly one hour (DT = 1 h).
    """
    p = np.asarray(price_signal, dtype=float)
    n = len(p)
    dt = 1.0
    eta = np.sqrt(ROUND_TRIP_EFFICIENCY)
    if n < 1 or not np.isfinite(p).all():
        raise ValueError("Invalid or empty hourly price signal")

    # Decision vector: [charge[n], discharge[n], soc[n+1], charge_mode[n]]
    ch = slice(0, n)
    dis = slice(n, 2 * n)
    soc = slice(2 * n, 3 * n + 1)
    mode = slice(3 * n + 1, 4 * n + 1)
    nvars = 4 * n + 1

    # scipy.milp minimises, so negate expected trading margin.
    objective = np.zeros(nvars)
    objective[ch] = p * dt
    objective[dis] = -(p - DEGRADATION_EUR_PER_MWH_DISCHARGED) * dt

    low = np.zeros(nvars)
    high = np.zeros(nvars)
    high[ch] = POWER_MW
    high[dis] = POWER_MW
    high[soc] = CAPACITY_MWH
    high[mode] = 1

    if charge_allowed is not None:
        mask = np.asarray(charge_allowed, dtype=bool)
        if mask.size != n:
            raise ValueError("charge_allowed has the wrong length")
        high[ch] = np.where(mask, POWER_MW, 0.0)
    if discharge_allowed is not None:
        mask = np.asarray(discharge_allowed, dtype=bool)
        if mask.size != n:
            raise ValueError("discharge_allowed has the wrong length")
        high[dis] = np.where(mask, POWER_MW, 0.0)

    integer = np.zeros(nvars, dtype=int)
    integer[mode] = 1

    # Rows: n SOC transition equations + 2 boundary equations + 2n mode rules.
    mat = lil_matrix((3 * n + 2, nvars), dtype=float)
    lb = np.full(3 * n + 2, -np.inf)
    ub = np.full(3 * n + 2, np.inf)

    for t in range(n):
        # SOC[t+1] = SOC[t] + eta*charge[t]*dt - discharge[t]*dt/eta
        mat[t, soc.start + t + 1] = 1
        mat[t, soc.start + t] = -1
        mat[t, ch.start + t] = -eta * dt
        mat[t, dis.start + t] = dt / eta
        lb[t] = ub[t] = 0

        # charge[t] <= POWER_MW * mode[t]
        rc = n + 2 + t
        mat[rc, ch.start + t] = 1
        mat[rc, mode.start + t] = -POWER_MW
        ub[rc] = 0

        # discharge[t] <= POWER_MW * (1 - mode[t])
        rd = 2 * n + 2 + t
        mat[rd, dis.start + t] = 1
        mat[rd, mode.start + t] = POWER_MW
        ub[rd] = POWER_MW

    # Disallow carrying unvalued energy across day boundaries.
    mat[n, soc.start] = 1
    mat[n + 1, soc.stop - 1] = 1
    lb[n] = ub[n] = INITIAL_SOC_MWH
    lb[n + 1] = ub[n + 1] = INITIAL_SOC_MWH

    result = milp(
        c=objective,
        integrality=integer,
        bounds=Bounds(low, high),
        constraints=LinearConstraint(mat.tocsr(), lb, ub),
        options={"disp": False},
    )
    if not result.success:
        raise RuntimeError(f"MILP failed: {result.message}")

    charge_mw = np.clip(result.x[ch], 0, POWER_MW)
    discharge_mw = np.clip(result.x[dis], 0, POWER_MW)
    soc_mwh = np.clip(result.x[soc][1:], 0, CAPACITY_MWH)
    return charge_mw, discharge_mw, soc_mwh


def actual_pnl(actual_prices, charge_mw, discharge_mw):
    """Cash settlement at actual price minus assumed discharge degradation."""
    return (
        (discharge_mw - charge_mw) * np.asarray(actual_prices)
        - DEGRADATION_EUR_PER_MWH_DISCHARGED * discharge_mw
    )  # each timestep is 1 hour, so MW * 1h = MWh


def complete_delivery_day(frame, date):
    """Validate full Berlin-local delivery day: 23 / 24 / 25 hourly MTUs."""
    local_start = pd.Timestamp(date).tz_localize(MARKET_TZ)
    local_end = pd.Timestamp(date + timedelta(days=1)).tz_localize(MARKET_TZ)
    expected_utc = pd.date_range(
        local_start, local_end, freq="h", inclusive="left"
    ).tz_convert("UTC")
    observed_utc = pd.DatetimeIndex(frame[TIME_COL])
    return observed_utc.equals(expected_utc)


def load_prices():
    if not INPUT_CSV.is_file():
        raise FileNotFoundError(
            f"Cannot find {INPUT_CSV}\nChange INPUT_CSV at the top of this script."
        )
    data = pd.read_csv(INPUT_CSV)
    missing = {TIME_COL, ACTUAL_COL, PRED_COL} - set(data.columns)
    if missing:
        raise ValueError(f"Input is missing columns: {sorted(missing)}")

    data[TIME_COL] = pd.to_datetime(data[TIME_COL], utc=True, errors="raise")
    for col in (ACTUAL_COL, PRED_COL):
        data[col] = pd.to_numeric(data[col], errors="raise")
    if data[[TIME_COL, ACTUAL_COL, PRED_COL]].isna().any().any():
        raise ValueError("Null timestamps or prices found")
    if not np.isfinite(data[[ACTUAL_COL, PRED_COL]].to_numpy()).all():
        raise ValueError("Infinite prices found")
    if data[TIME_COL].duplicated().any():
        raise ValueError("Duplicate UTC timestamps found")
    if (data[TIME_COL].dt.minute != 0).any() or (data[TIME_COL].dt.second != 0).any():
        raise ValueError("Expected 1-hour MTUs (pre-October 2025 data)")

    data = data.sort_values(TIME_COL).reset_index(drop=True)
    data["delivery_date"] = data[TIME_COL].dt.tz_convert(MARKET_TZ).dt.date
    return data


def max_drawdown_eur(daily_pnl):
    cumulative = np.r_[0, np.asarray(daily_pnl, dtype=float).cumsum()]
    drawdowns = cumulative - np.maximum.accumulate(cumulative)
    return float(drawdowns.min())


def main():
    prices = load_prices()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    daily_rows = []
    hourly_rows = []
    skipped = []

    for date, day in prices.groupby("delivery_date", sort=True):
        day = day.copy().reset_index(drop=True)
        if not complete_delivery_day(day, date):
            skipped.append(str(date))
            continue

        actual = day[ACTUAL_COL].to_numpy(dtype=float)
        forecast = day[PRED_COL].to_numpy(dtype=float)
        hours = day[TIME_COL].dt.tz_convert(MARKET_TZ).dt.hour.to_numpy()

        # Trading decisions are made from available signals, never actual prices.
        xgb = optimize_day(forecast)

        # Rule: charge 00:00-05:59 and discharge 17:00-20:59, Berlin local.
        night = hours < 6
        evening = (hours >= 17) & (hours < 21)
        fake_rule_prices = np.where(night, 0.0, np.where(evening, 100.0, 50.0))
        rule = optimize_day(fake_rule_prices, night, evening)

        # Hindsight-only theoretical upper bound (not a tradable strategy).
        oracle = optimize_day(actual)

        row = {"delivery_date": str(date), "hours": len(day)}
        for name, schedule in (("xgb", xgb), ("rule", rule), ("oracle", oracle)):
            charge, discharge, soc = schedule
            hourly_pnl = actual_pnl(actual, charge, discharge)
            day[f"{name}_charge_mw"] = charge
            day[f"{name}_discharge_mw"] = discharge
            day[f"{name}_soc_mwh"] = soc
            day[f"{name}_pnl_eur"] = hourly_pnl
            row[f"{name}_pnl_eur"] = float(hourly_pnl.sum())
            row[f"{name}_discharged_mwh"] = float(discharge.sum())

        row["no_trade_pnl_eur"] = 0.0
        row["xgb_vs_rule_eur"] = row["xgb_pnl_eur"] - row["rule_pnl_eur"]
        row["oracle_gap_eur"] = row["oracle_pnl_eur"] - row["xgb_pnl_eur"]
        row["xgb_expected_pnl_eur"] = float(
            actual_pnl(forecast, xgb[0], xgb[1]).sum()
        )
        daily_rows.append(row)
        hourly_rows.append(day)

    if not daily_rows:
        raise ValueError("No complete Berlin-local days found in input CSV")

    daily = pd.DataFrame(daily_rows)
    hourly = pd.concat(hourly_rows, ignore_index=True)
    strategies = ("xgb", "rule", "oracle", "no_trade")
    for name in strategies:
        daily[f"{name}_cumulative_eur"] = daily[f"{name}_pnl_eur"].cumsum()

    daily.to_csv(OUTPUT_DIR / "daily_pnl.csv", index=False)
    hourly.to_csv(OUTPUT_DIR / "hourly_dispatch.csv", index=False)

    summary_rows = []
    for name in strategies:
        pnl = daily[f"{name}_pnl_eur"]
        summary_rows.append({
            "strategy": name,
            "total_pnl_eur": float(pnl.sum()),
            "average_daily_pnl_eur": float(pnl.mean()),
            "profitable_days_pct": float((pnl > 0).mean() * 100),
            "max_drawdown_eur": max_drawdown_eur(pnl),
        })
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUTPUT_DIR / "summary.csv", index=False)

    fig, ax = plt.subplots(figsize=(11, 6))
    for name, label in (("xgb", "XGBoost"), ("rule", "Night/evening rule"),
                        ("oracle", "Perfect foresight (upper bound)"),
                        ("no_trade", "No trading")):
        ax.plot(pd.to_datetime(daily["delivery_date"]),
                daily[f"{name}_cumulative_eur"], label=label)
    ax.set(title="DE-LU BESS Arbitrage: Realised-price Simulated P&L",
           xlabel="Berlin delivery date", ylabel="Cumulative P&L (EUR)")
    ax.axhline(0, linewidth=0.7, color="grey")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "cumulative_pnl.png", dpi=160)
    plt.close(fig)

    print(f"Backtested {len(daily)} complete delivery days.")
    if skipped:
        print(f"Skipped {len(skipped)} incomplete days: {', '.join(skipped[:6])}")
    print("\n", summary.round(2).to_string(index=False))
    print(f"\nXGBoost - rule benchmark: EUR {daily['xgb_vs_rule_eur'].sum():,.2f}")
    oracle_total = daily["oracle_pnl_eur"].sum()
    if oracle_total > 0:
        print(f"Oracle profit capture: {daily['xgb_pnl_eur'].sum()/oracle_total:.1%}")
    print(f"\nFiles saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
