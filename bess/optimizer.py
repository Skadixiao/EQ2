"""Day-ahead BESS dispatch optimizer for the German DE-LU market.

Decisions use *predicted* prices only. Actual prices are carried into the
output for a separate backtest; they are NEVER used by the optimizer.

Run from the EQ2 project root:
    python bess/optimizer.py --input outputs/xgboost_optuna/test_predictions.csv

Dependency: pip install numpy pandas scipy
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MARKET_TZ = "Europe/Berlin"

# Simple demo assumptions — edit here if desired.
CAPACITY_MWH = 1.0
MAX_POWER_MW = 0.5
ROUND_TRIP_EFFICIENCY = 0.90
INITIAL_SOC = 0.50              # Fraction of capacity at the start of EACH day
FINAL_SOC = 0.50                # Fraction of capacity at the end of EACH day
INTERVAL_HOURS = 1.0            # This portfolio dataset is hourly


def optimize_day(predicted_prices: np.ndarray) -> dict[str, np.ndarray]:
    """Maximise forecast revenue with a MILP (no simultaneous charge/discharge)."""
    n = len(predicted_prices)
    eta = np.sqrt(ROUND_TRIP_EFFICIENCY)  # eta_charge * eta_discharge = 0.90

    # Decision vector: [charge MW, discharge MW, SOC MWh, charge-mode binary].
    ch = np.arange(n)
    dis = np.arange(n, 2 * n)
    soc = np.arange(2 * n, 3 * n + 1)
    mode = np.arange(3 * n + 1, 4 * n + 1)
    size = 4 * n + 1

    # scipy.milp MINIMISES: charging costs - discharge proceeds.
    cost = np.zeros(size)
    cost[ch] = predicted_prices * INTERVAL_HOURS
    cost[dis] = -predicted_prices * INTERVAL_HOURS

    lower = np.zeros(size)
    upper = np.r_[np.full(2 * n, MAX_POWER_MW),
                  np.full(n + 1, CAPACITY_MWH), np.ones(n)]
    lower[soc[0]] = upper[soc[0]] = INITIAL_SOC * CAPACITY_MWH
    lower[soc[-1]] = upper[soc[-1]] = FINAL_SOC * CAPACITY_MWH

    integer = np.zeros(size)
    integer[mode] = 1  # Binary because its bounds are [0, 1].

    # Per hour: charge <= max_power * mode
    #           discharge <= max_power * (1 - mode)
    #           SOC_next = SOC + eta*charge*dt - discharge*dt/eta
    A = lil_matrix((3 * n, size))
    row_lower = np.full(3 * n, -np.inf)
    row_upper = np.zeros(3 * n)
    for t in range(n):
        A[t, ch[t]] = 1
        A[t, mode[t]] = -MAX_POWER_MW

        A[n + t, dis[t]] = 1
        A[n + t, mode[t]] = MAX_POWER_MW
        row_upper[n + t] = MAX_POWER_MW

        A[2 * n + t, soc[t + 1]] = 1
        A[2 * n + t, soc[t]] = -1
        A[2 * n + t, ch[t]] = -eta * INTERVAL_HOURS
        A[2 * n + t, dis[t]] = INTERVAL_HOURS / eta
        row_lower[2 * n + t] = 0  # SOC balance is equality.

    result = milp(
        c=cost,
        integrality=integer,
        bounds=Bounds(lower, upper),
        constraints=LinearConstraint(A.tocsr(), row_lower, row_upper),
        options={"mip_rel_gap": 1e-7},
    )
    if not result.success:
        raise RuntimeError(f"Battery optimisation failed: {result.message}")

    x = result.x
    return {
        "charge_mw": np.maximum(x[ch], 0),
        "discharge_mw": np.maximum(x[dis], 0),
        "soc_start_mwh": x[soc[:-1]],
        "soc_end_mwh": x[soc[1:]],
    }


def expected_hours(delivery_date: str) -> pd.DatetimeIndex:
    """Local midnight to midnight: 23, 24, or 25 hours on DST days."""
    day = pd.Timestamp(delivery_date)
    start = day.tz_localize(MARKET_TZ).tz_convert("UTC")
    end = (day + pd.Timedelta(days=1)).tz_localize(MARKET_TZ).tz_convert("UTC")
    return pd.date_range(start, end, freq="h", inclusive="left")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", type=Path,
        default=PROJECT_ROOT / "outputs/xgboost_optuna/test_predictions.csv",
        help="XGBoost out-of-sample predictions CSV",
    )
    parser.add_argument(
        "--output", type=Path,
        default=PROJECT_ROOT / "bess/dispatch_schedule.csv",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    required = {"mtu_start_utc", "predicted_price_eur_mwh", "day_ahead_price_eur_mwh"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in input CSV: {sorted(missing)}")

    df["mtu_start_utc"] = pd.to_datetime(df["mtu_start_utc"], utc=True)
    if df["mtu_start_utc"].duplicated().any():
        raise ValueError("Duplicate UTC timestamps detected; check your input CSV.")
    for col in ("predicted_price_eur_mwh", "day_ahead_price_eur_mwh"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.sort_values("mtu_start_utc").copy()
    # Derive the correct German LOCAL delivery date, not the UTC calendar date.
    df["delivery_date"] = df["mtu_start_utc"].dt.tz_convert(MARKET_TZ).dt.strftime("%Y-%m-%d")

    schedules, skipped = [], []
    for delivery_date, day in df.groupby("delivery_date", sort=True):
        day = day.copy()
        times = pd.DatetimeIndex(day["mtu_start_utc"])
        if not times.equals(expected_hours(delivery_date)) or day[
            ["predicted_price_eur_mwh", "day_ahead_price_eur_mwh"]
        ].isna().any().any():
            skipped.append(delivery_date)
            continue  # Never optimise an incomplete local delivery day.

        decision = optimize_day(day["predicted_price_eur_mwh"].to_numpy(dtype=float))
        for name, values in decision.items():
            day[name] = values

        day["charge_mwh"] = day["charge_mw"] * INTERVAL_HOURS
        day["discharge_mwh"] = day["discharge_mw"] * INTERVAL_HOURS
        day["net_export_mwh"] = day["discharge_mwh"] - day["charge_mwh"]
        day["soc_end_pct"] = 100 * day["soc_end_mwh"] / CAPACITY_MWH
        day["forecast_cashflow_eur"] = day["predicted_price_eur_mwh"] * day["net_export_mwh"]
        schedules.append(day[[
            "mtu_start_utc", "delivery_date", "predicted_price_eur_mwh",
            "day_ahead_price_eur_mwh", "charge_mw", "discharge_mw",
            "charge_mwh", "discharge_mwh", "net_export_mwh",
            "soc_start_mwh", "soc_end_mwh", "soc_end_pct", "forecast_cashflow_eur",
        ]])

    if not schedules:
        raise ValueError("No complete German delivery days found. Check time resolution and input dates.")

    output = pd.concat(schedules, ignore_index=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    print(f"Saved {len(output):,} hourly dispatch rows ({len(schedules)} days) to: {args.output}")
    print(f"Total forecast gross P&L: EUR {output['forecast_cashflow_eur'].sum():,.2f}")
    if skipped:
        print(f"Skipped {len(skipped)} incomplete/invalid local dates: {', '.join(skipped[:8])}")
        if len(skipped) > 8:
            print("(Additional skipped dates not shown.)")


if __name__ == "__main__":
    main()
