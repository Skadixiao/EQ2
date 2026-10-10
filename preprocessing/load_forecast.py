from pathlib import Path
import pandas as pd


# --------------------------------------------------
# Paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[1]

RAW_DIR = BASE_DIR / "data" / "raw" / "load_forecast"
OUTPUT_FILE = BASE_DIR / "data" / "processed" / "load_forecast.csv"


# --------------------------------------------------
# Configuration
# --------------------------------------------------

MTU_COL = "MTU (UTC)"
ACTUAL_COL = "Actual Total Load (MW)"
FORECAST_COL = "Day-ahead Total Load Forecast (MW)"

ACTUAL_OUT = "actual_total_load_mw"
FORECAST_OUT = "day_ahead_load_forecast_mw"

CUTOFF = pd.Timestamp(
    "2025-10-01",
    tz="Europe/Berlin"
).tz_convert("UTC")


# --------------------------------------------------
# Preprocessing
# --------------------------------------------------

def preprocess_load_forecast():

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    files = sorted(
        [
            f for f in RAW_DIR.glob("*.csv")
            if (
                f.stem.isdigit()
                and len(f.stem) == 4
                and int(f.stem) <= CUTOFF.year
            )
        ],
        key=lambda f: int(f.stem)
    )

    if not files:
        raise FileNotFoundError(f"No CSV files found in {RAW_DIR}")

    temp_file = OUTPUT_FILE.with_suffix(".tmp")
    temp_file.unlink(missing_ok=True)

    first_file = True
    previous_end = None

    try:
        for file in files:

            print(f"Processing {file.name}...")

            df = pd.read_csv(
                file,
                usecols=[MTU_COL, ACTUAL_COL, FORECAST_COL],
                encoding="utf-8-sig"
            )

            # Parse UTC timestamp
            df["mtu_start_utc"] = pd.to_datetime(
                df[MTU_COL].str.slice(0, 16),
                format="%d/%m/%Y %H:%M",
                utc=True
            )

            # Apply cutoff
            df = df[df["mtu_start_utc"] < CUTOFF].copy()

            if df.empty:
                continue

            if df["mtu_start_utc"].duplicated().any():
                raise ValueError(f"Duplicate MTU in {file.name}")

            # Rename columns
            df = df.rename(columns={
                ACTUAL_COL: ACTUAL_OUT,
                FORECAST_COL: FORECAST_OUT
            })

            value_cols = [ACTUAL_OUT, FORECAST_OUT]

            # Convert to numeric
            for col in value_cols:
                df[col] = pd.to_numeric(
                    df[col],
                    errors="coerce"
                )

            # Convert to hourly resolution
            df["mtu_start_utc"] = df["mtu_start_utc"].dt.floor("h")

            grouped = df.groupby("mtu_start_utc")

            # Check complete hours
            counts = grouped.size()
            expected = counts.mode().iloc[0]

            if expected not in (1, 4) or (counts != expected).any():
                raise ValueError(
                    f"Incomplete or inconsistent hourly intervals in {file.name}"
                )

            # Hourly average (MW)
            hourly = grouped[value_cols].mean()

            # Keep missing values if any interval is missing
            valid_counts = grouped[value_cols].count()

            for col in value_cols:
                hourly.loc[
                    valid_counts[col] != counts, col
                ] = float("nan")

            hourly = hourly.reset_index()
            hourly = hourly.sort_values("mtu_start_utc")

            # Check overlap between files
            if previous_end is not None:
                if hourly["mtu_start_utc"].iloc[0] <= previous_end:
                    raise ValueError(f"Overlapping timestamps in {file.name}")

            previous_end = hourly["mtu_start_utc"].iloc[-1]

            # Check missing hours
            gaps = hourly["mtu_start_utc"].diff().dropna()
            missing_hours = int(
                (gaps / pd.Timedelta(hours=1) - 1).clip(lower=0).sum()
            )

            if missing_hours:
                print(f"  Warning: {missing_hours} missing hours")

            # Report missing values
            missing = hourly[value_cols].isna().sum()

            if missing.any():
                print(f"  Warning: missing values:\n{missing[missing > 0]}")

            hourly.to_csv(
                temp_file,
                mode="w" if first_file else "a",
                header=first_file,
                index=False,
                date_format="%Y-%m-%dT%H:%M:%SZ"
            )

            first_file = False

            print(f"  Saved {len(hourly):,} hourly rows")

        if first_file:
            raise ValueError("No valid data found.")

        temp_file.replace(OUTPUT_FILE)
        print(f"\nSaved to {OUTPUT_FILE}")

    finally:
        temp_file.unlink(missing_ok=True)


if __name__ == "__main__":
    preprocess_load_forecast()