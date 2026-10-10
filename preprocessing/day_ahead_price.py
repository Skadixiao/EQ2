from pathlib import Path
import pandas as pd


# --------------------------------------------------
# Paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[1]

RAW_DIR = BASE_DIR / "data" / "raw" / "day_ahead_price"
OUTPUT_FILE = BASE_DIR / "data" / "processed" / "day_ahead_price.csv"


# --------------------------------------------------
# Configuration
# --------------------------------------------------

MTU_COL = "MTU (UTC)"
SEQ_COL = "Sequence"
PRICE_COL = "Day-ahead Price (EUR/MWh)"

CUTOFF = pd.Timestamp(
    "2025-10-01",
    tz="Europe/Berlin"
).tz_convert("UTC")


# --------------------------------------------------
# Preprocessing
# --------------------------------------------------

def preprocess_day_ahead_price():

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
                usecols=[MTU_COL, SEQ_COL, PRICE_COL],
                encoding="utf-8-sig"
            )

            # Keep Sequence 1
            df = df[df[SEQ_COL] == "Sequence 1"].copy()

            if df.empty:
                continue

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

            # Validate original timestamps
            if df["mtu_start_utc"].duplicated().any():
                raise ValueError(f"Duplicate MTU in {file.name}")

            df[PRICE_COL] = pd.to_numeric(
                df[PRICE_COL],
                errors="raise"
            )

            if df[PRICE_COL].isna().any():
                raise ValueError(f"Missing prices in {file.name}")

            # Convert timestamps to hourly resolution
            df["mtu_start_utc"] = df["mtu_start_utc"].dt.floor("h")

            grouped = df.groupby("mtu_start_utc")[PRICE_COL]

            # Validate hourly coverage
            counts = grouped.size()
            expected = counts.mode().iloc[0]

            if expected not in (1, 4) or (counts != expected).any():
                raise ValueError(
                    f"Incomplete or inconsistent hourly intervals in {file.name}"
                )

            # Check prices are identical within each hour
            if (grouped.nunique() != 1).any():
                raise ValueError(
                    f"Different prices within the same hour in {file.name}"
                )

            # One price per hour
            hourly = grouped.first().reset_index()

            hourly = hourly.rename(columns={
                PRICE_COL: "day_ahead_price_eur_mwh"
            })

            hourly = hourly.sort_values("mtu_start_utc")

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
    preprocess_day_ahead_price()