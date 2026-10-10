from pathlib import Path
import pandas as pd
import re


# --------------------------------------------------
# Paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[1]

RAW_DIR = BASE_DIR / "data" / "raw" / "renewable_forecast"
OUTPUT_FILE = BASE_DIR / "data" / "processed" / "renewable_forecast.csv"


# --------------------------------------------------
# Configuration
# --------------------------------------------------

MTU_COL = "MTU (UTC)"
FORECAST_COL = "Day-ahead (MW)"

TECHNOLOGIES = {
    "offshore": "offshore_wind_forecast_mw",
    "onshore": "onshore_wind_forecast_mw",
    "solar": "solar_forecast_mw"
}

CUTOFF = pd.Timestamp(
    "2025-10-01",
    tz="Europe/Berlin"
).tz_convert("UTC")


# --------------------------------------------------
# Preprocessing
# --------------------------------------------------

def preprocess_renewable_forecast():

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    pattern = re.compile(
        r"^(offshore|onshore|solar)(\d{4})\.csv$"
    )

    files_by_year = {}

    for file in RAW_DIR.glob("*.csv"):

        match = pattern.match(file.name)

        if not match:
            continue

        technology, year = match.groups()
        year = int(year)

        if year > CUTOFF.year:
            continue

        files_by_year.setdefault(year, {})[technology] = file

    if not files_by_year:
        raise FileNotFoundError(
            f"No renewable CSV files found in {RAW_DIR}"
        )

    temp_file = OUTPUT_FILE.with_suffix(".tmp")
    temp_file.unlink(missing_ok=True)

    first_file = True
    previous_end = None

    try:
        for year in sorted(files_by_year):

            print(f"\nProcessing {year}...")

            missing_files = set(TECHNOLOGIES) - set(
                files_by_year[year]
            )

            if missing_files:
                raise FileNotFoundError(
                    f"Missing technologies for {year}: {missing_files}"
                )

            hourly_series = []

            for technology, output_col in TECHNOLOGIES.items():

                file = files_by_year[year][technology]

                print(f"  Reading {file.name}...")

                df = pd.read_csv(
                    file,
                    usecols=[MTU_COL, FORECAST_COL],
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
                    raise ValueError(
                        f"No data before cutoff in {file.name}"
                    )

                if df["mtu_start_utc"].duplicated().any():
                    raise ValueError(
                        f"Duplicate MTU in {file.name}"
                    )

                # Convert forecast to numeric
                df[FORECAST_COL] = pd.to_numeric(
                    df[FORECAST_COL],
                    errors="coerce"
                )

                # Convert to hourly resolution
                df["mtu_start_utc"] = df["mtu_start_utc"].dt.floor("h")

                grouped = df.groupby("mtu_start_utc")[FORECAST_COL]

                # Validate complete hours
                counts = grouped.size()
                expected = counts.mode().iloc[0]

                if expected not in (1, 4) or (counts != expected).any():
                    raise ValueError(
                        f"Incomplete hourly intervals in {file.name}"
                    )

                # Hourly average (MW)
                hourly = grouped.mean()

                # Preserve missing values
                valid_counts = grouped.count()

                hourly.loc[
                    valid_counts != counts
                ] = float("nan")

                hourly.name = output_col

                hourly_series.append(hourly)

            # Merge three hourly forecast series
            merged = pd.concat(
                hourly_series,
                axis=1,
                join="outer"
            )

            merged = merged.sort_index().reset_index()

            # Check overlap between years
            if previous_end is not None:
                if merged["mtu_start_utc"].iloc[0] <= previous_end:
                    raise ValueError(
                        f"Overlapping timestamps in {year}"
                    )

            previous_end = merged["mtu_start_utc"].iloc[-1]

            # Check missing hours
            gaps = merged["mtu_start_utc"].diff().dropna()

            missing_hours = int(
                (gaps / pd.Timedelta(hours=1) - 1).clip(lower=0).sum()
            )

            if missing_hours:
                print(f"  Warning: {missing_hours} missing hours")

            # Report missing values
            missing = merged.isna().sum()

            if missing.any():
                print("  Warning: missing values:")
                print(missing[missing > 0])

            merged.to_csv(
                temp_file,
                mode="w" if first_file else "a",
                header=first_file,
                index=False,
                date_format="%Y-%m-%dT%H:%M:%SZ"
            )

            first_file = False

            print(f"  Saved {len(merged):,} hourly rows")

        if first_file:
            raise ValueError("No valid renewable forecast data found.")

        temp_file.replace(OUTPUT_FILE)

        print(f"\nSaved to {OUTPUT_FILE}")

    finally:
        temp_file.unlink(missing_ok=True)


if __name__ == "__main__":
    preprocess_renewable_forecast()