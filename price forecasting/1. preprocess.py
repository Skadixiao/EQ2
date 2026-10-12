"""Preprocess ENTSO-E raw CSV exports into four hourly datasets.

Put this file in EQ2/scripts/ and run: python scripts/preprocess_market_data.py
"""

from pathlib import Path
import re

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = BASE_DIR / "data" / "raw"
OUT_DIR = BASE_DIR / "data" / "processed"
MTU_COL = "MTU (UTC)"
CUTOFF = pd.Timestamp("2025-10-01", tz="Europe/Berlin").tz_convert("UTC")

GEN_TYPES = {
    "Biomass": "biomass_generation_mw",
    "Fossil Brown coal/Lignite": "lignite_generation_mw",
    "Fossil Gas": "gas_generation_mw",
    "Fossil Hard coal": "hard_coal_generation_mw",
    "Fossil Oil": "oil_generation_mw",
    "Hydro Pumped Storage": "pumped_storage_generation_mw",
}
RENEWABLE_TYPES = {
    "offshore": "offshore_wind_forecast_mw",
    "onshore": "onshore_wind_forecast_mw",
    "solar": "solar_forecast_mw",
}


def yearly_files(folder):
    files = sorted(
        (f for f in (RAW_DIR / folder).glob("*.csv")
         if re.fullmatch(r"\d{4}", f.stem) and int(f.stem) <= CUTOFF.year),
        key=lambda f: int(f.stem),
    )
    if not files:
        raise FileNotFoundError(f"No yearly CSV files in {RAW_DIR / folder}")
    return files


def prepare(df):
    """Parse the beginning of each MTU interval and apply the market cutoff."""
    df["mtu_start_utc"] = pd.to_datetime(
        df[MTU_COL].str.slice(0, 16),
        format="%d/%m/%Y %H:%M",
        utc=True,
    )
    return df.loc[df["mtu_start_utc"] < CUTOFF].copy()


def hourly_mean(df, columns, label):
    """Average 15-minute MW observations to hourly; preserve missing data."""
    if df["mtu_start_utc"].duplicated().any():
        raise ValueError(f"Duplicate MTU in {label}")

    df["mtu_start_utc"] = df["mtu_start_utc"].dt.floor("h")
    groups = df.groupby("mtu_start_utc")
    counts = groups.size()
    expected = counts.mode().iloc[0]
    if expected not in (1, 4) or (counts != expected).any():
        raise ValueError(f"Incomplete or inconsistent hourly intervals in {label}")

    result = groups[columns].mean()
    valid = groups[columns].count()
    return result.where(valid.eq(counts, axis=0))


def prices():
    price_col = "Day-ahead Price (EUR/MWh)"
    for file in yearly_files("day_ahead_price"):
        df = pd.read_csv(file, usecols=[MTU_COL, "Sequence", price_col], encoding="utf-8-sig")
        df = prepare(df.loc[df["Sequence"] == "Sequence 1"].copy())
        if df.empty:
            continue
        if df["mtu_start_utc"].duplicated().any():
            raise ValueError(f"Duplicate MTU in {file.name}")

        df[price_col] = pd.to_numeric(df[price_col], errors="raise")
        if df[price_col].isna().any():
            raise ValueError(f"Missing prices in {file.name}")

        df["mtu_start_utc"] = df["mtu_start_utc"].dt.floor("h")
        groups = df.groupby("mtu_start_utc")[price_col]
        counts = groups.size()
        expected = counts.mode().iloc[0]
        if expected not in (1, 4) or (counts != expected).any():
            raise ValueError(f"Incomplete hourly prices in {file.name}")
        if (groups.nunique() != 1).any():
            raise ValueError(f"Different prices within the same hour in {file.name}")

        hourly = groups.first().rename("day_ahead_price_eur_mwh").reset_index()
        yield file.name, hourly


def loads():
    names = {
        "Actual Total Load (MW)": "actual_total_load_mw",
        "Day-ahead Total Load Forecast (MW)": "day_ahead_load_forecast_mw",
    }
    for file in yearly_files("load_forecast"):
        df = prepare(pd.read_csv(file, usecols=[MTU_COL, *names], encoding="utf-8-sig"))
        if df.empty:
            continue
        df = df.rename(columns=names)
        for col in names.values():
            df[col] = pd.to_numeric(df[col], errors="coerce")
        hourly = hourly_mean(df, list(names.values()), file.name).reset_index()
        yield file.name, hourly


def renewables():
    folder = RAW_DIR / "renewable_forecast"
    by_year = {}
    for file in folder.glob("*.csv"):
        match = re.fullmatch(r"(offshore|onshore|solar)(\d{4})\.csv", file.name)
        if match and int(match.group(2)) <= CUTOFF.year:
            by_year.setdefault(int(match.group(2)), {})[match.group(1)] = file
    if not by_year:
        raise FileNotFoundError(f"No renewable CSV files in {folder}")

    for year, files in sorted(by_year.items()):
        if set(files) != set(RENEWABLE_TYPES):
            raise FileNotFoundError(f"Missing renewable files for {year}: {set(RENEWABLE_TYPES) - set(files)}")

        series = []
        for tech, name in RENEWABLE_TYPES.items():
            file = files[tech]
            df = prepare(pd.read_csv(file, usecols=[MTU_COL, "Day-ahead (MW)"], encoding="utf-8-sig"))
            if df.empty:
                raise ValueError(f"No data before cutoff in {file.name}")
            df["Day-ahead (MW)"] = pd.to_numeric(df["Day-ahead (MW)"], errors="coerce")
            hourly = hourly_mean(df, ["Day-ahead (MW)"], file.name)
            series.append(hourly["Day-ahead (MW)"].rename(name))

        yield str(year), pd.concat(series, axis=1, join="outer").sort_index().reset_index()


def generation():
    type_col, value_col = "Production Type", "Generation (MW)"
    for file in yearly_files("actual_generation_per_type"):
        chunks = []
        reader = pd.read_csv(
            file, usecols=[MTU_COL, type_col, value_col],
            encoding="utf-8-sig", chunksize=100_000,
        )
        for chunk in reader:
            chunk = chunk.loc[chunk[type_col].isin(GEN_TYPES)].copy()
            if not chunk.empty:
                chunk = prepare(chunk)
                if not chunk.empty:
                    chunks.append(chunk[["mtu_start_utc", type_col, value_col]])
        if not chunks:
            continue

        df = pd.concat(chunks, ignore_index=True)
        if df.duplicated(["mtu_start_utc", type_col]).any():
            raise ValueError(f"Duplicate MTU / production type in {file.name}")
        df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
        df["hour"] = df["mtu_start_utc"].dt.floor("h")

        grouped = (df.groupby(["hour", type_col])[value_col]
                   .agg(mean="mean", count="size", valid="count")
                   .reset_index())
        incomplete = (grouped["count"] != 4) | (grouped["valid"] != 4)
        if incomplete.any():
            print(f"  Warning: {incomplete.sum():,} incomplete generation hours in {file.name}")
            grouped.loc[incomplete, "mean"] = float("nan")

        hourly = (grouped.pivot(index="hour", columns=type_col, values="mean")
                  .reindex(columns=list(GEN_TYPES))
                  .rename(columns=GEN_TYPES))
        hourly.index.name = "mtu_start_utc"
        yield file.name, hourly.sort_index().reset_index()


def save_dataset(name, frames):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    output = OUT_DIR / f"{name}.csv"
    temp = output.with_suffix(".tmp")
    temp.unlink(missing_ok=True)
    first, previous_end, total = True, None, 0

    try:
        for label, df in frames:
            if df.empty:
                continue
            print(f"{name}: {label}")
            df = df.sort_values("mtu_start_utc")
            if df["mtu_start_utc"].duplicated().any():
                raise ValueError(f"Duplicate hourly timestamps in {label}")
            if previous_end is not None and df["mtu_start_utc"].iloc[0] <= previous_end:
                raise ValueError(f"Overlapping timestamps in {label}")
            previous_end = df["mtu_start_utc"].iloc[-1]

            gaps = df["mtu_start_utc"].diff().dropna()
            missing_hours = int((gaps / pd.Timedelta(hours=1) - 1).clip(lower=0).sum())
            if missing_hours:
                print(f"  Warning: {missing_hours} missing hours")
            missing_values = df.isna().sum()
            if missing_values.any():
                print(f"  Warning: missing values: {missing_values[missing_values > 0].to_dict()}")

            df.to_csv(temp, mode="w" if first else "a", header=first, index=False,
                      date_format="%Y-%m-%dT%H:%M:%SZ")
            first = False
            total += len(df)

        if first:
            raise ValueError(f"No valid data found for {name}")
        temp.replace(output)
        print(f"Saved {total:,} rows -> {output}\n")
    finally:
        temp.unlink(missing_ok=True)


if __name__ == "__main__":
    save_dataset("day_ahead_price", prices())
    save_dataset("load_forecast", loads())
    save_dataset("renewable_forecast", renewables())
    save_dataset("actual_generation_per_type", generation())
