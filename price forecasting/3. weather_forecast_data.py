"""ECMWF IFS historical day-ahead forecasts for DE-LU power price modelling.

Uses Open-Meteo Single Runs archive (ECMWF IFS HRES 9 km), NOT reanalysis.
One 00 UTC model run is selected on the day before each DE-LU delivery date.

Example:
  pip install pandas requests
  python retrieve_ecmwf_price_weather.py --start 2024-03-15 --end 2024-03-21
  python retrieve_ecmwf_price_weather.py --start 2024-03-15 --end 2025-09-30

Output: data/processed/ecmwf_forecast_hourly.csv
Merge with your model dataset on mtu_start_utc (parse as UTC).
"""

import argparse
import time
from pathlib import Path

import pandas as pd
import requests

BASE_DIR = Path(__file__).resolve().parents[1]
API = "https://single-runs-api.open-meteo.com/v1/forecast"
VARIABLES = ["temperature_2m", "wind_speed_100m", "shortwave_radiation", "cloud_cover"]
# Representative sample points, not grid-weighted countrywide generation forecasts.
SITES = {
    "kiel": (54.32, 10.12),
    "hamburg": (53.55, 10.00),
    "bremen": (53.08, 8.80),
    "hannover": (52.37, 9.73),
    "berlin": (52.52, 13.41),
    "leipzig": (51.34, 12.37),
    "frankfurt": (50.11, 8.68),
    "stuttgart": (48.78, 9.18),
    "munich": (48.14, 11.58),
}
NORTH = ["kiel", "hamburg", "bremen", "hannover"]
SOUTH = ["frankfurt", "stuttgart", "munich"]


def get_daily_forecast(session, delivery_date):
    """Return the one issue-time forecast for every hourly MTU on a local date."""
    run_date = delivery_date - pd.Timedelta(days=1)
    run = run_date.strftime("%Y-%m-%dT00:00")  # initialization UTC, before auction
    params = {
        "run": run,
        "models": "ecmwf_ifs",  # IFS HRES 9km, not the 0.25-degree open-data subset
        "latitude": ",".join(str(x[0]) for x in SITES.values()),
        "longitude": ",".join(str(x[1]) for x in SITES.values()),
        "hourly": ",".join(VARIABLES),
        "timezone": "GMT",
        "forecast_days": 3,
        "wind_speed_unit": "ms",
        "temperature_unit": "celsius",
    }
    response = session.get(API, params=params, timeout=90)
    response.raise_for_status()
    sites_json = response.json()
    if isinstance(sites_json, dict):
        sites_json = [sites_json]
    if len(sites_json) != len(SITES):
        raise ValueError(f"Expected {len(SITES)} sites; got {len(sites_json)} for {run}")

    joined = None
    for (name, _), site_json in zip(SITES.items(), sites_json):
        h = site_json.get("hourly", {})
        if not all(key in h for key in ["time", *VARIABLES]):
            raise ValueError(f"Missing ECMWF weather variable in {name}, run {run}: {site_json}")
        series = pd.DataFrame(h)
        series["mtu_start_utc"] = pd.to_datetime(series.pop("time"), utc=True)
        series = series.set_index("mtu_start_utc").sort_index()
        # Open-Meteo shortwave_radiation is the *preceding* hour's average.
        # Radiation stamped 14:00 corresponds to delivery MTU [13:00, 14:00).
        series["shortwave_radiation"] = series["shortwave_radiation"].shift(-1)
        series = series.rename(columns={k: f"{name}_{k}" for k in VARIABLES})
        joined = series if joined is None else joined.join(series, how="inner")

    local_date = joined.index.tz_convert("Europe/Berlin").date
    joined = joined[local_date == delivery_date.date()].copy()
    expected_hours = len(pd.date_range(
        delivery_date.tz_localize("Europe/Berlin"),
        (delivery_date + pd.Timedelta(days=1)).tz_localize("Europe/Berlin"),
        freq="h", inclusive="left",
    ))
    if len(joined) != expected_hours:
        raise ValueError(f"Incomplete delivery date {delivery_date.date()}: "
                         f"found {len(joined)} hours, expected {expected_hours}")

    def avg(names, field):
        return joined[[f"{x}_{field}" for x in names]].mean(axis=1)

    out = pd.DataFrame(index=joined.index)
    out["temp_de_mean_c"] = avg(SITES, "temperature_2m")
    out["temp_de_min_c"] = joined[[f"{x}_temperature_2m" for x in SITES]].min(axis=1)
    out["wind_north_100m_ms"] = avg(NORTH, "wind_speed_100m")
    out["wind_de_mean_100m_ms"] = avg(SITES, "wind_speed_100m")
    out["solar_south_wm2"] = avg(SOUTH, "shortwave_radiation")
    out["solar_de_mean_wm2"] = avg(SITES, "shortwave_radiation")
    out["cloud_de_mean_pct"] = avg(SITES, "cloud_cover")
    out["temp_daily_mean_c"] = out["temp_de_mean_c"].mean()
    out["temp_daily_min_c"] = out["temp_de_mean_c"].min()
    out["temp_daily_max_c"] = out["temp_de_mean_c"].max()
    out["delivery_date_de"] = delivery_date.strftime("%Y-%m-%d")
    out["forecast_run_utc"] = run + ":00Z"
    out = out.reset_index()
    out["mtu_start_utc"] = out["mtu_start_utc"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    if out[list(out.columns.difference(["mtu_start_utc", "delivery_date_de", "forecast_run_utc"]))].isna().any().any():
        raise ValueError(f"Nulls in delivered forecast features for {delivery_date.date()}")
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2024-03-15", help="First German LOCAL delivery date")
    parser.add_argument("--end", default="2024-03-21", help="Last German LOCAL delivery date")
    parser.add_argument("--out", default="data/processed/ecmwf_forecast_hourly.csv")
    args = parser.parse_args()
    if pd.Timestamp(args.start) < pd.Timestamp("2024-03-15"):
        parser.error("ECMWF IFS individual runs are archived from 2024-03-14; "
                     "the first supported day-ahead delivery date is 2024-03-15.")
    dates = pd.date_range(args.start, args.end, freq="D")
    if dates.empty:
        parser.error("--end must be on or after --start")

    output = Path(args.out)
    if not output.is_absolute():
        output = BASE_DIR / output
    output.parent.mkdir(parents=True, exist_ok=True)
    
    # Incremental output: interrupted multi-month downloads can be resumed.
    completed = set()
    if output.exists():
        existing = pd.read_csv(output, usecols=["delivery_date_de"])
        completed = set(existing["delivery_date_de"].astype(str))

    with requests.Session() as session:
        for date in dates:
            label = date.strftime("%Y-%m-%d")
            if label in completed:
                print(f"Skipping {label} (already saved)")
                continue
            print(f"Retrieving ECMWF 00Z run for delivery {label}...")
            df = get_daily_forecast(session, date)
            df.to_csv(output, mode="a", header=not output.exists(), index=False)
            print(f"Saved {len(df)} MTUs for {label}")
            time.sleep(0.5)  # modest API request pacing
    print(f"Weather features saved in {output}")


if __name__ == "__main__":
    main()
