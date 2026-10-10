from pathlib import Path
import pandas as pd


# --------------------------------------------------
# Paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[1]

RAW_DIR = BASE_DIR / "data" / "raw" / "actual_generation_per_type"

OUTPUT_FILE = (
    BASE_DIR / "data" / "processed"
    / "actual_generation_per_type.csv"
)


# --------------------------------------------------
# Configuration
# --------------------------------------------------

MTU_COL = "MTU (UTC)"
TYPE_COL = "Production Type"
GEN_COL = "Generation (MW)"

# Keep selected generation types
TECHNOLOGIES = {
    "Biomass": "biomass_generation_mw",
    "Fossil Brown coal/Lignite": "lignite_generation_mw",
    "Fossil Gas": "gas_generation_mw",
    "Fossil Hard coal": "hard_coal_generation_mw",
    "Fossil Oil": "oil_generation_mw",
    "Hydro Pumped Storage": "pumped_storage_generation_mw"
}

# Exclude delivery periods from 1 Oct 2025
# German market local time
CUTOFF = pd.Timestamp(
    "2025-10-01",
    tz="Europe/Berlin"
).tz_convert("UTC")

CHUNK_SIZE = 100_000


# --------------------------------------------------
# Preprocessing
# --------------------------------------------------

def preprocess_actual_generation():

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # Automatically discover yearly CSV files
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
        raise FileNotFoundError(
            f"No yearly CSV files found in {RAW_DIR}"
        )

    temp_file = OUTPUT_FILE.with_suffix(".tmp")
    temp_file.unlink(missing_ok=True)

    first_file = True
    previous_end = None
    total_rows = 0

    try:

        # ------------------------------------------
        # Process one year at a time
        # ------------------------------------------

        for file in files:

            print(f"\nProcessing {file.name}...")

            filtered_chunks = []

            # Read large CSV in chunks
            reader = pd.read_csv(
                file,
                usecols=[
                    MTU_COL,
                    TYPE_COL,
                    GEN_COL
                ],
                encoding="utf-8-sig",
                chunksize=CHUNK_SIZE
            )

            for chunk in reader:

                # Keep selected generation types
                chunk = chunk[
                    chunk[TYPE_COL].isin(TECHNOLOGIES)
                ].copy()

                if chunk.empty:
                    continue

                # Extract MTU start time
                mtu_start = (
                    chunk[MTU_COL]
                    .str.split(" - ", n=1)
                    .str[0]
                    .str.strip()
                    .str.slice(0, 16)
                )

                chunk["mtu_start_utc"] = pd.to_datetime(
                    mtu_start,
                    format="%d/%m/%Y %H:%M",
                    utc=True,
                    errors="raise"
                )

                # Apply cutoff date
                chunk = chunk[
                    chunk["mtu_start_utc"] < CUTOFF
                ].copy()

                if chunk.empty:
                    continue

                # Convert generation to numeric
                # "n/e" and invalid values become NaN
                chunk[GEN_COL] = pd.to_numeric(
                    chunk[GEN_COL],
                    errors="coerce"
                )

                # Keep necessary columns
                chunk = chunk[
                    [
                        "mtu_start_utc",
                        TYPE_COL,
                        GEN_COL
                    ]
                ]

                filtered_chunks.append(chunk)

            if not filtered_chunks:
                print(f"  Skipping {file.name}: no valid data.")
                continue

            # Combine filtered chunks for this year
            df = pd.concat(
                filtered_chunks,
                ignore_index=True
            )

            print(f"  Selected rows: {len(df):,}")

            # ------------------------------------------
            # Validate original data
            # ------------------------------------------

            # Each technology should have at most
            # one record per 15-minute MTU
            duplicates = df.duplicated(
                subset=["mtu_start_utc", TYPE_COL]
            )

            if duplicates.any():
                raise ValueError(
                    f"Duplicate MTU/Production Type "
                    f"combinations in {file.name}"
                )

            # Report missing generation values
            missing_raw = df[GEN_COL].isna().sum()

            if missing_raw:
                print(
                    f"  Warning: {missing_raw:,} "
                    "missing generation values"
                )

            # ------------------------------------------
            # Convert 15-minute to hourly
            # ------------------------------------------

            df["hour"] = (
                df["mtu_start_utc"].dt.floor("h")
            )

            # Group by hour and generation type
            grouped = (
                df.groupby(
                    ["hour", TYPE_COL]
                )[GEN_COL]
                .agg(
                    mean="mean",
                    valid_count="count",
                    interval_count="size"
                )
                .reset_index()
            )

            # Require four complete 15-minute records
            incomplete = (
                (grouped["interval_count"] != 4)
                |
                (grouped["valid_count"] != 4)
            )

            if incomplete.any():
                print(
                    f"  Warning: {incomplete.sum():,} "
                    "incomplete hourly technology records"
                )

                # Do not calculate averages from
                # incomplete hourly data
                grouped.loc[
                    incomplete, "mean"
                ] = float("nan")

            # ------------------------------------------
            # Long to Wide
            # ------------------------------------------

            hourly = grouped.pivot(
                index="hour",
                columns=TYPE_COL,
                values="mean"
            )

            # Ensure consistent column order
            hourly = hourly.reindex(
                columns=list(TECHNOLOGIES.keys())
            )

            # Rename technology columns
            hourly = hourly.rename(
                columns=TECHNOLOGIES
            )

            # Rename time index
            hourly.index.name = "mtu_start_utc"

            hourly = (
                hourly
                .sort_index()
                .reset_index()
            )

            if hourly.empty:
                continue

            # ------------------------------------------
            # Validate hourly data
            # ------------------------------------------

            if hourly["mtu_start_utc"].duplicated().any():
                raise ValueError(
                    f"Duplicate hourly timestamps in {file.name}"
                )

            # Check overlap between years
            if previous_end is not None:
                if hourly["mtu_start_utc"].iloc[0] <= previous_end:
                    raise ValueError(
                        f"Overlapping timestamps in {file.name}"
                    )

            # Check missing hours
            gaps = (
                hourly["mtu_start_utc"]
                .diff()
                .dropna()
            )

            missing_hours = int(
                (
                    gaps / pd.Timedelta(hours=1) - 1
                ).clip(lower=0).sum()
            )

            if missing_hours:
                print(
                    f"  Warning: {missing_hours} missing hours"
                )

            # Report missing values
            missing = hourly.isna().sum()

            if missing.any():
                print("  Warning: missing hourly values:")
                print(missing[missing > 0])

            previous_end = (
                hourly["mtu_start_utc"].iloc[-1]
            )

            # ------------------------------------------
            # Save yearly results
            # ------------------------------------------

            hourly.to_csv(
                temp_file,
                mode="w" if first_file else "a",
                header=first_file,
                index=False,
                date_format="%Y-%m-%dT%H:%M:%SZ"
            )

            first_file = False
            total_rows += len(hourly)

            print(
                f"  Saved {len(hourly):,} hourly rows"
            )

        if first_file:
            raise ValueError(
                "No valid generation data found."
            )

        # Replace output after successful processing
        temp_file.replace(OUTPUT_FILE)

        print("\nPreprocessing completed.")
        print(f"Total hourly rows: {total_rows:,}")
        print(f"Output: {OUTPUT_FILE}")

    finally:
        temp_file.unlink(missing_ok=True)


if __name__ == "__main__":
    preprocess_actual_generation()