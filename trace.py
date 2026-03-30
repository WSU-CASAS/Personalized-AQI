import sys
import os
import pandas as pd
import numpy as np
from wear import detect_nonwear


def build_navigation_trace(full_path):
    """
    Build a compressed navigation trace:
      - Per-minute timeline
      - Rounded lat/lon (3 decimals)
      - Missing or non-wear = N/A
      - Only first timestamp of each distinct location state
    """

    # Load file
    df = pd.read_csv(full_path, header=0, skiprows=[1], on_bad_lines="skip", engine="python")

    required = ["stamp", "latitude", "longitude"]
    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Timestamp conversion
    df["stamp"] = pd.to_datetime(df["stamp"], errors="coerce")

    # Remove rows where timestamp could not be parsed
    df = df.dropna(subset=["stamp"])
    if df.empty:
        raise ValueError("No valid timestamps found after parsing.")

    # Sort chronologically
    df = df.sort_values("stamp")

    # Non-wear detection
    wear_mask = detect_nonwear(df)
    df["wear"] = wear_mask

    # GPS numeric conversion and forward-fill
    df["latitude"]  = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df["latitude"]  = df["latitude"].ffill()
    df["longitude"] = df["longitude"].ffill()

    df["latitude"]  = df["latitude"].astype(object).where(df["latitude"].notna(), "N/A")
    df["longitude"] = df["longitude"].astype(object).where(df["longitude"].notna(), "N/A")

    df = df[["stamp", "latitude", "longitude"]]

    start = df["stamp"].min()
    end = df["stamp"].max()

    # Build minute resolution timeline
    full_range = pd.date_range(start, end, freq="1min")
    full_df = pd.DataFrame({"stamp": full_range})

    # Merge nearest rows (±30s)
    merged = pd.merge_asof(full_df.sort_values("stamp"), df.sort_values("stamp"),
        on="stamp", direction="nearest", tolerance=pd.Timedelta("30s"))

    # Fill missing with N/A
    merged["latitude"] = merged["latitude"].where(merged["latitude"].notna(), "N/A")
    merged["longitude"] = merged["longitude"].where(merged["longitude"].notna(), "N/A")

    # Identify state transitions
    merged["loc"] = merged["latitude"].astype(str) + "," + merged["longitude"].astype(str)

    # Keep first entry of each distinct location state
    nav = merged.loc[
        merged["loc"].ne(merged["loc"].shift(1))
    ][["stamp", "latitude", "longitude"]]

    return nav.rename(columns={"stamp": "timestamp"}).reset_index(drop=True)

def assign_ema_locations(ema_file, base_dir):
    """
    For each EMA response, determine the location of the watch at the time
    of the response using the participant's navigation trace.

    ema_file: path to ema_air_cleanliness.csv
    base_dir: directory containing navigation traces
    """

    ema = pd.read_csv(ema_file)

    required_cols = ["Device", "Response_date", "Response_time", "Response"]
    for col in required_cols:
        if col not in ema.columns:
            raise ValueError(f"EMA file is missing required column: {col}")

    # Build full timestamp
    ema["Response_timestamp"] = pd.to_datetime(ema["Response_date"] + " " + ema["Response_time"], errors="coerce")

    results = [] # output rows

    # Cache loaded navtrace files to avoid re-reading
    nav_cache = {}

    for _, row in ema.iterrows():
        device = row["Device"]
        ts = row["Response_timestamp"]
        response = row["Response"]

        nav_file = os.path.join(base_dir, f"{device}_navtrace.csv")

        if not os.path.exists(nav_file):
            print(f"WARNING: No navigation trace found for device {device}")
            results.append([device, ts, response, "N/A", "N/A"])
            continue

        # Load navtrace once per device
        if device not in nav_cache:
            nav = pd.read_csv(nav_file)
            nav["timestamp"] = pd.to_datetime(nav["timestamp"], errors="coerce")
            nav_cache[device] = nav
        else:
            nav = nav_cache[device]

        # Find the most recent nav entry BEFORE or AT the EMA timestamp
        nav_before = nav[nav["timestamp"] <= ts]

        if nav_before.empty:
            # No earlier location; cannot assign
            lat = "N/A"
            lon = "N/A"
        else:
            last_row = nav_before.iloc[-1]
            lat = last_row["latitude"]
            lon = last_row["longitude"]

        results.append([device, ts, response, lat, lon])

    outdf = pd.DataFrame(results, columns=["Device", "Response_timestamp", "Response", "Latitude", "Longitude"])
    out_path = os.path.join(base_dir, "ema_locations.csv")
    outdf.to_csv(out_path, index=False)
    print(f"EMA-with-location file saved to: {out_path}")

def main():
    base_dir = ""
    ema_file = "ema_information.csv"

    # Generate navigation traces
    csv_files = sorted(
        f for f in os.listdir(base_dir)
        if f.lower().endswith(".csv") and not f.endswith("_navtrace.csv")
           and not f.startswith("ema_air")
    )

    print(f"Found {len(csv_files)} CSV files to process.\n")

    for filename in csv_files:
        full_path = os.path.join(base_dir, filename)
        print(f"Processing: {filename}")

        nav = build_navigation_trace(full_path)
        if nav is None:
            continue
        outname = filename[:-4] + "_navtrace.csv"
        outpath = os.path.join(base_dir, outname)
        nav.to_csv(outpath, index=False)
        print(f"  Output written: {outpath}\n")

    print("All navigation traces processed.")

    # Assign EMA responses to locations
    print("\nAssigning EMA responses to locations…")
    assign_ema_locations(ema_file, base_dir)

if __name__ == "__main__":
    main()
