"""
   Merge air quality information and navigation traces into one file.
   This version uses two downloaded epa files.
"""


import os
import math
import argparse
import numpy as np
import pandas as pd
from typing import Optional, Tuple
import datetime as dt
import requests as req
import json
import time
import requests as req
import datetime as dt
import weather

EMA_FILE = "ema_locations.csv"
EPA_FILE = "epa_data.csv"
PAM_FILE = "pamhourly.csv"
TEMP_FILE = "temperature.csv"
TRACE_DIR = "data"
OUT_CSV = "ema_trace_pam_joined.csv"


# Utilities
def is_naish(x) -> bool:
    if pd.isna(x):
        return True
    s = str(x).strip().lower()
    return s in {"na", "n/a", "nan", "none", ""}


def is_missing(x) -> bool:
    return pd.isna(x) or x == -1


def to_datetime_series(date_series, time_series):
    dt_str = date_series.astype(str).str.strip() + " " + time_series.astype(str).str.strip()
    return pd.to_datetime(dt_str, errors="coerce")


def haversine_m(lat1, lon1, lat2, lon2) -> float:
    if any(is_naish(v) for v in [lat1, lon1, lat2, lon2]):
        return float("inf")
    r = 6371000.0
    phi1 = math.radians(float(lat1))
    phi2 = math.radians(float(lat2))
    dphi = math.radians(float(lat2) - float(lat1))
    dlmb = math.radians(float(lon2) - float(lon1))
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlmb / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def nearest_time_match(df, target_dt):
    ts = df["timestamp"].to_numpy()
    i = ts.searchsorted(target_dt.to_datetime64())
    if i == 0:
        return df.iloc[0]
    if i >= len(df):
        return df.iloc[-1]
    before = df.iloc[i - 1]
    after = df.iloc[i]
    if abs((target_dt - before["timestamp"]).total_seconds()) <= \
       abs((after["timestamp"] - target_dt).total_seconds()):
        return before
    return after


def nearest_trace_location(trace_df, ema_dt):
    if trace_df is None or trace_df.empty:
        return None
    ts = trace_df["timestamp"].to_numpy()
    i = ts.searchsorted(ema_dt.to_datetime64())
    if i == 0:
        row = trace_df.iloc[0]
    elif i >= len(trace_df):
        row = trace_df.iloc[-1]
    else:
        before = trace_df.iloc[i - 1]
        after = trace_df.iloc[i]
        row = before if abs((ema_dt - before["timestamp"]).total_seconds()) <= \
                       abs((after["timestamp"] - ema_dt).total_seconds()) else after
    return float(row["latitude"]), float(row["longitude"]), row["timestamp"]


def best_epa_match(epa, ema_dt, lat, lon):
    subset = epa[epa["date"] == ema_dt.date()]
    if subset.empty:
        return None
    subset = subset.copy()
    subset["dist_m"] = subset.apply(lambda r: haversine_m(lat, lon, r["latitude"], r["longitude"]), axis=1)
    subset["time_diff_s"] = (subset["timestamp"] - ema_dt).abs().dt.total_seconds()
    return subset.sort_values(["dist_m", "time_diff_s"]).iloc[0]


def best_pam_match(pam, ema_dt, lat, lon):
    subset = pam[pam["date"] == ema_dt.date()]
    if subset.empty:
        return None
    subset = subset.copy()
    subset["dist_m"] = subset.apply(
        lambda r: haversine_m(lat, lon, r["Latitude"], r["Longitude"]), axis=1
    )
    subset["time_diff_s"] = (subset["timestamp"] - ema_dt).abs().dt.total_seconds()
    return subset.sort_values(["dist_m", "time_diff_s"]).iloc[0]


def sort_output(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values(
        by=["Participant", "Response_date", "Response_time"],
        ascending=[True, True, True],
    )


# Loaders
def load_trace_for_device(device):
    path = os.path.join(TRACE_DIR, f"{device}_navtrace.csv")
    if not os.path.exists(path):
        return None
    tr = pd.read_csv(path)
    tr["timestamp"] = pd.to_datetime(tr["timestamp"], errors="coerce")
    tr[["latitude", "longitude"]] = tr[["latitude", "longitude"]].ffill()
    return tr.dropna(subset=["timestamp", "latitude", "longitude"])


def load_ema():
    ema = pd.read_csv(EMA_FILE)
    ema["ema_dt"] = to_datetime_series(ema["Response_date"], ema["Response_time"])
    return ema.dropna(subset=["ema_dt"]).sort_values("ema_dt")


def load_epa():
    epa = pd.read_csv(EPA_FILE)
    epa["timestamp"] = pd.to_datetime(epa["timestamp"], errors="coerce")
    epa = epa.dropna(subset=["timestamp", "latitude", "longitude", "pm2.5"])
    epa["date"] = epa["timestamp"].dt.date
    return epa


def load_pam():
    pam = pd.read_csv(PAM_FILE)
    pam["timestamp"] = pd.to_datetime(pam["timestamp"], errors="coerce")
    pam = pam.dropna(subset=["timestamp", "Latitude", "Longitude", "correctedPM"])
    pam["date"] = pam["timestamp"].dt.date
    return pam


def load_temperature_csv():
    df = pd.read_csv(TEMP_FILE)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["wind_direction"] = df["wind_direction"].ffill().bfill()
    return df


def GetAqi(date, hour, latitude: float, longitude: float):
    sdate = date + "T" + hour + "-00"
    url = (
        "https://www.airnowapi.org/aq/observation/latlong/historical/"
        "?format=application/json"
        f"&latitude={latitude}"
        f"&longitude={longitude}"
        f"&date={sdate}"
        "&API_KEY=543F19A8-A961-43F1-9B68-929E9F157953"
    )
    params = {"format": "application/json"}
    while True:
        try:
            response = req.get(url, params=params)
            print("AirNow response:", response.status_code)
            if response.status_code == 429:   # rate limit
                print("AirNow rate-limited (429). Sleeping 5 minutes...")
                time.sleep(300)
                continue  # retry after sleep
            if response.ok:  # success
                result = response.json()
                if not result:
                    print('no result')
                    return [-1]
                return(result[0].get("AQI", None), date)
            response.raise_for_status()
        except Exception as e:
            print("AirNow AQI error:", e)
            return [-1]


def safe_get_aqi(ts, lat, lon):
    try:
        return GetAqi(ts.strftime("%Y-%m-%d"), ts.strftime("%H"), lat, lon)[0]
    except Exception:
        return None


def safe_get_weather_pm(dt_str, lat, lon):
    try:
        return weather.calc_weather(dt_str, lat, lon)
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repair-only", action="store_true")
    args = parser.parse_args()

    pam = load_pam()
    epa = load_epa()
    ema = load_ema()
    temp_csv = load_temperature_csv()

    trace_cache = {}

    # Output DF (new rows + copied rows)
    out_df = pd.DataFrame()

    # Load existing joined file
    existing_df = None
    existing_devices = set()
    copied_devices = set()

    existing_path = os.path.join("arc", OUT_CSV)
    if os.path.exists(existing_path):
        existing_df = pd.read_csv(existing_path)
        existing_devices = set(existing_df["Participant"].unique())

    if args.repair_only and os.path.exists(OUT_CSV):
        out_df = pd.read_csv(OUT_CSV)

    # Main loop
    for _, e in ema.iterrows():
        device = e["Device"]

        # Faster: copy existing device once
        if (
            existing_df is not None
            and device in existing_devices
            and device not in copied_devices
        ):
            device_rows = existing_df[existing_df["Participant"] == device]
            out_df = pd.concat([out_df, device_rows], ignore_index=True)
            copied_devices.add(device)
            continue

        # If already copied for this device, skip further EMA rows
        if device in copied_devices:
            continue

        ema_dt = e["ema_dt"]

        if args.repair_only and not out_df.empty:
            mask = (
                (out_df["Participant"] == device) &
                (out_df["Response_date"] == e["Response_date"]) &
                (out_df["Response_time"] == e["Response_time"])
            )
            if mask.any():
                row = out_df.loc[mask].iloc[0]
                if not any(is_missing(row[c]) for c in [
                    "Weather_pm25", "Weather_o3", "Weather_co",
                    "Aqi", "Temperature", "Wind_speed", "Wind_direction"
                ]):
                    continue

        # Trace lookup
        if device not in trace_cache:
            trace_cache[device] = load_trace_for_device(device)

        tr = trace_cache[device]
        trace_match = nearest_trace_location(tr, ema_dt)
        if trace_match is None:
            continue

        lat, lon, _ = trace_match

        # PAM match
        pam_match = best_pam_match(pam, ema_dt, lat, lon)
        if pam_match is None:
            continue

        epa_match = best_epa_match(epa, ema_dt, lat, lon)
        epa_pm = None
        if epa_match is not None:
            epa_pm = epa_match["pm2.5"]

        # External calls
        aq_value = safe_get_aqi(ema_dt, lat, lon)
        print("aq", aq_value)

        weather_vals = safe_get_weather_pm(ema_dt.strftime("%Y-%m-%d %H:%M:%S"), lat, lon)
        print("wvals", weather_vals)

        w_pm, w_o3, w_co = (
            weather_vals[5], weather_vals[6], weather_vals[8]
        ) if weather_vals and len(weather_vals) >= 9 else (None, None, None)

        # Temperature csv
        temp_match = nearest_time_match(temp_csv, ema_dt)
        csv_temp = temp_match["temperature"]
        csv_wind = temp_match["wind_speed"]
        csv_wdir = temp_match["wind_direction"]

        # Build output row
        new_row = {
            "Participant": device,
            "Response_date": e["Response_date"],
            "Response_time": e["Response_time"],
            "Response": e["Response"],
            "Watch_Latitude": lat,
            "Watch_Longitude": lon,
            "Weather_pm25": w_pm,
            "Weather_o3": w_o3,
            "Weather_co": w_co,
            "Aqi": aq_value,
            "Epa_pm25": epa_pm,
            "Temperature": csv_temp,
            "Wind_speed": csv_wind,
            "Wind_direction": csv_wdir,
            "PAM_Date": pam_match["timestamp"].date().isoformat(),
            "PAM_Time": pam_match["timestamp"].time().isoformat(),
            "PAM_pm": pam_match["correctedPM"],
            "PAM_Latitude": pam_match["Latitude"],
            "PAM_Longitude": pam_match["Longitude"],
            "PAM_Dist": pam_match["dist_m"],
        }

        out_df = pd.concat([out_df, pd.DataFrame([new_row])], ignore_index=True)
        out_df.replace(-1, np.nan, inplace=True)
        out_df = sort_output(out_df)
        out_df.to_csv(OUT_CSV, index=False)

    print(f"Finished. Wrote {len(out_df)} rows to {OUT_CSV}")


if __name__ == "__main__":
    main()
