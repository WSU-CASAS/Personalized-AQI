import numpy as np
import pandas as pd
from pathlib import Path
import wear

MAX_SPEED_MPS = 50
HOME_THRESHOLD_M = 100
WEAR_THRESHOLD = 0.5
MIN_POINTS_PER_DAY = 1
MAX_GAP_S = 86400  # 1 day
NORMALIZE = True
SHRINKAGE_K = 10


# Utils
def haversine_vec(lat1, lon1, lat2, lon2):
    R = 6371000.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))


def normalize_pid(stem):
    if NORMALIZE == False:
        return stem.split(".")[0]
    pid = stem.split("-")[0]
    if pid.endswith(("w", "b")):
        pid = pid[:-1]
    return pid


def compute_daily(df, home_lat, home_lon):
    df = df.copy()

    df["dist_from_home_m"] = haversine_vec(df.latitude, df.longitude, home_lat, home_lon)

    dt = (df.groupby("date")["tracked_at"] .diff() .dt.total_seconds() .fillna(0) .clip(0, 60))
    df["dt"] = dt

    daily = (
        df.groupby("date")
        .agg(
            total_distance_m=("step_dist_m", "sum"),
            max_distance_from_home_m=("dist_from_home_m", "max"),
            mean_distance_from_home_m=("dist_from_home_m", "mean"),
            observed_time_s=("dt", "sum"),
            away_time_s=("dt", lambda x: np.sum(
                x[df.loc[x.index, "dist_from_home_m"] > HOME_THRESHOLD_M]
            )),
            first_time=("tracked_at", "min"),
            last_time=("tracked_at", "max"),
            num_points=("dt", "count"),
        )
        .reset_index()
    )

    daily["span_s"] = (
        daily.last_time - daily.first_time
    ).dt.total_seconds()

    daily = daily[
        (daily.num_points >= MIN_POINTS_PER_DAY) &
        (daily.span_s < MAX_GAP_S)
    ].copy()

    if daily.empty:
        return daily

    daily["home_time_s"] = 86400 - daily.observed_time_s
    daily["time_at_home_ratio"] = (
        daily.home_time_s +
        (daily.observed_time_s - daily.away_time_s)
    ) / 86400

    return daily


# Location sanity checks
def filter_erroneous_locations(df):
    valid_range = (
        df.latitude.between(-90, 90) &
        df.longitude.between(-180, 180)
    )
    non_zero = ~(
        (df.latitude.abs() < 1e-6) &
        (df.longitude.abs() < 1e-6)
    )
    return df[valid_range & non_zero].copy()


# Load + preprocess (single file)
def load_single_file(path):
    df = pd.read_csv(path, skiprows=[1], low_memory=False)

    df["tracked_at"] = pd.to_datetime(df["stamp"], errors="coerce")
    df = df.dropna(subset=["latitude", "longitude", "tracked_at"])
    df = filter_erroneous_locations(df)

    if len(df) < 2:
        return None

    df = df.sort_values("tracked_at").reset_index(drop=True)

    df["wear_prob"] = wear.compute_wear_prob(df)
    df = df[df.wear_prob > WEAR_THRESHOLD].copy()
    if len(df) < 2:
        return None

    lat = df.latitude.values
    lon = df.longitude.values
    t = df.tracked_at.values

    dt = np.diff(t).astype("timedelta64[s]").astype(float)
    step_dist = haversine_vec(lat[:-1], lon[:-1], lat[1:], lon[1:])

    speed = np.zeros(len(df))
    speed[1:] = step_dist / np.clip(dt, 1, None)

    df["step_dist_m"] = np.concatenate([[0.0], step_dist])
    df["speed_mps"] = speed

    df.loc[df.speed_mps > MAX_SPEED_MPS, "step_dist_m"] = 0.0
    df["date"] = df.tracked_at.dt.date

    return df


# Staypoints
def extract_staypoints(df, radius_m=100, min_duration_s=600):
    staypoints = []
    df = df.sort_values("tracked_at").reset_index(drop=True)

    start_idx = 0
    for i in range(1, len(df)):
        d = haversine_vec(
            df.latitude.iloc[start_idx],
            df.longitude.iloc[start_idx],
            df.latitude.iloc[i],
            df.longitude.iloc[i],
        )

        if d > radius_m:
            duration = (
                df.tracked_at.iloc[i - 1] -
                df.tracked_at.iloc[start_idx]
            ).total_seconds()

            if duration >= min_duration_s:
                staypoints.append({
                    "latitude": df.latitude.iloc[start_idx:i].median(),
                    "longitude": df.longitude.iloc[start_idx:i].median(),
                    "duration_s": duration,
                })

            start_idx = i

    duration = (
        df.tracked_at.iloc[-1] -
        df.tracked_at.iloc[start_idx]
    ).total_seconds()

    if duration >= min_duration_s:
        staypoints.append({
            "latitude": df.latitude.iloc[start_idx:].median(),
            "longitude": df.longitude.iloc[start_idx:].median(),
            "duration_s": duration,
        })

    return pd.DataFrame(staypoints)


def summarize_staypoints_by_location(staypoints, merge_radius_m=100):
    locations = []

    for _, row in staypoints.iterrows():
        lat, lon, dur = row.latitude, row.longitude, row.duration_s

        assigned = False
        for loc in locations:
            d = haversine_vec(
                lat, lon,
                np.median(loc["latitudes"]),
                np.median(loc["longitudes"])
            )
            if d <= merge_radius_m:
                loc["durations"].append(dur)
                loc["latitudes"].append(lat)
                loc["longitudes"].append(lon)
                assigned = True
                break

        if not assigned:
            locations.append({
                "latitudes": [lat],
                "longitudes": [lon],
                "durations": [dur],
            })

    rows = []
    for loc in locations:
        rows.append({
            "latitude": np.median(loc["latitudes"]),
            "longitude": np.median(loc["longitudes"]),
            "num_visits": len(loc["durations"]),
            "mean_visit_duration_s": float(np.mean(loc["durations"])),
            "std_visit_duration_s": float(np.std(loc["durations"], ddof=0)),
        })

    return pd.DataFrame(rows)


def infer_home_location(df, decimals=3):
    df = df.copy()
    df["hour"] = df.tracked_at.dt.hour

    candidates = df[(df.hour >= 21) | (df.hour < 6)].copy()
    if candidates.empty:
        candidates = df[(df.hour <= 8) | (df.hour >= 20)].copy()

    if candidates.empty:
        endpoints = (
            df.sort_values("tracked_at")
              .groupby("date")
              .agg(latitude=("latitude", "first"),
                   longitude=("longitude", "first"))
        )
        endpoints2 = (
            df.sort_values("tracked_at")
              .groupby("date")
              .agg(latitude=("latitude", "last"),
                   longitude=("longitude", "last"))
        )
        candidates = pd.concat([endpoints, endpoints2]).reset_index(drop=True)

    candidates["lat_q"] = candidates.latitude.round(decimals)
    candidates["lon_q"] = candidates.longitude.round(decimals)

    counts = (
        candidates
        .value_counts(["lat_q", "lon_q"])
        .reset_index(name="count")
    )

    lat_q, lon_q = counts.iloc[0][["lat_q", "lon_q"]]
    home_points = candidates[
        (candidates.lat_q == lat_q) &
        (candidates.lon_q == lon_q)
    ]

    return (
        float(home_points.latitude.median()),
        float(home_points.longitude.median())
    )


def shrinkage_std(x, global_var, k=SHRINKAGE_K):
    x = np.asarray(x)
    n = len(x)

    if not np.isfinite(global_var):
        global_var = 0.0

    sample_var = np.var(x, ddof=1) if n >= 2 else 0.0
    lam = (n - 1) / (n - 1 + k) if n > 1 else 0.0
    shrunk_var = lam * sample_var + (1 - lam) * global_var

    if not np.isfinite(shrunk_var):
        return 0.0

    return float(np.sqrt(max(shrunk_var, 0.0)))


def compute_lag1_autocorr(x):
    x = np.asarray(x)

    if len(x) < 3:
        return 0.0

    x_mean = np.mean(x)
    denom = np.sum((x - x_mean) ** 2)

    if denom == 0 or not np.isfinite(denom):
        return 0.0

    val = np.sum((x[:-1] - x_mean) * (x[1:] - x_mean)) / denom

    if not np.isfinite(val):
        return 0.0

    return float(val)


def compute_slope(x):
    x = np.asarray(x)

    if len(x) < 2:
        return 0.0

    t = np.arange(len(x))

    try:
        slope = np.polyfit(t, x, 1)[0]
    except Exception:
        return 0.0

    if not np.isfinite(slope):
        return 0.0

    return float(slope)


def aggregate_daily(daily, pid):
    rows = {
        "id": pid,
        "num_valid_days": len(daily),
    }

    global_vars = {}
    for col in daily.columns:
        if col in {"date", "first_time", "last_time"}:
            continue
        vals = daily[col].values
        if len(vals) >= 2:
            gv = np.var(vals, ddof=1)
            global_vars[col] = gv if np.isfinite(gv) else 0.0
        else:
            global_vars[col] = 0.0

    for col in daily.columns:
        if col in {"date", "first_time", "last_time"}:
            continue

        vals = daily[col].values

        if len(vals) > 10:
            lo, hi = np.percentile(vals, [5, 95])
            vals = vals[(vals >= lo) & (vals <= hi)]

        if len(vals) == 0:
            rows[f"mean_{col}"] = 0.0
            rows[f"std_{col}"] = 0.0
            rows[f"ac1_{col}"] = 0.0
            rows[f"slope_{col}"] = 0.0
            continue

        rows[f"mean_{col}"] = float(np.mean(vals))
        rows[f"std_{col}"] = shrinkage_std(vals, global_vars[col])
        rows[f"ac1_{col}"] = compute_lag1_autocorr(vals)
        rows[f"slope_{col}"] = compute_slope(vals)

    if "max_distance_from_home_m" in daily.columns:
        if len(daily.max_distance_from_home_m) > 0:
            p95 = np.percentile(daily.max_distance_from_home_m, 95)
            rows["p95_max_distance_from_home_m"] = float(p95) if np.isfinite(p95) else 0.0
        else:
            rows["p95_max_distance_from_home_m"] = 0.0

    return rows


def main(input_dir):
    daily_dir = Path("daily")
    staypoint_dir = Path("staypoints")
    daily_dir.mkdir(exist_ok=True)
    staypoint_dir.mkdir(exist_ok=True)

    paths = list(Path(input_dir).glob("*.csv"))

    grouped = {}
    for p in paths:
        pid = normalize_pid(p.stem)
        grouped.setdefault(pid, []).append(p)

    agg_rows = []

    for pid, files in grouped.items():
        print(f"\nProcessing {pid}")

        dfs = []
        for f in sorted(files):
            df = load_single_file(f)
            if df is not None:
                dfs.append(df)

        if not dfs:
            continue

        df = (
            pd.concat(dfs)
            .drop_duplicates(subset="tracked_at")
            .sort_values("tracked_at")
            .reset_index(drop=True)
        )

        home_lat, home_lon = infer_home_location(df)
        daily = compute_daily(df, home_lat, home_lon)

        if daily.empty:
            continue

        daily.to_csv(daily_dir / f"{pid}_daily.csv", index=False)

        staypoints = extract_staypoints(df)
        if not staypoints.empty:
            sp_summary = summarize_staypoints_by_location(staypoints)
            sp_summary.to_csv(staypoint_dir / f"{pid}_staypoints.csv", index=False)

        agg = aggregate_daily(daily, pid)
        agg_rows.append(agg)

    pd.DataFrame(agg_rows).to_csv("location_markers.csv", index=False)
    print("\nSaved → location_markers.csv")


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python location.py data_dir")
        sys.exit(1)
    main(sys.argv[1])
