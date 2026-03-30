import sys
import os
import math
import pandas as pd
import matplotlib.pyplot as plt


# Utilities
def is_naish(x) -> bool:
    if pd.isna(x):
        return True
    s = str(x).strip().lower()
    return s in {"na", "n/a", "nan", "none", ""}


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


# Plotting function
def plot_trajectory_by_day(csv_file, center=True, normalize=True, plot_ema=True,
    plot_pam=True, pam_time_window_hours=1, save=True):
    """
    Privacy-preserving multi-day GPS trajectory plot with EMA and PAM overlays.
    Axes are shown in METERS relative to participant centroid.
    No legend is shown.
    """

    # Load NAVTRACE
    df = pd.read_csv(csv_file)
    df = df[df["latitude"] != "N/A"].copy()
    if df.empty:
        print("No usable GPS in this navtrace file.")
        return

    df["latitude"] = df["latitude"].astype(float)
    df["longitude"] = df["longitude"].astype(float)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["date"] = df["timestamp"].dt.date
    df = df.sort_values("timestamp")

    # Load EMA
    base_dir = os.path.dirname(csv_file)
    ema_file = os.path.join(base_dir, "ema_locations.csv")

    ema_dev = pd.DataFrame()
    if os.path.exists(ema_file):
        ema = pd.read_csv(ema_file)
        ema["Response Timestamp"] = pd.to_datetime(ema["Response Timestamp"])

        device = os.path.basename(csv_file).replace("_navtrace.csv", "")
        ema_dev = ema[ema["Device"] == device].copy()

        ema_dev = ema_dev[ema_dev["Latitude"] != "N/A"]
        if not ema_dev.empty:
            ema_dev["Latitude"] = ema_dev["Latitude"].astype(float)
            ema_dev["Longitude"] = ema_dev["Longitude"].astype(float)
            ema_dev["date"] = ema_dev["Response Timestamp"].dt.date

    # Load PAM
    pam_file = os.path.join(base_dir, "pamhourly.csv")
    pam = pd.DataFrame()

    if plot_pam and os.path.exists(pam_file):
        pam = pd.read_csv(pam_file)
        pam["timestamp"] = pd.to_datetime(pam["timestamp"])
        pam = pam[pam["Latitude"] != "N/A"]
        pam["Latitude"] = pam["Latitude"].astype(float)
        pam["Longitude"] = pam["Longitude"].astype(float)
        pam["correctedPM"] = pam["correctedPM"].astype(float)

    # Coordinate transforms
    if normalize:
        lat_min = df["latitude"].min()
        lon_min = df["longitude"].min()
    else:
        lat_min = lon_min = 0

    lat_shifted = df["latitude"] - lat_min
    lon_shifted = df["longitude"] - lon_min

    if center:
        lat_mean = lat_shifted.mean()
        lon_mean = lon_shifted.mean()
    else:
        lat_mean = lon_mean = 0

    def transform(lat, lon):
        return lat - lat_min - lat_mean, lon - lon_min - lon_mean

    df["lat_r"], df["lon_r"] = transform(df["latitude"], df["longitude"])

    if not ema_dev.empty:
        ema_dev["lat_r"], ema_dev["lon_r"] = transform(
            ema_dev["Latitude"], ema_dev["Longitude"]
        )

    if not pam.empty:
        pam["lat_r"], pam["lon_r"] = transform(
            pam["Latitude"], pam["Longitude"]
        )

    # Convert degrees → meters
    mean_lat = df["latitude"].mean()
    m_per_deg_lat = 111_000
    m_per_deg_lon = 111_000 * math.cos(math.radians(mean_lat))

    df["x_m"] = df["lon_r"] * m_per_deg_lon
    df["y_m"] = df["lat_r"] * m_per_deg_lat

    if not ema_dev.empty:
        ema_dev["x_m"] = ema_dev["lon_r"] * m_per_deg_lon
        ema_dev["y_m"] = ema_dev["lat_r"] * m_per_deg_lat

    if not pam.empty:
        pam["x_m"] = pam["lon_r"] * m_per_deg_lon
        pam["y_m"] = pam["lat_r"] * m_per_deg_lat

    # PAM matcher
    def best_pam_for_ema(row):
        if pam.empty:
            return None

        ema_dt = row["Response Timestamp"]

        # SAME-DAY MATCH (robust)
        subset = pam[pam["timestamp"].dt.date == ema_dt.date()]
        if subset.empty:
            return None

        subset = subset.copy()
        subset["dist_m"] = subset.apply(
            lambda r: haversine_m(
                row["Latitude"], row["Longitude"],
                r["Latitude"], r["Longitude"]
            ),
            axis=1,
        )
        subset["time_diff_s"] = (subset["timestamp"] - ema_dt).abs().dt.total_seconds()

        subset = subset.sort_values(["dist_m", "time_diff_s"])
        return subset.iloc[0]

    # Plot figure
    plt.figure(figsize=(9, 9))

    unique_days = sorted(df["date"].unique())
    cmap = plt.get_cmap("tab20").resampled(len(unique_days))
    day_colors = {day: cmap(i) for i, day in enumerate(unique_days)}

    for day in unique_days:
        d = df[df["date"] == day]
        plt.plot(d["x_m"], d["y_m"], "-o", markersize=3, linewidth=1.4, color=day_colors[day])

    # EMA + PAM overlays
    if plot_ema and not ema_dev.empty:
        for _, row in ema_dev.iterrows():
            ex, ey = row["x_m"], row["y_m"]
            day_color = day_colors.get(row["date"], "black")

            # EMA marker
            plt.scatter(ex, ey, s=90, color="red", edgecolor="black", linewidth=0.8, zorder=6)

            # PAM marker (triangle, same color as day)
            if plot_pam:
                pam_match = best_pam_for_ema(row)
                if pam_match is not None:
                    px, py = pam_match["x_m"], pam_match["y_m"]

                    plt.scatter(px, py, s=160, marker="^", facecolors="none",
                        edgecolors=day_color, linewidth=2.0, zorder=5)

    device = os.path.basename(csv_file).replace("_navtrace.csv", "")
    plt.title(f"")
    plt.xlabel("East–West distance (meters)")
    plt.ylabel("North–South distance (meters)")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.gca().set_aspect("equal", adjustable="box")

    if save:
        out_path = csv_file.replace(".csv", "_trajectory_with_ema_pam.png")
        plt.savefig(out_path, dpi=150)
        print(f"Saved → {out_path}")

    plt.show()


# CLI
def main():
    if len(sys.argv) < 2:
        print("Usage: python trajectory_plot_by_day.py <navtrace.csv>")
        sys.exit(1)

    csv_file = sys.argv[1]
    if not os.path.exists(csv_file):
        print(f"ERROR: File not found: {csv_file}")
        sys.exit(1)

    plot_trajectory_by_day(csv_file)


if __name__ == "__main__":
    main()
