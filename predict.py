"""
   Optional NOAA values for temperature, wind speed, wind direction
   MLP, GAM, causal forest
   SHAP weighting
   Compute composite air quality score, CompositeAQ(t)=k∑wk⋅zk(t)
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from pygam import LinearGAM, s, te
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score
from sklearn.utils import shuffle
from scipy.stats import pearsonr
from econml.dml import CausalForestDML
from sklearn.ensemble import RandomForestRegressor
import shap


SUMMARY_DIR = "daily"


# Utilities
def attenuate(exposure, homestay, alpha=0.5):
    return exposure * (1 - alpha * homestay)


def add_homestay_ratio(df):
    df["Response_date"] = pd.to_datetime(df["Response_date"]).dt.date
    enriched = []

    for pid, g in df.groupby("Participant"):
        #path = os.path.join(SUMMARY_DIR, f"{pid}_summary.csv")
        path = os.path.join(SUMMARY_DIR, f"{pid}_daily.csv")
        if not os.path.exists(path):
            continue

        summary = pd.read_csv(path)
        summary["date"] = pd.to_datetime(summary["date"], errors="coerce").dt.date
        summary = summary.dropna(subset=["date"])
        summary["homestay_ratio"] = summary["time_at_home_ratio"]

        merged = g.merge(
            summary[["date", "homestay_ratio"]],
            left_on="Response_date",
            right_on="date",
            how="left",
        )

        enriched.append(merged.drop(columns="date"))

    return pd.concat(enriched, ignore_index=True)


def estimate_upper_bound(df):
    cors = []

    for _, g in df.groupby("Participant"):
        y = g["Response"].values.astype(float)

        if len(y) < 3:
            continue
        if np.std(y) == 0:
            continue

        y_lag = np.roll(y, 1)
        y_lag[0] = np.nan

        valid = ~np.isnan(y_lag)
        if valid.sum() < 2:
            continue

        c = np.corrcoef(y[valid], y_lag[valid])[0, 1]
        if not np.isnan(c):
            cors.append(c)

    return np.nanmean(cors)


# Imputation
def impute_cf_cb(df, cols, group_col="Participant"):
    df = df.copy()
    for col in cols:  # impute using this participant's data
        df[col] = (
            df.groupby(group_col)[col]
              .transform(lambda s: s.interpolate(
                  method="linear",
                  limit_direction="both"
              ))
        )
    return df


def impute_nearest_time_global(
    df,
    cols,
    date_col="Response_date",
    time_col="Response_time",
):
    df = df.copy()

    df["_ts"] = pd.to_datetime(
        df[date_col].astype(str) + " " +
        df[time_col].astype(str),
        errors="coerce"
    )

    if df["_ts"].isna().any():
        raise ValueError("Invalid timestamps during global nearest-time imputation")

    ts_np = df["_ts"].values.astype("datetime64[ns]")

    for col in cols:
        missing_idx = df.index[df[col].isna()]
        if len(missing_idx) == 0:
            continue

        known_mask = df[col].notna()
        known_ts = ts_np[known_mask]
        known_vals = df.loc[known_mask, col].values

        for idx in missing_idx:
            t = ts_np[df.index.get_loc(idx)]
            nearest = np.argmin(np.abs(known_ts - t))
            df.at[idx, col] = known_vals[nearest]

        print(f"[impute-global] {col}: filled {len(missing_idx)} values")

    return df.drop(columns="_ts")


# Plots
def plot_timeseries_with_gaps(ax, time, values, linewidth=0.8, alpha=0.8):
    """
    Plot a time series, breaking the line at NaNs.
    """
    mask = ~np.isnan(values)
    ax.plot(time[mask], values[mask], linewidth=linewidth, alpha=alpha)


def plot_participant_stacked_aq(df, participant_id, out_path, start_date=None, end_date=None):
    """
    Generate a stacked AQ plot for a single participant:
      1) Weather PM2.5
      2) AQI
      3) PAM PM2.5
      4) Composite AQ score
    """

    # Filter + timestamp
    g = df[df["Participant"] == participant_id].copy()
    if g.empty:
        raise ValueError(f"No data for participant {participant_id}")

    g["timestamp"] = pd.to_datetime(
        g["Response_date"].astype(str) + " " +
        g["Response_time"].astype(str),
        errors="coerce"
    )
    g = g.dropna(subset=["timestamp"]).sort_values("timestamp")

    if start_date:
        g = g[g["timestamp"] >= pd.Timestamp(start_date)]
    if end_date:
        g = g[g["timestamp"] <= pd.Timestamp(end_date)]

    fig, axes = plt.subplots(nrows=4, sharex=True, figsize=(14, 10))

    plot_timeseries_with_gaps(axes[0], g["timestamp"], g["Weather_pm25"].values)
    axes[0].set_ylabel(r"Weather PM$_{2.5}$")

    plot_timeseries_with_gaps(axes[1], g["timestamp"], g["Aqi"].values)
    axes[1].set_ylabel("AQI")

    plot_timeseries_with_gaps(axes[2], g["timestamp"], g["PAM_pm"].values)
    axes[2].set_ylabel(r"PAM PM$_{2.5}$")

    plot_timeseries_with_gaps(axes[3], g["timestamp"], g["CompositeAQ"].values)
    axes[3].set_ylabel("Composite AQ")

    axes[-1].set_xlabel("Date")

    fig.suptitle(f"Air Quality Time Series — Participant {participant_id}", fontsize=14, y=0.98,)

    for ax in axes:
        ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(out_path, dpi=300)
    plt.close()

    print(f"Saved stacked AQ plot to {out_path}")


def plot_participant_overlay_aq(df, participant_id, out_path, start_date=None, end_date=None):
    """
    Overlay AQ-related time series for a single participant
    on one plot, emphasizing CompositeAQ.
    """

    g = df[df["Participant"] == participant_id].copy()
    if g.empty:
        raise ValueError(f"No data for participant {participant_id}")

    g["timestamp"] = pd.to_datetime(
        g["Response_date"].astype(str) + " " +
        g["Response_time"].astype(str),
        errors="coerce"
    )
    g = g.dropna(subset=["timestamp"]).sort_values("timestamp")

    if start_date:
        g = g[g["timestamp"] >= pd.Timestamp(start_date)]
    if end_date:
        g = g[g["timestamp"] <= pd.Timestamp(end_date)]

    fig, ax = plt.subplots(figsize=(14, 5))

    # muted background signals
    ax.plot(g["timestamp"], g["Weather_pm25"], label="Weather PM2.5", color="tab:blue", alpha=0.35, linewidth=1.0)
    ax.plot(g["timestamp"], g["Weather_co"], label="Weather CO", color="tab:purple", alpha=0.35, linewidth=1.0)
    ax.plot(g["timestamp"], g["Weather_o3"], label="Weather O3", color="tab:cyan", alpha=0.35, linewidth=1.0)
    ax.plot(g["timestamp"], g["Aqi"], label="AQI", color="tab:orange", alpha=0.35, linewidth=1.0)
    ax.plot(g["timestamp"], g["PAM_pm"], label="PAM PM2.5", color="tab:green", alpha=0.35, linewidth=1.0)

    # emphasized composite
    ax.plot(g["timestamp"], g["CompositeAQ"], label="Composite AQ", color="red", linewidth=1.8, alpha=0.95, zorder=10)

    ax.set_title(f"Air Quality Signals — Participant {participant_id}", fontsize=13)
    ax.set_xlabel("Date")
    ax.set_ylabel("Standardized value")
    ax.legend(loc="upper left", frameon=False, ncol=2)

    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

    print(f"Saved overlay AQ plot to {out_path}")


# Models
def mlp_model(random_state=42):
    return MLPRegressor(
        hidden_layer_sizes=(32, 16),
        activation="relu",
        alpha=0.01,
        max_iter=2000,
        random_state=random_state,
    )


def cross_validated_r2(df, features, target="Response", n_splits=10, null=False):
    X = df[features].values
    y = df["Response"].values

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    r2s = []

    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        if null:
            X_train = shuffle(X_train, random_state=42)
            X_test = shuffle(X_test, random_state=42)

        model = mlp_model()
        model.fit(X_train, y_train)

        r2s.append(r2_score(y_test, model.predict(X_test)))

    return np.mean(r2s), np.std(r2s)


def shap_mlp_analysis(df, features, n_background=200, n_explain=300):
    X = df[features].values
    y = df["Response"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = mlp_model()
    model.fit(X_scaled, y)

    background = shap.sample(X_scaled, n_background, random_state=42)
    explain = shap.sample(X_scaled, n_explain, random_state=42)

    explainer = shap.KernelExplainer(model.predict, background)
    shap_values = explainer.shap_values(explain)

    mean_abs = np.mean(np.abs(shap_values), axis=0)
    std_abs = np.std(np.abs(shap_values), axis=0)

    return (
        pd.DataFrame({
            "Feature": features,
            "MeanAbsSHAP": mean_abs,
            "StdAbsSHAP": std_abs,
        })
        .sort_values("MeanAbsSHAP", ascending=False)
    )


def compute_shap_weights(shap_df, aq_features):
    """
    Convert SHAP importance table into normalized weights
    for composite AQ score.
    """
    sub = shap_df[shap_df["Feature"].isin(aq_features)].copy()

    if sub.empty:
        raise ValueError("No AQ features found in SHAP table")

    weights = sub.set_index("Feature")["MeanAbsSHAP"]
    weights = weights / weights.sum()  # normalize

    return weights.to_dict()


def compute_composite_aq(df, aq_features, weights, scale=1.0):
    """
    Compute CompositeAQ(t) = k * sum_k w_k * z_k(t)
    Assumes z_k are already standardized.
    """
    df = df.copy()

    missing = [f for f in aq_features if f not in df.columns]
    if missing:
        raise ValueError(f"Missing AQ features: {missing}")

    comp = np.zeros(len(df))

    for f in aq_features:
        comp += weights[f] * df[f].values

    df["CompositeAQ"] = scale * comp
    return df


def composite(df, shap_df, aq_components):
    print("\n===== COMPOSITE AQ SCORE =====")
    weights = compute_shap_weights(shap_df, aq_components)
    print("CompositeAQ weights:")
    for k, v in weights.items():
        print(f"  {k}: {v:.3f}")
    df = compute_composite_aq(df, aq_components, weights)
    print("\nCompositeAQ summary:")
    print(df["CompositeAQ"].describe())

    # Participant-level stacked AQ plot
    PARTICIPANT_ID = "pnum"
    OUT_PLOT = f"stacked_aq_{PARTICIPANT_ID}.png"
    OUT_OVERLAY = f"overlay_aq_{PARTICIPANT_ID}.png"

    plot_participant_overlay_aq(df, participant_id=PARTICIPANT_ID, out_path=OUT_OVERLAY, start_date="2025-04-17", end_date="2025-06-03")
    return df


# GAM + causal forest unchanged
def fit_gam(df, features):
    X = df[features].values
    y = df["Response"].values

    terms = s(0)
    for i in range(1, len(features)):
        terms += s(i)

    aq_idx = features.index("Aqi")
    hs_idx = features.index("homestay_ratio")
    terms += te(aq_idx, hs_idx)

    return LinearGAM(terms).fit(X, y)


def causal_forest_analysis(df, features):
    Y = df["Response"].values
    T = df["Aqi"].values
    X = df[[f for f in features if f != "Aqi"]].values

    est = CausalForestDML(
        model_t=RandomForestRegressor(min_samples_leaf=50),
        model_y=RandomForestRegressor(min_samples_leaf=50),
        n_estimators=500,
        random_state=42,
    )

    est.fit(Y, T, X=X)
    te = est.effect(X)

    return np.mean(te), np.std(te)


# Compute correlation between composite values and EMA values
def compute_correlation(df):
    # --- pooled ---
    sub = df[["CompositeAQ", "Response"]].dropna()
    r, p = pearsonr(sub["CompositeAQ"], sub["Response"])
    print(f"\n===== PEARSON CORRELATION =====")
    print(f"Pooled r = {r:.3f}, R² = {r**2:.3f}, p = {p:.3g}")

    # --- participant-wise ---
    cors = []
    for pid, g in df.groupby("Participant"):
        g = g[["CompositeAQ", "Response"]].dropna()
        if len(g) < 3 or g["Response"].std() == 0:
            continue
        r_i, _ = pearsonr(g["CompositeAQ"], g["Response"])
        cors.append(r_i)

    cors = np.array(cors)
    print(f"Participant mean r = {cors.mean():.3f} ± {cors.std():.3f}")
    print(f"Participant mean R² ≈ {(cors**2).mean():.3f}")


def main():
    df = pd.read_csv("ema_trace_pam_joined.csv")
    df = add_homestay_ratio(df)
    df = df.dropna(subset=["homestay_ratio"])

    base_features = ["Aqi", "Weather_pm25", "PAM_pm"]

    more_features = [
        #"Aqi", "Weather_pm25", "PAM_pm",
        "PAM_Dist", "AQ_weighted", "WeatherPM_weighted", "PAMPM_weighted",
        "homestay_ratio",
        #"Temperature", "Wind_speed",
        #"Wind_direction",
        "Weather_o3", "Weather_co",
    ]

    aq_components = [
        #"Weather_pm25",
        "WeatherPM_weighted",
        "PAMPM_weighted",
        "AQ_weighted",
        "Weather_o3",
        "Weather_co",
    ]

    gam_features = ["Aqi", "Weather_pm25", "PAM_pm", "homestay_ratio"]

    df["WeatherPM_weighted"] = attenuate(df["Weather_pm25"], df["homestay_ratio"])
    df["PAMPM_weighted"] = attenuate(df["PAM_pm"], df["homestay_ratio"])

    cols = ["Response"] + list(set(more_features + gam_features))
    #df = impute_cf_cb(df, ["Aqi"])
    #df = impute_nearest_time_global(df, ["Aqi"])
    #df["Aqi"] = df["Aqi"].fillna(df["Aqi"].median())
    df = df.dropna(subset=["Aqi"])
    df["AQ_weighted"] = attenuate(df["Aqi"], df["homestay_ratio"])

    # Global z-score scaling
    scaler = StandardScaler()
    #df[cols] = scaler.fit_transform(df[cols])
    # Global z-score scaling (FEATURES ONLY — do not scale targets)
    feature_cols = list(set(more_features + gam_features))
    df[feature_cols] = StandardScaler().fit_transform(df[feature_cols])

    print(f"\nLoaded {len(df)} observations")
    print(f"Estimated upper-bound (lag-1 corr): {estimate_upper_bound(df):.3f}\n")

    print("===== 10-FOLD CV: BASE MLP =====")
    mu, sd = cross_validated_r2(df, base_features, target="Reponse")
    mu_null, sd_null = cross_validated_r2(df, base_features, target="Reponse", null=True)
    print(f"Observed R²: {mu:.3f} ± {sd:.3f}")
    print(f"Null R²:     {mu_null:.3f} ± {sd_null:.3f}")

    print("\n===== 10-FOLD CV: EXTENDED MLP =====")
    mu, sd = cross_validated_r2(df, more_features, target="Reponse")
    mu_null, sd_null = cross_validated_r2(df, more_features, target="Response", null=True)
    print(f"Observed R²: {mu:.3f} ± {sd:.3f}")
    print(f"Null R²:     {mu_null:.3f} ± {sd_null:.3f}")

    print("\n===== SHAP (MLP): GLOBAL FEATURE IMPORTANCE =====")
    shap_df = shap_mlp_analysis(df, more_features)
    print(shap_df.to_string(index=False))

    # Composite AQ score
    df = composite(df, shap_df, aq_components)

    print("\n===== GAM (interpretability only) =====")
    gam = fit_gam(df, gam_features)
    stats = gam.statistics_
    print(f"Pseudo R²: {stats['pseudo_r2']['explained_deviance']:.3f}")
    print(f"Effective DoF: {stats['edof']:.1f}")

    print("\n===== CAUSAL FOREST (AQ heterogeneity) =====")
    te_mean, te_sd = causal_forest_analysis(df, more_features)
    print(f"Mean AQ treatment effect: {te_mean:.3f}")
    print(f"SD   AQ treatment effect: {te_sd:.3f}")


if __name__ == "__main__":
    main()
