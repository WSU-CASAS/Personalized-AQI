"""
   Label data with whether the watch is worn or now.
   Uses a heuristic algorithm based on the standard deviation and value range of
   3D acceleration and 3D rotation over moving windows.
"""

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype, is_datetime64_any_dtype

# nanstd returns np.nan when there are no finite values
def _safe_nanstd(x: np.ndarray) -> float:
    m = np.isfinite(x)
    if not m.any():
        return np.nan
    return float(np.nanstd(x, ddof=0))

def detect_nonwear(data: pd.DataFrame, rate: int = 10, window_minutes: int = 1,
    accel_threshold: float = 0.01, rot_threshold: float = 0.03) -> pd.Series:
    """
    Sliding-window non-wear detection using std of accel/rotation vector magnitudes.
    Returns a boolean Series (True=worn, False=not worn) aligned to the ORIGINAL df.index.
    Policy: if a window has NO DATA (both stds are NaN), mark it as non-wear.
    """
    work = data.copy()

    # Ensure required numeric columns exist
    req = [
        "user_acceleration_x","user_acceleration_y","user_acceleration_z",
        "rotation_rate_x","rotation_rate_y","rotation_rate_z",
    ]
    for c in req:
        if c not in work.columns:
            work[c] = np.nan
        work[c] = pd.to_numeric(work[c], errors="coerce")

    # Timestamp for sorting (handles tz-aware, numeric epoch, strings)
    if "stamp" in work.columns:
        s = work["stamp"]
    else:
        s = pd.Series(work.index, index=work.index)

    if is_datetime64_any_dtype(s):
        ts = pd.to_datetime(s, utc=True, errors="coerce")
    elif is_numeric_dtype(s):
        s_num = pd.to_numeric(s, errors="coerce")
        med = pd.Series(s_num).median(skipna=True)
        unit = "ms" if (pd.notna(med) and med > 1e12) else "s"
        ts = pd.to_datetime(s_num, unit=unit, utc=True, errors="coerce")
    else:
        ts = pd.to_datetime(s, utc=True, errors="coerce")

    orig_index = work.index
    work = work.assign(_stamp_for_sort=ts).sort_values("_stamp_for_sort").reset_index(drop=True)

    # Vector magnitudes
    ax, ay, az = (work["user_acceleration_x"].astype("float64"),
                  work["user_acceleration_y"].astype("float64"),
                  work["user_acceleration_z"].astype("float64"))
    rx, ry, rz = (work["rotation_rate_x"].astype("float64"),
                  work["rotation_rate_y"].astype("float64"),
                  work["rotation_rate_z"].astype("float64"))

    accel_vm = np.sqrt(ax**2 + ay**2 + az**2)
    rot_vm   = np.sqrt(rx**2 + ry**2 + rz**2)

    n   = len(work)
    win = max(1, int(window_minutes * 60 * rate))
    worn = np.ones(n, dtype=bool)

    for i in range(0, n, win):
        j = min(i + win, n)

        sd_accel = _safe_nanstd(accel_vm[i:j].to_numpy())
        sd_rot   = _safe_nanstd(rot_vm[i:j].to_numpy())

        # >>> Your preference: treat NO DATA as non-wear
        if np.isnan(sd_accel) and np.isnan(sd_rot):
            worn[i:j] = False
            continue

        # If only one side has data, keep previous behavior:
        # require both channels present & below thresholds to mark non-wear.
        acc_ok = (not np.isnan(sd_accel)) and (sd_accel < accel_threshold)
        rot_ok = (not np.isnan(sd_rot))   and (sd_rot   < rot_threshold)

        if acc_ok and rot_ok:
            worn[i:j] = False

    # Map back to original order (position-based)
    worn_orig = pd.Series(worn, index=orig_index, name="wear")
    return worn_orig

def compute_wear_prob(df: pd.DataFrame, rate: int = 10, window_minutes: int = 1,
    accel_threshold: float = 0.01, rot_threshold: float = 0.03) -> pd.Series:
    worn = detect_nonwear(df, rate=rate, window_minutes=window_minutes,
        accel_threshold=accel_threshold, rot_threshold=rot_threshold)
    # Ensure it aligns exactly to df.index
    if not worn.index.equals(df.index):
        worn = worn.reindex(df.index)
    prob = worn.astype("float32")
    prob.name = "probability"
    # Convert True->1.0, False->0.0
    prob = prob.where(prob == 0.0, 1.0)
    return prob
