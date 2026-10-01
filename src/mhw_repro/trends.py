"""Mann–Kendall and Sen-slope summaries for annual MHW characteristics."""

import numpy as np
import pandas as pd
from scipy.stats import norm


DEFAULT_METRICS = [
    "event_frequency",
    "mean_event_duration_days",
    "mean_cumulative_intensity_c_days",
    "mean_maximum_intensity_c",
    "annual_mhw_days",
]


def mann_kendall_sen(year, value, alpha=0.05):
    """Return the original Mann–Kendall test and Sen slope after dropping NaNs.

    The variance includes the standard correction for tied values and the
    two-sided normal approximation uses the conventional continuity correction.
    """
    data = pd.DataFrame({"year": year, "value": value}).dropna()
    if len(data) < 3:
        return {
            "years_used": int(len(data)),
            "tau": np.nan,
            "p_value": np.nan,
            "sen_slope_per_year": np.nan,
            "trend": "insufficient data",
        }
    data = data.sort_values("year")
    x = data["value"].to_numpy(dtype=float)
    t = data["year"].to_numpy(dtype=float)
    n = len(data)
    if data["value"].nunique() < 2:
        return {
            "years_used": int(n),
            "tau": 0.0,
            "p_value": 1.0,
            "sen_slope_per_year": 0.0,
            "trend": "no trend",
        }

    score = 0.0
    slopes = []
    for earlier in range(n - 1):
        score += float(np.sign(x[earlier + 1 :] - x[earlier]).sum())
        slopes.extend(
            ((x[earlier + 1 :] - x[earlier]) / (t[earlier + 1 :] - t[earlier])).tolist()
        )
    _, tie_counts = np.unique(x, return_counts=True)
    tie_term = np.sum(tie_counts * (tie_counts - 1) * (2 * tie_counts + 5))
    variance = (n * (n - 1) * (2 * n + 5) - tie_term) / 18.0
    if score > 0:
        z_value = (score - 1) / np.sqrt(variance)
    elif score < 0:
        z_value = (score + 1) / np.sqrt(variance)
    else:
        z_value = 0.0
    p_value = 2 * (1 - norm.cdf(abs(z_value)))
    tau = score / (0.5 * n * (n - 1))
    slope = float(np.median(slopes))
    if p_value < alpha and slope > 0:
        trend = "increasing"
    elif p_value < alpha and slope < 0:
        trend = "decreasing"
    else:
        trend = "no trend"
    return {
        "years_used": int(n),
        "tau": float(tau),
        "p_value": float(p_value),
        "sen_slope_per_year": float(slope),
        "trend": trend,
    }


def trend_table(annual, metrics=None, alpha=0.05):
    """Build a tidy trend table for available annual metric columns."""
    metrics = DEFAULT_METRICS if metrics is None else metrics
    rows = []
    for metric in metrics:
        if metric not in annual.columns:
            continue
        row = {"metric": metric}
        row.update(mann_kendall_sen(annual["year"], annual[metric], alpha=alpha))
        rows.append(row)
    return pd.DataFrame(rows)
