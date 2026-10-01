"""Input discovery, normalization, and target-definition audits."""

from pathlib import Path

import numpy as np
import pandas as pd


COLUMN_ALIASES = {
    "date": "Date",
    "time": "Date",
    "sst": "sst_celsius",
    "sea_surface_temperature": "sst_celsius",
    "sst_celsius": "sst_celsius",
    "wind": "wind_speed",
    "wind_speed": "wind_speed",
    "precipitation": "tp_mm",
    "total_precipitation": "tp_mm",
    "tp_mm": "tp_mm",
    "climatology": "clim_mean",
    "climatological_mean": "clim_mean",
    "clim_mean": "clim_mean",
    "threshold": "threshold",
    "p90": "threshold",
    "90th_percentile": "threshold",
    "intensity": "intensity",
}

REQUIRED_FIELDS = {
    "Date",
    "sst_celsius",
    "wind_speed",
    "tp_mm",
    "clim_mean",
    "threshold",
}


def _canonical_name(name):
    cleaned = str(name).strip().lower().replace(" ", "_").replace("-", "_")
    return COLUMN_ALIASES.get(cleaned, str(name).strip())


def load_site_file(path):
    """Load one prepared site file and compute the manuscript target variables.

    The returned target is always recomputed as SST minus climatological mean.
    If a supplied ``intensity`` column is present, its agreement with this
    definition is reported in the returned audit dictionary.
    """
    path = Path(path)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        frame = pd.read_excel(path)
    elif path.suffix.lower() == ".csv":
        frame = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported input format: {path.suffix}")

    frame = frame.rename(columns={c: _canonical_name(c) for c in frame.columns})
    missing = REQUIRED_FIELDS.difference(frame.columns)
    if missing:
        raise ValueError(f"{path.name} is missing fields: {sorted(missing)}")

    frame["Date"] = pd.to_datetime(frame["Date"], dayfirst=True, errors="coerce")
    numeric = ["sst_celsius", "wind_speed", "tp_mm", "clim_mean", "threshold"]
    if "intensity" in frame.columns:
        numeric.append("intensity")
    for column in numeric:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    frame = frame.dropna(subset=["Date", *REQUIRED_FIELDS.difference({"Date"})])
    frame = frame.sort_values("Date")
    duplicate_dates = int(frame["Date"].duplicated().sum())
    frame = frame.drop_duplicates("Date", keep="first").reset_index(drop=True)
    if frame.empty:
        raise ValueError(f"{path.name} contains no usable daily records")

    target = frame["sst_celsius"] - frame["clim_mean"]
    threshold_anomaly = frame["threshold"] - frame["clim_mean"]
    audit = {
        "file": path.name,
        "rows": int(len(frame)),
        "start": frame["Date"].min(),
        "end": frame["Date"].max(),
        "duplicate_dates_removed": duplicate_dates,
        "missing_calendar_days": int(
            (frame["Date"].max() - frame["Date"].min()).days + 1 - len(frame)
        ),
        "target_mae_vs_supplied": np.nan,
        "target_max_abs_vs_supplied": np.nan,
    }
    if "intensity" in frame.columns:
        difference = frame["intensity"] - target
        audit["target_mae_vs_supplied"] = float(difference.abs().mean())
        audit["target_max_abs_vs_supplied"] = float(difference.abs().max())

    frame["intensity"] = target.astype(float)
    frame["threshold_anomaly"] = threshold_anomaly.astype(float)
    frame["month"] = frame["Date"].dt.month
    frame["doy"] = frame["Date"].dt.dayofyear
    frame["month_day"] = frame["Date"].dt.strftime("%m-%d")
    return frame, audit


def find_site_files(data_dir):
    """Return input-looking spreadsheet and CSV paths in stable name order."""
    data_dir = Path(data_dir)
    candidates = []
    for pattern in ("*.xlsx", "*.xls", "*.csv"):
        candidates.extend(data_dir.glob(pattern))
    excluded_words = {"forecast", "prediction", "result", "summary", "event"}
    return sorted(
        [
            path
            for path in set(candidates)
            if not any(word in path.name.lower() for word in excluded_words)
        ],
        key=lambda path: path.name,
    )


def print_audit(audit):
    """Print one compact input audit."""
    print(
        f"{audit['file']}: {audit['rows']:,} rows, "
        f"{audit['start'].date()} to {audit['end'].date()}, "
        f"missing days={audit['missing_calendar_days']}, "
        f"duplicates removed={audit['duplicate_dates_removed']}"
    )
    if np.isfinite(audit["target_mae_vs_supplied"]):
        print(
            "  supplied intensity audit: "
            f"MAE={audit['target_mae_vs_supplied']:.10f}, "
            f"max abs={audit['target_max_abs_vs_supplied']:.10f}"
        )

