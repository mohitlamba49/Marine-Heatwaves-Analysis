"""Marine heatwave event detection and annual characteristics."""

import numpy as np
import pandas as pd


def detect_events(frame, min_duration=5):
    """Label events that exceed the seasonal threshold for consecutive days.

    ``intensity`` must be SST minus the climatological mean, and
    ``threshold_anomaly`` must be the 90th-percentile threshold minus that
    climatological mean. A gap in dates breaks a run.
    """
    needed = {"Date", "intensity", "threshold_anomaly"}
    missing = needed.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing event-detection fields: {sorted(missing)}")
    if min_duration < 1:
        raise ValueError("min_duration must be a positive integer")

    dates = pd.to_datetime(frame["Date"]).reset_index(drop=True)
    intensity = pd.to_numeric(frame["intensity"], errors="coerce").reset_index(drop=True)
    threshold = pd.to_numeric(
        frame["threshold_anomaly"], errors="coerce"
    ).reset_index(drop=True)
    qualifying = (intensity > threshold) & intensity.notna() & threshold.notna()

    labels = np.zeros(len(frame), dtype=int)
    event_id = 0
    run_start = None
    for position in range(len(frame) + 1):
        at_end = position == len(frame)
        continues = False
        if not at_end and bool(qualifying.iloc[position]):
            if run_start is None:
                continues = True
            else:
                continues = (dates.iloc[position] - dates.iloc[position - 1]).days == 1

        if not at_end and bool(qualifying.iloc[position]) and continues:
            if run_start is None:
                run_start = position
            continue

        if run_start is not None:
            run_end = position
            if run_end - run_start >= min_duration:
                event_id += 1
                labels[run_start:run_end] = event_id
            run_start = position if (not at_end and bool(qualifying.iloc[position])) else None

    return pd.Series(labels, index=frame.index, name="event_id")


def _peak_category(maximum_ratio):
    if not np.isfinite(maximum_ratio):
        return "Unclassified"
    if maximum_ratio < 2:
        return "Moderate"
    if maximum_ratio < 3:
        return "Strong"
    if maximum_ratio < 4:
        return "Severe"
    return "Extreme"


def event_table(frame, event_labels=None, min_duration=5):
    """Summarize each MHW event using climatological-mean anomalies.

    Cumulative intensity is the sum of ``SST - climatological mean`` over all
    event days. It is not a sum of exceedance above the 90th percentile.
    """
    work = frame.copy().reset_index(drop=True)
    if event_labels is None:
        event_labels = detect_events(work, min_duration=min_duration)
    work["event_id"] = np.asarray(event_labels, dtype=int)
    rows = []
    for event_id, event in work[work["event_id"] > 0].groupby("event_id", sort=True):
        safe_threshold = event["threshold_anomaly"].replace(0, np.nan)
        ratios = event["intensity"] / safe_threshold
        maximum_ratio = float(ratios.max()) if ratios.notna().any() else np.nan
        rows.append(
            {
                "event_id": int(event_id),
                "start_date": event["Date"].min(),
                "end_date": event["Date"].max(),
                "event_year": int(event["Date"].min().year),
                "duration_days": int(len(event)),
                "mean_intensity_c": float(event["intensity"].mean()),
                "maximum_intensity_c": float(event["intensity"].max()),
                "cumulative_intensity_c_days": float(event["intensity"].sum()),
                "maximum_category_ratio": maximum_ratio,
                "peak_category": _peak_category(maximum_ratio),
            }
        )
    columns = [
        "event_id",
        "start_date",
        "end_date",
        "event_year",
        "duration_days",
        "mean_intensity_c",
        "maximum_intensity_c",
        "cumulative_intensity_c_days",
        "maximum_category_ratio",
        "peak_category",
    ]
    return pd.DataFrame(rows, columns=columns)


def annual_characteristics(frame, years=None, min_duration=5):
    """Return annual event frequency, event means, and calendar-year MHW days.

    Years without an event are zero for frequency and MHW days. Event-level
    mean duration, cumulative intensity, and maximum intensity remain NaN.
    """
    work = frame.copy().sort_values("Date").reset_index(drop=True)
    labels = detect_events(work, min_duration=min_duration)
    events = event_table(work, labels, min_duration=min_duration)
    work["event_id"] = labels.to_numpy()
    work["year"] = pd.to_datetime(work["Date"]).dt.year

    if years is None:
        years = range(int(work["year"].min()), int(work["year"].max()) + 1)
    annual = pd.DataFrame({"year": list(years)}).set_index("year")
    annual["event_frequency"] = 0
    annual["annual_mhw_days"] = 0

    if not events.empty:
        frequency = events.groupby("event_year").size()
        annual.loc[annual.index.intersection(frequency.index), "event_frequency"] = frequency
        grouped = events.groupby("event_year")
        annual["mean_event_duration_days"] = grouped["duration_days"].mean()
        annual["mean_cumulative_intensity_c_days"] = grouped[
            "cumulative_intensity_c_days"
        ].mean()
        annual["mean_maximum_intensity_c"] = grouped["maximum_intensity_c"].mean()
    else:
        annual["mean_event_duration_days"] = np.nan
        annual["mean_cumulative_intensity_c_days"] = np.nan
        annual["mean_maximum_intensity_c"] = np.nan

    mhw_days = work.loc[work["event_id"] > 0].groupby("year").size()
    annual.loc[annual.index.intersection(mhw_days.index), "annual_mhw_days"] = mhw_days
    annual["event_frequency"] = annual["event_frequency"].astype(int)
    annual["annual_mhw_days"] = annual["annual_mhw_days"].astype(int)
    return annual.reset_index(), events, labels

