"""Daily and annual evaluation metrics used in the manuscript."""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from .mhw import annual_characteristics, detect_events


def regression_metrics(observed, predicted):
    """Return RMSE and MAE for aligned finite pairs."""
    pairs = pd.DataFrame({"observed": observed, "predicted": predicted}).dropna()
    if pairs.empty:
        return {"n": 0, "rmse": np.nan, "mae": np.nan}
    return {
        "n": int(len(pairs)),
        "rmse": float(np.sqrt(mean_squared_error(pairs["observed"], pairs["predicted"]))),
        "mae": float(mean_absolute_error(pairs["observed"], pairs["predicted"])),
    }


def event_scores(observed_labels, predicted_labels):
    """Return day-level precision, recall, F1, and critical success index."""
    observed = np.asarray(observed_labels) > 0
    predicted = np.asarray(predicted_labels) > 0
    if observed.shape != predicted.shape:
        raise ValueError("Observed and predicted event labels must have equal length")
    hits = int(np.sum(observed & predicted))
    false_alarms = int(np.sum(~observed & predicted))
    misses = int(np.sum(observed & ~predicted))
    precision = hits / (hits + false_alarms) if hits + false_alarms else 0.0
    recall = hits / (hits + misses) if hits + misses else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    csi = hits / (hits + false_alarms + misses) if hits + false_alarms + misses else 0.0
    return {
        "hits": hits,
        "false_alarms": false_alarms,
        "misses": misses,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "csi": float(csi),
    }


def reconstruct_from_prediction(reference_frame, predicted_intensity, min_duration=5):
    """Apply the observed seasonal threshold to a predicted anomaly series."""
    predicted = reference_frame.copy().reset_index(drop=True)
    predicted["intensity"] = np.asarray(predicted_intensity, dtype=float)
    annual, events, labels = annual_characteristics(
        predicted, min_duration=min_duration
    )
    return annual, events, labels


def annual_characteristic_errors(observed_frame, predicted_intensity, min_duration=5):
    """Evaluate reconstructed annual MHW characteristics.

    Event-level means are compared only in years in which both observed and
    predicted series contain at least one event. Frequency and MHW days include
    all years and therefore retain genuine zero-event years.
    """
    observed_annual, _, observed_labels = annual_characteristics(
        observed_frame, min_duration=min_duration
    )
    predicted_annual, _, predicted_labels = reconstruct_from_prediction(
        observed_frame, predicted_intensity, min_duration=min_duration
    )
    merged = observed_annual.merge(
        predicted_annual, on="year", suffixes=("_observed", "_predicted")
    )
    metrics = {}
    for name in ["event_frequency", "annual_mhw_days"]:
        metrics[name] = regression_metrics(
            merged[f"{name}_observed"], merged[f"{name}_predicted"]
        )
    for name in [
        "mean_event_duration_days",
        "mean_cumulative_intensity_c_days",
        "mean_maximum_intensity_c",
    ]:
        metrics[name] = regression_metrics(
            merged[f"{name}_observed"], merged[f"{name}_predicted"]
        )
    metrics["event_days"] = event_scores(observed_labels, predicted_labels)
    return metrics, merged

