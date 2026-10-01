"""
MHW TWO-STAGE HISTORICAL BACKTEST
=================================
Purpose
-------
Validate the same logic used for the 2026 MHW projection on years for which
observations are already known.

Stage 1: forecast SST, wind speed, and precipitation with SARIMAX(1,1,1)
         + 3 annual Fourier harmonics using information available before the
         hindcast year.
Stage 2: feed those forecast predictors into the RF-GAM-LSTM stacking model
         to predict daily SST anomaly (stored as `intensity` in prepared files).
Then:   compare the predicted anomaly with the Hobday seasonal threshold
         anomaly and apply the >=5 consecutive day event rule.

This script is intentionally CONSOLE-ONLY: it does not save result files.

IMPORTANT IMPLEMENTATION NOTES
----------------------------------------------------------
1. In the location files, `intensity` is the climatological SST anomaly:
       intensity = sst_celsius - clim_mean
   (the script audits this numerically before fitting).
2. The MHW event criterion is NOT intensity > 0. The correct criterion is:
       intensity > (threshold - clim_mean)
   followed by >=5 consecutive qualifying days.
3. The final manuscript settings are RF(200 trees, depth 8, min split 5),
   GAM with four 10-spline smooths, and LSTM(32 units, dropout 0.20).
4. An earlier development notebook fitted the stacking meta-model on test
   outcomes. That is leakage for an independent evaluation. This backtest uses
   a leakage-free internal meta period: first 75% of the available historical
   record to fit base learners and final 25% to fit the linear meta-model.
   It then refits the base learners on all data available before the hindcast
   year and applies the already-estimated meta-model to the hindcast year.

Recommended sequence in Colab
------------------------------
A. First run: SMOKE = True, HINDCAST_YEARS = [2025], SITE_LIMIT = 1
B. Full 2025:  SMOKE = False, HINDCAST_YEARS = [2025], SITE_LIMIT = None
C. If successful, expand years only if desired.

Required optional packages:
    !pip -q install pygam statsmodels scikit-learn tensorflow openpyxl

The script auto-detects .xlsx/.xls/.csv location files in DATA_DIR and keeps
only files containing the required historical columns.
"""

import os
import glob
import math
import warnings
import random
from pathlib import Path
import numpy as np
import pandas as pd

from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    precision_score,
    recall_score,
    f1_score,
)

warnings.filterwarnings("ignore")

# ============================================================
# 0. USER SETTINGS
# ============================================================

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = str(REPO_ROOT / "data" / "raw")
HINDCAST_YEARS = [2025]        # Start with 2025 only
SITE_LIMIT = None              # Set 1 for first smoke run; None = all sites
SMOKE = False                  # True = quick execution check, NOT manuscript result
RANDOM_SEED = 42

# Historical information structure
MIN_TRAIN_START = "1983-01-01"
CLIM_START = "1983-01-01"
CLIM_END = "2012-12-31"
META_FRACTION = 0.25           # last 25% historical data for meta-model fitting
LSTM_WINDOW = 30

# Final manuscript settings
RF_N_ESTIMATORS = 200
RF_MAX_DEPTH = 8
RF_MIN_SAMPLES_SPLIT = 5
GAM_N_SPLINES = 10
LSTM_UNITS = 32
LSTM_DROPOUT = 0.20
LSTM_EPOCHS = 50
LSTM_BATCH_SIZE = 32

# Stage-1 predictor forecasting settings used in the 2026 notebooks
PREDICTOR_SARIMAX_ORDER = (1, 1, 1)
FOURIER_K = 3
FOURIER_PERIOD = 365.0

# In smoke mode only: much shorter fitting
SMOKE_LSTM_EPOCHS = 2
SMOKE_RF_TREES = 40
SMOKE_MAXITER_SARIMAX = 25

REQUIRED_COLUMNS = {
    "Date",
    "sst_celsius",
    "wind_speed",
    "tp_mm",
    "clim_mean",
    "threshold",
}

np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)


# ============================================================
# 1. BASIC HELPERS
# ============================================================

def banner(text, char="="):
    print("\n" + char * 90)
    print(text)
    print(char * 90)


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def safe_mean(x):
    x = np.asarray(x, dtype=float)
    return float(np.nanmean(x)) if np.isfinite(x).any() else np.nan


def read_location_file(path):
    ext = os.path.splitext(path)[1].lower()
    if ext in (".xlsx", ".xls"):
        df = pd.read_excel(path)
    elif ext == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported file type: {path}")

    df.columns = [str(c).strip() for c in df.columns]
    if "Date" not in df.columns:
        raise ValueError("No Date column")

    # dayfirst=True matches the prepared site files; errors='coerce' is audited.
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date").drop_duplicates("Date")

    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    # Keep numeric core clean.
    for c in ["sst_celsius", "wind_speed", "tp_mm", "clim_mean", "threshold"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    if "intensity" in df.columns:
        df["intensity"] = pd.to_numeric(df["intensity"], errors="coerce")
        df["intensity_supplied"] = df["intensity"]

    df = df.dropna(subset=[
        "sst_celsius", "wind_speed", "tp_mm", "clim_mean", "threshold"
    ]).reset_index(drop=True)

    df["month"] = df["Date"].dt.month
    df["doy"] = df["Date"].dt.dayofyear
    df["md"] = df["Date"].dt.strftime("%m-%d")
    df["sst_anomaly"] = df["sst_celsius"] - df["clim_mean"]
    df["intensity"] = df["sst_anomaly"]
    df["threshold_anomaly"] = df["threshold"] - df["clim_mean"]
    return df


def find_location_files(data_dir):
    candidates = []
    for pat in ("*.xlsx", "*.xls", "*.csv"):
        candidates.extend(glob.glob(os.path.join(data_dir, pat)))

    accepted = []
    rejected = []
    for path in sorted(set(candidates)):
        # Avoid obvious model output files.
        name = os.path.basename(path).lower()
        if any(k in name for k in [
            "forecast", "result", "prediction", "event", "summary", "2026 mean"
        ]):
            continue
        try:
            if path.lower().endswith(".csv"):
                peek = pd.read_csv(path, nrows=3)
            else:
                peek = pd.read_excel(path, nrows=3)
            cols = {str(c).strip() for c in peek.columns}
            if REQUIRED_COLUMNS.issubset(cols):
                accepted.append(path)
            else:
                rejected.append((path, sorted(REQUIRED_COLUMNS.difference(cols))))
        except Exception as e:
            rejected.append((path, [str(e)]))

    return accepted, rejected


def location_label(path):
    name = os.path.splitext(os.path.basename(path))[0]
    # Preserve useful filename but remove common phrases if present.
    for phrase in ["whole data for analysis ", "whole_data_for_analysis_"]:
        name = name.replace(phrase, "")
    return name


# ============================================================
# 2. TARGET / THRESHOLD AUDITS
# ============================================================

def audit_target_definition(df, label):
    """Check what the historical `intensity` column actually represents."""
    audit_values = df["intensity_supplied"] if "intensity_supplied" in df else df["intensity"]
    d1 = np.abs(audit_values - (df["sst_celsius"] - df["clim_mean"]))
    d2 = np.abs(audit_values - (df["sst_celsius"] - df["threshold"]))

    mae_clim = float(d1.mean())
    max_clim = float(d1.max())
    mae_thr = float(d2.mean())

    print(f"Target audit [{label}]")
    print(f"  MAE(intensity, SST - clim_mean) = {mae_clim:.12f}")
    print(f"  MAX(intensity, SST - clim_mean) = {max_clim:.12f}")
    print(f"  MAE(intensity, SST - threshold) = {mae_thr:.12f}")

    if max_clim < 1e-6:
        print("  PASS: intensity is the climatological SST anomaly.")
    elif mae_clim < mae_thr:
        print("  WARNING: intensity is closer to SST-clim_mean but not exactly identical.")
    else:
        print("  WARNING: intensity does not match SST-clim_mean; inspect before using results.")

    zero_flag = audit_values > 0
    threshold_flag = df["intensity"] > df["threshold_anomaly"]
    disagree = float(np.mean(zero_flag != threshold_flag))
    print(f"  Daily disagreement: (intensity>0) vs Hobday threshold = {100*disagree:.2f}%")
    return {
        "target_mae_clim": mae_clim,
        "target_max_clim": max_clim,
        "target_mae_threshold": mae_thr,
        "zero_vs_threshold_disagreement": disagree,
    }


def make_fixed_climatology_lookup(df_hist):
    """
    Use the historical file's already-computed Hobday climatology and threshold.
    Month-day is used as the key to avoid leap-year DOY shifts.
    """
    base = df_hist[(df_hist["Date"] >= CLIM_START) & (df_hist["Date"] <= CLIM_END)].copy()
    if base.empty:
        raise ValueError("No observations in 1983-2012 climatology reference period")

    lookup = (
        base.groupby("md", as_index=False)[["clim_mean", "threshold"]]
        .median()
        .sort_values("md")
    )
    lookup["threshold_anomaly"] = lookup["threshold"] - lookup["clim_mean"]
    return lookup


def attach_climatology(dates_df, lookup):
    out = dates_df.copy()
    out["md"] = pd.to_datetime(out["Date"]).dt.strftime("%m-%d")
    out = out.merge(lookup, on="md", how="left")

    # Feb 29 may be absent from the original 365-day files. If encountered,
    # interpolate from Feb 28 and Mar 01 rather than inventing a new climatology.
    if out[["clim_mean", "threshold", "threshold_anomaly"]].isna().any().any():
        feb29 = out["md"] == "02-29"
        if feb29.any():
            lut = lookup.set_index("md")
            for col in ["clim_mean", "threshold", "threshold_anomaly"]:
                if "02-28" in lut.index and "03-01" in lut.index:
                    val = 0.5 * (float(lut.loc["02-28", col]) + float(lut.loc["03-01", col]))
                    out.loc[feb29 & out[col].isna(), col] = val

    if out[["clim_mean", "threshold", "threshold_anomaly"]].isna().any().any():
        bad = out.loc[out["clim_mean"].isna(), "Date"].head().tolist()
        raise ValueError(f"Could not attach climatology to dates, examples: {bad}")
    return out


# ============================================================
# 3. FOURIER + STAGE-1 PREDICTOR FORECASTS
# ============================================================

def fourier_from_positions(n, period=FOURIER_PERIOD, k=FOURIER_K, start_pos=0, index=None):
    t = np.arange(start_pos, start_pos + n, dtype=float)
    arr = {}
    for j in range(1, k + 1):
        arr[f"sin_{j}"] = np.sin(2 * np.pi * j * t / period)
        arr[f"cos_{j}"] = np.cos(2 * np.pi * j * t / period)
    return pd.DataFrame(arr, index=index)


def forecast_one_predictor(y, n_future, smoke=False):
    y = pd.Series(np.asarray(y, dtype=float)).reset_index(drop=True)
    exog_hist = fourier_from_positions(len(y), start_pos=0)
    exog_future = fourier_from_positions(n_future, start_pos=len(y))

    model = SARIMAX(
        y,
        exog=exog_hist,
        order=PREDICTOR_SARIMAX_ORDER,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    if smoke:
        fit = model.fit(disp=False, maxiter=SMOKE_MAXITER_SARIMAX)
    else:
        fit = model.fit(disp=False)

    fc = fit.forecast(steps=n_future, exog=exog_future)
    return np.asarray(fc, dtype=float), bool(getattr(fit, "mle_retvals", {}).get("converged", True))


def forecast_environment(hist, target_dates, smoke=False):
    n = len(target_dates)
    rows = {"Date": pd.to_datetime(target_dates)}
    conv = {}

    # Smoke mode uses recent history only to keep the test quick. Full mode uses
    # all information available before the hindcast year.
    h = hist.copy()
    if smoke and len(h) > 1800:
        h = h.tail(1800).copy()

    for col in ["sst_celsius", "wind_speed", "tp_mm"]:
        pred, ok = forecast_one_predictor(h[col].values, n, smoke=smoke)
        rows[col] = pred
        conv[col] = ok

    future = pd.DataFrame(rows)
    # Match the 2026 notebooks' physical clipping for non-negative variables.
    future["wind_speed"] = future["wind_speed"].clip(lower=0)
    future["tp_mm"] = future["tp_mm"].clip(lower=0)
    future["month"] = future["Date"].dt.month
    future["doy"] = future["Date"].dt.dayofyear
    return future, conv


# ============================================================
# 4. EVENT DETECTION + METRICS
# ============================================================

def consecutive_event_labels(flag, min_duration=5):
    flag = np.asarray(flag, dtype=bool)
    keep = np.zeros(len(flag), dtype=bool)
    event_id = np.full(len(flag), -1, dtype=int)

    eid = 0
    i = 0
    while i < len(flag):
        if not flag[i]:
            i += 1
            continue
        j = i
        while j < len(flag) and flag[j]:
            j += 1
        if j - i >= min_duration:
            keep[i:j] = True
            event_id[i:j] = eid
            eid += 1
        i = j
    return keep, event_id


def event_summary(dates, anomaly, threshold_anomaly):
    dates = pd.to_datetime(pd.Series(dates)).reset_index(drop=True)
    anomaly = np.asarray(anomaly, dtype=float)
    threshold_anomaly = np.asarray(threshold_anomaly, dtype=float)

    qualifying = anomaly > threshold_anomaly
    mhw_days, event_id = consecutive_event_labels(qualifying, min_duration=5)

    events = []
    for eid in sorted(set(event_id[event_id >= 0])):
        idx = np.where(event_id == eid)[0]
        vals = anomaly[idx]
        thr = threshold_anomaly[idx]
        exceed = vals - thr
        # Hobday category multiplier relative to threshold excess above climatology:
        # category based on max anomaly / threshold_anomaly is not needed for backtest.
        events.append({
            "event_id": int(eid),
            "start": dates.iloc[idx[0]],
            "end": dates.iloc[idx[-1]],
            "duration": int(len(idx)),
            "mean_anomaly": float(np.mean(vals)),
            "max_anomaly": float(np.max(vals)),
            "cumulative_anomaly": float(np.sum(vals)),
            "max_exceedance": float(np.max(exceed)),
        })

    ev = pd.DataFrame(events)
    return mhw_days, ev


def summarize_annual_characteristics(dates, anomaly, threshold_anomaly):
    mhw_days, events = event_summary(dates, anomaly, threshold_anomaly)
    if len(events) == 0:
        return {
            "events": 0,
            "mhw_days": int(mhw_days.sum()),
            "mean_duration": np.nan,
            "mean_cumulative_anomaly": np.nan,
            "mean_max_anomaly": np.nan,
        }, mhw_days, events

    return {
        "events": int(len(events)),
        "mhw_days": int(mhw_days.sum()),
        "mean_duration": float(events["duration"].mean()),
        "mean_cumulative_anomaly": float(events["cumulative_anomaly"].mean()),
        "mean_max_anomaly": float(events["max_anomaly"].mean()),
    }, mhw_days, events


def classification_scores(obs_days, pred_days):
    obs = np.asarray(obs_days, dtype=int)
    pred = np.asarray(pred_days, dtype=int)
    tp = int(np.sum((obs == 1) & (pred == 1)))
    fp = int(np.sum((obs == 0) & (pred == 1)))
    fn = int(np.sum((obs == 1) & (pred == 0)))
    denom = tp + fp + fn
    csi = tp / denom if denom > 0 else np.nan
    return {
        "precision": precision_score(obs, pred, zero_division=0),
        "recall": recall_score(obs, pred, zero_division=0),
        "f1": f1_score(obs, pred, zero_division=0),
        "csi": csi,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


# ============================================================
# 5. RF-GAM-LSTM STACKING
# ============================================================

def import_ml_dependencies():
    try:
        from pygam import LinearGAM, s
    except ImportError as e:
        raise ImportError(
            "pygam is required; install requirements-full.txt"
        ) from e

    try:
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout
        from tensorflow.keras.callbacks import EarlyStopping
    except ImportError as e:
        raise ImportError(
            "tensorflow is required; install requirements-full.txt"
        ) from e

    tf.random.set_seed(RANDOM_SEED)
    return LinearGAM, s, Sequential, LSTM, Dense, Dropout, EarlyStopping


def build_lstm_model(n_features, epochs, verbose=0):
    LinearGAM, s, Sequential, LSTM, Dense, Dropout, EarlyStopping = import_ml_dependencies()
    model = Sequential([
        LSTM(LSTM_UNITS, input_shape=(LSTM_WINDOW, n_features)),
        Dropout(LSTM_DROPOUT),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse")
    return model, EarlyStopping


def make_sequences(features_scaled, target_scaled=None, window=LSTM_WINDOW):
    xs, ys, end_idx = [], [], []
    for i in range(window, len(features_scaled)):
        xs.append(features_scaled[i-window:i])
        end_idx.append(i)
        if target_scaled is not None:
            ys.append(target_scaled[i])
    xs = np.asarray(xs)
    end_idx = np.asarray(end_idx, dtype=int)
    if target_scaled is None:
        return xs, end_idx
    return xs, np.asarray(ys), end_idx


def fit_base_models(train_df, smoke=False):
    LinearGAM, s, Sequential, LSTM, Dense, Dropout, EarlyStopping = import_ml_dependencies()

    train_df = train_df.copy().reset_index(drop=True)
    train_df["month"] = train_df["Date"].dt.month
    train_df["doy"] = train_df["Date"].dt.dayofyear
    train_df["sst_anomaly"] = train_df["sst_celsius"] - train_df["clim_mean"]

    # RF
    rf = RandomForestRegressor(
        n_estimators=SMOKE_RF_TREES if smoke else RF_N_ESTIMATORS,
        max_depth=RF_MAX_DEPTH,
        min_samples_split=RF_MIN_SAMPLES_SPLIT,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    rf.fit(
        train_df[["sst_celsius", "wind_speed", "tp_mm", "month", "doy"]],
        train_df["intensity"],
    )

    # GAM
    Xg = train_df[["sst_celsius", "wind_speed", "tp_mm", "doy"]].values
    yg = train_df["intensity"].values
    gam = LinearGAM(
        s(0, n_splines=GAM_N_SPLINES)
        + s(1, n_splines=GAM_N_SPLINES)
        + s(2, n_splines=GAM_N_SPLINES)
        + s(3, n_splines=GAM_N_SPLINES)
    )
    # Smoothing is selected only on the fitting data.
    gam.gridsearch(Xg, yg, progress=False)

    # LSTM
    feat_cols = ["sst_anomaly", "wind_speed", "tp_mm"]
    X = train_df[feat_cols].values
    y = train_df[["intensity"]].values
    scaler_X = MinMaxScaler().fit(X)
    scaler_y = MinMaxScaler().fit(y)
    Xs = scaler_X.transform(X)
    ys = scaler_y.transform(y)
    Xseq, yseq, _ = make_sequences(Xs, ys, LSTM_WINDOW)
    if len(Xseq) == 0:
        raise ValueError("Not enough observations for LSTM sequences")

    lstm_model, EarlyStopping = build_lstm_model(n_features=len(feat_cols), epochs=0)
    callbacks = []
    if not smoke:
        callbacks = [EarlyStopping(monitor="loss", patience=5, restore_best_weights=True)]
    lstm_model.fit(
        Xseq,
        yseq,
        epochs=SMOKE_LSTM_EPOCHS if smoke else LSTM_EPOCHS,
        batch_size=LSTM_BATCH_SIZE,
        shuffle=False,
        verbose=0,
        callbacks=callbacks,
    )

    return {
        "rf": rf,
        "gam": gam,
        "lstm": lstm_model,
        "scaler_X": scaler_X,
        "scaler_y": scaler_y,
        "lstm_features": feat_cols,
    }


def predict_base_models(models, future_df, history_tail_df):
    future = future_df.copy().reset_index(drop=True)
    future["month"] = future["Date"].dt.month
    future["doy"] = future["Date"].dt.dayofyear
    future["sst_anomaly"] = future["sst_celsius"] - future["clim_mean"]

    # RF
    rf_pred = models["rf"].predict(
        future[["sst_celsius", "wind_speed", "tp_mm", "month", "doy"]]
    )

    # GAM
    gam_pred = models["gam"].predict(
        future[["sst_celsius", "wind_speed", "tp_mm", "doy"]].values
    )

    # LSTM uses the preceding 30 predictor days plus recursively available
    # forecast predictors. Its target predictions are NOT fed back as inputs.
    feat = models["lstm_features"]
    hist_tail = history_tail_df.copy().tail(LSTM_WINDOW)
    hist_tail["sst_anomaly"] = hist_tail["sst_celsius"] - hist_tail["clim_mean"]
    combined = pd.concat([hist_tail[feat], future[feat]], ignore_index=True)
    Xscaled = models["scaler_X"].transform(combined.values)
    Xseq = []
    for i in range(LSTM_WINDOW, len(Xscaled)):
        Xseq.append(Xscaled[i-LSTM_WINDOW:i])
    Xseq = np.asarray(Xseq)
    lstm_scaled = models["lstm"].predict(Xseq, verbose=0)
    lstm_pred = models["scaler_y"].inverse_transform(lstm_scaled).ravel()

    if not (len(rf_pred) == len(gam_pred) == len(lstm_pred) == len(future)):
        raise RuntimeError("Base prediction lengths are not aligned")

    return pd.DataFrame({
        "RF": rf_pred,
        "GAM": gam_pred,
        "LSTM": lstm_pred,
    })


def fit_leakage_free_stack(history_df, smoke=False):
    """
    Manuscript-consistent stacking:
      * first 75% historical observations: fit base learners
      * last 25%: temporally out-of-sample base predictions -> fit meta-model
      * refit base learners on all historical observations for hindcast use
    """
    history_df = history_df.sort_values("Date").reset_index(drop=True)
    split = int((1.0 - META_FRACTION) * len(history_df))
    if split <= LSTM_WINDOW + 100 or len(history_df) - split < 100:
        raise ValueError("Insufficient observations for base/meta temporal split")

    base_train = history_df.iloc[:split].copy()
    meta_period = history_df.iloc[split:].copy()

    base_models = fit_base_models(base_train, smoke=smoke)
    meta_pred = predict_base_models(base_models, meta_period, base_train)
    meta_y = meta_period["intensity"].values

    meta_model = LinearRegression().fit(meta_pred[["RF", "GAM", "LSTM"]], meta_y)
    meta_fit = meta_model.predict(meta_pred[["RF", "GAM", "LSTM"]])

    # Refit base learners using all information available before the hindcast year.
    final_models = fit_base_models(history_df, smoke=smoke)

    diagnostics = {
        "meta_start": meta_period["Date"].min(),
        "meta_end": meta_period["Date"].max(),
        "meta_n": len(meta_period),
        "meta_rmse": rmse(meta_y, meta_fit),
        "meta_mae": float(mean_absolute_error(meta_y, meta_fit)),
        "weights": np.asarray(meta_model.coef_, dtype=float),
        "intercept": float(meta_model.intercept_),
    }
    return final_models, meta_model, diagnostics


# ============================================================
# 6. ONE SITE / ONE HINDCAST YEAR
# ============================================================

def evaluate_site_year(df, label, year, smoke=False):
    origin = pd.Timestamp(year=year - 1, month=12, day=31)
    hist = df[(df["Date"] >= MIN_TRAIN_START) & (df["Date"] <= origin)].copy()
    obs = df[df["Date"].dt.year == year].copy()

    if len(hist) < 1000:
        raise ValueError(f"Too little historical data before {year}: n={len(hist)}")
    if len(obs) < 300:
        raise ValueError(f"Incomplete observed target year {year}: n={len(obs)}")

    lookup = make_fixed_climatology_lookup(hist)

    # Use exactly the dates available in the historical location file for that year.
    # This keeps the backtest compatible with prepared 365-day files that omit Feb 29.
    target_dates = obs["Date"].reset_index(drop=True)

    # ---------------- Stage 1 ----------------
    future_env, conv = forecast_environment(hist, target_dates, smoke=smoke)
    future_env = attach_climatology(future_env, lookup)
    future_env["sst_anomaly"] = future_env["sst_celsius"] - future_env["clim_mean"]

    # Observed year, using its supplied fixed climatology/threshold fields.
    obs = obs.sort_values("Date").reset_index(drop=True)
    obs_anom = obs["intensity"].values
    obs_thr_anom = (obs["threshold"] - obs["clim_mean"]).values

    # Direct baseline: forecast SST minus fixed climatology.
    direct_pred = future_env["sst_anomaly"].values

    # ---------------- Stage 2 ----------------
    final_models, meta_model, stack_diag = fit_leakage_free_stack(hist, smoke=smoke)
    base_future = predict_base_models(final_models, future_env, hist)
    stack_pred = meta_model.predict(base_future[["RF", "GAM", "LSTM"]])

    # ---------------- Daily anomaly accuracy ----------------
    direct_rmse = rmse(obs_anom, direct_pred)
    direct_mae = float(mean_absolute_error(obs_anom, direct_pred))
    stack_rmse = rmse(obs_anom, stack_pred)
    stack_mae = float(mean_absolute_error(obs_anom, stack_pred))

    # ---------------- Stage-1 predictor accuracy ----------------
    sst_rmse = rmse(obs["sst_celsius"], future_env["sst_celsius"])
    wind_rmse = rmse(obs["wind_speed"], future_env["wind_speed"])
    tp_rmse = rmse(obs["tp_mm"], future_env["tp_mm"])

    # ---------------- MHW event evaluation ----------------
    obs_chars, obs_days, obs_events = summarize_annual_characteristics(
        obs["Date"], obs_anom, obs_thr_anom
    )
    direct_chars, direct_days, direct_events = summarize_annual_characteristics(
        target_dates, direct_pred, future_env["threshold_anomaly"].values
    )
    stack_chars, stack_days, stack_events = summarize_annual_characteristics(
        target_dates, stack_pred, future_env["threshold_anomaly"].values
    )

    direct_cls = classification_scores(obs_days, direct_days)
    stack_cls = classification_scores(obs_days, stack_days)

    print(f"\n[{label} | {year}]")
    print("  Stage-1 RMSE: "
          f"SST={sst_rmse:.4f} C | Wind={wind_rmse:.4f} | Precip={tp_rmse:.4f}")
    print("  SARIMAX convergence: " + ", ".join(f"{k}={v}" for k, v in conv.items()))
    print("  Meta period: "
          f"{stack_diag['meta_start'].date()} to {stack_diag['meta_end'].date()} "
          f"(n={stack_diag['meta_n']})")
    w = stack_diag["weights"]
    print("  Meta model: "
          f"intercept={stack_diag['intercept']:.5f}; "
          f"RF={w[0]:.5f}, GAM={w[1]:.5f}, LSTM={w[2]:.5f}; "
          f"meta RMSE={stack_diag['meta_rmse']:.4f}")
    print("  Daily anomaly:")
    print(f"    Direct forecast-SST anomaly  RMSE={direct_rmse:.4f}  MAE={direct_mae:.4f}")
    print(f"    Two-stage stacking anomaly   RMSE={stack_rmse:.4f}  MAE={stack_mae:.4f}")
    print(f"    Paired RMSE gain (Direct-Stack) = {direct_rmse-stack_rmse:+.4f}")
    print("  MHW-day classification:")
    print(f"    Direct: Precision={direct_cls['precision']:.3f} Recall={direct_cls['recall']:.3f} "
          f"F1={direct_cls['f1']:.3f} CSI={direct_cls['csi']:.3f}")
    print(f"    Stack : Precision={stack_cls['precision']:.3f} Recall={stack_cls['recall']:.3f} "
          f"F1={stack_cls['f1']:.3f} CSI={stack_cls['csi']:.3f}")
    print("  Annual MHW characteristics (Observed | Direct | Stack):")
    for key in ["events", "mhw_days", "mean_duration", "mean_cumulative_anomaly", "mean_max_anomaly"]:
        a, b, c = obs_chars[key], direct_chars[key], stack_chars[key]
        def fmt(v):
            return "NA" if pd.isna(v) else (f"{int(v)}" if key in ["events", "mhw_days"] else f"{v:.4f}")
        print(f"    {key:25s} {fmt(a):>10s} | {fmt(b):>10s} | {fmt(c):>10s}")

    return {
        "site": label,
        "year": year,
        "n_days": len(obs),
        "sst_rmse": sst_rmse,
        "wind_rmse": wind_rmse,
        "tp_rmse": tp_rmse,
        "direct_rmse": direct_rmse,
        "direct_mae": direct_mae,
        "stack_rmse": stack_rmse,
        "stack_mae": stack_mae,
        "rmse_gain_direct_minus_stack": direct_rmse - stack_rmse,
        "direct_precision": direct_cls["precision"],
        "direct_recall": direct_cls["recall"],
        "direct_f1": direct_cls["f1"],
        "direct_csi": direct_cls["csi"],
        "stack_precision": stack_cls["precision"],
        "stack_recall": stack_cls["recall"],
        "stack_f1": stack_cls["f1"],
        "stack_csi": stack_cls["csi"],
        "obs_events": obs_chars["events"],
        "direct_events": direct_chars["events"],
        "stack_events": stack_chars["events"],
        "obs_mhw_days": obs_chars["mhw_days"],
        "direct_mhw_days": direct_chars["mhw_days"],
        "stack_mhw_days": stack_chars["mhw_days"],
        "meta_rf_weight": w[0],
        "meta_gam_weight": w[1],
        "meta_lstm_weight": w[2],
    }


# ============================================================
# 7. INTERNAL SELF-TEST (NO USER FILES, NO TENSORFLOW/PYGAM)
# ============================================================

def internal_self_test():
    """Test target audit, climatology, Stage 1, and event logic on synthetic data."""
    banner("INTERNAL SELF-TEST")
    rng = np.random.default_rng(123)
    dates = pd.date_range("2010-01-01", "2018-12-31", freq="D")
    # Remove leap days to mimic the prepared 365-day site files.
    dates = dates[~((dates.month == 2) & (dates.day == 29))]
    t = np.arange(len(dates))
    clim = 28 + 1.2 * np.sin(2*np.pi*t/365.0)
    threshold = clim + 0.6
    sst = clim + 0.3*np.sin(2*np.pi*t/40.0) + rng.normal(0, 0.12, len(t))
    wind = 5 + np.sin(2*np.pi*t/20.0) + rng.normal(0, 0.2, len(t))
    tp = np.clip(rng.gamma(0.8, 0.5, len(t)), 0, None)
    intensity = sst - clim
    syn = pd.DataFrame({
        "Date": dates,
        "sst_celsius": sst,
        "wind_speed": wind,
        "tp_mm": tp,
        "clim_mean": clim,
        "threshold": threshold,
        "intensity": intensity,
    })
    syn["month"] = syn["Date"].dt.month
    syn["doy"] = syn["Date"].dt.dayofyear
    syn["md"] = syn["Date"].dt.strftime("%m-%d")
    syn["sst_anomaly"] = syn["intensity"]
    syn["threshold_anomaly"] = syn["threshold"] - syn["clim_mean"]

    a = audit_target_definition(syn, "synthetic")
    assert a["target_max_clim"] < 1e-10
    lookup = make_fixed_climatology_lookup(syn)
    future_dates = syn[syn["Date"].dt.year == 2018]["Date"].reset_index(drop=True)
    hist = syn[syn["Date"] < "2018-01-01"].copy()
    env, conv = forecast_environment(hist, future_dates, smoke=True)
    env = attach_climatology(env, lookup)
    assert len(env) == len(future_dates)
    chars, days, events = summarize_annual_characteristics(
        syn.loc[syn["Date"].dt.year == 2018, "Date"],
        syn.loc[syn["Date"].dt.year == 2018, "intensity"],
        syn.loc[syn["Date"].dt.year == 2018, "threshold_anomaly"],
    )
    assert len(days) == len(future_dates)
    print("  PASS: target-definition audit")
    print("  PASS: climatology attachment")
    print("  PASS: Stage-1 SARIMAX predictor generation")
    print("  PASS: >=5-day Hobday event logic")
    print("  Note: full RF-GAM-LSTM fitting is tested when pygam and tensorflow are installed.")
    banner("SELF-TEST PASSED")


# ============================================================
# 8. MAIN EXECUTION
# ============================================================

def run_backtest():
    banner("MHW TWO-STAGE HISTORICAL BACKTEST")
    print("DATA_DIR       :", DATA_DIR)
    print("HINDCAST_YEARS :", HINDCAST_YEARS)
    print("SMOKE          :", SMOKE)
    print("SITE_LIMIT     :", SITE_LIMIT)

    files, rejected = find_location_files(DATA_DIR)
    if SITE_LIMIT is not None:
        files = files[:SITE_LIMIT]

    print(f"\nEligible historical location files detected: {len(files)}")
    for f in files:
        print("  ", os.path.basename(f))

    if len(files) == 0:
        raise FileNotFoundError(
            "No eligible location files detected. Each file must contain: "
            + ", ".join(sorted(REQUIRED_COLUMNS))
        )

    if SITE_LIMIT is None and len(files) != 15:
        print(f"\nWARNING: expected 15 location files but detected {len(files)}.")

    all_results = []
    failed = []

    for i, path in enumerate(files, 1):
        label = location_label(path)
        banner(f"SITE {i}/{len(files)}: {label}", char="-")
        try:
            df = read_location_file(path)
            print(f"Data: {df['Date'].min().date()} to {df['Date'].max().date()}, n={len(df)}")
            print("Columns include:", ", ".join(sorted(REQUIRED_COLUMNS)))
            audit_target_definition(df, label)

            for year in HINDCAST_YEARS:
                if year > int(df["Date"].dt.year.max()):
                    print(f"Skipping {year}: no observed target year in file")
                    continue
                res = evaluate_site_year(df, label, year, smoke=SMOKE)
                all_results.append(res)

        except Exception as e:
            failed.append((label, str(e)))
            print(f"\nFAILED [{label}]: {type(e).__name__}: {e}")

    if not all_results:
        banner("NO SUCCESSFUL BACKTESTS")
        if failed:
            print(pd.DataFrame(failed, columns=["site", "error"]).to_string(index=False))
        return None

    results = pd.DataFrame(all_results)

    banner("SITE-YEAR RESULTS")
    show_cols = [
        "site", "year", "sst_rmse",
        "direct_rmse", "stack_rmse", "rmse_gain_direct_minus_stack",
        "direct_f1", "stack_f1", "direct_csi", "stack_csi",
        "obs_events", "direct_events", "stack_events",
        "obs_mhw_days", "direct_mhw_days", "stack_mhw_days",
    ]
    print(results[show_cols].round(4).to_string(index=False))

    banner("AGGREGATE SUMMARY")
    print("Successful site-years:", len(results))
    print(f"Mean Stage-1 SST RMSE        : {results['sst_rmse'].mean():.4f} C")
    print(f"Mean direct anomaly RMSE     : {results['direct_rmse'].mean():.4f} C")
    print(f"Mean stacking anomaly RMSE   : {results['stack_rmse'].mean():.4f} C")
    print(f"Mean RMSE gain Direct-Stack  : {results['rmse_gain_direct_minus_stack'].mean():+.4f} C")
    print(f"Median RMSE gain             : {results['rmse_gain_direct_minus_stack'].median():+.4f} C")
    print(f"Stack RMSE win percentage    : {100*np.mean(results['stack_rmse'] < results['direct_rmse']):.1f}%")
    print(f"Mean direct F1 / CSI         : {results['direct_f1'].mean():.3f} / {results['direct_csi'].mean():.3f}")
    print(f"Mean stacking F1 / CSI       : {results['stack_f1'].mean():.3f} / {results['stack_csi'].mean():.3f}")
    print(f"Mean |MHW-day error| direct  : {np.mean(np.abs(results['direct_mhw_days']-results['obs_mhw_days'])):.3f}")
    print(f"Mean |MHW-day error| stack   : {np.mean(np.abs(results['stack_mhw_days']-results['obs_mhw_days'])):.3f}")
    print(f"Mean |event-count error| direct: {np.mean(np.abs(results['direct_events']-results['obs_events'])):.3f}")
    print(f"Mean |event-count error| stack : {np.mean(np.abs(results['stack_events']-results['obs_events'])):.3f}")

    banner("STACKING WEIGHTS")
    print(results[["site", "year", "meta_rf_weight", "meta_gam_weight", "meta_lstm_weight"]].round(4).to_string(index=False))

    if failed:
        banner("FAILED SITE-YEARS / FILES")
        print(pd.DataFrame(failed, columns=["site", "error"]).to_string(index=False))

    banner("RUN COMPLETE")
    print("Results were printed only; no manuscript table was overwritten.")

    return results


if __name__ == "__main__":
    # Set to True only if you want to test the script structure without user files.
    RUN_INTERNAL_SELF_TEST = False

    if RUN_INTERNAL_SELF_TEST:
        internal_self_test()
    else:
        run_backtest()
