# Full analysis algorithm — pseudocode

This file is documentation, not executable source code.

```text
FOR each of the 15 configured sites:
    load daily SST, wind speed, precipitation, climatological mean, and p90 threshold
    sort by date; reject duplicate dates; report gaps
    set anomaly = SST - climatological mean
    set threshold anomaly = p90 threshold - climatological mean

    OBSERVED CHARACTERIZATION
        mark days where anomaly > threshold anomaly
        retain runs of at least 5 consecutive calendar days
        summarize each event and annual characteristics
        run Mann-Kendall and Sen-slope analysis for 1983-2025

    DAILY RECONSTRUCTION BENCHMARK
        split chronologically: 1983-2016 training, 2017-2025 test
        fit SARIMAX, RF, XGBoost, LSTM, GAM, SVR, and AR-RF
        fit RF-GAM-LSTM stack using a chronological base/meta split inside training only
        predict each test day without using future observations
        calculate site-level RMSE and MAE
        reconstruct annual MHW characteristics from each model's predicted anomaly

    TWO-STAGE 2025 BACKTEST
        withhold all of 2025
        forecast 2025 SST, wind, and precipitation from data ending 2024-12-31
        transform these predictor forecasts to anomaly using the fitted stack
        detect predicted events and compare with observed 2025

    CONDITIONAL 2026 PROJECTION
        fit with information ending 2025-12-31
        generate predictor paths needed to initialize the 30-day sequence
        report projected events only for 2026-02-26 through 2026-12-31
        label every result as conditional on the statistically forecast predictors
END FOR

aggregate only after retaining site identifiers and experiment labels
do not mix daily-reconstruction metrics with two-stage forecast metrics
```

