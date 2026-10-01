# Conditional 2026 projection — pseudocode

This file is algorithm documentation. It does not represent subsequently observed 2026 ERA5 values.

```text
forecast origin = 2025-12-31
reported projection window = 2026-02-26 through 2026-12-31

FOR each site:
    use historical data ending on 2025-12-31
    retain the fixed 1983-2012 climatological mean and p90 threshold

    FOR predictor in [SST, wind speed, precipitation]:
        fit SARIMAX(1,1,1) with 3 annual Fourier harmonics
        forecast a continuous daily predictor path beginning 2026-01-01
    END FOR

    use the internally generated January-February path only to establish the
    lagged and 30-day rolling/sequence state needed on 2026-02-26
    do not insert observed 2026 values and do not refit models or ensemble weights

    apply the frozen RF, GAM, and LSTM base learners
    combine their daily anomaly predictions with the frozen linear meta-model
    retain predictions dated 2026-02-26 through 2026-12-31

    detect days where predicted anomaly > fixed threshold anomaly
    retain runs of at least 5 consecutive days
    for each retained event calculate:
        start and end date
        duration
        mean and maximum climatological-mean anomaly
        cumulative climatological-mean anomaly
        peak category ratio and category

    estimate the displayed residual margin from held-out historical residuals
    label it an approximate residual interval, not a complete probabilistic interval
END FOR

print site summary, detailed event table, and model diagnostics
```

The internal predictor path before 26 February is not part of the reported event-projection window. No RMSE or MAE is claimed for that interval in the manuscript.

