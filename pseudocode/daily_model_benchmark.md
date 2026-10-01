# Eight-model daily benchmark — pseudocode

This file records the exact modelling logic and settings but is not an executable replacement for the archived analysis notebooks.

## Shared chronological design

```text
training rows = dates from 1983-01-01 through 2016-12-31
test rows     = dates from 2017-01-01 through 2025-12-31
target        = SST - climatological mean

fit every scaler, hyperparameter search, feature transformation, and model on training data only
for lagged or rolling features, use values strictly earlier than the target day
preserve the date index when aligning predictions and observations
calculate RMSE and MAE separately at every site
```

## Model specifications

```text
SARIMAX-FOURIER
    response: daily anomaly
    exogenous fields: SST, wind, precipitation, 3 annual sine/cosine pairs
    select p and q from 0..3 by training AIC; use d=0
    fit on a complete daily index

RANDOM FOREST
    features: same-day SST, wind, precipitation, month, day of year
    parameters: 200 trees, max depth 8, minimum split 5, seed 42

XGBOOST
    features: same-day fields, seasonal variables,
              1/3/7-day lags, shifted 3/7/30-day rolling statistics
    parameters: 50 trees, learning rate 0.05, depth 3,
                subsample 0.8, column subsample 0.8, seed 42

LSTM
    input sequence: previous 30 days of anomaly, wind, and precipitation
    current-day values are excluded
    parameters: 32 units, dropout 0.20, maximum 50 epochs,
                batch size 32, early stopping, no shuffling, seed 42

GAM
    features: same-day SST, wind, precipitation, day of year
    four smooth terms, 10 splines per term, training-only smoothing grid search

SVR
    features: same-day SST, wind, precipitation
    standardize using training mean and standard deviation
    RBF kernel; select from 20 randomized settings with 3-fold time-series CV

AR-RF
    Random Forest features plus anomaly lags t-1, t-2, and t-3
    parameters: same RF settings
    at test time, document whether lagged anomalies are observed (one-step
    reconstruction) or recursively predicted (multi-step forecast)

RF-GAM-LSTM STACKING
    divide pre-test training rows chronologically:
        first 75% -> fit RF, GAM, and LSTM base learners
        last 25%  -> obtain out-of-sample base predictions and fit linear meta-model
    freeze meta-model
    refit base learners on all pre-test training rows
    generate test predictions and combine with frozen meta-model
```

## Evaluation and reconstruction

```text
for each site and model:
    compute daily RMSE and MAE over aligned finite 2017-2025 pairs
    apply the observed calendar-day threshold anomaly to predicted anomaly
    retain runs lasting at least 5 consecutive days
    calculate annual frequency and MHW days for every test year
    calculate event-level annual means only for years with predicted events
    compare event-level means only where both observed and predicted values exist
print the complete site-by-model tables
```

