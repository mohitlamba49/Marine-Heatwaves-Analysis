"""Deterministic synthetic smoke test; not a manuscript result."""

import numpy as np
import pandas as pd

from mhw_repro.mhw import annual_characteristics
from mhw_repro.trends import trend_table


dates = pd.date_range("2000-01-01", "2009-12-31", freq="D")
frame = pd.DataFrame({"Date": dates})
frame["clim_mean"] = 28.0 + np.sin(2 * np.pi * frame["Date"].dt.dayofyear / 365.0)
frame["threshold"] = frame["clim_mean"] + 0.5
frame["intensity"] = 0.1

# One increasingly long event in each year. Each event is safely above threshold.
for year in range(2000, 2010):
    duration = 5 + (year - 2000)
    mask = (frame["Date"] >= f"{year}-07-01") & (
        frame["Date"] < pd.Timestamp(f"{year}-07-01") + pd.Timedelta(days=duration)
    )
    frame.loc[mask, "intensity"] = 0.8

frame["sst_celsius"] = frame["clim_mean"] + frame["intensity"]
frame["threshold_anomaly"] = frame["threshold"] - frame["clim_mean"]

annual, events, labels = annual_characteristics(frame, min_duration=5)
trends = trend_table(annual)

assert len(events) == 10
assert int((labels > 0).sum()) == sum(range(5, 15))
assert annual["event_frequency"].eq(1).all()
assert annual["mean_event_duration_days"].tolist() == list(range(5, 15))

print("SYNTHETIC ANNUAL CHARACTERISTICS")
print(annual.to_string(index=False))
print("\nSYNTHETIC TREND RESULTS")
print(trends.round(6).to_string(index=False))
print("\nSMOKE TEST PASSED")

