import numpy as np
import pandas as pd

from mhw_repro.mhw import annual_characteristics, detect_events, event_table


def make_frame(intensity, start="2001-01-01"):
    dates = pd.date_range(start, periods=len(intensity), freq="D")
    return pd.DataFrame(
        {
            "Date": dates,
            "intensity": intensity,
            "threshold_anomaly": np.repeat(0.5, len(intensity)),
        }
    )


def test_four_days_do_not_qualify_but_five_days_do():
    frame = make_frame([0.1, *([0.8] * 4), 0.1, *([0.8] * 5), 0.1])
    labels = detect_events(frame, min_duration=5)
    assert int(labels.max()) == 1
    assert int((labels > 0).sum()) == 5


def test_date_gap_breaks_consecutive_run():
    frame = make_frame([0.8] * 6)
    frame.loc[3:, "Date"] = frame.loc[3:, "Date"] + pd.Timedelta(days=1)
    labels = detect_events(frame, min_duration=5)
    assert int((labels > 0).sum()) == 0


def test_severity_is_sum_relative_to_climatological_mean():
    frame = make_frame([0.6, 0.7, 0.8, 0.9, 1.0])
    events = event_table(frame, min_duration=5)
    assert len(events) == 1
    assert np.isclose(events.loc[0, "cumulative_intensity_c_days"], 4.0)
    # Threshold exceedance would be 1.5; the manuscript definition is 4.0.
    assert not np.isclose(events.loc[0, "cumulative_intensity_c_days"], 1.5)


def test_no_event_year_policy():
    dates = pd.date_range("2001-01-01", "2002-12-31", freq="D")
    frame = pd.DataFrame(
        {"Date": dates, "intensity": 0.1, "threshold_anomaly": 0.5}
    )
    mask = (frame["Date"] >= "2001-07-01") & (frame["Date"] <= "2001-07-05")
    frame.loc[mask, "intensity"] = 0.8
    annual, _, _ = annual_characteristics(frame, years=[2001, 2002])
    no_event = annual.loc[annual["year"] == 2002].iloc[0]
    assert no_event["event_frequency"] == 0
    assert no_event["annual_mhw_days"] == 0
    assert np.isnan(no_event["mean_event_duration_days"])

