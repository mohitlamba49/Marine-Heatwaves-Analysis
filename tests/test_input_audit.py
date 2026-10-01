import numpy as np
import pandas as pd

from mhw_repro.io import load_site_file


def test_target_is_recomputed_from_sst_and_climatology(tmp_path):
    path = tmp_path / "site.csv"
    pd.DataFrame(
        {
            "Date": pd.date_range("2001-01-01", periods=5),
            "sst_celsius": [29, 30, 31, 30, 29],
            "wind_speed": 5.0,
            "tp_mm": 0.0,
            "clim_mean": 28.0,
            "threshold": 28.5,
            "intensity": 999.0,
        }
    ).to_csv(path, index=False)
    frame, audit = load_site_file(path)
    assert np.allclose(frame["intensity"], [1, 2, 3, 2, 1])
    assert audit["target_mae_vs_supplied"] > 900

