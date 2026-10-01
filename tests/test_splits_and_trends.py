import numpy as np
import pandas as pd

from mhw_repro.splits import chronological_base_meta_split
from mhw_repro.trends import mann_kendall_sen


def test_chronological_split_is_ordered_and_non_overlapping():
    base, meta = chronological_base_meta_split(100, base_fraction=0.75)
    base_positions = np.arange(100)[base]
    meta_positions = np.arange(100)[meta]
    assert len(base_positions) == 75
    assert len(meta_positions) == 25
    assert base_positions.max() < meta_positions.min()


def test_event_mean_trend_uses_only_defined_years():
    years = pd.Series([2000, 2001, 2002, 2003, 2004])
    values = pd.Series([5.0, np.nan, 7.0, np.nan, 9.0])
    result = mann_kendall_sen(years, values)
    assert result["years_used"] == 3
    assert np.isclose(result["sen_slope_per_year"], 1.0)

