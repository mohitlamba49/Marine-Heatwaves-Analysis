import numpy as np
import pandas as pd

from mhw_repro.evaluation import event_scores, regression_metrics


def test_regression_metrics():
    result = regression_metrics([1.0, 2.0, 3.0], [1.0, 1.0, 4.0])
    assert result["n"] == 3
    assert np.isclose(result["mae"], 2 / 3)
    assert np.isclose(result["rmse"], np.sqrt(2 / 3))


def test_event_scores():
    observed = pd.Series([0, 1, 1, 0, 1])
    predicted = pd.Series([0, 1, 0, 1, 1])
    result = event_scores(observed, predicted)
    assert result["hits"] == 2
    assert result["false_alarms"] == 1
    assert result["misses"] == 1
    assert np.isclose(result["f1"], 2 / 3)
    assert np.isclose(result["csi"], 0.5)

