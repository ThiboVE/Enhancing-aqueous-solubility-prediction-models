import numpy as np
import pandas as pd
import pytest

from ml_enhance import FeatureImportance


class DummyModel:
    def __init__(self, coef: list[float]) -> None:
        self.coef_ = np.array(coef)


def test_basic_feature_importance():
    df = pd.DataFrame(
        {
            "estimator": [
                DummyModel([1, 2, 3]),
                DummyModel([1, 2, 3]),
            ]
        }
    )

    fi = FeatureImportance(df)
    result = fi.get_feature_importance(num_features=3, mode="full")

    assert not result.empty
    assert set(result["feature"]) == {"f0", "f1", "f2"}
    assert all(result["frequency"] == 1.0)


def test_provided_fi_used():
    df = pd.DataFrame(
        {
            "estimator": [None, None],
        }
    )
    df.index = [0, 1]

    provided = {
        0: pd.Series([1, 2], index=["a", "b"]),
        1: pd.Series([1, 2], index=["a", "b"]),
    }

    fi = FeatureImportance(df, includes_FI=True, provided_FI=provided)
    result = fi.get_feature_importance(mode="full")

    assert set(result["feature"]) == {"a", "b"}
    assert all(result["frequency"] == 1.0)


def test_missing_folds_raises():
    df = pd.DataFrame({"estimator": [None, None]})
    df.index = [0, 1]

    provided = {0: pd.Series([1, 2], index=["a", "b"])}

    with pytest.raises(ValueError):
        FeatureImportance(df, includes_FI=True, provided_FI=provided)


def test_two_stage_selection():
    df = pd.DataFrame({"estimator": [DummyModel([1, 2, 3])]})

    fi = FeatureImportance(df)
    result = fi.get_feature_importance(num_features=1, mode="two_stage")

    assert len(result) == 1
    assert result.iloc[0]["feature"] == "f2"


def test_weighting():
    df = pd.DataFrame(
        {
            "estimator": [
                DummyModel([1, 2]),
                DummyModel([1, 2]),
            ],
            "test_r2": [0.89, 0.5],
        }
    )

    fi = FeatureImportance(df)
    result = fi.get_feature_importance(mode="full", weight_by_score=True)

    f0 = result[result["feature"] == "f0"].iloc[0]
    f1 = result[result["feature"] == "f1"].iloc[0]

    assert np.isclose(f0["mean_importance"], 0.695)
    assert np.isclose(f1["mean_importance"], 1.39)


def test_inconsistent_features():
    df = pd.DataFrame({"estimator": [None, None]})
    df.index = [0, 1]

    provided = {
        0: pd.Series([1, 2], index=["a", "b"]),
        1: pd.Series([1, 2], index=["a", "c"]),  # mismatch
    }

    fi = FeatureImportance(df, includes_FI=True, provided_FI=provided)

    # should not crash, but features will differ
    result = fi.get_feature_importance(mode="full")

    assert set(result["feature"]) == {"a", "b", "c"}
