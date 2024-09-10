import pytest
import pandas as pd
import numpy as np
from feat_engine import MissingValueHandler


# Test data for missing value handling
@pytest.fixture
def sample_data() -> pd.DataFrame:
    return pd.DataFrame({
        'A': [1, 2, np.nan, 4],
        'B': [np.nan, 5, 6, np.nan],
        'C': [7, np.nan, 9, 10]
    })


def test_identify_missing(sample_data: pd.DataFrame) -> None:
    result = MissingValueHandler.identify_missing(sample_data)
    expected = pd.DataFrame({
        'A': [False, False, True, False],
        'B': [True, False, False, True],
        'C': [False, True, False, False]
    })
    pd.testing.assert_frame_equal(result, expected)


def test_missing_summary(sample_data: pd.DataFrame) -> None:
    result = MissingValueHandler.missing_summary(sample_data)
    expected = pd.Series({'A': 1, 'B': 2, 'C': 1})
    pd.testing.assert_series_equal(result, expected)


def test_drop_missing_rows(sample_data: pd.DataFrame) -> None:
    result = MissingValueHandler.drop_missing(sample_data, axis=0, how='all')
    expected = pd.DataFrame({
        'A': [1, 2, np.nan, 4],
        'B': [np.nan, 5, 6, np.nan],
        'C': [7, np.nan, 9, 10]
    }).reset_index(drop=True)
    pd.testing.assert_frame_equal(result.reset_index(drop=True), expected)


def test_drop_missing_columns(sample_data: pd.DataFrame) -> None:
    result = MissingValueHandler.drop_missing(sample_data, axis=1, how='all')
    expected = pd.DataFrame({
        'A': [1, 2, np.nan, 4],
        'B': [np.nan, 5, 6, np.nan],
        'C': [7, np.nan, 9, 10]
    })
    pd.testing.assert_frame_equal(result, expected)


def test_fill_missing_mean(sample_data: pd.DataFrame) -> None:
    result = MissingValueHandler.fill_missing(sample_data, strategy='mean')
    expected = pd.DataFrame({
        'A': [1, 2, 2.333333, 4],   # Mean of [1, 2, 4] = 2.333333
        'B': [5.5, 5, 6, 5.5],      # Mean of [5, 6] = 5.5
        'C': [7, 8.666667, 9, 10]   # Mean of [7, 9, 10] = 8.666667
    })
    pd.testing.assert_frame_equal(result, expected)


def test_fill_missing_constant(sample_data: pd.DataFrame) -> None:
    result = MissingValueHandler.fill_missing_constant(sample_data, fill_value=0)
    expected = pd.DataFrame({
        'A': [1.0, 2.0, 0.0, 4.0],
        'B': [0.0, 5.0, 6.0, 0.0],
        'C': [7.0, 0.0, 9.0, 10.0]
    })
    pd.testing.assert_frame_equal(result, expected)


def test_fill_missing_knn(sample_data: pd.DataFrame) -> None:
    result = MissingValueHandler.fill_missing_knn(sample_data, n_neighbors=2)
    expected = pd.DataFrame({
        'A': [1.0, 2.0, 3.0, 4.0],         # Use float64 instead of int64
        'B': [5.5, 5.0, 6.0, 5.5],         # Use float64 values as KNNImputer returns floats
        'C': [7.0, 8.0, 9.0, 10.0]         # Update to match actual imputed value
    })
    pd.testing.assert_frame_equal(result, expected, atol=1e-1)


def test_fill_missing_iterative(sample_data: pd.DataFrame) -> None:
    result = MissingValueHandler.fill_missing_iterative(sample_data)
    # The output will be an imputed DataFrame, which will vary slightly depending on the iterative imputation method.
    assert not result.isnull().values.any()  # Check that no NaN values remain


def test_add_missing_indicator(sample_data: pd.DataFrame) -> None:
    result = MissingValueHandler.add_missing_indicator(sample_data)
    expected = pd.DataFrame({
        'A': [1, 2, np.nan, 4],
        'B': [np.nan, 5, 6, np.nan],
        'C': [7, np.nan, 9, 10],
        'A_missing': [0, 0, 1, 0],
        'B_missing': [1, 0, 0, 1],
        'C_missing': [0, 1, 0, 0]
    })
    pd.testing.assert_frame_equal(result, expected)
