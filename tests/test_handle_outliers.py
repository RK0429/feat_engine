import pytest
import pandas as pd
from feat_engine.handle_outliers import OutlierHandler


@pytest.fixture
def sample_data() -> pd.DataFrame:
    """
    Fixture to provide a sample dataframe for testing.

    Returns:
    - pd.DataFrame: A sample dataframe for outlier testing.
    """
    data = {
        'x1': [10, 12, 13, 15, 20, 100],
        'x2': [50, 55, 52, 60, 70, 200],
        'x3': [30, 28, 35, 40, 45, 500]
    }
    return pd.DataFrame(data)


def test_z_score_detection(sample_data: pd.DataFrame) -> None:
    """
    Test Z-Score based outlier detection.

    Args:
    - sample_data (pd.DataFrame): Sample dataframe fixture.

    Asserts:
    - Whether the last row is detected as an outlier.
    - Other rows are not detected as outliers.
    """
    oh = OutlierHandler()
    z_outliers = oh.z_score_detection(sample_data, threshold=2)

    assert z_outliers.iloc[-1].all()  # Last row should have outliers
    assert not z_outliers.iloc[:-1].any().any()  # Other rows should not have outliers


def test_iqr_detection(sample_data: pd.DataFrame) -> None:
    """
    Test IQR based outlier detection.

    Args:
    - sample_data (pd.DataFrame): Sample dataframe fixture.

    Asserts:
    - Whether the last row is detected as an outlier.
    - Other rows are not detected as outliers.
    """
    oh = OutlierHandler()
    iqr_outliers = oh.iqr_detection(sample_data)

    assert iqr_outliers.iloc[-1].all()  # Last row should have outliers
    assert not iqr_outliers.iloc[:-1].any().any()  # Other rows should not have outliers


def test_isolation_forest_detection(sample_data: pd.DataFrame) -> None:
    """
    Test Isolation Forest based outlier detection.

    Args:
    - sample_data (pd.DataFrame): Sample dataframe fixture.

    Asserts:
    - Whether the last row is detected as an outlier.
    - Other rows are not detected as outliers.
    """
    oh = OutlierHandler()
    iso_forest_outliers = oh.isolation_forest_detection(sample_data)

    assert iso_forest_outliers.iloc[-1]  # Last row should be an outlier
    assert not iso_forest_outliers.iloc[:-1].any()  # Other rows should not be outliers


def test_dbscan_detection(sample_data: pd.DataFrame) -> None:
    """
    Test DBSCAN based outlier detection.

    Args:
    - sample_data (pd.DataFrame): Sample dataframe fixture.

    Asserts:
    - Whether the last row is detected as an outlier.
    - Other rows are not detected as outliers.
    """
    oh = OutlierHandler()
    dbscan_outliers = oh.dbscan_detection(sample_data, eps=20, min_samples=2)

    assert dbscan_outliers.iloc[-1]  # Last row should be an outlier
    assert not dbscan_outliers.iloc[:-1].any()  # Other rows should not be outliers


def test_robust_scaler(sample_data: pd.DataFrame) -> None:
    """
    Test RobustScaler transformation.

    Args:
    - sample_data (pd.DataFrame): Sample dataframe fixture.

    Asserts:
    - The scaled values are within a reasonable range.
    """
    oh = OutlierHandler()
    robust_scaled_df = oh.robust_scaler(sample_data, 0.2, 0.8)

    assert robust_scaled_df.max().max() <= 5  # Adjust the expected range
    assert robust_scaled_df.min().min() >= -5  # Ensure scaled values are within a reasonable range


def test_winsorization(sample_data: pd.DataFrame) -> None:
    """
    Test Winsorization of extreme values.

    Args:
    - sample_data (pd.DataFrame): Sample dataframe fixture.

    Asserts:
    - Extreme values are within a reasonable range after Winsorization.
    """
    oh = OutlierHandler()
    winsorized_df = oh.winsorization(sample_data, limits=(0.2, 0.2))

    assert winsorized_df.max().max() <= 100  # Ensure extreme high values are capped
    assert winsorized_df.min().min() >= 10  # Ensure extreme low values are capped


def test_cap_outliers(sample_data: pd.DataFrame) -> None:
    """
    Test capping outliers based on the IQR method.

    Args:
    - sample_data (pd.DataFrame): Sample dataframe fixture.

    Asserts:
    - Outliers are capped within the IQR bounds.
    """
    oh = OutlierHandler()
    capped_df = oh.cap_outliers(sample_data, method='iqr', range_ratio=0.4)

    assert capped_df.max().max() <= 100  # Ensure extreme high values are capped
    assert capped_df.min().min() >= 10  # Ensure extreme low values are capped
