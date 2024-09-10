import pytest
import pandas as pd
from feat_engine.temporal_features import TemporalFeatureEngineering


@pytest.fixture
def sample_data() -> pd.DataFrame:
    """
    Fixture to provide sample temporal data for testing.

    Returns:
    - pd.DataFrame: Sample dataframe with datetime and value columns.
    """
    data = {'date': ['2023-01-01 12:00:00', '2023-01-02 13:30:00', '2023-01-03 15:45:00'],
            'value': [100, 150, 200]}
    df = pd.DataFrame(data)
    df['date'] = pd.to_datetime(df['date'])
    return df


def test_convert_to_datetime(sample_data: pd.DataFrame) -> None:
    """
    Test datetime conversion.

    Args:
    - sample_data (pd.DataFrame): Sample dataframe.

    Asserts:
    - The column is correctly converted to datetime.
    """
    tfe = TemporalFeatureEngineering()
    df = tfe.convert_to_datetime(sample_data, 'date')
    assert pd.api.types.is_datetime64_any_dtype(df['date'])


def test_extract_date_parts(sample_data: pd.DataFrame) -> None:
    """
    Test extracting date parts like year, month, day, etc.

    Args:
    - sample_data (pd.DataFrame): Sample dataframe.

    Asserts:
    - The date parts (year, month, day, etc.) are correctly extracted.
    """
    tfe = TemporalFeatureEngineering()
    df = tfe.extract_date_parts(sample_data, 'date')

    assert 'year' in df.columns
    assert 'month' in df.columns
    assert 'day' in df.columns
    assert 'day_of_week' in df.columns
    assert 'hour' in df.columns
    assert 'minute' in df.columns
    assert 'second' in df.columns


def test_create_lag_features(sample_data: pd.DataFrame) -> None:
    """
    Test creating lag features.

    Args:
    - sample_data (pd.DataFrame): Sample dataframe.

    Asserts:
    - Lag features are correctly created for specified lags.
    """
    tfe = TemporalFeatureEngineering()
    df = tfe.create_lag_features(sample_data, 'value', [1, 2])

    assert 'value_lag_1' in df.columns
    assert 'value_lag_2' in df.columns


def test_create_rolling_features(sample_data: pd.DataFrame) -> None:
    """
    Test creating rolling features like rolling mean.

    Args:
    - sample_data (pd.DataFrame): Sample dataframe.

    Asserts:
    - Rolling feature columns are correctly created.
    """
    tfe = TemporalFeatureEngineering()
    df = tfe.create_rolling_features(sample_data, 'value', window_size=2, feature='mean')

    assert 'value_rolling_mean_2' in df.columns


def test_cyclical_features(sample_data: pd.DataFrame) -> None:
    """
    Test creation of cyclical features like sine and cosine transformation.

    Args:
    - sample_data (pd.DataFrame): Sample dataframe.

    Asserts:
    - Cyclical features are created for time-related columns.
    """
    tfe = TemporalFeatureEngineering()
    df = tfe.extract_date_parts(sample_data, 'date')
    df = tfe.cyclical_features(df, 'hour', max_value=24)

    assert 'hour_sin' in df.columns
    assert 'hour_cos' in df.columns


def test_resample_data(sample_data: pd.DataFrame) -> None:
    """
    Test resampling data with specific aggregation (e.g., sum).

    Args:
    - sample_data (pd.DataFrame): Sample dataframe.

    Asserts:
    - Data is correctly resampled and aggregated.
    """
    tfe = TemporalFeatureEngineering()
    df_resampled = tfe.resample_data(sample_data, 'date', rule='D', aggregation='sum')

    assert isinstance(df_resampled, pd.DataFrame)
    assert df_resampled.shape[0] == 3  # Expect 3 daily records
