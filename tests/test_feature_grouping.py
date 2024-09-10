import pytest
import pandas as pd
from feat_engine.feature_grouping import FeatureGrouping


@pytest.fixture
def sample_data() -> pd.DataFrame:
    """
    Fixture to provide a sample dataframe for testing.

    Returns:
    - pd.DataFrame: Sample dataframe.
    """
    data = {
        'customer_id': [1, 2, 1, 3, 2],
        'product_id': ['A', 'B', 'A', 'C', 'B'],
        'transaction_value': [100, 150, 200, 300, 100],
        'date': ['2023-01-01', '2023-01-10', '2023-02-01', '2023-02-15', '2023-03-01']
    }
    df = pd.DataFrame(data)
    df['date'] = pd.to_datetime(df['date'])  # Convert to datetime
    return df


def test_group_by_category(sample_data: pd.DataFrame) -> None:
    """
    Test grouping by a single category column.
    """
    fga = FeatureGrouping()
    grouped = fga.group_by_category(sample_data, 'customer_id', 'transaction_value', ['sum', 'mean', 'count'])

    # Verify the output shape and column names
    assert grouped.shape == (3, 4)  # 3 unique customer_id values and 4 columns (customer_id, sum, mean, count)
    assert set(grouped.columns) == {'customer_id', 'sum', 'mean', 'count'}

    # Verify correct aggregation
    assert grouped.loc[grouped['customer_id'] == 1, 'sum'].values[0] == 300  # Customer 1 spent 300 in total


def test_group_by_multiple_categories(sample_data: pd.DataFrame) -> None:
    """
    Test grouping by multiple category columns.
    """
    fga = FeatureGrouping()
    grouped = fga.group_by_multiple_categories(sample_data, ['customer_id', 'product_id'], 'transaction_value', ['sum', 'mean'])

    # Verify the output shape and column names
    assert grouped.shape == (3, 4)  # 4 unique combinations of customer_id and product_id
    assert set(grouped.columns) == {'customer_id', 'product_id', 'sum', 'mean'}

    # Verify correct aggregation
    assert grouped.loc[(grouped['customer_id'] == 1) & (grouped['product_id'] == 'A'), 'sum'].values[0] == 300


def test_time_based_aggregation(sample_data: pd.DataFrame) -> None:
    """
    Test time-based aggregation.
    """
    fga = FeatureGrouping()
    time_grouped = fga.aggregate_time_based(sample_data, 'date', 'transaction_value', 'M', 'sum')

    # Verify the output shape and column names
    assert time_grouped.shape == (3, 2)  # 3 unique months
    assert set(time_grouped.columns) == {'date', 'transaction_value'}

    # Verify correct aggregation
    assert time_grouped.loc[time_grouped['date'] == pd.Timestamp('2023-01-31'), 'transaction_value'].values[0] == 250


def test_rolling_aggregation(sample_data: pd.DataFrame) -> None:
    """
    Test rolling aggregation.
    """
    fga = FeatureGrouping()
    rolling_df = fga.rolling_aggregation(sample_data, 'transaction_value', window=2, metric='mean')

    # Verify the rolling aggregation results
    assert 'transaction_value_rolling_mean' in rolling_df.columns
    assert pd.isna(rolling_df['transaction_value_rolling_mean'].iloc[0])  # First row should be NaN
    assert rolling_df['transaction_value_rolling_mean'].iloc[1] == 125  # Mean of first 2 rows


def test_percentile_calculation(sample_data: pd.DataFrame) -> None:
    """
    Test percentile calculation for a grouped column.
    """
    fga = FeatureGrouping()
    percentiles = fga.calculate_percentiles(sample_data, 'customer_id', 'transaction_value', [0.25, 0.5, 0.75])

    # Verify the output shape and correct column names
    assert percentiles.shape == (9, 3)  # 3 customer IDs * 3 percentiles
    assert set(percentiles.columns) == {'customer_id', 'level_1', 'transaction_value'}

    # Check percentile values
    customer_1_median = percentiles.loc[(percentiles['customer_id'] == 1) & (percentiles['level_1'] == 0.5), 'transaction_value'].values[0]
    assert customer_1_median == 150  # Customer 1's median transaction value
