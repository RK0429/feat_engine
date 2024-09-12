import pytest
import pandas as pd
from pandas import DataFrame
from typing import Tuple
from feat_engine.operate_dataframe import DataFrameOperator


@pytest.fixture
def sample_dataframes() -> Tuple[DataFrame, DataFrame]:
    """
    Creates sample dataframes for testing.

    Returns:
        Tuple[DataFrame, DataFrame]: Two sample DataFrames for testing.
    """
    df1 = pd.DataFrame({
        'id': [1, 2, 3],
        'value1': [10, 20, 30]
    })

    df2 = pd.DataFrame({
        'id': [2, 3, 4],
        'value2': [200, 300, 400]
    })

    return df1, df2


@pytest.fixture
def df_with_missing() -> DataFrame:
    """
    Creates a sample dataframe with missing values.

    Returns:
        DataFrame: A sample DataFrame with missing values.
    """
    df = pd.DataFrame({
        'A': [1, 2, None],
        'B': [None, 5, 6],
        'C': [7, 8, 9]
    })

    return df


def test_hmerge(sample_dataframes: Tuple[DataFrame, DataFrame]) -> None:
    """
    Test horizontal merge of two dataframes.

    Args:
        sample_dataframes (Tuple[DataFrame, DataFrame]): Sample DataFrames for testing.
    """
    df1, df2 = sample_dataframes
    merged_df = DataFrameOperator.hmerge(df1, df2, on='id')

    assert 'value1' in merged_df.columns
    assert 'value2' in merged_df.columns
    assert len(merged_df) == 4  # Outer join should result in 4 rows.


def test_vmerge(sample_dataframes: Tuple[DataFrame, DataFrame]) -> None:
    """
    Test vertical merge of two dataframes.

    Args:
        sample_dataframes (Tuple[DataFrame, DataFrame]): Sample DataFrames for testing.
    """
    df1, df2 = sample_dataframes
    merged_df = DataFrameOperator.vmerge(df1, df2)

    assert len(merged_df) == 6  # The resulting dataframe should have 6 rows.
    assert 'id' in merged_df.columns
    assert 'value1' in merged_df.columns or 'value2' in merged_df.columns


def test_div() -> None:
    """
    Test division of a dataframe into selected columns and remaining columns.
    """
    df = pd.DataFrame({
        'A': [1, 2, 3],
        'B': [4, 5, 6],
        'C': [7, 8, 9]
    })

    selected, remaining = DataFrameOperator.div(df, cols=['A', 'B'])

    assert 'A' in selected.columns and 'B' in selected.columns
    assert 'C' in remaining.columns
    assert len(selected.columns) == 2
    assert len(remaining.columns) == 1


def test_drop_columns() -> None:
    """
    Test dropping specific columns from a dataframe.
    """
    df = pd.DataFrame({
        'A': [1, 2, 3],
        'B': [4, 5, 6],
        'C': [7, 8, 9]
    })

    df_dropped = DataFrameOperator.drop_columns(df, ['B'])

    assert 'B' not in df_dropped.columns
    assert len(df_dropped.columns) == 2  # One column should have been dropped.


def test_groupby() -> None:
    """
    Test grouping dataframe by a specific column and applying an aggregation function.
    """
    df = pd.DataFrame({
        'Category': ['A', 'B', 'A', 'B'],
        'Value': [10, 20, 30, 40]
    })

    grouped_df = DataFrameOperator.groupby(df, by=['Category'], agg_func='sum')

    assert len(grouped_df) == 2  # There are two unique categories.
    assert 'Value' in grouped_df.columns
    assert grouped_df.loc[grouped_df['Category'] == 'A', 'Value'].values[0] == 40


def test_apply_function() -> None:
    """
    Test applying a custom function to specific columns.
    """
    df = pd.DataFrame({
        'A': [1, 2, 3],
        'B': [4, 5, 6]
    })

    def square(x: int) -> int:
        return x * x

    df_applied = DataFrameOperator.apply_function(df, ['A', 'B'], square)

    assert df_applied['A'].tolist() == [1, 4, 9]
    assert df_applied['B'].tolist() == [16, 25, 36]


def test_filter_rows() -> None:
    """
    Test filtering rows based on a condition.
    """
    df = pd.DataFrame({
        'A': [1, 2, 3],
        'B': [4, 5, 6]
    })

    filtered_df = DataFrameOperator.filter_rows(df, 'A > 1')

    assert len(filtered_df) == 2  # Two rows should satisfy the condition.
    assert filtered_df['A'].tolist() == [2, 3]


def test_fill_missing(df_with_missing: DataFrame) -> None:
    """
    Test filling missing values in the dataframe.

    Args:
        df_with_missing (DataFrame): DataFrame with missing values.
    """
    filled_df = DataFrameOperator.fill_missing(df_with_missing, value=0)

    assert filled_df.isnull().sum().sum() == 0  # There should be no missing values.
    assert filled_df['A'].tolist() == [1, 2, 0]  # Missing values filled with 0.


def test_rename_columns() -> None:
    """
    Test renaming columns in the dataframe.
    """
    df = pd.DataFrame({
        'old_name': [1, 2, 3]
    })

    renamed_df = DataFrameOperator.rename_columns(df, {'old_name': 'new_name'})

    assert 'new_name' in renamed_df.columns
    assert 'old_name' not in renamed_df.columns


def test_change_column_types() -> None:
    """
    Test changing column types in the dataframe.
    """
    df = pd.DataFrame({
        'A': ['1', '2', '3'],
        'B': ['4.5', '5.5', '6.5']
    })

    df_converted = DataFrameOperator.change_column_types(df, {'A': 'int', 'B': 'float'})

    assert pd.api.types.is_integer_dtype(df_converted['A'].dtype)  # This checks for any integer type (int32 or int64)
    assert pd.api.types.is_float_dtype(df_converted['B'].dtype)    # This checks for any float type (float32 or float64)


def test_sort_values() -> None:
    """
    Test sorting the dataframe by a specific column.
    """
    df = pd.DataFrame({
        'A': [3, 1, 2],
        'B': [6, 4, 5]
    })

    sorted_df = DataFrameOperator.sort_values(df, by='A')

    assert sorted_df['A'].tolist() == [1, 2, 3]  # Sorted in ascending order.


def test_split_by_missing_values(df_with_missing: DataFrame) -> None:
    """
    Test splitting dataframe into columns with missing values and without missing values.

    Args:
        df_with_missing (DataFrame): DataFrame with missing values.
    """
    df_with, df_without = DataFrameOperator.split_by_missing_values(df_with_missing)

    assert 'A' in df_with.columns and 'B' in df_with.columns
    assert 'C' in df_without.columns
    assert 'C' not in df_with.columns
    assert 'A' not in df_without.columns
