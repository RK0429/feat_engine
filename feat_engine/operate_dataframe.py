import pandas as pd
from typing import List, Tuple, Union, Callable, Dict


class DataFrameOperator:
    """
    A class that covers basic DataFrame operations like horizontal merge, vertical merge,
    and dividing DataFrames by column labels.
    """

    @staticmethod
    def hmerge(df1: pd.DataFrame, df2: pd.DataFrame, on: Union[str, None] = None) -> pd.DataFrame:
        """
        Horizontally merge two DataFrames (similar to SQL JOIN).

        Args:
            df1 (pd.DataFrame): The first DataFrame.
            df2 (pd.DataFrame): The second DataFrame.
            on (str): The column name to join on. If not specified, joins on index.

        Returns:
            pd.DataFrame: A new DataFrame with horizontally merged data.
        """
        if on:
            return pd.merge(df1, df2, on=on, how='outer')
        else:
            return pd.concat([df1, df2], axis=1)

    @staticmethod
    def vmerge(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
        """
        Vertically merge two DataFrames by stacking them.

        Args:
            df1 (pd.DataFrame): The first DataFrame.
            df2 (pd.DataFrame): The second DataFrame.

        Returns:
            pd.DataFrame: A new DataFrame with vertically merged data.
        """
        return pd.concat([df1, df2], axis=0).reset_index(drop=True)

    @staticmethod
    def div(df: pd.DataFrame, cols: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Divide a DataFrame into two DataFrames: one with the specified columns and one without them.

        Args:
            df (pd.DataFrame): The input DataFrame.
            cols (List[str]): List of column names to separate from the original DataFrame.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Two DataFrames, one with the specified columns, and the other without them.
        """
        df_selected = df[cols]
        df_remaining = df.drop(columns=cols)
        return df_selected, df_remaining

    @staticmethod
    def drop_columns(df: pd.DataFrame, columns: List[Union[str, int]]) -> pd.DataFrame:
        """
        Drop specified columns from the DataFrame by name or index.

        Args:
            df (pd.DataFrame): The input DataFrame.
            columns (List[Union[str, int]]): List of column names or index positions to drop.

        Returns:
            pd.DataFrame: A DataFrame with the specified columns dropped.
        """
        return df.drop(columns=columns, axis=1)

    @staticmethod
    def groupby(df: pd.DataFrame, by: List[str], agg_func: Union[str, Dict[str, str]]) -> pd.DataFrame:
        """
        Perform a group-by operation on the DataFrame and apply an aggregation function.

        Args:
            df (pd.DataFrame): The input DataFrame.
            by (List[str]): List of columns to group by.
            agg_func (Union[str, Dict[str, str]]): Aggregation function (e.g., 'sum', 'mean', etc.) or a dictionary mapping columns to aggregation functions.

        Returns:
            pd.DataFrame: A DataFrame with grouped and aggregated data.
        """
        return df.groupby(by).agg(agg_func).reset_index()

    @staticmethod
    def apply_function(df: pd.DataFrame, columns: List[str], func: Callable) -> pd.DataFrame:
        """
        Apply a custom function to each element of the specified columns.

        Args:
            df (pd.DataFrame): The input DataFrame.
            columns (List[str]): List of column names to apply the function to.
            func (Callable): The function to apply to each element in the specified columns.

        Returns:
            pd.DataFrame: A DataFrame with the function applied to the specified columns.
        """
        df[columns] = df[columns].applymap(func)
        return df

    @staticmethod
    def filter_rows(df: pd.DataFrame, condition: str) -> pd.DataFrame:
        """
        Filter rows in the DataFrame based on a given condition (as a string).

        Args:
            df (pd.DataFrame): The input DataFrame.
            condition (str): The condition to filter rows by (e.g., "age > 30").

        Returns:
            pd.DataFrame: A new DataFrame with filtered rows.
        """
        return df.query(condition)

    @staticmethod
    def fill_missing(df: pd.DataFrame, value: float = 0) -> pd.DataFrame:
        """
        Fill missing values in the DataFrame with a specified value.

        Args:
            df (pd.DataFrame): The input DataFrame.
            value (float): The value to fill missing entries with.

        Returns:
            pd.DataFrame: A DataFrame with missing values filled.
        """
        return df.fillna(value)

    @staticmethod
    def rename_columns(df: pd.DataFrame, columns_dict: dict) -> pd.DataFrame:
        """
        Rename columns in the DataFrame based on a given dictionary.

        Args:
            df (pd.DataFrame): The input DataFrame.
            columns_dict (dict): A dictionary mapping old column names to new ones.

        Returns:
            pd.DataFrame: A DataFrame with renamed columns.
        """
        return df.rename(columns=columns_dict)

    @staticmethod
    def change_column_types(df: pd.DataFrame, columns_types: Dict[str, str]) -> pd.DataFrame:
        """
        Change the data types of specified columns.

        Args:
            df (pd.DataFrame): The input DataFrame.
            columns_types (Dict[str, str]): A dictionary mapping column names to target data types.

        Returns:
            pd.DataFrame: A DataFrame with the specified column types changed.
        """
        return df.astype(columns_types)

    @staticmethod
    def sort_values(df: pd.DataFrame, by: str, ascending: bool = True) -> pd.DataFrame:
        """
        Sort the DataFrame by a specific column.

        Args:
            df (pd.DataFrame): The input DataFrame.
            by (str): The column name to sort by.
            ascending (bool): Whether to sort in ascending order (default: True).

        Returns:
            pd.DataFrame: The sorted DataFrame.
        """
        return df.sort_values(by=by, ascending=ascending)

    @staticmethod
    def split_by_missing_values(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Splits the input DataFrame into two DataFrames: one with columns containing missing values
        and another with columns that do not have any missing values.

        Args:
            df (pd.DataFrame): The input DataFrame.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]:
                - A DataFrame with columns that contain missing values.
                - A DataFrame with columns that do not have any missing values.
        """
        # Select columns with and without missing values
        columns_with_missing = df.columns[df.isnull().any()]
        columns_without_missing = df.columns[~df.isnull().any()]

        # Create two separate DataFrames
        df_with_missing = df[columns_with_missing]
        df_without_missing = df[columns_without_missing]

        return df_with_missing, df_without_missing
