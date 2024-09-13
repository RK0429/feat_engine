import pandas as pd
from typing import List, Tuple, Union, Callable, Dict, Optional, Any


class DataFrameOperator:
    """
    A class that provides various DataFrame operations such as merging, concatenation,
    splitting, and other utility functions for DataFrame manipulation.
    """

    @staticmethod
    def merge_dataframes(
        df1: pd.DataFrame,
        df2: pd.DataFrame,
        on: Optional[Union[str, List[str]]] = None,
        how: str = 'inner',
        left_on: Optional[Union[str, List[str]]] = None,
        right_on: Optional[Union[str, List[str]]] = None,
        left_index: bool = False,
        right_index: bool = False,
        suffixes: Tuple[str, str] = ('_x', '_y'),
        indicator: bool = False,
        validate: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Merge two DataFrames using database-style joins.

        Args:
            df1 (pd.DataFrame): The first DataFrame.
            df2 (pd.DataFrame): The second DataFrame.
            on (Union[str, List[str], None], optional): Column or index level names to join on.
            how (str, optional): Type of merge to be performed ('left', 'right', 'outer', 'inner', 'cross'). Defaults to 'inner'.
            left_on (Union[str, List[str], None], optional): Column(s) from the left DataFrame to use as keys.
            right_on (Union[str, List[str], None], optional): Column(s) from the right DataFrame to use as keys.
            left_index (bool, optional): Use index from the left DataFrame as join key. Defaults to False.
            right_index (bool, optional): Use index from the right DataFrame as join key. Defaults to False.
            suffixes (Tuple[str, str], optional): Suffixes to apply to overlapping column names. Defaults to ('_x', '_y').
            indicator (bool, optional): Adds a column '_merge' with merge information. Defaults to False.
            validate (str, optional): Checks if merge is of specified type. Defaults to None.

        Returns:
            pd.DataFrame: A merged DataFrame.
        """
        return pd.merge(
            df1,
            df2,
            how=how,
            on=on,
            left_on=left_on,
            right_on=right_on,
            left_index=left_index,
            right_index=right_index,
            suffixes=suffixes,
            indicator=indicator,
            validate=validate
        )

    @staticmethod
    def concat_dataframes(
        dfs: List[pd.DataFrame],
        axis: int = 0,
        join: str = 'outer',
        ignore_index: bool = False,
        keys: Optional[List] = None,
        levels: Optional[List] = None,
        names: Optional[List[str]] = None,
        verify_integrity: bool = False,
        sort: bool = False,
        copy: bool = True
    ) -> pd.DataFrame:
        """
        Concatenate pandas objects along a particular axis.

        Args:
            dfs (List[pd.DataFrame]): List of DataFrames to concatenate.
            axis (int, optional): The axis to concatenate along (0 for index, 1 for columns). Defaults to 0.
            join (str, optional): How to handle indexes on other axes ('inner', 'outer'). Defaults to 'outer'.
            ignore_index (bool, optional): If True, do not use the index values along the concatenation axis. Defaults to False.
            keys (List, optional): Sequence of keys to use to construct a hierarchical index. Defaults to None.
            levels (List, optional): Specific levels to use for the hierarchical index. Defaults to None.
            names (List[str], optional): Names for the levels in the resulting hierarchical index. Defaults to None.
            verify_integrity (bool, optional): Check whether the new concatenated axis contains duplicates. Defaults to False.
            sort (bool, optional): Sort non-concatenation axis if not aligned. Defaults to False.
            copy (bool, optional): If False, do not copy data unnecessarily. Defaults to True.

        Returns:
            pd.DataFrame: The concatenated DataFrame.
        """
        return pd.concat(
            dfs,
            axis=axis,
            join=join,
            ignore_index=ignore_index,
            keys=keys,
            levels=levels,
            names=names,
            verify_integrity=verify_integrity,
            sort=sort,
            copy=copy
        )

    @staticmethod
    def split_dataframe(
        df: pd.DataFrame,
        columns: List[str]
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split a DataFrame into two DataFrames based on specified columns.

        Args:
            df (pd.DataFrame): The input DataFrame.
            columns (List[str]): List of column names to separate.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: A tuple containing:
                - DataFrame with the specified columns.
                - DataFrame without the specified columns.
        """
        missing_cols = set(columns) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Columns {missing_cols} not found in DataFrame.")
        df_selected = df[columns].copy()
        df_remaining = df.drop(columns=columns)
        return df_selected, df_remaining

    @staticmethod
    def drop_columns(
        df: pd.DataFrame,
        columns: List[Union[str, int]]
    ) -> pd.DataFrame:
        """
        Drop specified columns from the DataFrame by name or index position.

        Args:
            df (pd.DataFrame): The input DataFrame.
            columns (List[Union[str, int]]): List of column names or index positions to drop.

        Returns:
            pd.DataFrame: A DataFrame with the specified columns dropped.
        """
        columns_to_drop = []
        for col in columns:
            if isinstance(col, int):
                try:
                    columns_to_drop.append(df.columns[col])
                except IndexError:
                    raise IndexError(f"Column index {col} is out of bounds.")
            elif isinstance(col, str):
                if col in df.columns:
                    columns_to_drop.append(col)
                else:
                    raise ValueError(f"Column '{col}' not found in DataFrame.")
            else:
                raise TypeError("Columns must be a list of column names or index positions.")
        return df.drop(columns=columns_to_drop)

    @staticmethod
    def groupby(
        df: pd.DataFrame,
        by: Union[str, List[str]],
        agg_funcs: Union[str, List[str], Dict[str, Union[str, List[str]]]]
    ) -> pd.DataFrame:
        """
        Perform a group-by operation and apply aggregation functions.

        Args:
            df (pd.DataFrame): The input DataFrame.
            by (Union[str, List[str]]): Column(s) to group by.
            agg_funcs (Union[str, List[str], Dict[str, Union[str, List[str]]]]): Aggregation function(s).

        Returns:
            pd.DataFrame: A DataFrame with grouped and aggregated data.
        """
        return df.groupby(by).agg(agg_funcs).reset_index()

    @staticmethod
    def apply_function(
        df: pd.DataFrame,
        columns: List[str],
        func: Callable,
        element_wise: bool = True
    ) -> pd.DataFrame:
        """
        Apply a custom function to specified columns.

        Args:
            df (pd.DataFrame): The input DataFrame.
            columns (List[str]): List of column names to apply the function to.
            func (Callable): The function to apply.
            element_wise (bool, optional): If True, apply function element-wise. If False, apply column-wise. Defaults to True.

        Returns:
            pd.DataFrame: A DataFrame with the function applied to the specified columns.
        """
        missing_cols = set(columns) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Columns {missing_cols} not found in DataFrame.")

        df_copy = df.copy()
        if element_wise:
            df_copy[columns] = df_copy[columns].applymap(func)
        else:
            df_copy[columns] = df_copy[columns].apply(func)
        return df_copy

    @staticmethod
    def filter_rows(
        df: pd.DataFrame,
        condition: str
    ) -> pd.DataFrame:
        """
        Filter rows in the DataFrame based on a given condition.

        Args:
            df (pd.DataFrame): The input DataFrame.
            condition (str): The condition to filter rows by (e.g., "age > 30").

        Returns:
            pd.DataFrame: A new DataFrame with filtered rows.
        """
        return df.query(condition)

    @staticmethod
    def fill_missing(
        df: pd.DataFrame,
        value: Optional[Union[float, Dict[str, Union[float, str]]]] = 0,
        columns: Optional[List[str]] = None,
        method: Optional[str] = None,
        axis: Optional[int] = None,
        limit: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Fill missing values in the DataFrame.

        Args:
            df (pd.DataFrame): The input DataFrame.
            value (Union[float, Dict[str, Union[float, str]]], optional): Value to use for filling holes.
            columns (List[str], optional): Specific columns to fill missing values in.
            method (str, optional): Method to use for filling holes ('backfill', 'bfill', 'pad', 'ffill', None).
            axis (int, optional): Axis along which to fill missing values.
            limit (int, optional): Maximum number of consecutive NaNs to fill.

        Returns:
            pd.DataFrame: A DataFrame with missing values filled.
        """
        df_copy = df.copy()
        if columns:
            missing_cols = set(columns) - set(df.columns)
            if missing_cols:
                raise ValueError(f"Columns {missing_cols} not found in DataFrame.")
            df_copy[columns] = df_copy[columns].fillna(
                value=value, method=method, axis=axis, limit=limit
            )
        else:
            df_copy = df_copy.fillna(
                value=value, method=method, axis=axis, limit=limit
            )
        return df_copy

    @staticmethod
    def rename_columns(
        df: pd.DataFrame,
        columns_dict: Dict[str, str]
    ) -> pd.DataFrame:
        """
        Rename columns in the DataFrame based on a given dictionary.

        Args:
            df (pd.DataFrame): The input DataFrame.
            columns_dict (Dict[str, str]): A dictionary mapping old column names to new ones.

        Returns:
            pd.DataFrame: A DataFrame with renamed columns.
        """
        missing_cols = set(columns_dict.keys()) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Columns {missing_cols} not found in DataFrame.")
        return df.rename(columns=columns_dict)

    @staticmethod
    def change_column_types(
        df: pd.DataFrame,
        columns_types: Dict[str, Union[str, type]]
    ) -> pd.DataFrame:
        """
        Change the data types of specified columns.

        Args:
            df (pd.DataFrame): The input DataFrame.
            columns_types (Dict[str, Union[str, type]]): A dictionary mapping column names to target data types.

        Returns:
            pd.DataFrame: A DataFrame with the specified column types changed.
        """
        missing_cols = set(columns_types.keys()) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Columns {missing_cols} not found in DataFrame.")
        return df.astype(columns_types)

    @staticmethod
    def sort_values(
        df: pd.DataFrame,
        by: Union[str, List[str]],
        ascending: Union[bool, List[bool]] = True,
        inplace: bool = False,
        na_position: str = 'last'
    ) -> pd.DataFrame:
        """
        Sort the DataFrame by specified column(s).

        Args:
            df (pd.DataFrame): The input DataFrame.
            by (Union[str, List[str]]): Column name(s) to sort by.
            ascending (Union[bool, List[bool]], optional): Sort ascending vs. descending. Defaults to True.
            inplace (bool, optional): If True, perform operation in-place. Defaults to False.
            na_position (str, optional): 'first' puts NaNs at the beginning, 'last' puts NaNs at the end. Defaults to 'last'.

        Returns:
            pd.DataFrame: The sorted DataFrame.
        """
        return df.sort_values(
            by=by,
            ascending=ascending,
            inplace=inplace,
            na_position=na_position
        )

    @staticmethod
    def split_by_missing_values(
        df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split the DataFrame into two DataFrames based on missing values.

        Args:
            df (pd.DataFrame): The input DataFrame.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]:
                - DataFrame with columns that contain missing values.
                - DataFrame with columns that do not have any missing values.
        """
        columns_with_missing = df.columns[df.isnull().any()]
        columns_without_missing = df.columns[~df.isnull().any()]
        df_with_missing = df[columns_with_missing].copy()
        df_without_missing = df[columns_without_missing].copy()
        return df_with_missing, df_without_missing

    @staticmethod
    def drop_duplicates(
        df: pd.DataFrame,
        subset: Optional[List[str]] = None,
        keep: str = 'first',
        inplace: bool = False,
        ignore_index: bool = False
    ) -> pd.DataFrame:
        """
        Remove duplicate rows from the DataFrame.

        Args:
            df (pd.DataFrame): The input DataFrame.
            subset (List[str], optional): Columns to consider when identifying duplicates.
            keep (str, optional): Which duplicate to keep ('first', 'last', False). Defaults to 'first'.
            inplace (bool, optional): If True, perform operation in-place. Defaults to False.
            ignore_index (bool, optional): If True, reset index after dropping duplicates. Defaults to False.

        Returns:
            pd.DataFrame: The DataFrame with duplicates removed.
        """
        return df.drop_duplicates(
            subset=subset,
            keep=keep,
            inplace=inplace,
            ignore_index=ignore_index
        )

    @staticmethod
    def sample_dataframe(
        df: pd.DataFrame,
        n: Optional[int] = None,
        frac: Optional[float] = None,
        replace: bool = False,
        weights: Optional[Union[str, pd.Series]] = None,
        random_state: Optional[int] = None,
        axis: int = 0
    ) -> pd.DataFrame:
        """
        Return a random sample of items from an axis of object.

        Args:
            df (pd.DataFrame): The input DataFrame.
            n (int, optional): Number of items from axis to return.
            frac (float, optional): Fraction of axis items to return.
            replace (bool, optional): Sample with or without replacement. Defaults to False.
            weights (Union[str, pd.Series], optional): Weights for sampling.
            random_state (int, optional): Seed for the random number generator.
            axis (int, optional): Axis to sample. Defaults to 0.

        Returns:
            pd.DataFrame: A random sample of the DataFrame.
        """
        return df.sample(
            n=n,
            frac=frac,
            replace=replace,
            weights=weights,
            random_state=random_state,
            axis=axis
        )

    @staticmethod
    def pivot_table(
        df: pd.DataFrame,
        values: Optional[Union[str, List[str]]] = None,
        index: Optional[Union[str, List[str]]] = None,
        columns: Optional[Union[str, List[str]]] = None,
        aggfunc: Union[str, List[str], Dict[str, Union[str, List[str]]]] = 'mean',
        fill_value: Optional[Any] = None,
        margins: bool = False,
        dropna: bool = True,
        margins_name: str = 'All',
        observed: bool = False,
        sort: bool = True
    ) -> pd.DataFrame:
        """
        Create a spreadsheet-style pivot table as a DataFrame.

        Args:
            df (pd.DataFrame): The input DataFrame.
            values (Union[str, List[str]], optional): Column(s) to aggregate.
            index (Union[str, List[str]], optional): Keys to group by on the pivot table index.
            columns (Union[str, List[str]], optional): Keys to group by on the pivot table column.
            aggfunc (Union[str, List[str], Dict[str, Union[str, List[str]]]], optional): Aggregation function(s). Defaults to 'mean'.
            fill_value (Any, optional): Value to replace missing values with.
            margins (bool, optional): Add all rows/columns (subtotals). Defaults to False.
            dropna (bool, optional): Do not include columns whose entries are all NaN. Defaults to True.
            margins_name (str, optional): Name of the row/column that will contain the totals. Defaults to 'All'.
            observed (bool, optional): This only applies if any of the groupers are categoricals. Defaults to False.
            sort (bool, optional): Sort group keys. Defaults to True.

        Returns:
            pd.DataFrame: The pivot table.
        """
        return pd.pivot_table(
            df,
            values=values,
            index=index,
            columns=columns,
            aggfunc=aggfunc,
            fill_value=fill_value,
            margins=margins,
            dropna=dropna,
            margins_name=margins_name,
            observed=observed,
            sort=sort
        )
