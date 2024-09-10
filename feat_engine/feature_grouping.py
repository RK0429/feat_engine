# feature_grouping/group_aggregate.py
import pandas as pd


class FeatureGrouping:
    """
    This class provides generic methods for grouping and aggregating data using categorical or time-based features.
    """

    def group_by_category(self, df: pd.DataFrame, group_column: str, agg_column: str, metrics: list) -> pd.DataFrame:
        """
        Groups data by a categorical column and applies aggregation metrics.

        Args:
        - df (pd.DataFrame): The dataframe.
        - group_column (str): The column to group by.
        - agg_column (str): The column to apply the aggregation.
        - metrics (list): List of aggregation metrics (e.g., ['sum', 'mean', 'count']).

        Returns:
        - pd.DataFrame: Grouped and aggregated data.
        """
        return df.groupby(group_column)[agg_column].agg(metrics).reset_index()

    def group_by_multiple_categories(self, df: pd.DataFrame, group_columns: list, agg_column: str, metrics: list) -> pd.DataFrame:
        """
        Groups data by multiple categorical columns and applies aggregation metrics.

        Args:
        - df (pd.DataFrame): The dataframe.
        - group_columns (list): List of columns to group by.
        - agg_column (str): The column to apply the aggregation.
        - metrics (list): List of aggregation metrics.

        Returns:
        - pd.DataFrame: Grouped and aggregated data.
        """
        return df.groupby(group_columns)[agg_column].agg(metrics).reset_index()

    def aggregate_time_based(self, df: pd.DataFrame, date_column: str, agg_column: str, rule: str, metric: str) -> pd.DataFrame:
        """
        Aggregates time-based data with specified resampling rule and aggregation metric.

        Args:
        - df (pd.DataFrame): The dataframe.
        - date_column (str): Date or time-based column.
        - agg_column (str): Column to apply aggregation.
        - rule (str): Resampling rule (e.g., 'D' for daily, 'M' for monthly).
        - metric (str): Aggregation metric (e.g., 'sum', 'mean', 'count').

        Returns:
        - pd.DataFrame: Resampled and aggregated data.
        """
        df[date_column] = pd.to_datetime(df[date_column])
        resampled_df = df.resample(rule, on=date_column)[agg_column].agg(metric).reset_index()
        return resampled_df

    def rolling_aggregation(self, df: pd.DataFrame, agg_column: str, window: int, metric: str) -> pd.DataFrame:
        """
        Applies rolling aggregation on a numerical column.

        Args:
        - df (pd.DataFrame): The dataframe.
        - agg_column (str): The column to apply the rolling aggregation.
        - window (int): Window size for rolling.
        - metric (str): Metric for aggregation (e.g., 'sum', 'mean', 'std').

        Returns:
        - pd.DataFrame: Data with rolling aggregation applied.
        """
        if metric == 'sum':
            df[f'{agg_column}_rolling_sum'] = df[agg_column].rolling(window=window).sum()
        elif metric == 'mean':
            df[f'{agg_column}_rolling_mean'] = df[agg_column].rolling(window=window).mean()
        elif metric == 'std':
            df[f'{agg_column}_rolling_std'] = df[agg_column].rolling(window=window).std()
        else:
            raise ValueError("Unsupported metric. Use 'sum', 'mean', or 'std'.")
        return df

    def calculate_percentiles(self, df: pd.DataFrame, group_column: str, agg_column: str, percentiles: list) -> pd.DataFrame:
        """
        Calculates percentiles for a grouped column.

        Args:
        - df (pd.DataFrame): The dataframe.
        - group_column (str): Column to group by.
        - agg_column (str): Column to calculate percentiles.
        - percentiles (list): List of percentiles to calculate (e.g., [0.25, 0.5, 0.75]).

        Returns:
        - pd.DataFrame: Dataframe with calculated percentiles.
        """
        return df.groupby(group_column)[agg_column].quantile(percentiles).reset_index()
