import pandas as pd


class FeatureGrouping:
    """
    The FeatureGrouping class provides methods for grouping and aggregating data using categorical or time-based features.
    It supports grouping by single or multiple categories, time-based aggregation, rolling aggregation, and percentile calculation.
    """

    def group_by_category(self, df: pd.DataFrame, group_column: str, agg_column: str, metrics: list) -> pd.DataFrame:
        """
        Groups data by a categorical column and applies aggregation metrics to a specified column.

        Args:
            df (pd.DataFrame): The input DataFrame.
            group_column (str): The column to group by.
            agg_column (str): The column on which to apply the aggregation metrics.
            metrics (list): List of aggregation metrics (e.g., ['sum', 'mean', 'count']).

        Returns:
            pd.DataFrame: Grouped and aggregated data with one row per group.
        """
        return df.groupby(group_column)[agg_column].agg(metrics).reset_index()

    def group_by_multiple_categories(self, df: pd.DataFrame, group_columns: list, agg_column: str, metrics: list) -> pd.DataFrame:
        """
        Groups data by multiple categorical columns and applies aggregation metrics to a specified column.

        Args:
            df (pd.DataFrame): The input DataFrame.
            group_columns (list): List of columns to group by.
            agg_column (str): The column on which to apply the aggregation metrics.
            metrics (list): List of aggregation metrics (e.g., ['sum', 'mean', 'count']).

        Returns:
            pd.DataFrame: Grouped and aggregated data with one row per group.
        """
        return df.groupby(group_columns)[agg_column].agg(metrics).reset_index()

    def aggregate_time_based(self, df: pd.DataFrame, date_column: str, agg_column: str, rule: str, metric: str) -> pd.DataFrame:
        """
        Aggregates time-based data using a specified resampling rule and aggregation metric.

        Args:
            df (pd.DataFrame): The input DataFrame.
            date_column (str): The name of the date or time-based column.
            agg_column (str): The column on which to apply the aggregation.
            rule (str): The resampling rule (e.g., 'D' for daily, 'M' for monthly).
            metric (str): The aggregation metric (e.g., 'sum', 'mean', 'count').

        Returns:
            pd.DataFrame: Resampled and aggregated data.
        """
        df[date_column] = pd.to_datetime(df[date_column])
        resampled_df = df.resample(rule, on=date_column)[agg_column].agg(metric).reset_index()
        return resampled_df

    def rolling_aggregation(self, df: pd.DataFrame, agg_column: str, window: int, metric: str) -> pd.DataFrame:
        """
        Applies rolling aggregation over a specified window on a numerical column.

        Args:
            df (pd.DataFrame): The input DataFrame.
            agg_column (str): The column on which to apply the rolling aggregation.
            window (int): The size of the rolling window.
            metric (str): The metric for aggregation (e.g., 'sum', 'mean', 'std').

        Returns:
            pd.DataFrame: Data with rolling aggregation applied.
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
        Calculates specified percentiles for a grouped column.

        Args:
            df (pd.DataFrame): The input DataFrame.
            group_column (str): The column by which to group the data.
            agg_column (str): The column on which to calculate the percentiles.
            percentiles (list): List of percentiles to calculate (e.g., [0.25, 0.5, 0.75]).

        Returns:
            pd.DataFrame: DataFrame with the calculated percentiles.
        """
        return df.groupby(group_column)[agg_column].quantile(percentiles).reset_index()
