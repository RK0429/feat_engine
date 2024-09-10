import pandas as pd
import numpy as np


class TemporalFeatures:
    """
    TemporalFeatures provides methods for extracting, transforming, and handling temporal data.
    These methods include datetime conversions, date part extraction, time differences, lag features,
    rolling statistics, cyclical features, and data resampling.
    """

    def __init__(self) -> None:
        """
        Initialize the TemporalFeatures class.
        """
        pass

    def convert_to_datetime(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        """
        Convert a column to datetime format.

        Args:
            df (pd.DataFrame): Input DataFrame.
            column (str): Column name to convert to datetime.

        Returns:
            pd.DataFrame: DataFrame with the column converted to datetime.
        """
        df[column] = pd.to_datetime(df[column])
        return df

    def extract_date_parts(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        """
        Extract year, month, day, day of the week, hour, minute, second, and weekend indicator from a datetime column.

        Args:
            df (pd.DataFrame): Input DataFrame.
            column (str): Name of the datetime column.

        Returns:
            pd.DataFrame: DataFrame with extracted date parts such as year, month, day, etc.
        """
        df['year'] = df[column].dt.year
        df['month'] = df[column].dt.month
        df['day'] = df[column].dt.day
        df['day_of_week'] = df[column].dt.dayofweek
        df['hour'] = df[column].dt.hour
        df['minute'] = df[column].dt.minute
        df['second'] = df[column].dt.second
        df['is_weekend'] = df[column].dt.dayofweek >= 5
        return df

    def create_time_difference(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        """
        Create a time difference between consecutive rows in a datetime column.

        Args:
            df (pd.DataFrame): Input DataFrame.
            column (str): Name of the datetime column.

        Returns:
            pd.DataFrame: DataFrame with a new column 'time_diff' for time differences.
        """
        df['time_diff'] = df[column].diff()
        return df

    def create_lag_features(self, df: pd.DataFrame, column: str, lags: list) -> pd.DataFrame:
        """
        Create lag features based on specified lag values.

        Args:
            df (pd.DataFrame): Input DataFrame.
            column (str): Column for which to create lag features.
            lags (list): List of lag values to generate.

        Returns:
            pd.DataFrame: DataFrame with new lag feature columns.
        """
        for lag in lags:
            df[f'{column}_lag_{lag}'] = df[column].shift(lag)
        return df

    def create_rolling_features(self, df: pd.DataFrame, column: str, window_size: int, feature: str = 'mean') -> pd.DataFrame:
        """
        Create rolling statistics such as mean, sum, and standard deviation over a given window size.

        Args:
            df (pd.DataFrame): Input DataFrame.
            column (str): Column for which to calculate rolling statistics.
            window_size (int): Size of the rolling window.
            feature (str): Type of rolling statistic to calculate ('mean', 'sum', 'std').

        Returns:
            pd.DataFrame: DataFrame with rolling feature columns.
        """
        if feature == 'mean':
            df[f'{column}_rolling_mean_{window_size}'] = df[column].rolling(window=window_size).mean()
        elif feature == 'sum':
            df[f'{column}_rolling_sum_{window_size}'] = df[column].rolling(window=window_size).sum()
        elif feature == 'std':
            df[f'{column}_rolling_std_{window_size}'] = df[column].rolling(window=window_size).std()
        else:
            raise ValueError("Unsupported feature. Use 'mean', 'sum', or 'std'.")
        return df

    def cyclical_features(self, df: pd.DataFrame, column: str, max_value: int) -> pd.DataFrame:
        """
        Create cyclical features for time-related columns (e.g., hours, months) using sine and cosine transformations.

        Args:
            df (pd.DataFrame): Input DataFrame.
            column (str): Name of the cyclical column (e.g., 'hour').
            max_value (int): Maximum value of the cyclical feature (e.g., 24 for hours, 12 for months).

        Returns:
            pd.DataFrame: DataFrame with new columns for sine and cosine transformations of the cyclical feature.
        """
        df[f'{column}_sin'] = np.sin(2 * np.pi * df[column] / max_value)
        df[f'{column}_cos'] = np.cos(2 * np.pi * df[column] / max_value)
        return df

    def resample_data(self, df: pd.DataFrame, column: str, rule: str, aggregation: str = 'sum') -> pd.DataFrame:
        """
        Resample the DataFrame based on a given frequency and aggregation method.

        Args:
            df (pd.DataFrame): Input DataFrame.
            column (str): Name of the datetime column.
            rule (str): Resampling frequency (e.g., 'W' for weekly, 'M' for monthly).
            aggregation (str): Aggregation method to apply during resampling ('sum', 'mean').

        Returns:
            pd.DataFrame: Resampled DataFrame with aggregated values.
        """
        df.set_index(column, inplace=True)
        if aggregation == 'sum':
            resampled_df = df.resample(rule).sum()
        elif aggregation == 'mean':
            resampled_df = df.resample(rule).mean()
        else:
            raise ValueError("Unsupported aggregation method. Use 'sum' or 'mean'.")
        df.reset_index(inplace=True)
        return resampled_df
