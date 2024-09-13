import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from typing import Optional, List, Union, Dict


class DatetimeConverter(BaseEstimator, TransformerMixin):
    """
    Converts specified columns to datetime format.
    """

    def __init__(self, columns: Union[str, List[str]], format: Optional[str] = None, errors: str = 'raise'):
        """
        Initializes the DatetimeConverter.

        Args:
            columns (str or List[str]): Column name or list of column names to convert to datetime.
            format (str, optional): Datetime format to use for parsing. Default is None.
            errors (str): How to handle errors. 'raise' will raise an exception, 'coerce' will set invalid parsing to NaT, 'ignore' will return the original input. Default is 'raise'.
        """
        self.columns = [columns] if isinstance(columns, str) else columns
        self.format = format
        self.errors = errors

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'DatetimeConverter':
        """
        Fit method does nothing as no fitting is required.

        Args:
            X (pd.DataFrame): Input DataFrame.
            y (pd.Series, optional): Target variable (ignored).

        Returns:
            DatetimeConverter: Returns self.
        """
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Converts specified columns to datetime.

        Args:
            X (pd.DataFrame): Input DataFrame.

        Returns:
            pd.DataFrame: DataFrame with specified columns converted to datetime.
        """
        X_transformed = X.copy()
        for col in self.columns:
            if col in X_transformed.columns:
                X_transformed[col] = pd.to_datetime(X_transformed[col], format=self.format, errors=self.errors)
            else:
                raise ValueError(f"Column '{col}' not found in input DataFrame.")
        return X_transformed


class DatePartExtractor(BaseEstimator, TransformerMixin):
    """
    Extracts date parts from datetime columns.
    """

    def __init__(self, column: str, parts: Optional[List[str]] = None, prefix: Optional[str] = None):
        """
        Initializes the DatePartExtractor.

        Args:
            column (str): Name of the datetime column.
            parts (List[str], optional): List of date parts to extract. Default is all parts.
                Supported parts: 'year', 'month', 'day', 'hour', 'minute', 'second', 'dayofweek', 'is_weekend', 'quarter', 'dayofyear', 'weekofyear'.
            prefix (str, optional): Prefix to add to the extracted feature names.
        """
        self.column = column
        self.parts = parts or ['year', 'month', 'day', 'hour', 'minute', 'second', 'dayofweek', 'is_weekend']
        self.prefix = prefix or ''
        self.supported_parts = {
            'year': 'year',
            'month': 'month',
            'day': 'day',
            'hour': 'hour',
            'minute': 'minute',
            'second': 'second',
            'dayofweek': 'dayofweek',
            'weekday_name': 'weekday_name',  # Not in pandas >= 1.0.0
            'is_weekend': 'is_weekend',
            'quarter': 'quarter',
            'dayofyear': 'dayofyear',
            'weekofyear': 'weekofyear'
        }

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'DatePartExtractor':
        """
        Fit method does nothing as no fitting is required.

        Args:
            X (pd.DataFrame): Input DataFrame.
            y (pd.Series, optional): Target variable (ignored).

        Returns:
            DatePartExtractor: Returns self.
        """
        # Check if parts are valid
        invalid_parts = set(self.parts) - set(self.supported_parts.keys())
        if invalid_parts:
            raise ValueError(f"Unsupported date parts: {invalid_parts}")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Extracts specified date parts from the datetime column.

        Args:
            X (pd.DataFrame): Input DataFrame.

        Returns:
            pd.DataFrame: DataFrame with extracted date parts.
        """
        X_transformed = X.copy()
        if self.column not in X_transformed.columns:
            raise ValueError(f"Column '{self.column}' not found in input DataFrame.")
        if not pd.api.types.is_datetime64_any_dtype(X_transformed[self.column]):
            raise TypeError(f"Column '{self.column}' is not of datetime dtype.")
        dt_series = X_transformed[self.column]
        for part in self.parts:
            feature_name = f"{self.prefix}{part}"
            if part == 'year':
                X_transformed[feature_name] = dt_series.dt.year
            elif part == 'month':
                X_transformed[feature_name] = dt_series.dt.month
            elif part == 'day':
                X_transformed[feature_name] = dt_series.dt.day
            elif part == 'hour':
                X_transformed[feature_name] = dt_series.dt.hour
            elif part == 'minute':
                X_transformed[feature_name] = dt_series.dt.minute
            elif part == 'second':
                X_transformed[feature_name] = dt_series.dt.second
            elif part == 'dayofweek':
                X_transformed[feature_name] = dt_series.dt.dayofweek
            elif part == 'weekday_name':
                X_transformed[feature_name] = dt_series.dt.day_name()
            elif part == 'is_weekend':
                X_transformed[feature_name] = dt_series.dt.dayofweek >= 5
            elif part == 'quarter':
                X_transformed[feature_name] = dt_series.dt.quarter
            elif part == 'dayofyear':
                X_transformed[feature_name] = dt_series.dt.dayofyear
            elif part == 'weekofyear':
                X_transformed[feature_name] = dt_series.dt.isocalendar().week
            else:
                raise ValueError(f"Unsupported date part: {part}")
        return X_transformed


class TimeDifferenceTransformer(BaseEstimator, TransformerMixin):
    """
    Creates time difference between consecutive rows in a datetime column.
    """

    def __init__(self, column: str, new_column_name: Optional[str] = None, periods: int = 1):
        """
        Initializes the TimeDifferenceTransformer.

        Args:
            column (str): Name of the datetime column.
            new_column_name (str, optional): Name of the new column to store time differences. Default is 'time_diff'.
            periods (int): Number of periods to calculate difference over. Default is 1.
        """
        self.column = column
        self.new_column_name = new_column_name or 'time_diff'
        self.periods = periods

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'TimeDifferenceTransformer':
        """
        Fit method does nothing as no fitting is required.

        Args:
            X (pd.DataFrame): Input DataFrame.
            y (pd.Series, optional): Target variable (ignored).

        Returns:
            TimeDifferenceTransformer: Returns self.
        """
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates time differences.

        Args:
            X (pd.DataFrame): Input DataFrame.

        Returns:
            pd.DataFrame: DataFrame with new time difference column.
        """
        X_transformed = X.copy()
        if self.column not in X_transformed.columns:
            raise ValueError(f"Column '{self.column}' not found in input DataFrame.")
        if not pd.api.types.is_datetime64_any_dtype(X_transformed[self.column]):
            raise TypeError(f"Column '{self.column}' is not of datetime dtype.")
        X_transformed[self.new_column_name] = X_transformed[self.column].diff(periods=self.periods)
        return X_transformed


class LagFeatureCreator(BaseEstimator, TransformerMixin):
    """
    Creates lag features for specified columns.
    """

    def __init__(self, columns: Union[str, List[str]], lags: List[int]):
        """
        Initializes the LagFeatureCreator.

        Args:
            columns (str or List[str]): Column name(s) for which to create lag features.
            lags (List[int]): List of lag periods.
        """
        self.columns = [columns] if isinstance(columns, str) else columns
        self.lags = lags

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'LagFeatureCreator':
        """
        Fit method does nothing as no fitting is required.

        Args:
            X (pd.DataFrame): Input DataFrame.
            y (pd.Series, optional): Target variable (ignored).

        Returns:
            LagFeatureCreator: Returns self.
        """
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Creates lag features.

        Args:
            X (pd.DataFrame): Input DataFrame.

        Returns:
            pd.DataFrame: DataFrame with new lag feature columns.
        """
        X_transformed = X.copy()
        for col in self.columns:
            if col not in X_transformed.columns:
                raise ValueError(f"Column '{col}' not found in input DataFrame.")
            for lag in self.lags:
                X_transformed[f'{col}_lag_{lag}'] = X_transformed[col].shift(lag)
        return X_transformed


class RollingFeatureCreator(BaseEstimator, TransformerMixin):
    """
    Creates rolling statistics for specified columns.
    """

    def __init__(self, columns: Union[str, List[str]], window_size: int, statistics: List[str] = ['mean']):
        """
        Initializes the RollingFeatureCreator.

        Args:
            columns (str or List[str]): Column name(s) for which to calculate rolling statistics.
            window_size (int): Size of the rolling window.
            statistics (List[str]): List of rolling statistics to calculate ('mean', 'sum', 'std', 'min', 'max').
        """
        self.columns = [columns] if isinstance(columns, str) else columns
        self.window_size = window_size
        self.statistics = statistics
        self.supported_statistics = ['mean', 'sum', 'std', 'min', 'max']

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'RollingFeatureCreator':
        """
        Fit method does nothing as no fitting is required.

        Args:
            X (pd.DataFrame): Input DataFrame.
            y (pd.Series, optional): Target variable (ignored).

        Returns:
            RollingFeatureCreator: Returns self.
        """
        # Check if statistics are valid
        invalid_stats = set(self.statistics) - set(self.supported_statistics)
        if invalid_stats:
            raise ValueError(f"Unsupported statistics: {invalid_stats}")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Creates rolling features.

        Args:
            X (pd.DataFrame): Input DataFrame.

        Returns:
            pd.DataFrame: DataFrame with new rolling feature columns.
        """
        X_transformed = X.copy()
        for col in self.columns:
            if col not in X_transformed.columns:
                raise ValueError(f"Column '{col}' not found in input DataFrame.")
            for stat in self.statistics:
                feature_name = f'{col}_rolling_{stat}_{self.window_size}'
                if stat == 'mean':
                    X_transformed[feature_name] = X_transformed[col].rolling(window=self.window_size).mean()
                elif stat == 'sum':
                    X_transformed[feature_name] = X_transformed[col].rolling(window=self.window_size).sum()
                elif stat == 'std':
                    X_transformed[feature_name] = X_transformed[col].rolling(window=self.window_size).std()
                elif stat == 'min':
                    X_transformed[feature_name] = X_transformed[col].rolling(window=self.window_size).min()
                elif stat == 'max':
                    X_transformed[feature_name] = X_transformed[col].rolling(window=self.window_size).max()
                else:
                    raise ValueError(f"Unsupported statistic: {stat}")
        return X_transformed


class CyclicalFeaturesEncoder(BaseEstimator, TransformerMixin):
    """
    Encodes cyclical features using sine and cosine transformations.
    """

    def __init__(self, columns: Union[str, List[str]], max_values: Union[int, List[int]]):
        """
        Initializes the CyclicalFeaturesEncoder.

        Args:
            columns (str or List[str]): Column name(s) to encode.
            max_values (int or List[int]): Maximum value(s) of the cyclical features.
                If columns is a list, max_values should be a list of the same length.
        """
        self.columns = [columns] if isinstance(columns, str) else columns
        self.max_values = [max_values] if isinstance(max_values, int) else max_values
        if len(self.columns) != len(self.max_values):
            raise ValueError("The length of 'columns' and 'max_values' must be equal.")

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'CyclicalFeaturesEncoder':
        """
        Fit method does nothing as no fitting is required.

        Args:
            X (pd.DataFrame): Input DataFrame.
            y (pd.Series, optional): Target variable (ignored).

        Returns:
            CyclicalFeaturesEncoder: Returns self.
        """
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Encodes cyclical features.

        Args:
            X (pd.DataFrame): Input DataFrame.

        Returns:
            pd.DataFrame: DataFrame with new sine and cosine encoded columns.
        """
        X_transformed = X.copy()
        for col, max_val in zip(self.columns, self.max_values):
            if col not in X_transformed.columns:
                raise ValueError(f"Column '{col}' not found in input DataFrame.")
            X_transformed[f'{col}_sin'] = np.sin(2 * np.pi * X_transformed[col] / max_val)
            X_transformed[f'{col}_cos'] = np.cos(2 * np.pi * X_transformed[col] / max_val)
        return X_transformed


class DataResampler(BaseEstimator, TransformerMixin):
    """
    Resamples the DataFrame based on a given frequency and aggregation method.
    """

    def __init__(self, datetime_column: str, rule: str, aggregation_methods: Union[str, Dict[str, str]] = 'sum'):
        """
        Initializes the DataResampler.

        Args:
            datetime_column (str): Name of the datetime column.
            rule (str): Resampling frequency (e.g., 'W' for weekly, 'M' for monthly).
            aggregation_methods (str or Dict[str, str]): Aggregation method(s) to apply during resampling.
                If a string is provided, the same method is applied to all columns.
                If a dict is provided, it should map column names to aggregation methods.
                Supported methods include 'sum', 'mean', 'min', 'max', etc.
        """
        self.datetime_column = datetime_column
        self.rule = rule
        self.aggregation_methods = aggregation_methods

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'DataResampler':
        """
        Fit method does nothing as no fitting is required.

        Args:
            X (pd.DataFrame): Input DataFrame.
            y (pd.Series, optional): Target variable (ignored).

        Returns:
            DataResampler: Returns self.
        """
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Resamples the DataFrame.

        Args:
            X (pd.DataFrame): Input DataFrame.

        Returns:
            pd.DataFrame: Resampled DataFrame with aggregated values.
        """
        X_transformed = X.copy()
        if self.datetime_column not in X_transformed.columns:
            raise ValueError(f"Column '{self.datetime_column}' not found in input DataFrame.")
        if not pd.api.types.is_datetime64_any_dtype(X_transformed[self.datetime_column]):
            raise TypeError(f"Column '{self.datetime_column}' is not of datetime dtype.")
        X_transformed.set_index(self.datetime_column, inplace=True)
        resampled_df = X_transformed.resample(self.rule).agg(self.aggregation_methods)
        resampled_df.reset_index(inplace=True)
        return resampled_df
