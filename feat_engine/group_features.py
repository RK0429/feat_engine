import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from typing import Optional, Union, List, Dict


class GroupByFeatureGenerator(BaseEstimator, TransformerMixin):
    """
    A transformer that generates aggregated features by grouping data based on categorical or time-based features.
    It supports grouping by single or multiple categories, time-based aggregation, rolling aggregation,
    and percentile calculation.
    """

    def __init__(
        self,
        group_by: Union[str, List[str]],
        aggregations: Dict[str, List[str]],
        suffix: Optional[str] = None,
        as_index: bool = False,
    ):
        """
        Initializes the GroupByFeatureGenerator.

        Args:
            group_by (str or List[str]): Column(s) to group by.
            aggregations (Dict[str, List[str]]): Dictionary specifying columns to aggregate and their aggregation functions.
                Example: {'col1': ['mean', 'sum'], 'col2': ['max']}.
            suffix (str, optional): Suffix to add to the aggregated feature names. Default is None.
            as_index (bool): Whether to keep the group columns as index. Default is False.
        """
        self.group_by = [group_by] if isinstance(group_by, str) else group_by
        self.aggregations = aggregations
        self.suffix = suffix
        self.as_index = as_index
        self.grouped_df_: Optional[pd.DataFrame] = None

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'GroupByFeatureGenerator':
        """
        Fits the transformer by performing group-by aggregations.

        Args:
            X (pd.DataFrame): Input DataFrame.
            y (pd.Series, optional): Target variable (ignored).

        Returns:
            GroupByFeatureGenerator: Returns self.
        """
        grouped = X.groupby(self.group_by, as_index=self.as_index).agg(self.aggregations)
        # Flatten MultiIndex columns
        grouped.columns = [
            f"{col[0]}_{col[1]}" + (f"_{self.suffix}" if self.suffix else "") for col in grouped.columns.to_flat_index()
        ]
        self.grouped_df_ = grouped.reset_index() if not self.as_index else grouped
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Merges the aggregated features back into the original DataFrame.

        Args:
            X (pd.DataFrame): Input DataFrame.

        Returns:
            pd.DataFrame: DataFrame with new aggregated features.
        """
        if self.grouped_df_ is None:
            raise ValueError("The transformer has not been fitted yet!")
        X_transformed = X.merge(self.grouped_df_, on=self.group_by, how='left')
        return X_transformed


class TimeBasedAggregator(BaseEstimator, TransformerMixin):
    """
    A transformer that performs time-based aggregation using resampling rules.
    """

    def __init__(
        self,
        datetime_column: str,
        aggregations: Dict[str, Union[str, List[str]]],
        rule: str,
        suffix: Optional[str] = None,
    ):
        """
        Initializes the TimeBasedAggregator.

        Args:
            datetime_column (str): The name of the datetime column.
            aggregations (Dict[str, Union[str, List[str]]]): Dictionary specifying columns to aggregate and their aggregation functions.
                Example: {'col1': 'sum', 'col2': ['mean', 'max']}.
            rule (str): Resampling frequency (e.g., 'D' for daily, 'M' for monthly).
            suffix (str, optional): Suffix to add to the aggregated feature names. Default is None.
        """
        self.datetime_column = datetime_column
        self.aggregations = aggregations
        self.rule = rule
        self.suffix = suffix
        self.resampled_df_: Optional[pd.DataFrame] = None

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'TimeBasedAggregator':
        """
        Fits the transformer by resampling and aggregating the data.

        Args:
            X (pd.DataFrame): Input DataFrame.
            y (pd.Series, optional): Target variable (ignored).

        Returns:
            TimeBasedAggregator: Returns self.
        """
        X_resampled = X.copy()
        if not pd.api.types.is_datetime64_any_dtype(X_resampled[self.datetime_column]):
            X_resampled[self.datetime_column] = pd.to_datetime(X_resampled[self.datetime_column])
        X_resampled.set_index(self.datetime_column, inplace=True)
        resampled = X_resampled.resample(self.rule).agg(self.aggregations)
        # Flatten MultiIndex columns
        resampled.columns = [
            f"{col[0]}_{col[1]}" + (f"_{self.suffix}" if self.suffix else "") if isinstance(col, tuple) else col
            for col in resampled.columns.to_flat_index()
        ]
        self.resampled_df_ = resampled.reset_index()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Merges the resampled and aggregated features back into the original DataFrame.

        Args:
            X (pd.DataFrame): Input DataFrame.

        Returns:
            pd.DataFrame: DataFrame with new time-based aggregated features.
        """
        if self.resampled_df_ is None:
            raise ValueError("The transformer has not been fitted yet!")
        X_transformed = X.copy()
        if not pd.api.types.is_datetime64_any_dtype(X_transformed[self.datetime_column]):
            X_transformed[self.datetime_column] = pd.to_datetime(X_transformed[self.datetime_column])
        X_transformed = X_transformed.merge(self.resampled_df_, on=self.datetime_column, how='left')
        return X_transformed


class RollingAggregator(BaseEstimator, TransformerMixin):
    """
    A transformer that applies rolling window aggregations on numerical columns.
    """

    def __init__(
        self,
        columns: Union[str, List[str]],
        window: int,
        statistics: List[str],
        group_by: Optional[Union[str, List[str]]] = None,
        min_periods: int = 1,
        center: bool = False,
        suffix: Optional[str] = None,
    ):
        """
        Initializes the RollingAggregator.

        Args:
            columns (str or List[str]): Column(s) on which to apply the rolling aggregation.
            window (int): The size of the rolling window.
            statistics (List[str]): List of metrics for aggregation (e.g., ['sum', 'mean', 'std']).
            group_by (str or List[str], optional): Column(s) to group by before applying rolling aggregation.
            min_periods (int): Minimum number of observations in window required to have a value. Default is 1.
            center (bool): Set the labels at the center of the window. Default is False.
            suffix (str, optional): Suffix to add to the rolling feature names. Default is None.
        """
        self.columns = [columns] if isinstance(columns, str) else columns
        self.window = window
        self.statistics = statistics
        self.group_by = [group_by] if isinstance(group_by, str) else group_by
        self.min_periods = min_periods
        self.center = center
        self.suffix = suffix

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'RollingAggregator':
        """
        Fit method does nothing as no fitting is required.

        Args:
            X (pd.DataFrame): Input DataFrame.
            y (pd.Series, optional): Target variable (ignored).

        Returns:
            RollingAggregator: Returns self.
        """
        supported_statistics = ['mean', 'sum', 'std', 'min', 'max']
        invalid_stats = set(self.statistics) - set(supported_statistics)
        if invalid_stats:
            raise ValueError(f"Unsupported statistics: {invalid_stats}")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Applies rolling aggregations to the specified columns.

        Args:
            X (pd.DataFrame): Input DataFrame.

        Returns:
            pd.DataFrame: DataFrame with rolling aggregation features added.
        """
        X_transformed = X.copy()
        if self.group_by:
            grouped = X_transformed.groupby(self.group_by)
        else:
            grouped = X_transformed

        for col in self.columns:
            for stat in self.statistics:
                feature_name = f"{col}_rolling_{stat}_{self.window}"
                if self.suffix:
                    feature_name += f"_{self.suffix}"
                rolling = grouped[col].rolling(
                    window=self.window,
                    min_periods=self.min_periods,
                    center=self.center
                )
                if stat == 'mean':
                    X_transformed[feature_name] = rolling.mean().reset_index(level=self.group_by, drop=True)
                elif stat == 'sum':
                    X_transformed[feature_name] = rolling.sum().reset_index(level=self.group_by, drop=True)
                elif stat == 'std':
                    X_transformed[feature_name] = rolling.std().reset_index(level=self.group_by, drop=True)
                elif stat == 'min':
                    X_transformed[feature_name] = rolling.min().reset_index(level=self.group_by, drop=True)
                elif stat == 'max':
                    X_transformed[feature_name] = rolling.max().reset_index(level=self.group_by, drop=True)
                else:
                    raise ValueError(f"Unsupported statistic: {stat}")
        return X_transformed


class PercentileCalculator(BaseEstimator, TransformerMixin):
    """
    A transformer that calculates specified percentiles for grouped data.
    """

    def __init__(
        self,
        group_by: Union[str, List[str]],
        column: str,
        percentiles: List[float],
        suffix: Optional[str] = None,
    ):
        """
        Initializes the PercentileCalculator.

        Args:
            group_by (str or List[str]): Column(s) to group by.
            column (str): Column on which to calculate the percentiles.
            percentiles (List[float]): List of percentiles to calculate (e.g., [0.25, 0.5, 0.75]).
            suffix (str, optional): Suffix to add to the percentile feature names. Default is None.
        """
        self.group_by = [group_by] if isinstance(group_by, str) else group_by
        self.column = column
        self.percentiles = percentiles
        self.suffix = suffix
        self.percentile_df_: Optional[pd.DataFrame] = None

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'PercentileCalculator':
        """
        Fits the transformer by calculating the percentiles.

        Args:
            X (pd.DataFrame): Input DataFrame.
            y (pd.Series, optional): Target variable (ignored).

        Returns:
            PercentileCalculator: Returns self.
        """
        grouped = X.groupby(self.group_by)[self.column].quantile(self.percentiles)
        self.percentile_df_ = grouped.unstack(level=-1).reset_index()
        percentile_cols = [
            f"{self.column}_percentile_{int(p * 100)}" + (f"_{self.suffix}" if self.suffix else "")
            for p in self.percentiles
        ]
        self.percentile_df_.columns = self.group_by + percentile_cols
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Merges the calculated percentiles back into the original DataFrame.

        Args:
            X (pd.DataFrame): Input DataFrame.

        Returns:
            pd.DataFrame: DataFrame with percentile features added.
        """
        if self.percentile_df_ is None:
            raise ValueError("The transformer has not been fitted yet!")
        X_transformed = X.merge(self.percentile_df_, on=self.group_by, how='left')
        return X_transformed
