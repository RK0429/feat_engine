import pandas as pd
from sklearn.preprocessing import (
    MinMaxScaler,
    StandardScaler,
    RobustScaler,
    MaxAbsScaler,
    Normalizer
)
# from sklearn.pipeline import Pipeline
# from sklearn.experimental import enable_iterative_imputer  # Enable the experimental API
from sklearn.compose import ColumnTransformer
from typing import Optional, Any, Dict


class ScalingNormalizer:
    """
    A utility class to perform scaling and normalization of data.
    """

    def __init__(self, method: str = 'standard', **kwargs: Any):
        """
        Initialize the ScalingNormalizer class with a specified scaling method.

        Args:
        - method (str): Scaling or normalization method ('minmax', 'standard', 'robust', 'maxabs', 'l2').
        - **kwargs: Additional parameters for the scaling method.
        """
        self.method: str = method
        self.kwargs: Any = kwargs
        self.scaler: Any = self._get_scaler()

    def _get_scaler(self) -> Any:
        """
        Retrieve the scaler object based on the specified method.
        """
        if self.method == 'minmax':
            return MinMaxScaler(**self.kwargs)
        elif self.method == 'standard':
            return StandardScaler(**self.kwargs)
        elif self.method == 'robust':
            return RobustScaler(**self.kwargs)
        elif self.method == 'maxabs':
            return MaxAbsScaler(**self.kwargs)
        elif self.method == 'l2':
            return Normalizer(norm='l2', **self.kwargs)
        else:
            raise ValueError(f"Unsupported scaling method: {self.method}")

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'ScalingNormalizer':
        """
        Fit the scaler to the data.

        Args:
        - X (pd.DataFrame): Input data.
        - y: Not used, but included for compatibility.

        Returns:
        - self
        """
        self.scaler.fit(X, y)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the data using the fitted scaler.

        Args:
        - X (pd.DataFrame): Input data.

        Returns:
        - pd.DataFrame: Transformed data.
        """
        transformed = self.scaler.transform(X)
        if isinstance(X, pd.DataFrame):
            return pd.DataFrame(transformed, columns=X.columns, index=X.index)
        return pd.DataFrame(transformed)

    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        Fit the scaler to the data and transform it.

        Args:
        - X (pd.DataFrame): Input data.
        - y: Not used, but included for compatibility.

        Returns:
        - pd.DataFrame: Transformed data.
        """
        transformed = self.scaler.fit_transform(X, y)
        if isinstance(X, pd.DataFrame):
            return pd.DataFrame(transformed, columns=X.columns, index=X.index)
        return pd.DataFrame(transformed)

    @staticmethod
    def create_column_transformer(column_methods: Dict[str, str]) -> ColumnTransformer:
        """
        Create a ColumnTransformer to apply different scaling methods to different columns.

        Args:
        - column_methods (dict): Dictionary mapping column names to scaling methods.
                                 Example: {'column1': 'minmax', 'column2': 'standard'}

        Returns:
        - ColumnTransformer: scikit-learn's ColumnTransformer object.
        """
        transformers = []
        for column, method in column_methods.items():
            scaler = ScalingNormalizer(method=method).scaler
            transformers.append((f"{method}_scaler_{column}", scaler, [column]))

        return ColumnTransformer(transformers=transformers, remainder='passthrough')
