import pandas as pd
from sklearn.preprocessing import (
    MinMaxScaler,
    StandardScaler,
    RobustScaler,
    MaxAbsScaler,
    Normalizer
)
from sklearn.compose import ColumnTransformer
from typing import Optional, Any, Dict


class ScalingNormalizer:
    """
    A utility class for scaling and normalizing data using various methods such as Min-Max scaling, standard scaling,
    robust scaling, max absolute scaling, and L2 normalization.
    """

    def __init__(self, method: str = 'standard', **kwargs: Any):
        """
        Initializes the ScalingNormalizer class with a specified scaling or normalization method.

        Args:
            method (str): The scaling or normalization method to use ('minmax', 'standard', 'robust', 'maxabs', 'l2').
                          Default is 'standard'.
            **kwargs (Any): Additional parameters to pass to the scaling or normalization method.
        """
        self.method: str = method
        self.kwargs: Any = kwargs
        self.scaler: Any = self._get_scaler()

    def _get_scaler(self) -> Any:
        """
        Retrieves the appropriate scaler object based on the specified method.

        Returns:
            Any: The scaler or normalizer object corresponding to the specified method.
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
        Fits the scaler or normalizer to the input data.

        Args:
            X (pd.DataFrame): The input data to be scaled or normalized.
            y (Optional[pd.Series]): Not used in the scaling process, provided for compatibility.

        Returns:
            ScalingNormalizer: Returns the instance of the class after fitting.
        """
        self.scaler.fit(X, y)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms the input data using the fitted scaler or normalizer.

        Args:
            X (pd.DataFrame): The input data to transform.

        Returns:
            pd.DataFrame: Transformed data in the form of a DataFrame.
        """
        transformed = self.scaler.transform(X)
        if isinstance(X, pd.DataFrame):
            return pd.DataFrame(transformed, columns=X.columns, index=X.index)
        return pd.DataFrame(transformed)

    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        Fits the scaler or normalizer to the data and transforms it.

        Args:
            X (pd.DataFrame): The input data to scale or normalize.
            y (Optional[pd.Series]): Not used in the scaling process, provided for compatibility.

        Returns:
            pd.DataFrame: The transformed data in the form of a DataFrame.
        """
        transformed = self.scaler.fit_transform(X, y)
        if isinstance(X, pd.DataFrame):
            return pd.DataFrame(transformed, columns=X.columns, index=X.index)
        return pd.DataFrame(transformed)

    @staticmethod
    def create_column_transformer(column_methods: Dict[str, str]) -> ColumnTransformer:
        """
        Creates a ColumnTransformer to apply different scaling or normalization methods to different columns.

        Args:
            column_methods (Dict[str, str]): A dictionary mapping column names to scaling or normalization methods.
                                             Example: {'column1': 'minmax', 'column2': 'standard'}

        Returns:
            ColumnTransformer: A ColumnTransformer object to apply specified methods to different columns.
        """
        transformers = []
        for column, method in column_methods.items():
            scaler = ScalingNormalizer(method=method).scaler
            transformers.append((f"{method}_scaler_{column}", scaler, [column]))

        return ColumnTransformer(transformers=transformers, remainder='passthrough')
