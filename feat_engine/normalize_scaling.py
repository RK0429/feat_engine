import pandas as pd
import numpy as np
from sklearn.preprocessing import (
    MinMaxScaler,
    StandardScaler,
    RobustScaler,
    MaxAbsScaler,
    Normalizer,
    QuantileTransformer,
    PowerTransformer
)
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from typing import Optional, Any, Dict, List, Union


class ScalingNormalizer(BaseEstimator, TransformerMixin):
    """
    A utility class for scaling and normalizing data using various methods such as Min-Max scaling,
    standard scaling, robust scaling, max absolute scaling, and normalization. Non-specified columns are left unchanged.
    """

    def __init__(
        self,
        method: Union[str, Dict[str, str]] = 'standard',
        columns: Optional[List[str]] = None,
        **kwargs: Any
    ):
        """
        Initializes the ScalingNormalizer class with a specified scaling or normalization method.

        Args:
            method (Union[str, Dict[str, str]]): The scaling or normalization method to use.
                Can be a single method for all columns or a dictionary mapping column names to methods.
                Supported methods:
                    - 'minmax'
                    - 'standard'
                    - 'robust'
                    - 'maxabs'
                    - 'l1'
                    - 'l2'
                    - 'max'
                    - 'quantile'
                    - 'power'
            columns (List[str], optional): List of columns to scale or normalize. If None, all numeric columns are used.
            **kwargs (Any): Additional parameters to pass to the scaling or normalization methods.
        """
        self.method = method
        self.columns = columns
        self.kwargs = kwargs
        self.scalers: Dict[str, Any] = {}

    def _get_scaler(self, method: str) -> Any:
        """
        Retrieves the appropriate scaler or normalizer object based on the specified method.

        Args:
            method (str): The scaling method to use.

        Returns:
            Any: The scaler or normalizer object corresponding to the specified method.
        """
        if method == 'minmax':
            return MinMaxScaler(**self.kwargs)
        elif method == 'standard':
            return StandardScaler(**self.kwargs)
        elif method == 'robust':
            return RobustScaler(**self.kwargs)
        elif method == 'maxabs':
            return MaxAbsScaler(**self.kwargs)
        elif method in ['l1', 'l2', 'max']:
            return Normalizer(norm=method, **self.kwargs)
        elif method == 'quantile':
            return QuantileTransformer(**self.kwargs)
        elif method == 'power':
            return PowerTransformer(**self.kwargs)
        else:
            raise ValueError(f"Unsupported scaling method: {method}")

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'ScalingNormalizer':
        """
        Fits the scaler or normalizer to the specified columns of the input data.

        Args:
            X (pd.DataFrame): The input data to be scaled or normalized.
            y (pd.Series, optional): Not used in the scaling process, provided for compatibility.

        Returns:
            ScalingNormalizer: Returns the instance after fitting.
        """
        if self.columns is None:
            # Default to numeric columns
            self.columns = X.select_dtypes(include=[np.number]).columns.tolist()

        if not self.columns:
            raise ValueError("No columns to scale or normalize.")

        if isinstance(self.method, str):
            # Same method for all columns
            scaler = self._get_scaler(self.method)
            scaler.fit(X[self.columns])
            self.scalers = {col: scaler for col in self.columns}
        elif isinstance(self.method, dict):
            # Different methods per column
            for col in self.columns:
                method = self.method.get(col, 'standard')  # Default to 'standard' if not specified
                scaler = self._get_scaler(method)
                scaler.fit(X[[col]])
                self.scalers[col] = scaler
        else:
            raise ValueError("Method must be a string or a dictionary mapping columns to methods.")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms the specified columns of the input data using the fitted scalers or normalizers.

        Args:
            X (pd.DataFrame): The input data to transform.

        Returns:
            pd.DataFrame: Transformed data with specified columns scaled or normalized.
        """
        X_transformed = X.copy()
        if self.columns is not None:
            for col in self.columns:
                if col in X.columns:
                    scaler = self.scalers[col]
                    X_transformed[col] = scaler.transform(X[[col]])
                else:
                    raise ValueError(f"Column '{col}' not found in the input data.")
        return X_transformed

    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        Fits the scalers or normalizers to the specified columns of the data and transforms it.

        Args:
            X (pd.DataFrame): The input data to scale or normalize.
            y (pd.Series, optional): Not used in the scaling process, provided for compatibility.

        Returns:
            pd.DataFrame: The transformed data with specified columns scaled or normalized.
        """
        self.fit(X, y)
        return self.transform(X)

    def inverse_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Inverses the transformation on the specified columns of the input data.

        Args:
            X (pd.DataFrame): The transformed data to inverse transform.

        Returns:
            pd.DataFrame: Original data with specified columns inverse transformed.
        """
        X_inv_transformed = X.copy()
        if self.columns is not None:
            for col in self.columns:
                if col in X.columns:
                    scaler = self.scalers[col]
                    if hasattr(scaler, 'inverse_transform'):
                        X_inv_transformed[col] = scaler.inverse_transform(X[[col]])
                    else:
                        raise ValueError(f"Scaler for column '{col}' does not support inverse_transform.")
                else:
                    raise ValueError(f"Column '{col}' not found in the input data.")
        return X_inv_transformed

    @staticmethod
    def create_column_transformer(
        column_methods: Dict[str, str],
        remainder: str = 'passthrough',
        **kwargs: Any
    ) -> ColumnTransformer:
        """
        Creates a ColumnTransformer to apply different scaling or normalization methods to different columns.

        Args:
            column_methods (Dict[str, str]): A dictionary mapping column names to scaling or normalization methods.
                                             Example: {'column1': 'minmax', 'column2': 'standard'}
            remainder (str): Strategy for handling remaining columns. Defaults to 'passthrough'.
            **kwargs (Any): Additional parameters to pass to the scaling or normalization methods.

        Returns:
            ColumnTransformer: A ColumnTransformer object to apply specified methods to different columns.
        """
        transformers = []
        for column, method in column_methods.items():
            scaler = ScalingNormalizer(method=method, columns=[column], **kwargs)
            transformers.append((f"{method}_scaler_{column}", scaler, [column]))

        return ColumnTransformer(transformers=transformers, remainder=remainder)

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """
        Get parameters for this estimator.

        Args:
            deep (bool): If True, will return the parameters for this estimator and contained subobjects that are estimators.

        Returns:
            Dict[str, Any]: Parameter names mapped to their values.
        """
        params = {'method': self.method, 'columns': self.columns}
        params.update(self.kwargs)
        return params

    def set_params(self, **params: Any) -> 'ScalingNormalizer':
        """
        Set the parameters of this estimator.

        Args:
            **params: Estimator parameters.

        Returns:
            ScalingNormalizer: Returns self.
        """
        for key, value in params.items():
            setattr(self, key, value)
        return self
