import numpy as np
import pandas as pd
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    PowerTransformer,
    QuantileTransformer,
    FunctionTransformer
)
from sklearn.base import BaseEstimator, TransformerMixin
from typing import List, Optional, Union, Any, Dict
from scipy.stats import boxcox


class FeatureTransformer(BaseEstimator, TransformerMixin):
    """
    A transformer class that applies various feature transformations to numerical data,
    including logarithmic, square root, power, scaling, and other transformations.
    """

    def __init__(
        self,
        columns: Optional[List[str]] = None,
        transformations: Optional[Union[str, List[str], Dict[str, str]]] = None,
        method: str = 'yeo-johnson',
        output_distribution: str = 'normal',
        **kwargs: Any
    ):
        """
        Initializes the FeatureTransformer.

        Args:
            columns (List[str], optional): List of column names to transform. If None, all numeric columns are used.
            transformations (Union[str, List[str], Dict[str, str]], optional): Transformation(s) to apply.
                Can be a single string, a list of strings, or a dictionary mapping columns to transformations.
                Supported transformations include:
                    - 'log'
                    - 'sqrt'
                    - 'power'
                    - 'boxcox'
                    - 'zscore'
                    - 'minmax'
                    - 'quantile'
                    - 'rank'
                    - 'dft' (Discrete Fourier Transform)
            method (str): Method to use for power transformations ('yeo-johnson' or 'box-cox'). Default is 'yeo-johnson'.
            output_distribution (str): Desired output distribution for quantile transformation ('normal' or 'uniform').
            **kwargs: Additional keyword arguments for specific transformers.
        """
        self.columns = columns
        self.transformations = transformations
        self.method = method
        self.output_distribution = output_distribution
        self.kwargs = kwargs
        self.transformers_: Dict[str, Any] = {}

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'FeatureTransformer':
        """
        Fits the transformer to the data.

        Args:
            X (pd.DataFrame): Input DataFrame.
            y (pd.Series, optional): Target variable (not used).

        Returns:
            self
        """
        X = X.copy()
        if self.columns is None:
            self.columns = X.select_dtypes(include=[np.number]).columns.tolist()

        if isinstance(self.transformations, str):
            transformations = {col: self.transformations for col in self.columns}
        elif isinstance(self.transformations, list):
            transformations = {col: self.transformations[i % len(self.transformations)] for i, col in enumerate(self.columns)}
        elif isinstance(self.transformations, dict):
            transformations = self.transformations
        else:
            raise ValueError("transformations must be a string, list, or dictionary.")

        for col in self.columns:
            trans = transformations.get(col)
            if trans == 'log':
                self.transformers_[col] = FunctionTransformer(
                    func=lambda x: np.log(x.replace(0, np.nan)),
                    inverse_func=np.exp,
                    check_inverse=False
                )
            elif trans == 'sqrt':
                self.transformers_[col] = FunctionTransformer(
                    func=np.sqrt,
                    inverse_func=np.square,
                    check_inverse=False
                )
            elif trans == 'power':
                self.transformers_[col] = PowerTransformer(method=self.method, **self.kwargs)
            elif trans == 'boxcox':
                self.transformers_[col] = FunctionTransformer(
                    func=lambda x: boxcox(x.clip(lower=1e-6))[0],
                    check_inverse=False
                )
            elif trans == 'zscore':
                self.transformers_[col] = StandardScaler(**self.kwargs)
            elif trans == 'minmax':
                self.transformers_[col] = MinMaxScaler(**self.kwargs)
            elif trans == 'quantile':
                self.transformers_[col] = QuantileTransformer(
                    output_distribution=self.output_distribution, **self.kwargs
                )
            elif trans == 'rank':
                self.transformers_[col] = FunctionTransformer(
                    func=lambda x: x.rank(),
                    check_inverse=False
                )
            elif trans == 'dft':
                self.transformers_[col] = FunctionTransformer(
                    func=lambda x: np.fft.fft(x.to_numpy()).real,
                    check_inverse=False
                )
            else:
                raise ValueError(f"Unsupported transformation: {trans}")

            self.transformers_[col].fit(X[[col]])
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms the input DataFrame.

        Args:
            X (pd.DataFrame): Input DataFrame.

        Returns:
            pd.DataFrame: Transformed DataFrame.
        """
        X_transformed = X.copy()
        for col, transformer in self.transformers_.items():
            if col in X.columns:
                X_transformed[col] = transformer.transform(X[[col]])
            else:
                raise ValueError(f"Column '{col}' not found in input DataFrame.")
        return X_transformed

    def inverse_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Inverse transforms the input DataFrame.

        Args:
            X (pd.DataFrame): Transformed DataFrame.

        Returns:
            pd.DataFrame: Original DataFrame.
        """
        X_inv = X.copy()
        for col, transformer in self.transformers_.items():
            if hasattr(transformer, 'inverse_transform'):
                X_inv[col] = transformer.inverse_transform(X[[col]])
            else:
                raise ValueError(f"Transformer for column '{col}' does not support inverse_transform.")
        return X_inv

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """
        Get parameters for this estimator.

        Args:
            deep (bool): If True, will return the parameters for this estimator and contained subobjects that are estimators.

        Returns:
            Dict[str, Any]: Parameter names mapped to their values.
        """
        return {
            'columns': self.columns,
            'transformations': self.transformations,
            'method': self.method,
            'output_distribution': self.output_distribution,
            **self.kwargs
        }

    def set_params(self, **params: Any) -> 'FeatureTransformer':
        """
        Set the parameters of this estimator.

        Args:
            **params: Estimator parameters.

        Returns:
            FeatureTransformer: Returns self.
        """
        for key, value in params.items():
            setattr(self, key, value)
        return self
