import pandas as pd
# import numpy as np
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer  # Enable the experimental API
from sklearn.impute import IterativeImputer


class MissingValueHandler:
    """
    A class to handle missing values in datasets with various strategies.
    """

    @staticmethod
    def identify_missing(data: pd.DataFrame) -> pd.DataFrame:
        return data.isnull()

    @staticmethod
    def missing_summary(data: pd.DataFrame) -> pd.Series:
        return data.isnull().sum()

    @staticmethod
    def drop_missing(data: pd.DataFrame, axis: int = 0, how: str = 'any') -> pd.DataFrame:
        return data.dropna(axis=axis, how=how)

    @staticmethod
    def fill_missing(data: pd.DataFrame, strategy: str = 'mean') -> pd.DataFrame:
        imputer = SimpleImputer(strategy=strategy)
        filled_data = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)
        return filled_data

    @staticmethod
    def fill_missing_constant(data: pd.DataFrame, fill_value: float | int | str) -> pd.DataFrame:
        imputer = SimpleImputer(strategy='constant', fill_value=fill_value)
        filled_data = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)
        return filled_data

    @staticmethod
    def fill_missing_knn(data: pd.DataFrame, n_neighbors: int = 5) -> pd.DataFrame:
        imputer = KNNImputer(n_neighbors=n_neighbors)
        filled_data = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)
        return filled_data

    @staticmethod
    def fill_missing_iterative(data: pd.DataFrame) -> pd.DataFrame:
        """
        Fills missing values using Iterative Imputer (which is experimental).
        """
        imputer = IterativeImputer()
        filled_data = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)
        return filled_data

    @staticmethod
    def add_missing_indicator(data: pd.DataFrame) -> pd.DataFrame:
        data_with_indicators = data.copy()
        for column in data.columns:
            data_with_indicators[column + '_missing'] = data[column].isnull().astype(int)
        return data_with_indicators
