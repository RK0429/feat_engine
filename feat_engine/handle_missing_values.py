import pandas as pd
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer  # Enable the experimental API
from sklearn.impute import IterativeImputer


class MissingValueHandler:
    """
    A class to handle missing values in datasets using various strategies such as simple imputation,
    KNN-based imputation, and iterative imputation.
    """

    @staticmethod
    def identify_missing(data: pd.DataFrame) -> pd.DataFrame:
        """
        Identifies missing values in the dataset.

        Args:
            data (pd.DataFrame): The input DataFrame.

        Returns:
            pd.DataFrame: A DataFrame of the same shape as the input, with boolean values
                          indicating where values are missing (True for missing values).
        """
        return data.isnull()

    @staticmethod
    def missing_summary(data: pd.DataFrame) -> pd.Series:
        """
        Provides a summary of missing values for each column in the dataset.

        Args:
            data (pd.DataFrame): The input DataFrame.

        Returns:
            pd.Series: A Series indicating the number of missing values in each column.
        """
        return data.isnull().sum()

    @staticmethod
    def drop_missing(data: pd.DataFrame, axis: int = 0, how: str = 'any') -> pd.DataFrame:
        """
        Drops rows or columns with missing values.

        Args:
            data (pd.DataFrame): The input DataFrame.
            axis (int): Specifies whether to drop rows (0) or columns (1). Default is 0 (drop rows).
            how (str): Specifies how to determine if a row or column is missing:
                        - 'any': If any NA values are present, drop.
                        - 'all': If all values are NA, drop.

        Returns:
            pd.DataFrame: The DataFrame with missing rows or columns dropped.
        """
        return data.dropna(axis=axis, how=how)

    @staticmethod
    def fill_missing(data: pd.DataFrame, strategy: str = 'mean') -> pd.DataFrame:
        """
        Fills missing values using a specified strategy, such as mean, median, or most frequent.

        Args:
            data (pd.DataFrame): The input DataFrame.
            strategy (str): The strategy to use for imputing missing values ('mean', 'median',
                            'most_frequent', or 'constant').

        Returns:
            pd.DataFrame: The DataFrame with missing values filled according to the strategy.
        """
        imputer = SimpleImputer(strategy=strategy)
        filled_data = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)
        return filled_data

    @staticmethod
    def fill_missing_constant(data: pd.DataFrame, fill_value: float | int | str) -> pd.DataFrame:
        """
        Fills missing values with a specified constant value.

        Args:
            data (pd.DataFrame): The input DataFrame.
            fill_value (float | int | str): The constant value to use for filling missing values.

        Returns:
            pd.DataFrame: The DataFrame with missing values filled by the constant value.
        """
        imputer = SimpleImputer(strategy='constant', fill_value=fill_value)
        filled_data = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)
        return filled_data

    @staticmethod
    def fill_missing_knn(data: pd.DataFrame, n_neighbors: int = 5) -> pd.DataFrame:
        """
        Fills missing values using K-Nearest Neighbors (KNN) imputation.

        Args:
            data (pd.DataFrame): The input DataFrame.
            n_neighbors (int): The number of neighboring samples to use for imputation.

        Returns:
            pd.DataFrame: The DataFrame with missing values filled using KNN imputation.
        """
        imputer = KNNImputer(n_neighbors=n_neighbors)
        filled_data = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)
        return filled_data

    @staticmethod
    def fill_missing_iterative(data: pd.DataFrame) -> pd.DataFrame:
        """
        Fills missing values using Iterative Imputer, which models each feature with missing values
        as a function of other features and uses that to impute the missing values.

        Args:
            data (pd.DataFrame): The input DataFrame.

        Returns:
            pd.DataFrame: The DataFrame with missing values filled using Iterative Imputer.
        """
        imputer = IterativeImputer()
        filled_data = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)
        return filled_data

    @staticmethod
    def add_missing_indicator(data: pd.DataFrame) -> pd.DataFrame:
        """
        Adds a binary indicator column for each feature, showing where missing values were located.

        Args:
            data (pd.DataFrame): The input DataFrame.

        Returns:
            pd.DataFrame: The original DataFrame with additional indicator columns for missing values
                          (one for each original column, with _missing appended to its name).
        """
        data_with_indicators = data.copy()
        for column in data.columns:
            data_with_indicators[column + '_missing'] = data[column].isnull().astype(int)
        return data_with_indicators
