import numpy as np
import pandas as pd
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, PowerTransformer, QuantileTransformer
)
from scipy.stats import boxcox


class FeatureTransformer:
    """
    The `FeatureTransformer` class provides various methods for transforming numerical features
    in a dataset using techniques such as log transformation, power transformation, and scaling.
    """

    def __init__(self) -> None:
        """
        Initializes the `FeatureTransformer` class.
        """
        pass

    def log_transform(self, df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
        """
        Apply log transformation to specified columns in the DataFrame.

        Args:
            df (pd.DataFrame): Input DataFrame containing numerical columns.
            columns (list[str]): List of column names to apply log transformation to.

        Returns:
            pd.DataFrame: DataFrame with log-transformed columns.
        """
        df[columns] = np.log(df[columns].replace(0, np.nan))  # Log of 0 is undefined, replaced with NaN
        return df

    def sqrt_transform(self, df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
        """
        Apply square root transformation to specified columns in the DataFrame.

        Args:
            df (pd.DataFrame): Input DataFrame containing numerical columns.
            columns (list[str]): List of column names to apply square root transformation to.

        Returns:
            pd.DataFrame: DataFrame with square root-transformed columns.
        """
        df[columns] = np.sqrt(df[columns])
        return df

    def power_transform(self, df: pd.DataFrame, columns: list[str], method: str = 'yeo-johnson') -> pd.DataFrame:
        """
        Apply power transformation (Yeo-Johnson or Box-Cox) to specified columns in the DataFrame.

        Args:
            df (pd.DataFrame): Input DataFrame containing numerical columns.
            columns (list[str]): List of column names to apply power transformation to.
            method (str): Method of power transformation to use ('yeo-johnson' or 'box-cox'). Box-Cox requires positive data.

        Returns:
            pd.DataFrame: DataFrame with power-transformed columns.
        """
        pt = PowerTransformer(method=method)
        df[columns] = pt.fit_transform(df[columns])
        return df

    def boxcox_transform(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        """
        Apply Box-Cox transformation to a specified column in the DataFrame (only applicable to positive data).

        Args:
            df (pd.DataFrame): Input DataFrame containing numerical columns.
            column (str): Column name to apply Box-Cox transformation to.

        Returns:
            pd.DataFrame: DataFrame with the Box-Cox-transformed column.
        """
        df[column], _ = boxcox(df[column].clip(lower=1e-6))  # Clip values to avoid zero or negative inputs
        return df

    def zscore_standardization(self, df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
        """
        Apply Z-score standardization to specified columns in the DataFrame.

        Args:
            df (pd.DataFrame): Input DataFrame containing numerical columns.
            columns (list[str]): List of column names to apply Z-score standardization to.

        Returns:
            pd.DataFrame: DataFrame with standardized columns, where each column has mean 0 and variance 1.
        """
        scaler = StandardScaler()
        df[columns] = scaler.fit_transform(df[columns])
        return df

    def min_max_scaling(self, df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
        """
        Apply min-max scaling to specified columns in the DataFrame.

        Args:
            df (pd.DataFrame): Input DataFrame containing numerical columns.
            columns (list[str]): List of column names to apply min-max scaling to.

        Returns:
            pd.DataFrame: DataFrame with scaled columns, where each value is between 0 and 1.
        """
        scaler = MinMaxScaler()
        df[columns] = scaler.fit_transform(df[columns])
        return df

    def quantile_transform(self, df: pd.DataFrame, columns: list[str], output_distribution: str = 'normal') -> pd.DataFrame:
        """
        Apply quantile transformation to specified columns in the DataFrame.

        Args:
            df (pd.DataFrame): Input DataFrame containing numerical columns.
            columns (list[str]): List of column names to apply quantile transformation to.
            output_distribution (str): Desired output distribution ('normal' or 'uniform').

        Returns:
            pd.DataFrame: DataFrame with quantile-transformed columns.
        """
        qt = QuantileTransformer(output_distribution=output_distribution)
        df[columns] = qt.fit_transform(df[columns])
        return df

    def rank_transform(self, df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
        """
        Apply rank transformation to specified columns in the DataFrame.

        Args:
            df (pd.DataFrame): Input DataFrame containing numerical columns.
            columns (list[str]): List of column names to apply rank transformation to.

        Returns:
            pd.DataFrame: DataFrame with rank-transformed columns, where each value is replaced by its rank.
        """
        df[columns] = df[columns].rank()
        return df

    def discrete_fourier_transform(self, df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
        """
        Apply discrete Fourier transform to specified columns in the DataFrame.

        Args:
            df (pd.DataFrame): Input DataFrame containing numerical columns.
            columns (list[str]): List of column names to apply Fourier transformation to.

        Returns:
            pd.DataFrame: DataFrame with Fourier-transformed columns, retaining the real part of the transformation.
        """
        df[columns] = np.fft.fft(df[columns].to_numpy(), axis=0).real
        return df
