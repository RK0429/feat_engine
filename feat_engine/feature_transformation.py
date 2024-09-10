import numpy as np
import pandas as pd
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, PowerTransformer, QuantileTransformer
)
from scipy.stats import boxcox


class FeatureTransformation:
    """
    FeatureTransformation class provides various feature transformation methods.
    """

    def __init__(self):
        pass

    def log_transform(self, df: pd.DataFrame, columns: list) -> pd.DataFrame:
        """
        Apply log transformation to specified columns.

        Args:
        - df (pd.DataFrame): Input DataFrame.
        - columns (list): List of column names to apply log transformation.

        Returns:
        - pd.DataFrame: DataFrame with log-transformed columns.
        """
        df[columns] = np.log(df[columns].replace(0, np.nan))  # Log of 0 is undefined
        return df

    def sqrt_transform(self, df: pd.DataFrame, columns: list) -> pd.DataFrame:
        """
        Apply square root transformation to specified columns.

        Args:
        - df (pd.DataFrame): Input DataFrame.
        - columns (list): List of column names to apply square root transformation.

        Returns:
        - pd.DataFrame: DataFrame with square root-transformed columns.
        """
        df[columns] = np.sqrt(df[columns])
        return df

    def power_transform(self, df: pd.DataFrame, columns: list, method: str = 'yeo-johnson') -> pd.DataFrame:
        """
        Apply power transformation (Yeo-Johnson or Box-Cox) to specified columns.

        Args:
        - df (pd.DataFrame): Input DataFrame.
        - columns (list): List of column names to apply power transformation.
        - method (str): 'yeo-johnson' or 'box-cox'. Box-Cox is only applicable to positive data.

        Returns:
        - pd.DataFrame: DataFrame with power-transformed columns.
        """
        pt = PowerTransformer(method=method)
        df[columns] = pt.fit_transform(df[columns])
        return df

    def boxcox_transform(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        """
        Apply Box-Cox transformation to a specified column (only for positive data).

        Args:
        - df (pd.DataFrame): Input DataFrame.
        - column (str): Column name to apply Box-Cox transformation.

        Returns:
        - pd.DataFrame: DataFrame with Box-Cox-transformed column.
        """
        df[column], _ = boxcox(df[column].clip(lower=1e-6))  # Clip values to avoid zero or negative
        return df

    def zscore_standardization(self, df: pd.DataFrame, columns: list) -> pd.DataFrame:
        """
        Apply Z-score standardization to specified columns.

        Args:
        - df (pd.DataFrame): Input DataFrame.
        - columns (list): List of column names to apply Z-score standardization.

        Returns:
        - pd.DataFrame: DataFrame with standardized columns.
        """
        scaler = StandardScaler()
        df[columns] = scaler.fit_transform(df[columns])
        return df

    def min_max_scaling(self, df: pd.DataFrame, columns: list) -> pd.DataFrame:
        """
        Apply min-max scaling to specified columns.

        Args:
        - df (pd.DataFrame): Input DataFrame.
        - columns (list): List of column names to apply min-max scaling.

        Returns:
        - pd.DataFrame: DataFrame with scaled columns.
        """
        scaler = MinMaxScaler()
        df[columns] = scaler.fit_transform(df[columns])
        return df

    def quantile_transform(self, df: pd.DataFrame, columns: list, output_distribution: str = 'normal') -> pd.DataFrame:
        """
        Apply quantile transformation to specified columns.

        Args:
        - df (pd.DataFrame): Input DataFrame.
        - columns (list): List of column names to apply quantile transformation.
        - output_distribution (str): 'normal' or 'uniform'.

        Returns:
        - pd.DataFrame: DataFrame with quantile-transformed columns.
        """
        qt = QuantileTransformer(output_distribution=output_distribution)
        df[columns] = qt.fit_transform(df[columns])
        return df

    def rank_transform(self, df: pd.DataFrame, columns: list) -> pd.DataFrame:
        """
        Apply rank transformation to specified columns.

        Args:
        - df (pd.DataFrame): Input DataFrame.
        - columns (list): List of column names to apply rank transformation.

        Returns:
        - pd.DataFrame: DataFrame with rank-transformed columns.
        """
        df[columns] = df[columns].rank()
        return df

    def discrete_fourier_transform(self, df: pd.DataFrame, columns: list) -> pd.DataFrame:
        """
        Apply discrete Fourier transform to specified columns.

        Args:
        - df (pd.DataFrame): Input DataFrame.
        - columns (list): List of column names to apply Fourier transformation.

        Returns:
        - pd.DataFrame: DataFrame with Fourier-transformed columns.
        """
        df[columns] = np.fft.fft(df[columns].to_numpy(), axis=0).real
        return df
