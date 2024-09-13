import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.base import BaseEstimator, TransformerMixin
from typing import Optional


class TargetBasedEncoder(BaseEstimator, TransformerMixin):
    """
    A transformer that provides various methods for creating target-based features,
    including target mean encoding, smoothed target mean encoding, count encoding,
    cross-validated target encoding, and Weight of Evidence (WoE).
    """

    def __init__(
        self,
        method: str = 'target_mean',
        target_col: str = '',
        group_col: str = '',
        smoothing: Optional[int] = None,
        n_splits: int = 5
    ):
        """
        Initializes the TargetBasedEncoder.

        Args:
            method (str): The method to use for encoding. Options: 'target_mean', 'smoothed_mean', 'count',
                          'cross_validated_mean', 'woe'.
            target_col (str): The name of the target column.
            group_col (str): The name of the categorical column to encode.
            smoothing (int, optional): Smoothing parameter for smoothed target mean encoding (default: None).
            n_splits (int): The number of splits for cross-validated encoding (default: 5).
        """
        self.method = method
        self.target_col = target_col
        self.group_col = group_col
        self.smoothing = smoothing
        self.n_splits = n_splits

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'TargetBasedEncoder':
        """
        Fit method for compatibility with the scikit-learn API.

        Args:
            X (pd.DataFrame): Input DataFrame.
            y (pd.Series, optional): Target variable (ignored).

        Returns:
            TargetBasedEncoder: Returns self.
        """
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Applies the selected encoding method to the DataFrame.

        Args:
            X (pd.DataFrame): Input DataFrame.

        Returns:
            pd.DataFrame: DataFrame with the target-based feature encoding applied.
        """
        if self.method == 'target_mean':
            return self._target_mean_encoding(X)
        elif self.method == 'smoothed_mean':
            if self.smoothing is None:
                raise ValueError("Smoothing parameter 'm' is required for smoothed mean encoding.")
            return self._smoothed_target_mean_encoding(X)
        elif self.method == 'count':
            return self._count_encoding(X)
        elif self.method == 'cross_validated_mean':
            return self._cross_validated_target_encoding(X)
        elif self.method == 'woe':
            return self._calculate_woe(X)
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def _target_mean_encoding(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply target mean encoding to the group column.

        Args:
            df (pd.DataFrame): Input DataFrame.

        Returns:
            pd.DataFrame: DataFrame with target mean encoding.
        """
        mean_encoding = df.groupby(self.group_col)[self.target_col].mean()
        return df[self.group_col].map(mean_encoding).to_frame(f'{self.group_col}_target_mean')

    def _smoothed_target_mean_encoding(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply smoothed target mean encoding to the group column.

        Args:
            df (pd.DataFrame): Input DataFrame.

        Returns:
            pd.DataFrame: DataFrame with smoothed target mean encoding.
        """
        global_mean = df[self.target_col].mean()
        agg = df.groupby(self.group_col)[self.target_col].agg(['mean', 'count'])
        smoothed_mean = (agg['count'] * agg['mean'] + self.smoothing * global_mean) / (agg['count'] + self.smoothing)
        return df[self.group_col].map(smoothed_mean).to_frame(f'{self.group_col}_smoothed_target_mean')

    def _count_encoding(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply count encoding to the group column.

        Args:
            df (pd.DataFrame): Input DataFrame.

        Returns:
            pd.DataFrame: DataFrame with count encoding.
        """
        counts = df[self.group_col].value_counts()
        return df[self.group_col].map(counts).to_frame(f'{self.group_col}_count')

    def _cross_validated_target_encoding(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply cross-validated target mean encoding to the group column.

        Args:
            df (pd.DataFrame): Input DataFrame.

        Returns:
            pd.DataFrame: DataFrame with cross-validated target encoding.
        """
        kf = KFold(n_splits=self.n_splits, shuffle=True)
        df_encoded = df.copy()
        df_encoded['encoded'] = 0

        for train_idx, val_idx in kf.split(df):
            train_df, val_df = df.iloc[train_idx], df.iloc[val_idx]
            mean_encoding = train_df.groupby(self.group_col)[self.target_col].mean()
            df_encoded.loc[val_idx, 'encoded'] = val_df[self.group_col].map(mean_encoding)

        return df_encoded[['encoded']].rename(columns={'encoded': f'{self.group_col}_cv_target_mean'})

    def _calculate_woe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Weight of Evidence (WoE) for the group column based on the target variable.

        Args:
            df (pd.DataFrame): Input DataFrame.

        Returns:
            pd.DataFrame: DataFrame with WoE encoding.
        """
        pos_prob = df.groupby(self.group_col)[self.target_col].mean()
        neg_prob = 1 - pos_prob
        woe = np.log(pos_prob / neg_prob)
        return df[self.group_col].map(woe).to_frame(f'{self.group_col}_woe')
