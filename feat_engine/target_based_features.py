import pandas as pd
import numpy as np
from sklearn.model_selection import KFold


class TargetBasedFeatures:
    """
    Class for creating target-based features for machine learning models.
    This includes various encodings like target mean encoding, smoothed target mean encoding,
    count encoding, and cross-validated target encoding.
    """

    def __init__(self) -> None:
        pass

    def target_mean_encoding(self, df: pd.DataFrame, target_col: str, group_col: str) -> pd.Series:
        """
        Apply target mean encoding to the specified column.

        Args:
        - df (pd.DataFrame): DataFrame containing data.
        - target_col (str): Name of the target column.
        - group_col (str): Name of the categorical column to group by.

        Returns:
        - pd.Series: A Series with the target mean encoded values.
        """
        mean_encoding = df.groupby(group_col)[target_col].mean()
        return df[group_col].map(mean_encoding)

    def smoothed_target_mean_encoding(self, df: pd.DataFrame, target_col: str, group_col: str, m: int) -> pd.Series:
        """
        Apply smoothed target mean encoding with regularization.

        Args:
        - df (pd.DataFrame): DataFrame containing data.
        - target_col (str): Name of the target column.
        - group_col (str): Name of the categorical column to group by.
        - m (int): Smoothing parameter.

        Returns:
        - pd.Series: A Series with the smoothed target mean encoded values.
        """
        global_mean = df[target_col].mean()
        agg = df.groupby(group_col)[target_col].agg(['mean', 'count'])
        smoothed_mean = (agg['count'] * agg['mean'] + m * global_mean) / (agg['count'] + m)
        return df[group_col].map(smoothed_mean)

    def count_encoding(self, df: pd.DataFrame, group_col: str) -> pd.Series:
        """
        Apply count encoding to the specified column.

        Args:
        - df (pd.DataFrame): DataFrame containing data.
        - group_col (str): Name of the categorical column to group by.

        Returns:
        - pd.Series: A Series with the count encoded values.
        """
        counts = df[group_col].value_counts()
        return df[group_col].map(counts)

    def cross_validated_target_encoding(self, df: pd.DataFrame, target_col: str, group_col: str, n_splits: int = 5) -> pd.Series:
        """
        Apply cross-validated target encoding to avoid data leakage.

        Args:
        - df (pd.DataFrame): DataFrame containing data.
        - target_col (str): Name of the target column.
        - group_col (str): Name of the categorical column to group by.
        - n_splits (int): Number of cross-validation splits.

        Returns:
        - pd.Series: A Series with the cross-validated target encoded values.
        """
        kf = KFold(n_splits=n_splits, shuffle=True)
        df['encoded'] = 0
        for train_idx, val_idx in kf.split(df):
            train_df, val_df = df.iloc[train_idx], df.iloc[val_idx]
            mean_encoding = train_df.groupby(group_col)[target_col].mean()
            df.loc[val_idx, 'encoded'] = val_df[group_col].map(mean_encoding)
        return df['encoded']

    def calculate_woe(self, df: pd.DataFrame, target_col: str, group_col: str) -> pd.Series:
        """
        Calculate Weight of Evidence (WoE) for a categorical feature based on the target.

        Args:
        - df (pd.DataFrame): DataFrame containing data.
        - target_col (str): Name of the target column.
        - group_col (str): Name of the categorical column to group by.

        Returns:
        - pd.Series: A Series with the WoE encoded values.
        """
        pos_prob = df.groupby(group_col)[target_col].mean()
        neg_prob = 1 - pos_prob
        woe = np.log(pos_prob / neg_prob)
        return df[group_col].map(woe)
