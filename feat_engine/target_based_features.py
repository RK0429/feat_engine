import pandas as pd
import numpy as np
from sklearn.model_selection import KFold


class TargetBasedFeatures:
    """
    The TargetBasedFeatures class provides various methods for creating target-based features.
    These methods include target mean encoding, smoothed target mean encoding, count encoding,
    and cross-validated target encoding, which are useful for feature engineering in machine learning models.
    """

    def __init__(self) -> None:
        """Initialize the TargetBasedFeatures class."""
        pass

    def target_mean_encoding(self, df: pd.DataFrame, target_col: str, group_col: str) -> pd.Series:
        """
        Apply target mean encoding to a categorical column.

        Target mean encoding computes the mean of the target variable for each category of the group column,
        which can help provide a useful signal in a predictive model.

        Args:
            df (pd.DataFrame): The input dataframe containing the data.
            target_col (str): The name of the target column.
            group_col (str): The name of the categorical column to group by.

        Returns:
            pd.Series: A series with the target mean encoded values for each instance in the dataframe.
        """
        mean_encoding = df.groupby(group_col)[target_col].mean()
        return df[group_col].map(mean_encoding)

    def smoothed_target_mean_encoding(self, df: pd.DataFrame, target_col: str, group_col: str, m: int) -> pd.Series:
        """
        Apply smoothed target mean encoding with regularization to a categorical column.

        This technique smooths the target mean encoding by shrinking the group mean towards the global mean,
        helping to reduce the variance for small sample sizes within groups.

        Args:
            df (pd.DataFrame): The input dataframe containing the data.
            target_col (str): The name of the target column.
            group_col (str): The name of the categorical column to group by.
            m (int): The smoothing parameter, which controls the regularization strength.

        Returns:
            pd.Series: A series with the smoothed target mean encoded values for each instance in the dataframe.
        """
        global_mean = df[target_col].mean()
        agg = df.groupby(group_col)[target_col].agg(['mean', 'count'])
        smoothed_mean = (agg['count'] * agg['mean'] + m * global_mean) / (agg['count'] + m)
        return df[group_col].map(smoothed_mean)

    def count_encoding(self, df: pd.DataFrame, group_col: str) -> pd.Series:
        """
        Apply count encoding to a categorical column.

        Count encoding replaces each category with the number of times it appears in the dataset,
        providing a simple yet effective way to encode categorical data.

        Args:
            df (pd.DataFrame): The input dataframe containing the data.
            group_col (str): The name of the categorical column to encode.

        Returns:
            pd.Series: A series with the count encoded values for each instance in the dataframe.
        """
        counts = df[group_col].value_counts()
        return df[group_col].map(counts)

    def cross_validated_target_encoding(self, df: pd.DataFrame, target_col: str, group_col: str, n_splits: int = 5) -> pd.Series:
        """
        Apply cross-validated target encoding to a categorical column to avoid data leakage.

        Cross-validated target encoding creates encoded values using out-of-fold predictions in a K-fold setup,
        ensuring that the encoded values are not influenced by the corresponding target value in the same fold.

        Args:
            df (pd.DataFrame): The input dataframe containing the data.
            target_col (str): The name of the target column.
            group_col (str): The name of the categorical column to encode.
            n_splits (int): The number of cross-validation splits (default: 5).

        Returns:
            pd.Series: A series with the cross-validated target encoded values for each instance in the dataframe.
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
        Calculate Weight of Evidence (WoE) for a categorical column based on the target variable.

        WoE is a popular encoding method used in credit scoring, which measures the strength of separation
        between the positive and negative classes for each category.

        Args:
            df (pd.DataFrame): The input dataframe containing the data.
            target_col (str): The name of the target column.
            group_col (str): The name of the categorical column to calculate WoE.

        Returns:
            pd.Series: A series with the WoE encoded values for each instance in the dataframe.
        """
        pos_prob = df.groupby(group_col)[target_col].mean()
        neg_prob = 1 - pos_prob
        woe = np.log(pos_prob / neg_prob)
        return df[group_col].map(woe)
