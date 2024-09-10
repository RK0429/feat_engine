import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import RobustScaler
from scipy.stats import zscore, mstats


class OutlierHandling:
    """
    A class that provides several methods for detecting and handling outliers in datasets.
    """

    def __init__(self) -> None:
        pass

    def z_score_detection(self, df: pd.DataFrame, threshold: float = 3.0) -> pd.DataFrame:
        """
        Detects outliers using Z-Score method.

        Args:
        - df (pd.DataFrame): Input dataframe.
        - threshold (float): Z-score threshold beyond which values are considered outliers (default: 3.0).

        Returns:
        - pd.DataFrame: Boolean dataframe indicating True for outliers.
        """
        z_scores = np.abs(zscore(df))
        return pd.DataFrame(z_scores > threshold, index=df.index, columns=df.columns)

    def iqr_detection(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detects outliers using the Interquartile Range (IQR) method.

        Args:
        - df (pd.DataFrame): Input dataframe.

        Returns:
        - pd.DataFrame: Boolean dataframe indicating True for outliers.
        """
        Q1 = df.quantile(0.25)
        Q3 = df.quantile(0.75)
        IQR = Q3 - Q1
        is_outlier = (df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))
        return is_outlier

    def isolation_forest_detection(self, df: pd.DataFrame, contamination: float = 0.1) -> pd.Series:
        """
        Detects outliers using the Isolation Forest method.

        Args:
        - df (pd.DataFrame): Input dataframe.
        - contamination (float): The proportion of outliers in the data (default: 0.1).

        Returns:
        - pd.Series: Boolean series indicating True for outliers.
        """
        iso_forest = IsolationForest(contamination=contamination)
        outliers = iso_forest.fit_predict(df)
        return pd.Series(outliers == -1, index=df.index)

    def dbscan_detection(self, df: pd.DataFrame, eps: float = 0.5, min_samples: int = 5) -> pd.Series:
        """
        Detects outliers using the DBSCAN method.

        Args:
        - df (pd.DataFrame): Input dataframe.
        - eps (float): The maximum distance between two samples to be considered as neighbors.
        - min_samples (int): The number of samples required to form a cluster.

        Returns:
        - pd.Series: Boolean series indicating True for outliers.
        """
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        outliers = dbscan.fit_predict(df)
        return pd.Series(outliers == -1, index=df.index)

    def robust_scaler(self, df: pd.DataFrame, lower_percentile: float = 0.01, upper_percentile: float = 0.99) -> pd.DataFrame:
        """
        Scales the data using RobustScaler to reduce the impact of outliers.

        Args:
        - df (pd.DataFrame): Input dataframe.

        Returns:
        - pd.DataFrame: Scaled dataframe.
        """
        scaler = RobustScaler()
        capped_df = df.clip(lower=df.quantile(lower_percentile), upper=df.quantile(upper_percentile), axis=1)
        scaled_df = pd.DataFrame(scaler.fit_transform(capped_df), columns=df.columns, index=df.index)
        return scaled_df

    def winsorization(self, df: pd.DataFrame, limits: tuple = (0.05, 0.05)) -> pd.DataFrame:
        """
        Limits extreme values in the data using Winsorization.

        Args:
        - df (pd.DataFrame): Input dataframe.
        - limits (tuple): The fraction of data to be Winsorized from the bottom and top (default: 5%).

        Returns:
        - pd.DataFrame: Winsorized dataframe.
        """
        return df.apply(lambda col: pd.Series(mstats.winsorize(col, limits=limits), index=df.index))

    def cap_outliers(self, df: pd.DataFrame, method: str = 'iqr', range_ratio: float = 0.8) -> pd.DataFrame:
        """
        Caps outliers by setting them to a maximum or minimum threshold.

        Args:
        - df (pd.DataFrame): Input dataframe.
        - method (str): The method to use for capping ('iqr' is supported).

        Returns:
        - pd.DataFrame: Dataframe with capped outliers.
        """
        if method == 'iqr':
            Q1 = df.quantile(0.25)
            Q3 = df.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - range_ratio * IQR / 2
            upper_bound = Q3 + range_ratio * IQR / 2
            return df.clip(lower=lower_bound, upper=upper_bound, axis=1)
        else:
            raise ValueError("Unsupported method. Currently only 'iqr' is supported.")
