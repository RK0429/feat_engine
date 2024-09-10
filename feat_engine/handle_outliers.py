import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import RobustScaler
from scipy.stats import zscore, mstats


class OutlierHandler:
    """
    A class that provides methods for detecting and handling outliers in datasets.
    Supports Z-Score, IQR, Isolation Forest, DBSCAN, RobustScaler, Winsorization, and capping methods.
    """

    def __init__(self) -> None:
        """
        Initialize the OutlierHandler class.
        """
        pass

    def z_score_detection(self, df: pd.DataFrame, threshold: float = 3.0) -> pd.DataFrame:
        """
        Detect outliers using the Z-Score method.

        Args:
            df (pd.DataFrame): Input dataframe.
            threshold (float): Z-score threshold beyond which values are considered outliers (default: 3.0).

        Returns:
            pd.DataFrame: A boolean dataframe indicating True for outliers.
        """
        z_scores = np.abs(zscore(df))
        return pd.DataFrame(z_scores > threshold, index=df.index, columns=df.columns)

    def iqr_detection(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect outliers using the Interquartile Range (IQR) method.

        The IQR method identifies outliers as points that fall below the 1.5 times the IQR
        below the first quartile (Q1) or above the third quartile (Q3).

        Args:
            df (pd.DataFrame): Input dataframe.

        Returns:
            pd.DataFrame: A boolean dataframe indicating True for outliers.
        """
        Q1 = df.quantile(0.25)
        Q3 = df.quantile(0.75)
        IQR = Q3 - Q1
        is_outlier = (df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))
        return is_outlier

    def isolation_forest_detection(self, df: pd.DataFrame, contamination: float = 0.1) -> pd.Series:
        """
        Detect outliers using the Isolation Forest method.

        Isolation Forest isolates observations by randomly selecting a feature and then
        randomly selecting a split value between the maximum and minimum values of the selected feature.

        Args:
            df (pd.DataFrame): Input dataframe.
            contamination (float): The proportion of outliers in the data (default: 0.1).

        Returns:
            pd.Series: A boolean series indicating True for outliers.
        """
        iso_forest = IsolationForest(contamination=contamination)
        outliers = iso_forest.fit_predict(df)
        return pd.Series(outliers == -1, index=df.index)

    def dbscan_detection(self, df: pd.DataFrame, eps: float = 0.5, min_samples: int = 5) -> pd.Series:
        """
        Detect outliers using the DBSCAN (Density-Based Spatial Clustering of Applications with Noise) method.

        DBSCAN is a density-based clustering algorithm that marks points in low-density regions as outliers.

        Args:
            df (pd.DataFrame): Input dataframe.
            eps (float): The maximum distance between two samples to be considered as neighbors.
            min_samples (int): The minimum number of samples required to form a dense region.

        Returns:
            pd.Series: A boolean series indicating True for outliers.
        """
        numeric_df = df.select_dtypes(include=[np.number])
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        outliers = dbscan.fit_predict(numeric_df)
        return pd.Series(outliers == -1, index=df.index)

    def robust_scaler(self, df: pd.DataFrame, lower_percentile: float = 0.01, upper_percentile: float = 0.99) -> pd.DataFrame:
        """
        Scale data using the RobustScaler method, which is less sensitive to outliers.

        The data is scaled by subtracting the median and dividing by the interquartile range, making it robust to outliers.

        Args:
            df (pd.DataFrame): Input dataframe.
            lower_percentile (float): Lower bound for clipping the data (default: 0.01).
            upper_percentile (float): Upper bound for clipping the data (default: 0.99).

        Returns:
            pd.DataFrame: The scaled dataframe.
        """
        scaler = RobustScaler()
        capped_df = df.clip(lower=df.quantile(lower_percentile), upper=df.quantile(upper_percentile), axis=1)
        scaled_df = pd.DataFrame(scaler.fit_transform(capped_df), columns=df.columns, index=df.index)
        return scaled_df

    def winsorization(self, df: pd.DataFrame, limits: tuple = (0.05, 0.05)) -> pd.DataFrame:
        """
        Apply Winsorization to limit extreme values in the data.

        Winsorization transforms extreme values (both top and bottom) into less extreme values
        based on specified limits.

        Args:
            df (pd.DataFrame): Input dataframe.
            limits (tuple): The fraction of data to be Winsorized from the bottom and top (default: 5%).

        Returns:
            pd.DataFrame: The Winsorized dataframe.
        """
        return df.apply(lambda col: pd.Series(mstats.winsorize(col, limits=limits), index=df.index))

    def cap_outliers(self, df: pd.DataFrame, method: str = 'iqr', range_ratio: float = 0.8) -> pd.DataFrame:
        """
        Cap outliers by setting values beyond a threshold to a maximum or minimum value.

        Args:
            df (pd.DataFrame): Input dataframe.
            method (str): The method to use for capping outliers. Only 'iqr' is supported.
            range_ratio (float): The ratio to adjust the IQR range for capping (default: 0.8).

        Returns:
            pd.DataFrame: The dataframe with capped outliers.
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
