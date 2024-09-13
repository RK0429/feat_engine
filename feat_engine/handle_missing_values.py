import pandas as pd
from typing import List, Optional, Union, Any, Dict
from sklearn.experimental import enable_iterative_imputer  # noqa F401
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.base import RegressorMixin, ClassifierMixin


class MissingValueHandler:
    """
    A class to handle missing values in datasets using various strategies such as simple imputation,
    KNN-based imputation, iterative imputation, and machine learning models.
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
    def missing_summary(data: pd.DataFrame) -> pd.DataFrame:
        """
        Provides a summary of missing values for each column in the dataset.

        Args:
            data (pd.DataFrame): The input DataFrame.

        Returns:
            pd.DataFrame: A DataFrame with columns 'missing_count' and 'missing_percentage'.
        """
        missing_count = data.isnull().sum()
        missing_percentage = 100 * missing_count / len(data)
        summary = pd.DataFrame({
            'missing_count': missing_count,
            'missing_percentage': missing_percentage
        })
        return summary

    @staticmethod
    def drop_missing(
        data: pd.DataFrame,
        axis: int = 0,
        how: str = 'any',
        thresh: Optional[int] = None,
        subset: Optional[List[str]] = None,
        inplace: bool = False
    ) -> pd.DataFrame:
        """
        Drops rows or columns with missing values.

        Args:
            data (pd.DataFrame): The input DataFrame.
            axis (int, optional): Specifies whether to drop rows (0) or columns (1). Default is 0 (drop rows).
            how (str, optional): 'any' or 'all'. If 'any', drop if any NA values are present. If 'all', drop if all values are NA.
            thresh (int, optional): Require that many non-NA values. Overrides 'how'.
            subset (List[str], optional): Labels along the axis to consider.
            inplace (bool, optional): If True, perform operation in-place.

        Returns:
            pd.DataFrame: The DataFrame with missing rows or columns dropped.
        """
        return data.dropna(axis=axis, how=how, thresh=thresh, subset=subset, inplace=inplace)

    @staticmethod
    def drop_missing_threshold(
        data: pd.DataFrame,
        threshold: float = 0.5,
        axis: int = 0,
        inplace: bool = False
    ) -> pd.DataFrame:
        """
        Drops rows or columns with missing values that exceed a specified threshold.

        Args:
            data (pd.DataFrame): The input DataFrame.
            threshold (float): The maximum allowed proportion of missing values (between 0 and 1). Default is 0.5.
            axis (int): Specifies whether to drop rows (0) or columns (1). Default is 0 (drop rows).
            inplace (bool, optional): If True, perform operation in-place.

        Returns:
            pd.DataFrame: The DataFrame with rows or columns dropped based on the missing value threshold.
        """
        if not 0 <= threshold <= 1:
            raise ValueError("Threshold must be between 0 and 1.")
        thresh = int((1 - threshold) * data.shape[axis])
        return data.dropna(axis=axis, thresh=thresh, inplace=inplace)

    @staticmethod
    def fill_missing(
        data: pd.DataFrame,
        strategy: Union[str, Dict[str, str]] = 'mean',
        fill_value: Optional[Any] = None,
        columns: Optional[List[str]] = None,
        inplace: bool = False
    ) -> pd.DataFrame:
        """
        Fills missing values in the DataFrame using specified strategies.

        Args:
            data (pd.DataFrame): The input DataFrame.
            strategy (Union[str, Dict[str, str]]): The imputation strategy ('mean', 'median', 'most_frequent', 'constant') or
                                                  a dictionary mapping column names to strategies.
            fill_value (Any, optional): When strategy='constant', used to fill missing values.
            columns (List[str], optional): List of columns to impute. If None, all columns are imputed.
            inplace (bool, optional): If True, perform operation in-place.

        Returns:
            pd.DataFrame: The DataFrame with missing values filled according to the strategy.
        """
        if not inplace:
            data = data.copy()

        if columns is None:
            columns = data.columns.tolist()

        if isinstance(strategy, str):
            imputer = SimpleImputer(strategy=strategy, fill_value=fill_value)
            data[columns] = imputer.fit_transform(data[columns])
        elif isinstance(strategy, dict):
            for col in columns:
                col_strategy = strategy.get(col, 'mean')
                imputer = SimpleImputer(strategy=col_strategy, fill_value=fill_value)
                data[[col]] = imputer.fit_transform(data[[col]])
        else:
            raise ValueError("Strategy must be a string or a dictionary mapping columns to strategies.")
        return data

    @staticmethod
    def fill_missing_knn(
        data: pd.DataFrame,
        n_neighbors: int = 5,
        weights: str = 'uniform',
        metric: str = 'nan_euclidean',
        columns: Optional[List[str]] = None,
        inplace: bool = False
    ) -> pd.DataFrame:
        """
        Fills missing values using K-Nearest Neighbors (KNN) imputation.

        Args:
            data (pd.DataFrame): The input DataFrame.
            n_neighbors (int): Number of neighboring samples to use for imputation.
            weights (str): Weight function used in prediction ('uniform' or 'distance').
            metric (str): Distance metric for searching neighbors.
            columns (List[str], optional): List of columns to impute. If None, all columns are imputed.
            inplace (bool, optional): If True, perform operation in-place.

        Returns:
            pd.DataFrame: The DataFrame with missing values filled using KNN imputation.
        """
        if not inplace:
            data = data.copy()

        if columns is None:
            columns = data.columns.tolist()

        imputer = KNNImputer(n_neighbors=n_neighbors, weights=weights, metric=metric)
        data[columns] = imputer.fit_transform(data[columns])
        return data

    @staticmethod
    def fill_missing_iterative(
        data: pd.DataFrame,
        estimator: Optional[RegressorMixin] = None,
        columns: Optional[List[str]] = None,
        inplace: bool = False,
        **kwargs: Any
    ) -> pd.DataFrame:
        """
        Fills missing values using Iterative Imputer.

        Args:
            data (pd.DataFrame): The input DataFrame.
            estimator (RegressorMixin, optional): The estimator to use at each step of the imputation. If None, BayesianRidge is used.
            columns (List[str], optional): List of columns to impute. If None, all columns are imputed.
            inplace (bool, optional): If True, perform operation in-place.
            **kwargs: Additional keyword arguments to pass to IterativeImputer.

        Returns:
            pd.DataFrame: The DataFrame with missing values filled using Iterative Imputer.
        """
        if not inplace:
            data = data.copy()

        if columns is None:
            columns = data.columns.tolist()

        imputer = IterativeImputer(estimator=estimator, **kwargs)
        data[columns] = imputer.fit_transform(data[columns])
        return data

    @staticmethod
    def fill_missing_ml(
        data: pd.DataFrame,
        target_column: str,
        model: Optional[Union[RegressorMixin, ClassifierMixin]] = None,
        inplace: bool = False,
        **kwargs: Any
    ) -> pd.DataFrame:
        """
        Fills missing values in the target column using a machine learning model trained on the other columns.

        Args:
            data (pd.DataFrame): The input DataFrame.
            target_column (str): The name of the column with missing values to impute.
            model (Union[RegressorMixin, ClassifierMixin], optional): The machine learning model to use. If None, RandomForestRegressor or RandomForestClassifier is used.
            inplace (bool, optional): If True, perform operation in-place.
            **kwargs: Additional keyword arguments to pass to the model.

        Returns:
            pd.DataFrame: The DataFrame with missing values in the target column filled using the model.
        """
        if not inplace:
            data = data.copy()

        df_complete = data.dropna(subset=[target_column])
        df_missing = data[data[target_column].isnull()]

        if df_missing.empty:
            return data

        X_train = df_complete.drop(columns=[target_column])
        y_train = df_complete[target_column]
        X_predict = df_missing.drop(columns=[target_column])

        # Handle categorical features
        X_full = pd.concat([X_train, X_predict])
        X_full_encoded = pd.get_dummies(X_full, drop_first=True)
        X_train_encoded = X_full_encoded.iloc[:len(X_train)]
        X_predict_encoded = X_full_encoded.iloc[len(X_train):]

        if y_train.dtype.kind in 'biufc':
            model = model or RandomForestRegressor(**kwargs)
        else:
            model = model or RandomForestClassifier(**kwargs)

        model.fit(X_train_encoded, y_train)
        predicted_values = model.predict(X_predict_encoded)

        data.loc[data[target_column].isnull(), target_column] = predicted_values
        return data

    @staticmethod
    def add_missing_indicator(
        data: pd.DataFrame,
        columns: Optional[List[str]] = None,
        inplace: bool = False
    ) -> pd.DataFrame:
        """
        Adds a binary indicator column for each feature, showing where missing values were located.

        Args:
            data (pd.DataFrame): The input DataFrame.
            columns (List[str], optional): List of columns to create indicators for. If None, only columns with missing values are used.
            inplace (bool, optional): If True, perform operation in-place.

        Returns:
            pd.DataFrame: The original DataFrame with additional indicator columns for missing values.
        """
        if not inplace:
            data = data.copy()

        # If columns is None, select only the columns with missing values
        if columns is None:
            columns = data.columns[data.isnull().any()].tolist()

        # Add missing value indicators for the specified columns
        for column in columns:
            data[f"{column}_missing"] = data[column].isnull().astype(int)

        return data

    @staticmethod
    def fill_missing_ffill(
        data: pd.DataFrame,
        columns: Optional[List[str]] = None,
        inplace: bool = False,
        limit: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Fills missing values using forward fill method.

        Args:
            data (pd.DataFrame): The input DataFrame.
            columns (List[str], optional): List of columns to forward fill. If None, all columns are used.
            inplace (bool, optional): If True, perform operation in-place.
            limit (int, optional): The maximum number of consecutive NaNs to fill.

        Returns:
            pd.DataFrame: The DataFrame with missing values filled using forward fill.
        """
        if not inplace:
            data = data.copy()

        data.fillna(method='ffill', axis=0, limit=limit, inplace=True)
        return data

    @staticmethod
    def fill_missing_bfill(
        data: pd.DataFrame,
        columns: Optional[List[str]] = None,
        inplace: bool = False,
        limit: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Fills missing values using backward fill method.

        Args:
            data (pd.DataFrame): The input DataFrame.
            columns (List[str], optional): List of columns to backward fill. If None, all columns are used.
            inplace (bool, optional): If True, perform operation in-place.
            limit (int, optional): The maximum number of consecutive NaNs to fill.

        Returns:
            pd.DataFrame: The DataFrame with missing values filled using backward fill.
        """
        if not inplace:
            data = data.copy()

        data.fillna(method='bfill', axis=0, limit=limit, inplace=True)
        return data

    @staticmethod
    def interpolate_missing(
        data: pd.DataFrame,
        method: str = 'linear',
        axis: int = 0,
        limit: Optional[int] = None,
        inplace: bool = False,
        **kwargs: Any
    ) -> pd.DataFrame:
        """
        Fills missing values using interpolation.

        Args:
            data (pd.DataFrame): The input DataFrame.
            method (str, optional): Interpolation method. Defaults to 'linear'.
            axis (int, optional): Axis along which to interpolate. Defaults to 0.
            limit (int, optional): Maximum number of consecutive NaNs to fill.
            inplace (bool, optional): If True, perform operation in-place.
            **kwargs: Additional keyword arguments to pass to interpolate.

        Returns:
            pd.DataFrame: The DataFrame with missing values filled using interpolation.
        """
        if not inplace:
            data = data.copy()

        data.interpolate(method=method, axis=axis, limit=limit, inplace=True, **kwargs)
        return data
