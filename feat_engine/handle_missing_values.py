import pandas as pd
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer  # Enable the experimental API
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.base import RegressorMixin


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
    def drop_missing_threshold(data: pd.DataFrame, threshold: float = 0.5, axis: int = 0) -> pd.DataFrame:
        """
        Drops rows or columns with missing values that exceed a specified threshold.

        Args:
            data (pd.DataFrame): The input DataFrame.
            threshold (float): The proportion of allowed missing values before dropping. Default is 0.5.
            axis (int): Specifies whether to drop rows (0) or columns (1). Default is 0 (drop rows).

        Returns:
            pd.DataFrame: The DataFrame with rows or columns dropped based on the missing value threshold.
        """
        return data.dropna(thresh=int((1 - threshold) * data.shape[axis]), axis=axis)

    @staticmethod
    def fill_missing(data: pd.DataFrame, strategy_num: str = 'mean', strategy_cat: str = 'most_frequent', knn: bool = False, n_neighbors: int = 5) -> pd.DataFrame:
        """
        Fills missing values in the DataFrame using specified strategies for numerical and categorical columns.
        Optionally applies KNN imputation for both numerical and categorical data.

        Args:
            data (pd.DataFrame): The input DataFrame.
            strategy_num (str): The strategy to use for imputing missing values in numerical columns ('mean', 'median', 'most_frequent', or 'constant').
            strategy_cat (str): The strategy to use for imputing missing values in categorical columns ('most_frequent' or 'constant').
            knn (bool): Whether to use KNN imputation for both numerical and categorical columns. Default is False.
            n_neighbors (int): The number of neighboring samples to use for KNN imputation if knn is True.

        Returns:
            pd.DataFrame: The DataFrame with missing values filled according to the strategy or KNN imputation.
        """
        # Separate numerical and categorical columns
        num_cols = data.select_dtypes(include=['number']).columns
        cat_cols = data.select_dtypes(include=['object']).columns
        filled_data = data.copy()

        if knn:
            # Handle KNN imputation for both numerical and categorical columns
            if not cat_cols.empty:
                encoders = {col: LabelEncoder() for col in cat_cols}
                for col in cat_cols:
                    filled_data[col] = encoders[col].fit_transform(filled_data[col].astype(str))

            imputer_knn = KNNImputer(n_neighbors=n_neighbors)
            filled_data = pd.DataFrame(imputer_knn.fit_transform(filled_data), columns=data.columns)

            if not cat_cols.empty:
                for col in cat_cols:
                    filled_data[col] = encoders[col].inverse_transform(filled_data[col].astype(int))
        else:
            # Impute missing values for numerical columns
            if not num_cols.empty:
                imputer_num = SimpleImputer(strategy=strategy_num)
                filled_data[num_cols] = pd.DataFrame(imputer_num.fit_transform(data[num_cols]), columns=num_cols)

            # Impute missing values for categorical columns
            if not cat_cols.empty:
                imputer_cat = SimpleImputer(strategy=strategy_cat)
                filled_data[cat_cols] = pd.DataFrame(imputer_cat.fit_transform(data[cat_cols]), columns=cat_cols)

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
    def fill_missing_ml_regression(data: pd.DataFrame, target_column: str, model: RegressorMixin = RandomForestRegressor(), test_size: float = 0.2) -> pd.DataFrame:
        """
        Fills missing values in the target column using a machine learning regression model trained on the other columns.

        Args:
            data (pd.DataFrame): The input DataFrame.
            target_column (str): The name of the column with missing values to impute.
            model (Any): The machine learning model to use for regression. Default is RandomForestRegressor.
            test_size (float): The proportion of the data to use as the test set when evaluating the model. Default is 0.2.

        Returns:
            pd.DataFrame: The DataFrame with missing values in the target column filled using regression.
        """
        # Split data into rows with and without missing values in the target column
        df_complete = data.dropna(subset=[target_column])
        df_missing = data[data[target_column].isnull()]

        # If no missing data in the target column, return the original data
        if df_missing.empty:
            return data

        # Features and target for training
        X = df_complete.drop(columns=[target_column])
        y = df_complete[target_column]

        # Prepare the data for prediction (rows with missing target)
        X_missing = df_missing.drop(columns=[target_column])

        # Train the model
        model.fit(X, y)

        # Predict missing values
        predicted_values = model.predict(X_missing)

        # Fill missing values in the original data
        data.loc[data[target_column].isnull(), target_column] = predicted_values

        return data

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
