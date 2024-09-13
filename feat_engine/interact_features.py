import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.base import BaseEstimator, TransformerMixin
from typing import List, Tuple, Optional


class PolynomialFeaturesTransformer(BaseEstimator, TransformerMixin):
    """
    Generates polynomial features for specified features in the input DataFrame.
    """

    def __init__(
        self,
        degree: int = 2,
        include_bias: bool = False,
        interaction_only: bool = False,
        features: Optional[List[str]] = None
    ):
        """
        Initializes the PolynomialFeaturesTransformer.

        Args:
            degree (int): Degree of polynomial features to generate. Default is 2.
            include_bias (bool): Whether to include a bias column. Default is False.
            interaction_only (bool): If True, only interaction features are produced (no powers of single features). Default is False.
            features (List[str], optional): List of feature names to generate polynomial interactions. If None, all numeric features are used.
        """
        self.degree = degree
        self.include_bias = include_bias
        self.interaction_only = interaction_only
        self.features = features
        self.poly = None

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'PolynomialFeaturesTransformer':
        """
        Fits the transformer to the data.

        Args:
            X (pd.DataFrame): Input DataFrame.
            y (pd.Series, optional): Target variable (not used).

        Returns:
            self
        """
        if self.features is None:
            self.features = X.select_dtypes(include=[np.number]).columns.tolist()
        self.poly = PolynomialFeatures(
            degree=self.degree,
            include_bias=self.include_bias,
            interaction_only=self.interaction_only
        )
        self.poly.fit(X[self.features])  # type: ignore
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms the input DataFrame by adding polynomial features.

        Args:
            X (pd.DataFrame): Input DataFrame.

        Returns:
            pd.DataFrame: DataFrame with polynomial features added.
        """
        if self.poly is None:
            raise RuntimeError("You must fit the transformer before transforming the data.")

        X_transformed = X.copy()
        poly_features = self.poly.transform(X[self.features])
        feature_names = self.poly.get_feature_names_out(self.features)
        df_poly = pd.DataFrame(poly_features, columns=feature_names, index=X.index)

        # Remove columns that already exist in X to avoid duplicates
        df_poly = df_poly.drop(columns=self.features, errors='ignore')
        X_transformed = pd.concat([X_transformed, df_poly], axis=1)
        return X_transformed


class ProductFeaturesTransformer(BaseEstimator, TransformerMixin):
    """
    Creates product interaction features between specified pairs of features.
    """

    def __init__(self, feature_pairs: Optional[List[Tuple[str, str]]] = None):
        """
        Initializes the ProductFeaturesTransformer.

        Args:
            feature_pairs (List[Tuple[str, str]], optional): List of tuples representing feature pairs to create product features.
                                                             If None, all possible pairs of numeric features are used.
        """
        self.feature_pairs = feature_pairs

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'ProductFeaturesTransformer':
        """
        Fits the transformer to the data.

        Args:
            X (pd.DataFrame): Input DataFrame.
            y (pd.Series, optional): Target variable (not used).

        Returns:
            self
        """
        if self.feature_pairs is None:
            numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
            self.feature_pairs = [
                (f1, f2) for i, f1 in enumerate(numeric_features)
                for f2 in numeric_features[i + 1:]
            ]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms the input DataFrame by adding product features.

        Args:
            X (pd.DataFrame): Input DataFrame.

        Returns:
            pd.DataFrame: DataFrame with product features added.
        """
        if self.feature_pairs is None:
            raise ValueError("feature_pairs is None. Fit the transformer or provide feature pairs before calling transform.")

        X_transformed = X.copy()
        for (f1, f2) in self.feature_pairs:
            if f1 in X.columns and f2 in X.columns:
                X_transformed[f'{f1}_x_{f2}'] = X[f1] * X[f2]
            else:
                raise ValueError(f"Features '{f1}' and/or '{f2}' not found in DataFrame.")
        return X_transformed


class ArithmeticCombinationsTransformer(BaseEstimator, TransformerMixin):
    """
    Generates arithmetic combination features for specified feature pairs.
    """

    def __init__(
        self,
        feature_pairs: Optional[List[Tuple[str, str]]] = None,
        operations: Optional[List[str]] = None
    ):
        """
        Initializes the ArithmeticCombinationsTransformer.

        Args:
            feature_pairs (List[Tuple[str, str]], optional): List of tuples representing feature pairs for arithmetic combinations.
                                                             If None, all possible pairs of numeric features are used.
            operations (List[str], optional): List of arithmetic operations to apply. Options are 'add', 'subtract', 'multiply', 'divide'.
                                              Default is ['add', 'subtract'].
        """
        self.feature_pairs = feature_pairs
        self.operations = operations or ['add', 'subtract']

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'ArithmeticCombinationsTransformer':
        """
        Fits the transformer to the data.

        Args:
            X (pd.DataFrame): Input DataFrame.
            y (pd.Series, optional): Target variable (not used).

        Returns:
            self
        """
        if self.feature_pairs is None:
            numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
            self.feature_pairs = [
                (f1, f2) for i, f1 in enumerate(numeric_features)
                for f2 in numeric_features[i + 1:]
            ]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms the input DataFrame by adding arithmetic combination features.

        Args:
            X (pd.DataFrame): Input DataFrame.

        Returns:
            pd.DataFrame: DataFrame with arithmetic combination features added.
        """
        if self.feature_pairs is None:
            raise ValueError("feature_pairs is None. Fit the transformer or provide feature pairs before calling transform.")

        X_transformed = X.copy()
        for (f1, f2) in self.feature_pairs:
            if f1 not in X.columns or f2 not in X.columns:
                raise ValueError(f"Features '{f1}' and/or '{f2}' not found in DataFrame.")
            if 'add' in self.operations:
                X_transformed[f'{f1}_plus_{f2}'] = X[f1] + X[f2]
            if 'subtract' in self.operations:
                X_transformed[f'{f1}_minus_{f2}'] = X[f1] - X[f2]
            if 'multiply' in self.operations:
                X_transformed[f'{f1}_times_{f2}'] = X[f1] * X[f2]
            if 'divide' in self.operations:
                with np.errstate(divide='ignore', invalid='ignore'):
                    division_result = X[f1] / X[f2]
                    division_result.replace([np.inf, -np.inf], np.nan, inplace=True)
                    X_transformed[f'{f1}_div_{f2}'] = division_result
        return X_transformed


class CrossedFeaturesTransformer(BaseEstimator, TransformerMixin):
    """
    Creates crossed interaction features for specified categorical variable pairs.
    """

    def __init__(self, feature_pairs: Optional[List[Tuple[str, str]]] = None):
        """
        Initializes the CrossedFeaturesTransformer.

        Args:
            feature_pairs (List[Tuple[str, str]], optional): List of tuples representing pairs of categorical features to create crossed features.
                                                             If None, all possible pairs of categorical features are used.
        """
        self.feature_pairs = feature_pairs

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'CrossedFeaturesTransformer':
        """
        Fits the transformer to the data.

        Args:
            X (pd.DataFrame): Input DataFrame.
            y (pd.Series, optional): Target variable (not used).

        Returns:
            self
        """
        if self.feature_pairs is None:
            categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
            self.feature_pairs = [
                (f1, f2) for i, f1 in enumerate(categorical_features)
                for f2 in categorical_features[i + 1:]
            ]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms the input DataFrame by adding crossed features.

        Args:
            X (pd.DataFrame): Input DataFrame.

        Returns:
            pd.DataFrame: DataFrame with crossed features added.
        """
        if self.feature_pairs is None:
            raise ValueError("feature_pairs is None. Fit the transformer or provide feature pairs before calling transform.")

        X_transformed = X.copy()
        for (f1, f2) in self.feature_pairs:
            if f1 not in X.columns or f2 not in X.columns:
                raise ValueError(f"Features '{f1}' and/or '{f2}' not found in DataFrame.")
            X_transformed[f'{f1}_{f2}_crossed'] = X[f1].astype(str) + '_' + X[f2].astype(str)
        return X_transformed
