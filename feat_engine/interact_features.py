import pandas as pd
from sklearn.preprocessing import PolynomialFeatures


class FeatureInteraction:
    """
    The `FeatureInteraction` class provides methods to generate various types of feature interactions,
    including polynomial features, product features, arithmetic combinations, and crossed features for categorical variables.
    """

    def __init__(self) -> None:
        """
        Initializes the `FeatureInteraction` class.
        """
        pass

    def polynomial_features(self, df: pd.DataFrame, features: list[str], degree: int = 2) -> pd.DataFrame:
        """
        Generates polynomial features for specified features in the input DataFrame.

        Args:
            df (pd.DataFrame): Input DataFrame containing the features to be used.
            features (list[str]): List of feature names to generate polynomial interactions.
            degree (int): Degree of polynomial features to generate. Default is 2.

        Returns:
            pd.DataFrame: DataFrame containing both original features and generated polynomial features.
        """
        poly = PolynomialFeatures(degree=degree, include_bias=False)
        poly_features = poly.fit_transform(df[features])
        poly_feature_names = poly.get_feature_names_out(features)
        df_poly = pd.DataFrame(poly_features, columns=poly_feature_names, index=df.index)
        return pd.concat([df, df_poly], axis=1)

    def product_features(self, df: pd.DataFrame, feature_pairs: list[tuple[str, str]]) -> pd.DataFrame:
        """
        Creates product interaction features between specified pairs of features.

        Args:
            df (pd.DataFrame): Input DataFrame containing the features.
            feature_pairs (list[tuple[str, str]]): List of tuples representing feature pairs to create product features.

        Returns:
            pd.DataFrame: DataFrame containing the original data and new product interaction features.
        """
        for (f1, f2) in feature_pairs:
            df[f'{f1}_x_{f2}'] = df[f1] * df[f2]
        return df

    def arithmetic_combinations(self, df: pd.DataFrame, feature_pairs: list[tuple[str, str]], operations: list[str] = ['add', 'subtract']) -> pd.DataFrame:
        """
        Generates arithmetic combination features for specified feature pairs, applying operations such as addition,
        subtraction, multiplication, and division.

        Args:
            df (pd.DataFrame): Input DataFrame containing the features.
            feature_pairs (list[tuple[str, str]]): List of tuples representing feature pairs for arithmetic combinations.
            operations (list[str]): List of arithmetic operations to apply. Default includes 'add' and 'subtract'.
                                    Other options are 'multiply' and 'divide'.

        Returns:
            pd.DataFrame: DataFrame containing both the original features and newly generated arithmetic combination features.
        """
        for (f1, f2) in feature_pairs:
            if 'add' in operations:
                df[f'{f1}_plus_{f2}'] = df[f1] + df[f2]
            if 'subtract' in operations:
                df[f'{f1}_minus_{f2}'] = df[f1] - df[f2]
            if 'multiply' in operations:
                df[f'{f1}_times_{f2}'] = df[f1] * df[f2]
            if 'divide' in operations and (df[f2] != 0).all():  # Avoid division by zero
                df[f'{f1}_div_{f2}'] = df[f1] / df[f2]
        return df

    def crossed_features(self, df: pd.DataFrame, feature_pairs: list[tuple[str, str]]) -> pd.DataFrame:
        """
        Creates crossed interaction features for specified categorical variable pairs, combining them into a new feature.

        Args:
            df (pd.DataFrame): Input DataFrame containing the categorical features.
            feature_pairs (list[tuple[str, str]]): List of tuples representing pairs of categorical features to create crossed features.

        Returns:
            pd.DataFrame: DataFrame containing the original features and newly created crossed features.
        """
        for (f1, f2) in feature_pairs:
            df[f'{f1}_{f2}_crossed'] = df[f1].astype(str) + '_' + df[f2].astype(str)
        return df
