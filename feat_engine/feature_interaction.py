import pandas as pd
from sklearn.preprocessing import PolynomialFeatures


class FeatureInteraction:
    """
    The FeatureInteraction class provides methods to generate various feature interactions.
    """

    def __init__(self) -> None:
        pass

    def polynomial_features(self, df: pd.DataFrame, features: list, degree: int = 2) -> pd.DataFrame:
        """
        Generates polynomial features for specified features.

        Args:
        - df (pd.DataFrame): Input DataFrame.
        - features (list): List of features to generate polynomial interactions.
        - degree (int): Degree of polynomial features. Default is 2.

        Returns:
        - pd.DataFrame: DataFrame containing the original and polynomial features.
        """
        poly = PolynomialFeatures(degree=degree, include_bias=False)
        poly_features = poly.fit_transform(df[features])
        poly_feature_names = poly.get_feature_names_out(features)
        df_poly = pd.DataFrame(poly_features, columns=poly_feature_names, index=df.index)
        return pd.concat([df, df_poly], axis=1)

    def product_features(self, df: pd.DataFrame, feature_pairs: list) -> pd.DataFrame:
        """
        Creates product interaction features between specified feature pairs.

        Args:
        - df (pd.DataFrame): Input DataFrame.
        - feature_pairs (list of tuples): List of feature pairs to create product features.

        Returns:
        - pd.DataFrame: DataFrame containing product interaction features.
        """
        for (f1, f2) in feature_pairs:
            df[f'{f1}_x_{f2}'] = df[f1] * df[f2]
        return df

    def arithmetic_combinations(self, df: pd.DataFrame, feature_pairs: list, operations: list = ['add', 'subtract']) -> pd.DataFrame:
        """
        Generates arithmetic combination features for specified feature pairs.

        Args:
        - df (pd.DataFrame): Input DataFrame.
        - feature_pairs (list of tuples): List of feature pairs for arithmetic combinations.
        - operations (list): List of operations to apply ('add', 'subtract', 'multiply', 'divide').

        Returns:
        - pd.DataFrame: DataFrame containing arithmetic combination features.
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

    def crossed_features(self, df: pd.DataFrame, feature_pairs: list) -> pd.DataFrame:
        """
        Creates crossed interaction features for categorical variables.

        Args:
        - df (pd.DataFrame): Input DataFrame.
        - feature_pairs (list of tuples): List of categorical feature pairs to create crossed features.

        Returns:
        - pd.DataFrame: DataFrame containing crossed interaction features.
        """
        for (f1, f2) in feature_pairs:
            df[f'{f1}_{f2}_crossed'] = df[f1].astype(str) + '_' + df[f2].astype(str)
        return df
