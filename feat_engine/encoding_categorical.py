import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder
import category_encoders as ce


class CategoricalEncoder:
    """
    Class `CategoricalEncoder` provides various methods for encoding categorical variables.
    """

    def __init__(self) -> None:
        self.encoders: dict = {}

    def label_encoding(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        """
        Apply label encoding to a specified column.

        Args:
        - df (pd.DataFrame): Input DataFrame.
        - column (str): Column to apply label encoding on.

        Returns:
        - pd.DataFrame: DataFrame with the encoded column.
        """
        le = LabelEncoder()
        df[f"{column}_encoded"] = le.fit_transform(df[column])
        self.encoders[column] = le
        return df

    def one_hot_encoding(self, df: pd.DataFrame, column: str, drop_first: bool = False) -> pd.DataFrame:
        """
        Apply one-hot encoding to a specified column.

        Args:
        - df (pd.DataFrame): Input DataFrame.
        - column (str): Column to apply one-hot encoding on.
        - drop_first (bool): Whether to drop the first category to avoid multicollinearity.

        Returns:
        - pd.DataFrame: DataFrame with the one-hot encoded columns.
        """
        ohe = OneHotEncoder(sparse_output=False, drop='first' if drop_first else None)
        encoded = ohe.fit_transform(df[[column]])
        df_encoded = pd.DataFrame(encoded, columns=ohe.get_feature_names_out([column]), index=df.index)
        df = pd.concat([df, df_encoded], axis=1)
        self.encoders[column] = ohe
        return df

    def ordinal_encoding(self, df: pd.DataFrame, column: str, categories: list) -> pd.DataFrame:
        """
        Apply ordinal encoding to a specified column with predefined category order.

        Args:
        - df (pd.DataFrame): Input DataFrame.
        - column (str): Column to apply ordinal encoding on.
        - categories (list): List specifying the order of categories.

        Returns:
        - pd.DataFrame: DataFrame with the ordinally encoded column.
        """
        oe = OrdinalEncoder(categories=[categories])
        df[f"{column}_encoded"] = oe.fit_transform(df[[column]])
        self.encoders[column] = oe
        return df

    def binary_encoding(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        """
        Apply binary encoding to a specified column.

        Args:
        - df (pd.DataFrame): Input DataFrame.
        - column (str): Column to apply binary encoding on.

        Returns:
        - pd.DataFrame: DataFrame with the binary encoded columns.
        """
        be = ce.BinaryEncoder(cols=[column])
        df = be.fit_transform(df)
        self.encoders[column] = be
        return df

    def target_encoding(self, df: pd.DataFrame, column: str, target: str) -> pd.DataFrame:
        """
        Apply target encoding to a specified column.

        Args:
        - df (pd.DataFrame): Input DataFrame.
        - column (str): Column to apply target encoding on.
        - target (str): Target column to compute the encoding.

        Returns:
        - pd.DataFrame: DataFrame with the target encoded column.
        """
        te = ce.TargetEncoder(cols=[column])
        df[f"{column}_encoded"] = te.fit_transform(df[column], df[target])
        self.encoders[column] = te
        return df

    def frequency_encoding(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        """
        Apply frequency encoding to a specified column.

        Args:
        - df (pd.DataFrame): Input DataFrame.
        - column (str): Column to apply frequency encoding on.

        Returns:
        - pd.DataFrame: DataFrame with the frequency encoded column.
        """
        freq = df[column].value_counts(normalize=True)
        df[f"{column}_encoded"] = df[column].map(freq)
        return df
