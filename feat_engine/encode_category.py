import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder
import category_encoders as ce


class CategoricalEncoder:
    """
    Class `CategoricalEncoder` provides various methods for encoding categorical variables,
    including label encoding, one-hot encoding, ordinal encoding, binary encoding, target encoding,
    and frequency encoding.
    """

    def __init__(self) -> None:
        """
        Initializes the `CategoricalEncoder` class and stores encoders used for encoding each column.

        Attributes:
        - encoders (dict): A dictionary to store the encoders used for each column.
        """
        self.encoders: dict = {}

    def label_encoding(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        """
        Applies label encoding to a specified column, transforming categorical values into integer labels.

        Args:
            df (pd.DataFrame): Input DataFrame containing the column to be encoded.
            column (str): The name of the column to apply label encoding.

        Returns:
            pd.DataFrame: The DataFrame with the encoded column appended as '{column}_encoded'.
        """
        le = LabelEncoder()
        df[f"{column}_encoded"] = le.fit_transform(df[column])
        self.encoders[column] = le
        return df

    def one_hot_encoding(self, df: pd.DataFrame, column: str, drop_first: bool = False) -> pd.DataFrame:
        """
        Applies one-hot encoding to a specified column, converting categorical values into a series of binary columns.

        Args:
            df (pd.DataFrame): Input DataFrame containing the column to be encoded.
            column (str): The name of the column to apply one-hot encoding.
            drop_first (bool): Whether to drop the first category to avoid multicollinearity. Default is False.

        Returns:
            pd.DataFrame: The DataFrame with the one-hot encoded columns appended.
        """
        ohe = OneHotEncoder(sparse_output=False, drop='first' if drop_first else None)
        encoded = ohe.fit_transform(df[[column]])
        df_encoded = pd.DataFrame(encoded, columns=ohe.get_feature_names_out([column]), index=df.index)
        df = pd.concat([df, df_encoded], axis=1)
        self.encoders[column] = ohe
        return df

    def ordinal_encoding(self, df: pd.DataFrame, column: str, categories: list) -> pd.DataFrame:
        """
        Applies ordinal encoding to a specified column, encoding categories based on a predefined order.

        Args:
            df (pd.DataFrame): Input DataFrame containing the column to be encoded.
            column (str): The name of the column to apply ordinal encoding.
            categories (list): List specifying the order of categories for ordinal encoding.

        Returns:
            pd.DataFrame: The DataFrame with the ordinally encoded column appended as '{column}_encoded'.
        """
        oe = OrdinalEncoder(categories=[categories])
        df[f"{column}_encoded"] = oe.fit_transform(df[[column]])
        self.encoders[column] = oe
        return df

    def binary_encoding(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        """
        Applies binary encoding to a specified column, encoding categorical values into binary representations.

        Args:
            df (pd.DataFrame): Input DataFrame containing the column to be encoded.
            column (str): The name of the column to apply binary encoding.

        Returns:
            pd.DataFrame: The DataFrame with the binary encoded columns.
        """
        be = ce.BinaryEncoder(cols=[column])
        df = be.fit_transform(df)
        self.encoders[column] = be
        return df

    def target_encoding(self, df: pd.DataFrame, column: str, target: str) -> pd.DataFrame:
        """
        Applies target encoding to a specified column, encoding categorical values based on their relationship
        to a target variable.

        Args:
            df (pd.DataFrame): Input DataFrame containing the column to be encoded.
            column (str): The name of the column to apply target encoding.
            target (str): The target column used to compute the encoding.

        Returns:
            pd.DataFrame: The DataFrame with the target encoded column appended as '{column}_encoded'.
        """
        te = ce.TargetEncoder(cols=[column])
        df[f"{column}_encoded"] = te.fit_transform(df[column], df[target])
        self.encoders[column] = te
        return df

    def frequency_encoding(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        """
        Applies frequency encoding to a specified column, encoding categories based on their frequency of occurrence.

        Args:
            df (pd.DataFrame): Input DataFrame containing the column to be encoded.
            column (str): The name of the column to apply frequency encoding.

        Returns:
            pd.DataFrame: The DataFrame with the frequency encoded column appended as '{column}_encoded'.
        """
        freq = df[column].value_counts(normalize=True)
        df[f"{column}_encoded"] = df[column].map(freq)
        return df
