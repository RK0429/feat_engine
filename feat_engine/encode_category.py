import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder
import category_encoders as ce
from typing import List


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

    def label_encoding(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """
        Applies label encoding to the specified columns, transforming categorical values into integer labels.

        Args:
            df (pd.DataFrame): Input DataFrame containing the columns to be encoded.
            columns (List[str]): List of column names to apply label encoding.

        Returns:
            pd.DataFrame: The DataFrame with the encoded columns appended as '{column}_encoded'.
        """
        for column in columns:
            le = LabelEncoder()
            df[f"{column}_encoded"] = le.fit_transform(df[column])
            self.encoders[column] = le
        return df

    def one_hot_encoding(self, df: pd.DataFrame, columns: List[str], drop_first: bool = False) -> pd.DataFrame:
        """
        Applies one-hot encoding to the specified columns, converting categorical values into a series of binary columns.

        Args:
            df (pd.DataFrame): Input DataFrame containing the columns to be encoded.
            columns (List[str]): List of column names to apply one-hot encoding.
            drop_first (bool): Whether to drop the first category to avoid multicollinearity. Default is False.

        Returns:
            pd.DataFrame: The DataFrame with the one-hot encoded columns appended.
        """
        for column in columns:
            ohe = OneHotEncoder(sparse_output=False, drop='first' if drop_first else None)
            encoded = ohe.fit_transform(df[[column]])
            df_encoded = pd.DataFrame(encoded, columns=ohe.get_feature_names_out([column]), index=df.index)
            df = pd.concat([df, df_encoded], axis=1)
            self.encoders[column] = ohe
        return df

    def ordinal_encoding(self, df: pd.DataFrame, columns: List[str], categories: List[List[str]]) -> pd.DataFrame:
        """
        Applies ordinal encoding to the specified columns, encoding categories based on a predefined order.

        Args:
            df (pd.DataFrame): Input DataFrame containing the columns to be encoded.
            columns (List[str]): List of column names to apply ordinal encoding.
            categories (List[List[str]]): List of lists specifying the order of categories for each column.

        Returns:
            pd.DataFrame: The DataFrame with the ordinally encoded columns appended as '{column}_encoded'.
        """
        for i, column in enumerate(columns):
            oe = OrdinalEncoder(categories=[categories[i]])
            df[f"{column}_encoded"] = oe.fit_transform(df[[column]])
            self.encoders[column] = oe
        return df

    def binary_encoding(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """
        Applies binary encoding to the specified columns, encoding categorical values into binary representations.

        Args:
            df (pd.DataFrame): Input DataFrame containing the columns to be encoded.
            columns (List[str]): List of column names to apply binary encoding.

        Returns:
            pd.DataFrame: The DataFrame with the binary encoded columns.
        """
        be = ce.BinaryEncoder(cols=columns)
        df = be.fit_transform(df)
        for column in columns:
            self.encoders[column] = be
        return df

    def target_encoding(self, df: pd.DataFrame, columns: List[str], target: str) -> pd.DataFrame:
        """
        Applies target encoding to the specified columns, encoding categorical values based on their relationship
        to a target variable.

        Args:
            df (pd.DataFrame): Input DataFrame containing the columns to be encoded.
            columns (List[str]): List of column names to apply target encoding.
            target (str): The target column used to compute the encoding.

        Returns:
            pd.DataFrame: The DataFrame with the target encoded columns appended as '{column}_encoded'.
        """
        te = ce.TargetEncoder(cols=columns)
        df_encoded = te.fit_transform(df[columns], df[target])
        df = pd.concat([df, df_encoded.add_suffix("_encoded")], axis=1)
        for column in columns:
            self.encoders[column] = te
        return df

    def frequency_encoding(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """
        Applies frequency encoding to the specified columns, encoding categories based on their frequency of occurrence.

        Args:
            df (pd.DataFrame): Input DataFrame containing the columns to be encoded.
            columns (List[str]): List of column names to apply frequency encoding.

        Returns:
            pd.DataFrame: The DataFrame with the frequency encoded columns appended as '{column}_encoded'.
        """
        for column in columns:
            freq = df[column].value_counts(normalize=True)
            df[f"{column}_encoded"] = df[column].map(freq)
        return df
