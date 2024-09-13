import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder
from sklearn.base import BaseEstimator, TransformerMixin
import category_encoders as ce
from typing import List, Optional, Dict, Any, Union


class CategoricalEncoder(BaseEstimator, TransformerMixin):
    """
    Class `CategoricalEncoder` provides methods for encoding categorical variables,
    including label encoding, one-hot encoding, ordinal encoding, binary encoding,
    target encoding, frequency encoding, and more.

    The class is designed to be compatible with scikit-learn pipelines.
    """

    def __init__(
        self,
        encoding_methods: Optional[Union[str, Dict[str, str]]] = 'one_hot',
        columns: Optional[List[str]] = None,
        categories: Optional[Dict[str, List[str]]] = None,
        target: Optional[str] = None,
        drop_first: bool = False,
        **kwargs: Any
    ):
        """
        Initializes the CategoricalEncoder.

        Args:
            encoding_methods (Union[str, Dict[str, str]], optional): The encoding method(s) to use.
                Can be a string specifying the method to use for all columns, or a dictionary mapping
                column names to encoding methods. Supported methods include:
                - 'label': Label Encoding
                - 'one_hot': One-Hot Encoding
                - 'ordinal': Ordinal Encoding
                - 'binary': Binary Encoding
                - 'target': Target Encoding
                - 'frequency': Frequency Encoding
            columns (List[str], optional): List of columns to encode. If None, all object-type columns will be encoded.
            categories (Dict[str, List[str]], optional): Dictionary mapping column names to list of categories for ordinal encoding.
            target (str, optional): Target column name, required for target encoding.
            drop_first (bool, optional): Whether to drop the first category in one-hot encoding to avoid multicollinearity.
            **kwargs: Additional keyword arguments to pass to the underlying encoders.
        """
        self.encoding_methods = encoding_methods
        self.columns = columns
        self.categories = categories
        self.target = target
        self.drop_first = drop_first
        self.kwargs = kwargs
        self.encoders: Dict[str, Any] = {}

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'CategoricalEncoder':
        """
        Fits the encoders to the data.

        Args:
            X (pd.DataFrame): Input DataFrame.
            y (pd.Series, optional): Target variable. Required for target encoding.

        Returns:
            CategoricalEncoder: Fitted encoder.
        """
        # Determine columns to encode
        if self.columns is None:
            self.columns = X.select_dtypes(include=['object', 'category']).columns.tolist()

        # Handle encoding methods
        if isinstance(self.encoding_methods, str):
            encoding_methods = {col: self.encoding_methods for col in self.columns}
        elif isinstance(self.encoding_methods, dict):
            encoding_methods = self.encoding_methods
        else:
            raise ValueError("encoding_methods should be a string or a dictionary mapping columns to methods.")

        for column in self.columns:
            method = encoding_methods.get(column, 'one_hot')
            if method == 'label':
                encoder = LabelEncoder()
                encoder.fit(X[column].astype(str))
                self.encoders[column] = ('label', encoder)
            elif method == 'one_hot':
                encoder = OneHotEncoder(
                    sparse_output=False,
                    drop='first' if self.drop_first else None,
                    handle_unknown='ignore'
                )
                encoder.fit(X[[column]])
                self.encoders[column] = ('one_hot', encoder)
            elif method == 'ordinal':
                categories = self.categories.get(column) if self.categories else 'auto'
                encoder = OrdinalEncoder(categories=[categories] if categories != 'auto' else 'auto', handle_unknown='use_encoded_value', unknown_value=-1)
                encoder.fit(X[[column]])
                self.encoders[column] = ('ordinal', encoder)
            elif method == 'binary':
                encoder = ce.BinaryEncoder(cols=[column], **self.kwargs)
                encoder.fit(X)
                self.encoders[column] = ('binary', encoder)
            elif method == 'target':
                if y is None:
                    raise ValueError("y cannot be None for target encoding.")
                encoder = ce.TargetEncoder(cols=[column], **self.kwargs)
                encoder.fit(X, y)
                self.encoders[column] = ('target', encoder)
            elif method == 'frequency':
                freq = X[column].value_counts(normalize=True)
                self.encoders[column] = ('frequency', freq)
            else:
                raise ValueError(f"Unsupported encoding method: {method}")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms the data using the fitted encoders.

        Args:
            X (pd.DataFrame): Input DataFrame.

        Returns:
            pd.DataFrame: Transformed DataFrame.
        """
        X_transformed = X.copy()
        for column, (method, encoder) in self.encoders.items():
            if method == 'label':
                X_transformed[column] = encoder.transform(X_transformed[column].astype(str))
            elif method == 'one_hot':
                encoded = encoder.transform(X_transformed[[column]])
                encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out([column]), index=X_transformed.index)
                X_transformed = pd.concat([X_transformed, encoded_df], axis=1)
                X_transformed.drop(columns=[column], inplace=True)
            elif method == 'ordinal':
                X_transformed[column] = encoder.transform(X_transformed[[column]])
            elif method == 'binary':
                encoded_df = encoder.transform(X_transformed)
                X_transformed = pd.concat([X_transformed.drop(columns=[column]), encoded_df], axis=1)
            elif method == 'target':
                encoded = encoder.transform(X_transformed)
                X_transformed[column] = encoded[column]
            elif method == 'frequency':
                X_transformed[column] = X_transformed[column].map(encoder).fillna(0)
            else:
                raise ValueError(f"Unsupported encoding method: {method}")
        return X_transformed

    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        Fits the encoders and transforms the data.

        Args:
            X (pd.DataFrame): Input DataFrame.
            y (pd.Series, optional): Target variable. Required for target encoding.

        Returns:
            pd.DataFrame: Transformed DataFrame.
        """
        return self.fit(X, y).transform(X)

    def inverse_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Inverse transforms the data back to original categories.

        Args:
            X (pd.DataFrame): Encoded DataFrame.

        Returns:
            pd.DataFrame: DataFrame with original categories.
        """
        X_inv = X.copy()
        for column, (method, encoder) in self.encoders.items():
            if method == 'label':
                X_inv[column] = encoder.inverse_transform(X_inv[column].astype(int))
            elif method == 'one_hot':
                # Need to reconstruct original column from one-hot encoded columns
                feature_names = encoder.get_feature_names_out([column])
                encoded_cols = [col for col in feature_names if col in X_inv.columns]
                if not encoded_cols:
                    continue  # No encoded columns present
                one_hot_values = X_inv[encoded_cols].values
                categories = encoder.categories_[0]
                if self.drop_first:
                    categories = categories[1:]  # If drop_first, the first category is missing
                indices = one_hot_values.argmax(axis=1)
                X_inv[column] = [categories[idx] if one_hot_values[i].sum() > 0 else None for i, idx in enumerate(indices)]
                X_inv.drop(columns=encoded_cols, inplace=True)
            elif method == 'ordinal':
                categories = encoder.categories_[0]
                X_inv[column] = X_inv[column].apply(lambda x: categories[int(x)] if 0 <= x < len(categories) else None)
            elif method == 'binary':
                # Binary encoding cannot be inversely transformed
                pass
            elif method == 'target':
                # Target encoding cannot be inversely transformed
                pass
            elif method == 'frequency':
                # Frequency encoding cannot be inversely transformed
                pass
            else:
                raise ValueError(f"Unsupported encoding method: {method}")
        return X_inv

    def get_feature_names_out(self, input_features: Optional[List[str]] = None) -> List[str]:
        """
        Get output feature names for transformation.

        Args:
            input_features (List[str], optional): List of input feature names. If None, uses self.columns.

        Returns:
            List[str]: List of output feature names.
        """
        output_features = []
        for column, (method, encoder) in self.encoders.items():
            if method == 'one_hot':
                feature_names = encoder.get_feature_names_out([column])
                output_features.extend(feature_names)
            elif method == 'binary':
                binary_feature_names = encoder.get_feature_names()
                output_features.extend(binary_feature_names)
            else:
                output_features.append(column)
        return output_features

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """
        Get parameters for this estimator.

        Args:
            deep (bool): If True, will return the parameters for this estimator and contained subobjects that are estimators.

        Returns:
            Dict[str, Any]: Parameter names mapped to their values.
        """
        params = {
            'encoding_methods': self.encoding_methods,
            'columns': self.columns,
            'categories': self.categories,
            'target': self.target,
            'drop_first': self.drop_first,
            **self.kwargs
        }
        if deep:
            for encoder in self.encoders.values():
                if hasattr(encoder, 'get_params'):
                    params.update(encoder.get_params(deep=deep))
        return params

    def set_params(self, **params: Any) -> 'CategoricalEncoder':
        """
        Set the parameters of this estimator.

        Args:
            **params: Estimator parameters.

        Returns:
            CategoricalEncoder: Returns self.
        """
        for key, value in params.items():
            setattr(self, key, value)
        return self
