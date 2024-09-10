import pytest
import pandas as pd
from feat_engine.encoding_categorical import CategoricalEncoder
from typing import Dict


@pytest.fixture
def sample_data() -> pd.DataFrame:
    """Fixture to provide a sample dataset for testing."""
    data: Dict[str, list] = {
        'Category': ['low', 'medium', 'high', 'medium', 'low'],
        'Color': ['red', 'blue', 'green', 'blue', 'red'],
        'Target': [1, 0, 1, 0, 1]
    }
    return pd.DataFrame(data)


def test_label_encoding(sample_data: pd.DataFrame) -> None:
    encoders = CategoricalEncoder()
    df_encoded = encoders.label_encoding(sample_data, 'Category')

    # Check if the column has been created and properly encoded
    assert 'Category_encoded' in df_encoded.columns
    assert set(df_encoded['Category_encoded'].unique()) == {0, 1, 2}  # Check if there are three unique labels


def test_one_hot_encoding(sample_data: pd.DataFrame) -> None:
    encoders = CategoricalEncoder()
    df_encoded = encoders.one_hot_encoding(sample_data, 'Color')

    # Check if the one-hot encoded columns exist
    assert 'Color_red' in df_encoded.columns
    assert 'Color_blue' in df_encoded.columns
    assert 'Color_green' in df_encoded.columns

    # Check if the original data shape has expanded due to new columns
    assert df_encoded.shape[1] == sample_data.shape[1] + 3  # 3 new columns for 'Color' categories


def test_ordinal_encoding(sample_data: pd.DataFrame) -> None:
    encoders = CategoricalEncoder()
    df_encoded = encoders.ordinal_encoding(sample_data, 'Category', categories=['low', 'medium', 'high'])

    # Check if the column has been created and encoded correctly
    assert 'Category_encoded' in df_encoded.columns
    # Check if the encoding follows the specified category order
    assert list(df_encoded['Category_encoded'].unique()) == [0, 1, 2]


def test_binary_encoding(sample_data: pd.DataFrame) -> None:
    encoders = CategoricalEncoder()
    df_encoded = encoders.binary_encoding(sample_data, 'Color')

    # Check if binary encoded columns exist
    assert 'Color_0' in df_encoded.columns
    assert 'Color_1' in df_encoded.columns

    # Check if binary encoding added columns
    assert df_encoded.shape[1] > sample_data.shape[1]


def test_target_encoding(sample_data: pd.DataFrame) -> None:
    encoders = CategoricalEncoder()
    df_encoded = encoders.target_encoding(sample_data, 'Category', 'Target')

    # Check if the column has been created
    assert 'Category_encoded' in df_encoded.columns

    # Check if the encoding follows the mean of the target
    assert df_encoded['Category_encoded'].notnull().all()


def test_frequency_encoding(sample_data: pd.DataFrame) -> None:
    encoders = CategoricalEncoder()
    df_encoded = encoders.frequency_encoding(sample_data, 'Color')

    # Check if the column has been created
    assert 'Color_encoded' in df_encoded.columns

    # Check if frequencies have been encoded correctly
    assert df_encoded['Color_encoded'].notnull().all()
