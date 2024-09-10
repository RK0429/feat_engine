import pytest
import pandas as pd
from feat_engine.interact_features import FeatureInteraction


@pytest.fixture
def sample_data() -> pd.DataFrame:
    """Fixture to provide a sample dataset for testing."""
    data = {
        'x1': [1, 2, 3, 4],
        'x2': [4, 5, 6, 7],
        'x3': [7, 8, 9, 10],
        'Category1': ['A', 'B', 'A', 'B'],
        'Category2': ['X', 'Y', 'X', 'Y']
    }
    return pd.DataFrame(data)


def test_polynomial_features(sample_data: pd.DataFrame) -> None:
    """Test polynomial feature generation."""
    fi = FeatureInteraction()
    df_poly = fi.polynomial_features(sample_data, ['x1', 'x2'], degree=2)

    # Verify the correct number of polynomial features were created
    assert 'x1' in df_poly.columns
    assert 'x2' in df_poly.columns
    assert 'x1^2' in df_poly.columns
    assert 'x1 x2' in df_poly.columns
    assert 'x2^2' in df_poly.columns

    # Check the shape of the resulting DataFrame (2 original + 5 polynomial = 7 columns)
    assert df_poly.shape[1] == sample_data.shape[1] + 5  # 5 new polynomial columns for degree=2


def test_product_features(sample_data: pd.DataFrame) -> None:
    """Test product feature generation."""
    fi = FeatureInteraction()
    df_product = fi.product_features(sample_data, [('x1', 'x2'), ('x1', 'x3')])

    # Verify the product features were created
    assert 'x1_x_x2' in df_product.columns
    assert 'x1_x_x3' in df_product.columns

    # Check the shape of the resulting DataFrame (2 new product features)
    assert df_product.shape[1] == sample_data.shape[1]


def test_arithmetic_combinations(sample_data: pd.DataFrame) -> None:
    """Test arithmetic combination feature generation."""
    fi = FeatureInteraction()
    df_arithmetic = fi.arithmetic_combinations(sample_data, [('x1', 'x2'), ('x2', 'x3')], operations=['add', 'subtract', 'multiply', 'divide'])

    # Verify the arithmetic combination features were created
    assert 'x1_plus_x2' in df_arithmetic.columns
    assert 'x1_minus_x2' in df_arithmetic.columns
    assert 'x1_times_x2' in df_arithmetic.columns
    assert 'x1_div_x2' in df_arithmetic.columns

    assert 'x2_plus_x3' in df_arithmetic.columns
    assert 'x2_minus_x3' in df_arithmetic.columns
    assert 'x2_times_x3' in df_arithmetic.columns
    assert 'x2_div_x3' in df_arithmetic.columns

    # Check the shape of the resulting DataFrame
    assert df_arithmetic.shape[1] == sample_data.shape[1]  # 8 new arithmetic combination columns


def test_crossed_features(sample_data: pd.DataFrame) -> None:
    """Test categorical feature crossing."""
    fi = FeatureInteraction()
    df_crossed = fi.crossed_features(sample_data, [('Category1', 'Category2')])

    # Verify the crossed features were created
    assert 'Category1_Category2_crossed' in df_crossed.columns

    # Check the values of the crossed feature
    expected_values = ['A_X', 'B_Y', 'A_X', 'B_Y']
    assert df_crossed['Category1_Category2_crossed'].tolist() == expected_values

    # Check the shape of the resulting DataFrame
    assert df_crossed.shape[1] == sample_data.shape[1]
