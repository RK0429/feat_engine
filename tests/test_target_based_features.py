import pytest
import pandas as pd
from feat_engine.target_based_features import TargetBasedFeatures


@pytest.fixture
def sample_data() -> pd.DataFrame:
    """
    Fixture to provide a sample dataframe for testing.

    Returns:
    - pd.DataFrame: Sample dataframe with categorical and target columns.
    """
    data = {
        'category': ['A', 'B', 'A', 'B', 'C', 'A', 'C', 'B'],
        'target': [1, 0, 1, 0, 1, 1, 0, 1]
    }
    return pd.DataFrame(data)


def test_target_mean_encoding(sample_data: pd.DataFrame) -> None:
    """
    Test target mean encoding.

    Asserts:
    - The mean encoding result is correct for each category.
    """
    tbf = TargetBasedFeatures()
    mean_encoded = tbf.target_mean_encoding(sample_data, 'target', 'category')

    expected_means = {'A': 1.0, 'B': 0.333333, 'C': 0.5}

    # Assert that the mean encoding is correct for each category
    for category, mean_val in expected_means.items():
        assert mean_encoded[sample_data['category'] == category].mean() == pytest.approx(mean_val, 0.01)


def test_smoothed_target_mean_encoding(sample_data: pd.DataFrame) -> None:
    """
    Test smoothed target mean encoding.

    Asserts:
    - The smoothed mean encoding result is correct.
    """
    tbf = TargetBasedFeatures()
    smoothed_encoded = tbf.smoothed_target_mean_encoding(sample_data, 'target', 'category', m=2)

    # Check if the smoothed mean encoding produces valid results
    assert smoothed_encoded[sample_data['category'] == 'A'].mean() > 0.8
    assert smoothed_encoded[sample_data['category'] == 'B'].mean() > 0.3
    assert smoothed_encoded[sample_data['category'] == 'C'].mean() > 0.4


def test_count_encoding(sample_data: pd.DataFrame) -> None:
    """
    Test count encoding.

    Asserts:
    - The count encoding result is correct.
    """
    tbf = TargetBasedFeatures()
    count_encoded = tbf.count_encoding(sample_data, 'category')

    expected_counts = {'A': 3, 'B': 3, 'C': 2}

    # Assert that the count encoding is correct for each category
    for category, count_val in expected_counts.items():
        assert count_encoded[sample_data['category'] == category].mean() == count_val


def test_cross_validated_target_encoding(sample_data: pd.DataFrame) -> None:
    """
    Test cross-validated target encoding.

    Asserts:
    - The cross-validated target mean encoding is calculated without data leakage.
    """
    tbf = TargetBasedFeatures()
    cv_encoded = tbf.cross_validated_target_encoding(sample_data, 'target', 'category', n_splits=3)

    # Cross-validated encoding should be different from direct mean encoding to avoid data leakage
    mean_encoded = tbf.target_mean_encoding(sample_data, 'target', 'category')

    assert not cv_encoded.equals(mean_encoded)  # Ensure cross-validation worked


def test_calculate_woe(sample_data: pd.DataFrame) -> None:
    """
    Test calculation of Weight of Evidence (WoE).

    Asserts:
    - The WoE values are calculated correctly for each category.
    """
    tbf = TargetBasedFeatures()
    woe_encoded = tbf.calculate_woe(sample_data, 'target', 'category')

    # The WoE should give finite values for all categories
    assert not woe_encoded.isna().any()
    assert woe_encoded[sample_data['category'] == 'A'].iloc[0] > 0
    assert woe_encoded[sample_data['category'] == 'B'].iloc[0] < 0
