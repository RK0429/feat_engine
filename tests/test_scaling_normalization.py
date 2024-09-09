import pytest
import pandas as pd
import numpy as np
from feat_engine.scaling_normalization import ScalingNormalizer
# from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import NotFittedError


@pytest.fixture
def sample_data():
    return pd.DataFrame({
        'A': [1, 2, 3, 4, 5],
        'B': [10, 20, 30, 40, 50],
        'C': [100, 200, 300, 400, 500]
    })


def test_standard_scaler(sample_data):
    scaler = ScalingNormalizer(method='standard')
    transformed = scaler.fit_transform(sample_data)

    expected_mean = [0, 0, 0]  # After standard scaling, the mean should be 0
    expected_std = [1, 1, 1]   # The standard deviation should be 1 (population std)

    assert np.allclose(transformed.mean(), expected_mean, atol=1e-6)
    assert np.allclose(transformed.std(ddof=0), expected_std, atol=1e-6)  # Use ddof=0 for population std


def test_minmax_scaler(sample_data):
    scaler = ScalingNormalizer(method='minmax')
    transformed = scaler.fit_transform(sample_data)

    expected_min = [0, 0, 0]  # After Min-Max scaling, min should be 0
    expected_max = [1, 1, 1]  # After Min-Max scaling, max should be 1

    assert np.allclose(transformed.min(), expected_min, atol=1e-6)
    assert np.allclose(transformed.max(), expected_max, atol=1e-6)


def test_robust_scaler(sample_data):
    scaler = ScalingNormalizer(method='robust')
    transformed = scaler.fit_transform(sample_data)

    # The median is 0 and the IQR (interquartile range) should be 1 after robust scaling
    expected_median = [0, 0, 0]

    assert np.allclose(transformed.median(), expected_median, atol=1e-6)


def test_l2_normalizer(sample_data):
    scaler = ScalingNormalizer(method='l2')
    transformed = scaler.fit_transform(sample_data)

    # Each row vector's L2 norm (Euclidean length) should be 1 after normalization
    row_norms = np.linalg.norm(transformed, axis=1)
    expected_norm = [1, 1, 1, 1, 1]

    assert np.allclose(row_norms, expected_norm, atol=1e-6)


def test_not_fitted_error(sample_data):
    scaler = ScalingNormalizer(method='standard')
    with pytest.raises(NotFittedError):
        scaler.transform(sample_data)


def test_column_transformer(sample_data):
    column_methods = {
        'A': 'minmax',
        'B': 'standard',
        'C': 'robust'
    }
    column_transformer = ScalingNormalizer.create_column_transformer(column_methods)
    transformed = column_transformer.fit_transform(sample_data)

    # The transformed array should have the same number of rows and columns
    assert transformed.shape == sample_data.shape


def test_pipeline_with_logistic_regression(sample_data):
    X = sample_data[['A', 'B', 'C']]
    y = [0, 1, 0, 1, 0]  # Some dummy target values for testing

    pipeline = Pipeline([
        ('scaler', ScalingNormalizer(method='minmax').scaler),
        ('model', LogisticRegression())
    ])

    pipeline.fit(X, y)
    predictions = pipeline.predict(X)

    # Ensure that the predictions are binary (0 or 1)
    assert set(predictions) <= {0, 1}
