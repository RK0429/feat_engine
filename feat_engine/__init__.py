# feat_engine/__init__.py

from .handle_missing_values import MissingValueHandler
from .normalize_scaling import ScalingNormalizer
from .encode_category import CategoricalEncoder
from .interact_features import FeatureInteraction
from .transform_features import FeatureTransformer
from .reduce_dimension import DimensionReducer
from .cluster_data import DataClustering
from .handle_outliers import OutlierHandler
from .temporal_features import TemporalFeatures
from .group_features import FeatureGrouping
from .target_based_features import TargetBasedFeatures
from .visualize_data import DataVisualizer
from .solve_classification import ClassificationSolver
from .solve_regression import RegressionSolver

__all__ = [
    'MissingValueHandler',
    'ScalingNormalizer',
    'CategoricalEncoder',
    'FeatureInteraction',
    'FeatureTransformer',
    'DimensionReducer',
    'DataClustering',
    'OutlierHandler',
    'TemporalFeatures',
    'FeatureGrouping',
    'TargetBasedFeatures',
    'DataVisualizer',
    'ClassificationSolver',
    'RegressionSolver'
]
