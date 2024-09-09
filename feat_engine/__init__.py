# feat_engine/__init__.py

from .handling_missing_values import MissingValueHandler
from .scaling_normalization import ScalingNormalizer
# from .encoding_categorical import CategoricalEncoder
# from .feature_interaction import FeatureInteraction
# from .feature_transformation import FeatureTransformation
# from .dimensionality_reduction import DimensionalityReducer
# from .handling_outliers import OutlierHandler
# from .temporal_features import TemporalFeatureEngineer
# from .feature_grouping import FeatureGrouping
# from .target_based_features import TargetBasedFeatures

__all__ = [
    'MissingValueHandler',
    'ScalingNormalizer',
    # 'CategoricalEncoder',
    # 'FeatureInteraction',
    # 'FeatureTransformation',
    # 'DimensionalityReducer',
    # 'OutlierHandler',
    # 'TemporalFeatureEngineer',
    # 'FeatureGrouping',
    # 'TargetBasedFeatures'
]
