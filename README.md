
# feat_engine

`feat_engine` is a comprehensive feature engineering library designed to simplify and streamline feature processing tasks for machine learning models. It offers a wide range of tools for encoding categorical features, handling missing values, scaling and normalizing data, dimensionality reduction, visualizing data, and more.

## Features

The `feat_engine` package provides the following modules to help with various stages of feature engineering:

- **`encode_category.py`**: Encode categorical features with methods like one-hot encoding and label encoding.
- **`group_features.py`**: Group and aggregate features based on categorical or time-based columns.
- **`handle_missing_values.py`**: Handle missing data by filling or dropping missing values and visualizing missing data patterns.
- **`handle_outliers.py`**: Detect and handle outliers using methods like Z-Score, IQR, Isolation Forest, DBSCAN, Winsorization, and more.
- **`interact_features.py`**: Create interaction features, polynomial combinations, and more from existing features.
- **`normalize_scaling.py`**: Apply normalization and scaling techniques including Min-Max scaling, Z-score standardization, and robust scaling.
- **`reduce_dimension.py`**: Reduce the dimensionality of feature sets using methods such as PCA, LDA, t-SNE, UMAP, and autoencoders.
- **`target_based_features.py`**: Create target-based encodings like target mean encoding, smoothed target mean encoding, and cross-validated target encoding.
- **`temporal_features.py`**: Extract and transform time-based features, including creating rolling windows, lag features, and cyclical transformations.
- **`transform_features.py`**: Apply mathematical transformations such as logarithmic, square root, power transformations, and more.
- **`visualize_data.py`**: Visualize datasets using correlation heatmaps, distribution plots, missing value heatmaps, outlier detection, and more.

## Installation

You can install the package by cloning the repository and installing dependencies:

```bash
git clone https://github.com/your-username/feat_engine.git
cd feat_engine
pip install -r requirements.txt
```

## Usage

Here's a brief overview of how to use some of the key features from `feat_engine`:

### Encoding Categorical Features

```python
from feat_engine.encode_category import CategoryEncoder
import pandas as pd

df = pd.DataFrame({
    'category': ['A', 'B', 'C', 'A', 'B']
})

encoder = CategoryEncoder()
encoded_df = encoder.one_hot_encode(df, 'category')
print(encoded_df)
```

### Handling Missing Values

```python
from feat_engine.handle_missing_values import MissingValueHandler

df = pd.DataFrame({
    'feature1': [1, 2, None, 4],
    'feature2': [None, 2, 3, 4]
})

mv_handler = MissingValueHandler()
filled_df = mv_handler.fill_missing_values(df, method='mean')
print(filled_df)
```

### Scaling Features

```python
from feat_engine.normalize_scaling import FeatureScaler

scaler = FeatureScaler()
scaled_df = scaler.min_max_scale(df, columns=['feature1', 'feature2'])
print(scaled_df)
```

### Reducing Dimensions

```python
from feat_engine.reduce_dimension import DimensionReducer

df = pd.DataFrame({
    'feature1': [1, 2, 3, 4],
    'feature2': [2, 3, 4, 5],
    'feature3': [3, 4, 5, 6]
})

reducer = DimensionReducer()
reduced_df = reducer.pca(df, n_components=2)
print(reduced_df)
```

### Visualizing Data

```python
from feat_engine.visualize_data import DataVisualizer

visualizer = DataVisualizer()
visualizer.plot_correlation_heatmap(df)
```

## Testing

The package includes test cases to ensure functionality. Run tests with:

```bash
pytest tests/
```

Make sure to set the backend for `matplotlib` to `Agg` during testing to avoid issues with Tkinter in non-GUI environments.

```python
import matplotlib
matplotlib.use('Agg')
```

## Documentation

For more information on the package, see [Read The Docs](https://feat-engine.readthedocs.io/en/latest/).

## Contributing

Contributions are welcome! Feel free to open issues or submit pull requests.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
