import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA, TruncatedSVD, FactorAnalysis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.manifold import TSNE, Isomap
from sklearn.exceptions import NotFittedError
import umap.umap_ as umap
from typing import Optional, Any

# Conditional import of keras depending on version
try:
    from keras.layers import Input, Dense  # type: ignore
    from keras.models import Model  # type: ignore
except ImportError:
    from tensorflow.keras.layers import Input, Dense  # type: ignore
    from tensorflow.keras.models import Model  # type: ignore


class PCAReducer(BaseEstimator, TransformerMixin):
    """
    Dimensionality reduction using Principal Component Analysis (PCA).
    """

    def __init__(self, n_components: int = 2, **kwargs: Any):
        """
        Initializes the PCAReducer.

        Args:
            n_components (int): Number of principal components to keep.
            **kwargs: Additional keyword arguments for sklearn.decomposition.PCA.
        """
        self.n_components = n_components
        self.kwargs = kwargs
        self.pca = PCA(n_components=self.n_components, **self.kwargs)
        self.columns_ = None

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'PCAReducer':
        """
        Fits the PCA model to the data.

        Args:
            X (pd.DataFrame): Input DataFrame.
            y (pd.Series, optional): Target variable (not used).

        Returns:
            self
        """
        self.columns_ = X.columns
        self.pca.fit(X)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms the input DataFrame using the fitted PCA model.

        Args:
            X (pd.DataFrame): Input DataFrame.

        Returns:
            pd.DataFrame: Transformed DataFrame.
        """
        if self.pca is None:
            raise NotFittedError("This PCAReducer instance is not fitted yet.")
        X_pca = self.pca.transform(X)
        component_names = [f'PC{i+1}' for i in range(self.n_components)]
        return pd.DataFrame(X_pca, columns=component_names, index=X.index)

    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        Fits the PCA model and transforms the input DataFrame.

        Args:
            X (pd.DataFrame): Input DataFrame.
            y (pd.Series, optional): Target variable (not used).

        Returns:
            pd.DataFrame: Transformed DataFrame.
        """
        return self.fit(X).transform(X)


class LDAReducer(BaseEstimator, TransformerMixin):
    """
    Dimensionality reduction using Linear Discriminant Analysis (LDA).
    """

    def __init__(self, n_components: int = 2, **kwargs: Any):
        """
        Initializes the LDAReducer.

        Args:
            n_components (int): Number of linear discriminants to retain.
            **kwargs: Additional keyword arguments for sklearn.discriminant_analysis.LinearDiscriminantAnalysis.
        """
        self.n_components = n_components
        self.kwargs = kwargs
        self.lda = LDA(n_components=self.n_components, **self.kwargs)
        self.columns_ = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'LDAReducer':
        """
        Fits the LDA model to the data.

        Args:
            X (pd.DataFrame): Input DataFrame.
            y (pd.Series): Target variable.

        Returns:
            self
        """
        self.columns_ = X.columns
        self.lda.fit(X, y)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms the input DataFrame using the fitted LDA model.

        Args:
            X (pd.DataFrame): Input DataFrame.

        Returns:
            pd.DataFrame: Transformed DataFrame.
        """
        if self.lda is None:
            raise NotFittedError("This LDAReducer instance is not fitted yet.")
        X_lda = self.lda.transform(X)
        component_names = [f'LD{i+1}' for i in range(self.n_components)]
        return pd.DataFrame(X_lda, columns=component_names, index=X.index)

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """
        Fits the LDA model and transforms the input DataFrame.

        Args:
            X (pd.DataFrame): Input DataFrame.
            y (pd.Series): Target variable.

        Returns:
            pd.DataFrame: Transformed DataFrame.
        """
        return self.fit(X, y).transform(X)


class SVDReducer(BaseEstimator, TransformerMixin):
    """
    Dimensionality reduction using Truncated Singular Value Decomposition (SVD).
    """

    def __init__(self, n_components: int = 2, **kwargs: Any):
        """
        Initializes the SVDReducer.

        Args:
            n_components (int): Number of singular values to keep.
            **kwargs: Additional keyword arguments for sklearn.decomposition.TruncatedSVD.
        """
        self.n_components = n_components
        self.kwargs = kwargs
        self.svd = TruncatedSVD(n_components=self.n_components, **self.kwargs)
        self.columns_ = None

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'SVDReducer':
        """
        Fits the SVD model to the data.

        Args:
            X (pd.DataFrame): Input DataFrame.
            y (pd.Series, optional): Target variable (not used).

        Returns:
            self
        """
        self.columns_ = X.columns
        self.svd.fit(X)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms the input DataFrame using the fitted SVD model.

        Args:
            X (pd.DataFrame): Input DataFrame.

        Returns:
            pd.DataFrame: Transformed DataFrame.
        """
        if self.svd is None:
            raise NotFittedError("This SVDReducer instance is not fitted yet.")
        X_svd = self.svd.transform(X)
        component_names = [f'SVD{i+1}' for i in range(self.n_components)]
        return pd.DataFrame(X_svd, columns=component_names, index=X.index)

    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        Fits the SVD model and transforms the input DataFrame.

        Args:
            X (pd.DataFrame): Input DataFrame.
            y (pd.Series, optional): Target variable (not used).

        Returns:
            pd.DataFrame: Transformed DataFrame.
        """
        return self.fit(X).transform(X)


class FactorAnalysisReducer(BaseEstimator, TransformerMixin):
    """
    Dimensionality reduction using Factor Analysis.
    """

    def __init__(self, n_components: int = 2, **kwargs: Any):
        """
        Initializes the FactorAnalysisReducer.

        Args:
            n_components (int): Number of factors to retain.
            **kwargs: Additional keyword arguments for sklearn.decomposition.FactorAnalysis.
        """
        self.n_components = n_components
        self.kwargs = kwargs
        self.fa = FactorAnalysis(n_components=self.n_components, **self.kwargs)
        self.columns_ = None

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'FactorAnalysisReducer':
        """
        Fits the Factor Analysis model to the data.

        Args:
            X (pd.DataFrame): Input DataFrame.
            y (pd.Series, optional): Target variable (not used).

        Returns:
            self
        """
        self.columns_ = X.columns
        self.fa.fit(X)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms the input DataFrame using the fitted Factor Analysis model.

        Args:
            X (pd.DataFrame): Input DataFrame.

        Returns:
            pd.DataFrame: Transformed DataFrame.
        """
        if self.fa is None:
            raise NotFittedError("This FactorAnalysisReducer instance is not fitted yet.")
        X_fa = self.fa.transform(X)
        component_names = [f'FA{i+1}' for i in range(self.n_components)]
        return pd.DataFrame(X_fa, columns=component_names, index=X.index)

    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        Fits the Factor Analysis model and transforms the input DataFrame.

        Args:
            X (pd.DataFrame): Input DataFrame.
            y (pd.Series, optional): Target variable (not used).

        Returns:
            pd.DataFrame: Transformed DataFrame.
        """
        return self.fit(X).transform(X)


class TSNEReducer(BaseEstimator, TransformerMixin):
    """
    Dimensionality reduction using t-Distributed Stochastic Neighbor Embedding (t-SNE).
    """

    def __init__(self, n_components: int = 2, perplexity: float = 30.0, **kwargs: Any):
        """
        Initializes the TSNEReducer.

        Args:
            n_components (int): Number of dimensions to reduce to.
            perplexity (float): Perplexity parameter for t-SNE.
            **kwargs: Additional keyword arguments for sklearn.manifold.TSNE.
        """
        self.n_components = n_components
        self.perplexity = perplexity
        self.kwargs = kwargs
        self.tsne = TSNE(n_components=self.n_components, perplexity=self.perplexity, **self.kwargs)
        self.columns_ = None
        self.fitted = False

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'TSNEReducer':
        """
        Fits the t-SNE model to the data.

        Args:
            X (pd.DataFrame): Input DataFrame.
            y (pd.Series, optional): Target variable (not used).

        Returns:
            self
        """
        self.columns_ = X.columns
        self.tsne.fit(X)
        self.fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms the input DataFrame using the fitted t-SNE model.

        Args:
            X (pd.DataFrame): Input DataFrame.

        Returns:
            pd.DataFrame: Transformed DataFrame.
        """
        if not self.fitted:
            raise NotFittedError("This TSNEReducer instance is not fitted yet.")
        X_tsne = self.tsne.fit_transform(X)
        component_names = [f'tSNE{i+1}' for i in range(self.n_components)]
        return pd.DataFrame(X_tsne, columns=component_names, index=X.index)

    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        Fits the t-SNE model and transforms the input DataFrame.

        Args:
            X (pd.DataFrame): Input DataFrame.
            y (pd.Series, optional): Target variable (not used).

        Returns:
            pd.DataFrame: Transformed DataFrame.
        """
        self.fit(X)
        return self.transform(X)


class UMAPReducer(BaseEstimator, TransformerMixin):
    """
    Dimensionality reduction using Uniform Manifold Approximation and Projection (UMAP).
    """

    def __init__(self, n_components: int = 2, **kwargs: Any):
        """
        Initializes the UMAPReducer.

        Args:
            n_components (int): Number of dimensions to reduce to.
            **kwargs: Additional keyword arguments for umap.UMAP.
        """
        self.n_components = n_components
        self.kwargs = kwargs
        self.umap = umap.UMAP(n_components=self.n_components, **self.kwargs)
        self.columns_ = None
        self.fitted = False

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'UMAPReducer':
        """
        Fits the UMAP model to the data.

        Args:
            X (pd.DataFrame): Input DataFrame.
            y (pd.Series, optional): Target variable.

        Returns:
            self
        """
        self.columns_ = X.columns
        self.umap.fit(X, y=y)
        self.fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms the input DataFrame using the fitted UMAP model.

        Args:
            X (pd.DataFrame): Input DataFrame.

        Returns:
            pd.DataFrame: Transformed DataFrame.
        """
        if not self.fitted:
            raise NotFittedError("This UMAPReducer instance is not fitted yet.")
        X_umap = self.umap.transform(X)
        component_names = [f'UMAP{i+1}' for i in range(self.n_components)]
        return pd.DataFrame(X_umap, columns=component_names, index=X.index)

    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        Fits the UMAP model and transforms the input DataFrame.

        Args:
            X (pd.DataFrame): Input DataFrame.
            y (pd.Series, optional): Target variable.

        Returns:
            pd.DataFrame: Transformed DataFrame.
        """
        return self.fit(X, y=y).transform(X)


class IsomapReducer(BaseEstimator, TransformerMixin):
    """
    Dimensionality reduction using Isomap.
    """

    def __init__(self, n_components: int = 2, n_neighbors: int = 5, **kwargs: Any):
        """
        Initializes the IsomapReducer.

        Args:
            n_components (int): Number of dimensions to reduce to.
            n_neighbors (int): Number of neighbors to use when computing geodesic distances.
            **kwargs: Additional keyword arguments for sklearn.manifold.Isomap.
        """
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.kwargs = kwargs
        self.isomap = Isomap(n_components=self.n_components, n_neighbors=self.n_neighbors, **self.kwargs)
        self.columns_ = None

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'IsomapReducer':
        """
        Fits the Isomap model to the data.

        Args:
            X (pd.DataFrame): Input DataFrame.
            y (pd.Series, optional): Target variable (not used).

        Returns:
            self
        """
        self.columns_ = X.columns
        self.isomap.fit(X)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms the input DataFrame using the fitted Isomap model.

        Args:
            X (pd.DataFrame): Input DataFrame.

        Returns:
            pd.DataFrame: Transformed DataFrame.
        """
        X_isomap = self.isomap.transform(X)
        component_names = [f'Isomap{i+1}' for i in range(self.n_components)]
        return pd.DataFrame(X_isomap, columns=component_names, index=X.index)

    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        Fits the Isomap model and transforms the input DataFrame.

        Args:
            X (pd.DataFrame): Input DataFrame.
            y (pd.Series, optional): Target variable (not used).

        Returns:
            pd.DataFrame: Transformed DataFrame.
        """
        return self.fit(X).transform(X)


class AutoencoderReducer(BaseEstimator, TransformerMixin):
    """
    Dimensionality reduction using Autoencoders.
    """

    def __init__(self, encoding_dim: int = 10, epochs: int = 50, batch_size: int = 32, optimizer: str = 'adam', loss: str = 'mse', **kwargs: Any):
        """
        Initializes the AutoencoderReducer.

        Args:
            encoding_dim (int): Size of the encoding layer.
            epochs (int): Number of training epochs.
            batch_size (int): Batch size for training.
            optimizer (str): Optimizer to use for training.
            loss (str): Loss function to use for training.
            **kwargs: Additional keyword arguments for keras models.
        """
        self.encoding_dim = encoding_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.loss = loss
        self.kwargs = kwargs
        self.autoencoder = None
        self.encoder = None
        self.columns_ = None

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'AutoencoderReducer':
        """
        Fits the Autoencoder model to the data.

        Args:
            X (pd.DataFrame): Input DataFrame.
            y (pd.Series, optional): Target variable (not used).

        Returns:
            self
        """
        self.columns_ = X.columns
        input_dim = X.shape[1]
        input_layer = Input(shape=(input_dim,))
        encoded = Dense(self.encoding_dim, activation='relu')(input_layer)
        decoded = Dense(input_dim, activation='sigmoid')(encoded)

        self.autoencoder = Model(input_layer, decoded)
        self.autoencoder.compile(optimizer=self.optimizer, loss=self.loss)  # type: ignore

        # Train the autoencoder
        self.autoencoder.fit(  # type: ignore
            X.values, X.values,
            epochs=self.epochs,
            batch_size=self.batch_size,
            shuffle=True,
            verbose=0
        )

        # Create encoder model
        self.encoder = Model(inputs=input_layer, outputs=encoded)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms the input DataFrame using the trained Autoencoder.

        Args:
            X (pd.DataFrame): Input DataFrame.

        Returns:
            pd.DataFrame: Transformed DataFrame.
        """
        if self.encoder is None:
            raise NotFittedError("This AutoencoderReducer instance is not fitted yet.")
        X_encoded = self.encoder.predict(X.values)
        component_names = [f'AE{i+1}' for i in range(self.encoding_dim)]
        return pd.DataFrame(X_encoded, columns=component_names, index=X.index)

    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        Fits the Autoencoder model and transforms the input DataFrame.

        Args:
            X (pd.DataFrame): Input DataFrame.
            y (pd.Series, optional): Target variable (not used).

        Returns:
            pd.DataFrame: Transformed DataFrame.
        """
        self.fit(X)
        return self.transform(X)
