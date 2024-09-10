import pandas as pd
from sklearn.decomposition import PCA, TruncatedSVD, FactorAnalysis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.manifold import TSNE, Isomap
import umap.umap_ as umap  # Correct import
from keras.layers import Input, Dense
from keras.models import Model


class DimensionReducer:
    """
    The DimensionReducer class provides methods for reducing the dimensionality of datasets
    using various linear and non-linear techniques.
    It includes traditional algorithms like PCA and LDA, as well as advanced methods like
    t-SNE, UMAP, Isomap, and Autoencoders.
    """

    def __init__(self) -> None:
        """
        Initialize the DimensionReducer class.
        """
        pass

    def pca(self, df: pd.DataFrame, n_components: int) -> pd.DataFrame:
        """
        Perform Principal Component Analysis (PCA) to reduce the dimensionality of the dataset.

        PCA identifies the directions (principal components) that capture the maximum variance
        in the data, projecting it into a lower-dimensional space.

        Args:
            df (pd.DataFrame): The input dataframe containing the features.
            n_components (int): The number of principal components to keep.

        Returns:
            pd.DataFrame: The transformed dataframe with reduced dimensionality.
        """
        pca = PCA(n_components=n_components)
        pca_result = pca.fit_transform(df)
        return pd.DataFrame(pca_result)

    def lda(self, df: pd.DataFrame, labels: pd.Series, n_components: int) -> pd.DataFrame:
        """
        Perform Linear Discriminant Analysis (LDA) for supervised dimensionality reduction.

        LDA finds the linear combinations of features that best separate classes
        in the data.

        Args:
            df (pd.DataFrame): The input dataframe containing the features.
            labels (pd.Series): The class labels corresponding to the data points.
            n_components (int): The number of linear discriminants to retain.

        Returns:
            pd.DataFrame: The transformed dataframe with reduced dimensionality.
        """
        lda = LDA(n_components=n_components)
        lda_result = lda.fit_transform(df, labels)
        return pd.DataFrame(lda_result)

    def svd(self, df: pd.DataFrame, n_components: int) -> pd.DataFrame:
        """
        Perform Singular Value Decomposition (SVD) to reduce the dimensionality of the dataset.

        SVD is effective for working with sparse matrices and uncovering latent semantic
        structures in the data.

        Args:
            df (pd.DataFrame): The input dataframe containing the features.
            n_components (int): The number of singular values to retain.

        Returns:
            pd.DataFrame: The transformed dataframe with reduced dimensionality.
        """
        svd = TruncatedSVD(n_components=n_components)
        svd_result = svd.fit_transform(df)
        return pd.DataFrame(svd_result)

    def factor_analysis(self, df: pd.DataFrame, n_components: int) -> pd.DataFrame:
        """
        Perform Factor Analysis for dimensionality reduction.

        Factor Analysis models observed variables as linear combinations of potential factors,
        reducing the dimensionality of the dataset.

        Args:
            df (pd.DataFrame): The input dataframe containing the features.
            n_components (int): The number of factors to retain.

        Returns:
            pd.DataFrame: The transformed dataframe with reduced dimensionality.
        """
        fa = FactorAnalysis(n_components=n_components)
        fa_result = fa.fit_transform(df)
        return pd.DataFrame(fa_result)

    def tsne(self, df: pd.DataFrame, n_components: int = 2, perplexity: int = 30) -> pd.DataFrame:
        """
        Perform t-Distributed Stochastic Neighbor Embedding (t-SNE) for non-linear dimensionality reduction.

        t-SNE is primarily used for visualizing high-dimensional datasets, preserving local
        structures in a lower-dimensional space.

        Args:
            df (pd.DataFrame): The input dataframe containing the features.
            n_components (int): The number of dimensions to reduce to (default: 2).
            perplexity (int): Controls the balance between local and global aspects of the data (default: 30).

        Returns:
            pd.DataFrame: The transformed dataframe with reduced dimensionality.
        """
        tsne = TSNE(n_components=n_components, perplexity=perplexity)
        tsne_result = tsne.fit_transform(df)
        return pd.DataFrame(tsne_result)

    def umap(self, df: pd.DataFrame, n_components: int = 2) -> pd.DataFrame:
        """
        Perform Uniform Manifold Approximation and Projection (UMAP) for non-linear dimensionality reduction.

        UMAP is useful for visualizing high-dimensional data with faster computation
        and better preservation of global structures than t-SNE.

        Args:
            df (pd.DataFrame): The input dataframe containing the features.
            n_components (int): The number of dimensions to reduce to (default: 2).

        Returns:
            pd.DataFrame: The transformed dataframe with reduced dimensionality.
        """
        reducer = umap.UMAP(n_components=n_components)
        umap_result = reducer.fit_transform(df)
        return pd.DataFrame(umap_result)

    def isomap(self, df: pd.DataFrame, n_components: int = 2, n_neighbors: int = 2) -> pd.DataFrame:
        """
        Perform Isomap for non-linear dimensionality reduction based on geodesic distances.

        Isomap preserves the global structure of the data and is useful for manifold learning.

        Args:
            df (pd.DataFrame): The input dataframe containing the features.
            n_components (int): The number of dimensions to reduce to (default: 2).
            n_neighbors (int): The number of neighbors to use when computing geodesic distances (default: 2).

        Returns:
            pd.DataFrame: The transformed dataframe with reduced dimensionality.
        """
        isomap = Isomap(n_components=n_components, n_neighbors=n_neighbors)
        isomap_result = isomap.fit_transform(df)
        return pd.DataFrame(isomap_result, index=df.index)

    def autoencoder(self, df: pd.DataFrame, encoding_dim: int = 10, epochs: int = 50, batch_size: int = 10) -> pd.DataFrame:
        """
        Perform dimensionality reduction using Autoencoder.

        Autoencoders are neural networks trained to reconstruct the input data,
        with the hidden layer representing a lower-dimensional encoding.

        Args:
            df (pd.DataFrame): The input dataframe containing the features.
            encoding_dim (int): The size of the encoding layer (default: 10).
            epochs (int): The number of training iterations (default: 50).
            batch_size (int): The size of batches for training (default: 10).

        Returns:
            pd.DataFrame: The reduced representation of the data learned by the autoencoder.
        """
        input_dim = df.shape[1]
        input_layer = Input(shape=(input_dim,))
        encoded = Dense(encoding_dim, activation='relu')(input_layer)
        decoded = Dense(input_dim, activation='sigmoid')(encoded)

        autoencoder = Model(input_layer, decoded)
        autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

        # Train the autoencoder
        autoencoder.fit(df, df, epochs=epochs, batch_size=batch_size, verbose=0)

        # Get the encoded representation
        encoder = Model(input_layer, encoded)
        encoded_data = encoder.predict(df)
        return pd.DataFrame(encoded_data)
