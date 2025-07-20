import pandas as pd
from sklearn.decomposition import PCA as sklearn_PCA

from aixchem.transform.base import Transformation


class Decomposition(Transformation):
    """
    Base class for decomposition techniques (e.g., PCA, t-SNE).

    The computed embedding replaces the specified columns in dataset.X.
    By default, inplace is False, so the original dataset remains unchanged.
    """
    def __init__(self, inplace=False):
        super().__init__(inplace=inplace)

        self.model = None
        self.name = None

    def _fit(self, dataset, **kwargs):
        """
        Fit the decomposition model on the specified columns of dataset.X.

        :param dataset: The Dataset instance to fit.
        :param kwargs: Additional keyword arguments for the model's fit method.
        :return: self.
        """
        if hasattr(self.model, "fit"):
            self.model.fit(dataset.X[self.columns], **kwargs)

        return self

    def _transform(self, dataset, **kwargs):
        """
        Transform the dataset using the decomposition model.

        This method first checks for a transform method; if not available,
        it attempts to use fit_transform().

        :param dataset: The Dataset instance to transform.
        :param kwargs: Additional keyword arguments for the model's transform/fit_transform method.
        :return: The transformed Dataset instance.
        :raises ValueError: If the model has no transform or fit_transform method.
        """
        if hasattr(self.model, "transform"):
            embedding = self.model.transform(dataset.X[self.columns], **kwargs)
        elif hasattr(self.model, "fit_transform"):
            embedding = self.model.fit_transform(dataset.X[self.columns], **kwargs)
        else:
            raise ValueError(f"Model {self.model} has no transform or fit_transform method.")

        # Construct column names using self.name and index the embedding columns.
        new_cols = [f"{self.name}{i + 1}" for i in range(embedding.shape[1])]

        dataset.X = pd.DataFrame(embedding, index=dataset.X.index, columns=new_cols)

        return dataset


class PCA(Decomposition):
    """
    Principal Component Analysis (PCA) transformer.

    Example usage:
        pca = PCA(n_components=2)
        transformed_dataset = pca.fit_transform(dataset, columns=['col1', 'col2'])
    """
    def __init__(self, n_components=2, inplace=False, **params):
        super().__init__(inplace=inplace)
        self.model = sklearn_PCA(n_components=n_components, **params)
        self.name = "PC"

    @property
    def loadings(self):
        """
        Get the PCA loadings (eigenvectors).

        :return: The PCA loadings as a DataFrame.
        """
        if hasattr(self.model, "components_"):
            return pd.DataFrame(self.model.components_.T, index=self.columns, columns=[f"PC{i + 1}" for i in range(self.model.n_components_)])
        else:
            raise AttributeError("PCA model has no components_ attribute.")
    
    @property
    def summary(self):
        """
        Get the PCA summary including explained variance and loadings.

        :return: A DataFrame containing explained variance and loadings.
        """
        if hasattr(self.model, "explained_variance_ratio_"):
            
            summary = pd.DataFrame(
            [[
                float(self.model.explained_variance_ratio_[idx]),
                float(self.model.explained_variance_ratio_.cumsum()[idx]),
                float(self.model.singular_values_[idx])
                ] 
                for idx in range(self.model.n_components_)
            ], 
            columns=["Variance %", "Cum. Variance %", "Singular Value"], 
            index=[f"PC{i + 1}" for i in range(self.model.n_components_)])
            return summary
                    
        else:
            raise AttributeError("PCA model has no explained_variance_ratio_ or loadings attribute.")
        