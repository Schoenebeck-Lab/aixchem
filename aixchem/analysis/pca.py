import pandas as pd
import numpy as np

from sklearn.decomposition import PCA


class PrincipalComponentAnalysis:

    np.random.seed(42)
    rng = np.random.RandomState(42)

    def __init__(self, df, n_components=4, **kwargs):
        """Wrapper class for sklearn.PCA.

        :param df:              pd.DataFrame containing the features.
        :param n_components:    Numnber of principal components to be considered, defaults to 4
        """
        # Generate PC names
        names = [f"PC{i + 1}" for i in range(n_components)]

        # Build Model
        self.model = PCA(n_components=n_components, random_state=self.rng, **kwargs)

        # Format PCs, PC loadings and loadings matrix
        self.components = pd.DataFrame(self.model.fit_transform(df), columns=names, index=df.index)
        self.loadings = pd.DataFrame(self.model.components_.T, columns=names, index=df.columns)
        self.loadings_matrix = self.loadings * np.sqrt(self.model.explained_variance_)

        # Format PCA summary
        self.summary = pd.DataFrame()

        for i, name in enumerate(names):
            self.summary.at[name, "Variance %"] = self.model.explained_variance_ratio_[i]
            self.summary.at[name, "Cum. Variance %"] = self.model.explained_variance_ratio_.cumsum()[i]
            self.summary.at[name, "Singular Value"] = self.model.singular_values_[i]
    