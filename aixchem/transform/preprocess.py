import numpy as np
from sklearn.preprocessing import StandardScaler

from aixchem.transform.base import Transformation


class Scaler(Transformation):
    """
    A generic scaler transformer that wraps an external scaler (e.g., from sklearn).

    Example usage:
        scaler = Scaler(StandardScaler)
        scaler.fit_transform(dataset, columns=['col1', 'col2'])
    
    This transformer fits on the specified columns of dataset.X and then replaces those
    columns with the scaled features. If inplace is False, a copy of the Dataset is used.
    
    :param scaler: The external scaler class to use (e.g., StandardScaler).
    :param inplace: If False, a copy of the Dataset is created before applying the transformation.
    :param params: Additional parameters to pass to the scaler constructor.
    """
    def __init__(self, scaler=StandardScaler, inplace=True, **params):
        super().__init__(inplace=inplace)

        self.params.update(params)
        self.params["scaler"] = scaler.__name__

        # Instantiate the scaler with the specified parameters.
        self.scaler = scaler(**params)

    def _fit(self, dataset, **kwargs):
        """
        Fit the scaler on the columns specified in self.columns of dataset.X.

        :param dataset: The Dataset instance containing features in dataset.X.
        :param kwargs: Additional arguments for the scaler's fit() method.
        :return: self.
        """
        self.scaler.fit(dataset.X[self.columns], **kwargs)

        return self
    
    def _transform(self, dataset, **kwargs):
        """
        Scale the specified columns in dataset.X.

        :param dataset: The Dataset instance to transform.
        :param kwargs: Additional arguments for the scaler's transform() method.
        :return: The transformed Dataset instance.
        :raises ValueError: If the transformer has not been fitted (i.e. self.columns is not set).
        """
        if self.columns is None:
            raise ValueError("The transformer has not been fitted yet. Call fit() or fit_transform() first.")
        
        dataset.X[self.columns] = self.scaler.transform(dataset.X[self.columns], **kwargs)

        return dataset


class CorrelationAnalyzer(Transformation):
    """
    A transformer that calculates the correlation matrix for the dataset's features 
    and drops highly correlated features based on a given threshold.

    This transformer computes the correlation matrix in _fit() and uses it in _transform()
    to drop columns with any correlation value in the upper triangle above the specified threshold.
    The correlation matrix before and after dropping is stored in self.matrix and self.matrix_after, respectively.

    :param method: The correlation method to use (default: "pearson").
    :param threshold: The correlation threshold above which features are considered redundant (default: 0.8).
    :param sort: Whether to sort the columns before computing the correlation matrix (default: True).
    :param inplace: If False, a copy of the Dataset is created before applying the transformation.
    """
    def __init__(self, method="pearson", threshold=0.8, sort=True, inplace=True):
        super().__init__(inplace=inplace)

        self.params["method"] = method
        self.params["threshold"] = threshold
        self.params["sort"] = sort

        self.matrix = None         # Correlation matrix before dropping features.
        self.matrix_after = None   # Correlation matrix after dropping features.

    def _fit(self, dataset, **kwargs):
        """
        Compute and store the correlation matrix for the specified columns of dataset.X.

        :param dataset: The Dataset instance containing features in dataset.X.
        :param kwargs: Additional arguments for the correlation computation.
        :return: self.
        :raises ValueError: If any specified column is missing.
        """
        if self.params.get("sort"):
            # Select columns for analysis and sort index for reproducibility.
            data = dataset.X[self.columns].sort_index(axis=1)
        else:
            data = dataset.X[self.columns]

        self.matrix = data.corr(method=self.params["method"], **kwargs)

        return self
    
    def _transform(self, dataset):
        """
        Drop columns from dataset.X that have any correlation value in the upper triangle 
        exceeding the specified threshold. The operation is performed in-place.

        :param dataset: The Dataset instance to transform.
        :return: The modified Dataset instance.
        """
        thr = self.params.get("threshold")
        # Compute the upper triangle of the absolute correlation matrix.
        umatrix = self.umatrix.abs()

        # Identify columns to drop: any column with at least one value >= threshold in the upper triangle
        dataset.drop(columns=[c for c in umatrix.columns if any(umatrix[c] >= thr)])

        self.matrix_after = self.matrix.loc[dataset.X.columns, dataset.X.columns]

        return dataset
    
    @property
    def umatrix(self):
        """
        Compute and return the upper triangle of the correlation matrix.
        
        :return: A DataFrame representing the upper triangle of the correlation matrix.
        :raises ValueError: If the correlation matrix is not computed.
        """
        if self.matrix is None:
            raise ValueError("Correlation matrix is not computed. Call fit() first.")
        
        return self.matrix.where(np.triu(np.ones(self.matrix.shape), k=1).astype(bool))
