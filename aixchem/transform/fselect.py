import pandas as pd

from aixchem.transform.base import Transformation


class FeatureSelector(Transformation):
    """
    Abstract base class for feature selectors.
    """
    def __init__(self, inplace=True):
        super().__init__(inplace=inplace)

        self.ranking = None


class FeatureSeparation(FeatureSelector):
    """
    A feature selector that ranks features based on the relative univariate separation
    between two specified instances.

    For each feature, the separation is computed as the absolute difference between
    the values at two reference indices (idx and idy), divided by the feature's range.
    The range is calculated either as the full spread (max - min) or using a quantile-based range.

    :param quantiles: A tuple (low, high) used to compute the range via quantiles.
                      If None, the full range (max - min) is used.
    """
    def __init__(self, quantiles=(0.01, 0.99), inplace=True):
        super().__init__(inplace=inplace)

        self.quantiles = quantiles

    def _fit(self, dataset, idx, idy):
        """
        Compute a separation ranking for each feature in the specified columns of dataset.X.

        :param dataset: The Dataset instance containing features in dataset.X.
        :param idx: The index or label for the first reference instance.
        :param idy: The index or label for the second reference instance.
        :return: self.
        :raises ValueError: If either idx or idy is not found in dataset.X.index.
        """
        # Use only the specified columns (set by the base fit() method).
        data = dataset.X[self.columns]

        if idx not in data.index or idy not in data.index:
            raise ValueError("Both idx and idy must exist in dataset.X.index.")

        self.ranking = pd.Series(name="% Separation", dtype=float)

        for c in data.columns:

            column = data[c]

            if self.quantiles is None:
                xrange = column.max() - column.min()
            else:
                xrange = column.quantile(self.quantiles[1]) - column.quantile(self.quantiles[0])

            if xrange == 0:
                self.ranking.at[c] = 0.0
            else:
                delta = float(column.loc[idx] - column.loc[idy])
                self.ranking.at[c] = abs(delta) / abs(xrange)

        self.ranking = self.ranking.sort_values(ascending=False)

        return self
    
    def _transform(self, dataset, threshold=None, n_best=None):
        """
        Transform the dataset by dropping features that do not meet the selection criteria.

        Either a minimum threshold for the separation ratio or a number of top features (n_best)
        must be specified.

        :param dataset: The Dataset instance to transform.
        :param threshold: Keep features with a separation ratio >= threshold.
        :param n_best: Alternatively, keep only the top n_best features.
        :return: The modified Dataset instance.
        :raises ValueError: If neither threshold nor n_best is specified.
        """
        if threshold is not None:
            # Get the selected features
            selected = self.ranking[self.ranking >= threshold].index
        elif n_best is not None:
            selected = self.ranking.head(n_best).index
        else:
            raise ValueError("Either threshold or n_best must be specified.")

        dataset.drop(columns=[c for c in dataset.X.columns if c not in selected])

        return dataset
