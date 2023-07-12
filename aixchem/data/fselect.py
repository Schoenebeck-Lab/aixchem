import numpy as np
import pandas as pd


class TwoSampleFeatureSelection:

    def __init__(self, df, idx, idy, n_best=None, quantiles=(0.01, 0.99)):
        """
        Perform two sample feature selection on a given pd.DataFrame. Features 
        are ranked according to the percentage of separation that is provided 
        between the two references with respect to the total distribution.

        :param df:          The pd.DataFrame containing the features to be ranked.
        :param idx:         The index of the row containing the features for the first reference.
        :param idy:         The index of the row containing the features for the second reference.
        :param n_best:      The number of features that should be in the final selection.
                            If None, The feature dimensionality is set to half the number of samples in df,
                            defaults to None
        :param quantiles:   Select whether or not to use quantiles for the calcualtion of the feature range 
                            of the distribution. If None, min and max values of the distribution are used,
                            defaults to (0.01, 0.99)
        """
        self.ranking = pd.Series(name="Rank", dtype=np.float64)

        for column in df.columns:

            # Calculate feature range across the distribution
            if quantiles is None:
                xrange = df[column].max() - df[column].min()
            else:
                xrange = df[column].quantile(quantiles[1]) - df[column].quantile(quantiles[0])

            # Calculate delta between selection references
            delta = float(df[column].loc[idx] - df[column].loc[idy])

            # Save percentual difference between references with respect to the distribution
            self.ranking.at[column] = abs(delta)/abs(xrange)

        self.ranking = self.ranking.sort_values(ascending=False)

        if n_best is None:
            n_best = int(df.shape[0] / 2)

        self.selection = self.ranking.iloc[:n_best]
