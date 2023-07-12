import numpy as np


class CorrelationAnalysis:

    def __init__(self, df, absolute=True, method="pearson", **kwargs):
        """Wrapper class to perform correlation analysis on a pd.DataFrame.

        :param df:          pd.DataFrame containing the features to check.
        :param absolute:    Select whether or not absolute correlation values should be given,
                            defaults to True
        :param method:      Method to use, defaults to "pearson"
        """

        # Correlation matrix
        self.matrix = df.corr(method=method, **kwargs)

        # Upper triangle of the correlation matrix
        self.umatrix = self.matrix.where(np.triu(np.ones(self.matrix.shape), k=1).astype(np.bool))

        if absolute:
            self.matrix, self.umatrix = self.matrix.abs(), self.umatrix.abs()

    def filter(self, threshold=0.9):
        """Filter correlation matrix according to a threshold.

        :param threshold:   Correlation threshold to apply, 
                            defaults to 0.9

        :return:            Names of columns exceeding the threshold
        """
        
        # Ensure absolute values are used
        umatrix = self.umatrix.abs()

        # Find columns with correlation greater than threshold
        above_thr = [column for column in umatrix.columns if any(umatrix[column] >= threshold)]
        
        return above_thr
