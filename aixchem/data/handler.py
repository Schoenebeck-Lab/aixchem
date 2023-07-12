import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler, StandardScaler


class TabularDataHandler:

    # Set random state for reproducibility
    np.random.seed(42)
    rng = np.random.RandomState(42)

    # Define scalers
    scalers = {
        "minmax": MinMaxScaler(feature_range=(0, 1)),
        "std": StandardScaler()
        }

    def __init__(self, data, index=None, y=None, sep=";", **kwargs):
        """Handler class for tabular data.

        :param data:    pd.DataFrame or path to .csv-file
        :param index:   Name of the column to be used as index, defaults to None
        :param y:       Label column, defaults to None
        :param sep:     Separator in the .csv-file, defaults to ";"
        """

        # Load raw data either from df or from csv file
        if isinstance(data, pd.DataFrame):
            self.raw = data
        else:
            try:
                self.raw = pd.read_csv(data, index_col=index, sep=sep, **kwargs)
            except Exception as e:
                pass
        
        # Copy raw data to work data
        self.X = self.raw.copy()

        # Split label column if provided
        self.y = self.X.pop(y) if y is not None else None

    def drop(self, rows=None, cols=None):
        """Drop rows or columns.

        :param rows: Rows to drop, defaults to None
        :param cols: Columns to drop, defaults to None
        """
        if rows is not None:
            self.X = self.X.drop(index=rows)
            if self.y is not None:
                self.y = self.y.drop(index=rows)

        if cols is not None:
            self.X = self.X.drop(columns=cols)

    def scale(self, cols=None, scaler="std"):
        """Scale the specified columns.

        :param cols:        Columns to scale. If None, all columns are scaled, defaults to None
        :param scaler:      Either sklearn.Scaler or key from self.scalers, defaults to "std"
        :return:            sklearn.Scaler (for later use in inverse scaling etc.)
        """
        scaler = scaler if not isinstance(scaler, str) else self.scalers[scaler]
        cols = cols if cols is not None else self.X.columns
        self.X[cols] = scaler.fit_transform(self.X[cols])

        return scaler



    