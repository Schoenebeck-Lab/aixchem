from pathlib import Path

import pandas as pd
import numpy as np


class Dataset:
    """
    A flexible Dataset class that unifies various data sources.

    :param data: Data source; can be a pandas DataFrame, numpy array, pathlib.Path, or CSV file path (str).
    :param target: Specification for the target column(s). Can be a string, list of strings, pandas Series, pandas DataFrame, or numpy array.
    :param index: Specification for the index column(s). Can be a string, list of strings, integer, or list of integers.
    :param store_raw: If True, store a copy of the raw data in the attribute `raw`.
    :param pd_kwargs: Additional keyword arguments for the pandas DataFrame constructor or pd.read_csv.
    """
    def __init__(self, data, target=None, index=None, store_raw=False, **pd_kwargs):

        self.X = self._process_data(data, index, **pd_kwargs)
        self.y = self._process_target(target, index) if target is not None else None

        # Store raw data if desired
        self.raw = self._process_data(data, index, **pd_kwargs) if store_raw else None

    def _process_data(self, data, index, **pd_kwargs):
        """
        Convert the input data into a pandas DataFrame.

        :param data: Data source.
        :param index: Index column(s) specification.
        :param pd_kwargs: Additional keyword arguments for DataFrame constructor or pd.read_csv.
        :return: A pandas DataFrame representing the data.
        :raises ValueError: If the data type is unsupported or the file is not a CSV.
        """
        if isinstance(data, pd.DataFrame):
            return data.copy()
        
        elif isinstance(data, np.ndarray):
            return pd.DataFrame(data, index=index, **pd_kwargs)
        
        else:
            if Path(data).suffix.lower() == ".csv":
                return pd.read_csv(data, index_col=index, **pd_kwargs)
            else:
                raise ValueError("Unsupported data type. Data must be a DataFrame, numpy array, or CSV file path.")

    def _process_target(self, target, index):
        """
        Process the target data and return it as a pandas DataFrame.

        If target is specified as a string or list of strings, those columns are removed from self.X.
        If target is directly provided as a pandas Series, DataFrame, or numpy array, it is converted accordingly.

        :param target: Target specification.
        :param index: Index for converting numpy arrays (optional).
        :return: A DataFrame representing the target(s).
        :raises ValueError: If the target type is unsupported or the target columns are not found.
        """
        if isinstance(target, pd.Series):
            return target.to_frame()

        elif isinstance(target, pd.DataFrame):
            return target

        elif isinstance(target, np.ndarray):
            return pd.Series(target.flatten(), index=index).to_frame()
        
        elif isinstance(target, str) and target in self.X.columns:
            return self.X.pop(target).to_frame()
        
        elif isinstance(target, list) and all([isinstance(t, str) for t in target]):
            return pd.concat([self.X.pop(t) for t in target], axis=1)
            
        else:
            raise ValueError(f"Unsupported target type: {type(target)}")   

    def drop(self, rows=None, columns=None):
        """
        Drop specified rows and/or columns from the dataset.

        Rows are dropped from both the feature set (X) and the target (y), if present.
        Columns are dropped only from the feature set (X).

        :param rows: Row labels or list of row labels to drop.
        :param columns: Column labels or list of column labels to drop from the features.
        :return: The modified Dataset instance.
        """
        if rows is not None:
            self.X.drop(rows, axis=0, inplace=True)
            if self.y is not None:
                self.y.drop(rows, axis=0, inplace=True)

        if columns is not None:
            self.X.drop(columns, axis=1, inplace=True)

        return self
    
    def dropna(self, axis=0):
        """
        Drop rows or columns with NaN values from the dataset.
        
        For axis=0, rows with any NaN values in self.X (or self.X and self.y if present) are dropped.
        For axis=1, columns in self.X with any NaN values are dropped.
        
        :param axis: 0 to drop rows, 1 to drop columns from self.X.
        :return: The modified Dataset instance.
        :raises ValueError: If an unsupported axis value is provided.
        """
        if axis == 0:
            # Combine X and y if y exists; otherwise, just use X.
            data = pd.concat([self.X, self.y], axis=1) if self.y is not None else self.X
            mask = ~data.isna().any(axis=1)
            self.X = self.X.loc[mask]
            if self.y is not None:
                self.y = self.y.loc[mask]

        elif axis == 1:
            # Drop columns in self.X with any NaN values.
            self.X = self.X.drop(self.X.columns[self.X.isna().any()], axis=1)

        else:
            raise ValueError("Axis must be 0 (rows) or 1 (columns).")
        
        return self
    
    def copy(self):
        """
        Create a deep copy of the Dataset instance.

        :return: A new Dataset instance with copies of self.X, self.y, and self.raw.
        """
        dataset = Dataset.__new__(Dataset)
        dataset.X = self.X.copy()
        dataset.y = self.y.copy() if self.y is not None else None
        dataset.raw = self.raw.copy() if self.raw is not None else None

        return dataset
