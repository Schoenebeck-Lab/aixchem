
from abc import ABC, abstractmethod

from aixchem.dataset import Dataset


class Transformation(ABC):
    """
    Abstract base class for data transformation steps.

    This class provides a common interface for all transformations in the framework.
    It supports both in-place modifications and returning a modified copy of the Dataset.
    
    If `inplace` is True (default), the transformation modifies the input Dataset directly.
    If `inplace` is False, a copy of the Dataset is created before applying the transformation.
    """
    def __init__(self, inplace=True):
        super().__init__()

        # Dictionary to store transformation parameters.
        self.params = {}
        self.inplace = inplace
        
        # The columns to be transformed (set during fit).
        self.columns = None 

    @abstractmethod
    def _fit(self, dataset: Dataset, **kwargs):
        """
        Internal method to fit the transformer to the dataset.
        Child classes should override this method to perform any necessary fitting.

        :param dataset: The Dataset instance to fit.
        :param kwargs: Additional keyword arguments.
        :return: self.
        """
        return self
    
    def fit(self, dataset: Dataset, columns=None, **kwargs):
        """
        Fit the transformer to the dataset.

        This method processes the optional 'columns' parameter and stores the column names to be used
        during the transformation, then delegates the actual fitting to the _fit() method.

        :param dataset: The Dataset instance to fit.
        :param columns: The columns on which to apply the transformation.
                        If None, all columns in dataset.X are used.
        :param kwargs: Additional keyword arguments to pass to _fit().
        :return: self.
        :raises ValueError: If 'columns' is not None, a string, or a list of strings.
        """
        if columns is None:
            self.columns = list(dataset.X.columns)
        elif isinstance(columns, str):
            self.columns = [columns]
        elif isinstance(columns, list):
            self.columns = columns
        else:
            raise ValueError("Columns parameter must be None, a string, or a list of strings.")
        
        return self._fit(dataset, **kwargs)
    
    @abstractmethod
    def _transform(self, dataset: Dataset, **kwargs):
        """
        Internal method to transform the dataset.
        Child classes should override this method to implement the core transformation logic.

        :param dataset: The Dataset instance to transform.
        :param kwargs: Additional keyword arguments.
        :return: The transformed Dataset instance.
        """
        return dataset
    
    def transform(self, dataset: Dataset, **kwargs):
        """
        Transform the dataset.

        If `inplace` is False, a copy of the Dataset is created before applying the transformation.
        The actual transformation is delegated to the _transform() method.

        :param dataset: The Dataset instance to transform.
        :param kwargs: Additional keyword arguments to pass to _transform().
        :return: The transformed Dataset instance.
        """
        if not self.inplace:
            dataset = dataset.copy()

        return self._transform(dataset, **kwargs)

    def fit_transform(self, dataset: Dataset, **kwargs):
        """
        Fit the transformer to the dataset and then transform it.

        :param dataset: The Dataset instance to fit and transform.
        :param kwargs: Additional keyword arguments passed to fit().
        :return: The transformed Dataset instance.
        """
        return self.fit(dataset, **kwargs).transform(dataset)
        