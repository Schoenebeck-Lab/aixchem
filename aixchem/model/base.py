import inspect
from abc import ABC, abstractmethod

from aixchem.dataset import Dataset


class Model(ABC):

    def __init__(self, model, **params):
        # Get all valid parameters for the model's __init__ method
        valid_params = set(inspect.signature(model.__init__).parameters.keys())

        # Store valid parameters
        self.params = {k: v for k, v in params.items() if k in valid_params and v is not None}

        # Raise warning if invalid parameters are provided
        invalid_params = [p for p in params.keys() if p not in self.params.keys()]
        if invalid_params:
           print(f"Warning: The following parameters are not valid for {model.__name__}: {invalid_params}")
           
        # Initialize the model with the provided parameters
        self.model = model(**self.params)
        self.params.update({"model": model.__name__})

    def fit(self, dataset: Dataset, **kwargs):
        self.model.fit(dataset.X, dataset.y, **kwargs)
        return self
    
    def predict(self, dataset: Dataset, **kwargs):
        return self.model.predict(dataset.X, **kwargs)
    
    @abstractmethod
    def _score(self, dataset: Dataset, **kwargs):
        """
        Abstract method to compute the score of the model.
        Must be implemented by subclasses.
        :param dataset: The Dataset instance to score.
        :param kwargs: Additional keyword arguments.
        :return: The score of the model.
        """
        pass

    def score(self, dataset: Dataset, **kwargs):
        return self._score(dataset, **kwargs)
