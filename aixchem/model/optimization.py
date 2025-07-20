import pandas as pd
import itertools as it


class Optimization:
    """
    Base class for optimization algorithms.
    """

    def __init__(self, obj, params=None):
        """
        Initialize the optimization with a model or transformer class and a parameter grid.

        :param obj: The class of the model to be optimized.
        :param params: Dictionary of hyperparameters with lists of values, defaults to empty dict
        """
        self.obj = obj
        self.params = params or {}
        self.grid = self._get_grid()

        self.results = None

    def _get_grid(self):
        """
        Generate a grid of objects based on the provided parameters.
        This method creates all possible combinations of the parameters specified in self.params
        and returns a list of instantiated objects.

        :return: List of instantiated objects with all combinations of (valid) parameters.
        """
        def get_param_grid(params):
            keys, value_lists = zip(*params.items())
            for values in it.product(*value_lists):
                yield dict(zip(keys, values))

        return [self.obj(**param) for param in get_param_grid(self.params)]

    def _execute_task(self, obj, dataset, kwargs):
        """
        Fit and score a model with the given dataset and kwargs.

        :param obj: Model instance to fit and score
        :param dataset: Dataset to fit on
        :param kwargs: Additional fit parameters
        :return: Merged dictionary of params and score
        """
        obj.fit(dataset, **kwargs)
        
        return obj.params | obj.score(dataset, **kwargs)

    def run(self, dataset, njobs=1, **kwargs):
        """
        Run the optimization process on the dataset using the specified number of jobs.

        Note: If access to model attributes (like `.silhouettes` for Clusterers) is required after optimization,
        run with `njobs=1` (sequential mode). In parallel mode, each model is copied into a
        separate process, so changes to attributes (e.g., from `.fit()` or `.score()`) do not
        propagate back to the original objects in `self.grid`.

        :param dataset: The Dataset instance to optimize.
        :param njobs: Number of parallel jobs to run. If 1, runs sequentially.
        :param kwargs: Additional keyword arguments for the model's fit method.
        :return: DataFrame where each row corresponds to one parameter combination's results.
        """
        if njobs == 1:
            # Run sequentially
            results = [self._execute_task(obj, dataset, kwargs) for obj in self.grid]
        else:
            from joblib import Parallel, delayed
            # Run in parallel
            results = Parallel(n_jobs=njobs)(
                delayed(self._execute_task)(obj, dataset, kwargs)
                for obj in self.grid
            )

        self.results = pd.DataFrame(results)

        return self.results
