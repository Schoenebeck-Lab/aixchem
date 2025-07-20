import numpy as np
from scipy.spatial.distance import cdist
from sklearn.metrics import silhouette_score, silhouette_samples, davies_bouldin_score, calinski_harabasz_score

from aixchem.model.base import Model
from aixchem.model.optimization import Optimization

class Clusterer(Model):
    """
    Base class for clustering techniques (e.g., KMeans, DBSCAN).

    The computed cluster labels replace the specified columns in dataset.X.
    By default, inplace is False, so the original dataset remains unchanged.
    """
    def __init__(self, model, **params):
        super().__init__(model=model, **params)

        self.silhouettes = None

    def _score(self, dataset):

        clusters = self.predict(dataset)
        
        results = {
            "Inertia": self.model.inertia_,
            "Distortion": sum(np.min(cdist(dataset.X, self.model.cluster_centers_, 'euclidean'), axis=1)) / dataset.X.shape[0],
            "Silhouette Score": silhouette_score(dataset.X, clusters),
            "Davies-Bouldin Score": davies_bouldin_score(dataset.X, clusters),
            "Calinski-Harabasz Score": calinski_harabasz_score(dataset.X, clusters)
        }

        self.silhouettes = silhouette_samples(dataset.X, clusters)

        return results


class ClusterRobustness(Optimization):

    def __init__(self, clusterer, random_states=list(range(0, 10))):

        # Retrieve parameters from original object
        params = {k: [v] for k, v in clusterer.params.items()}

        # Set new random states
        params["model"] = [clusterer.model.__class__]
        params["random_state"] = random_states
        
        super().__init__(Clusterer, params)

    def _execute_task(self, obj, dataset, kwargs):

        obj.fit(dataset, **kwargs)
        
        return {"random_state": obj.params["random_state"]} | dict(zip(dataset.X.index, obj.predict(dataset)))

    def run(self, dataset, **kwargs):
        super().run(dataset, **kwargs)

        self.results = self.results.set_index("random_state", drop=True).T

        return self.results

    def check_candidates(self, candidates):
     
        # Ensure reference(s) are a list
        candidates = candidates if isinstance(candidates, list) else [candidates]
            
        # One-Hot-Encode clusters (1 = in the same cluster as candidates, 0 = not in the same cluster)
        for col in self.results.columns:
            self.results[col] = self.results[col].isin(self.results[col][candidates]).astype(int)

        # Return percentage of times each instance is clustered with the references
        return self.results.sum(axis=1) / len(self.results.columns)
