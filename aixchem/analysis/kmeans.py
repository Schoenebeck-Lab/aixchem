import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples, davies_bouldin_score, calinski_harabasz_score

from scipy.spatial.distance import cdist



class KMeansClustering:

    np.random.seed(42)
    rng = np.random.RandomState(42)

    def __init__(self):
        """
        Wrapper class for sklean.KMeans to facilitate optimization and interpretation of results.
        """

        self.model = None
        self.clusters = None

        self.metrics = pd.DataFrame()
        self.metrics.index.name = "k"

    def run(self, df, k=3, n_init=5000, **kwargs):
        """Function to run KMeans clustering. Kwargs can be specified for the sklearn class.

        :param df:      pd.DataFrame containing the features for clustering.
        :param k:       Cluster number k, defaults to 3
        :param n_init:  Number of initializations per run 
                        (Default value is increased to ensure identification of the global minimum of inertia), 
                        defaults to 5000
        """

        self.model = KMeans(n_clusters=k, random_state=self.rng, n_init=n_init, **kwargs).fit(df)

        self.clusters = pd.Series(self.model.labels_, name="Cluster", index=df.index, dtype=np.int64)

        self.metrics.at[k, "Inertia"] = self.model.inertia_
        self.metrics.at[k, "Distortion"] = sum(np.min(cdist(df, self.model.cluster_centers_, 'euclidean'), axis=1)) / df.shape[0]
        self.metrics.at[k, "Silhouette Score"] = silhouette_score(df, self.clusters)
        self.metrics.at[k, "Davies-Bouldin Score"] = davies_bouldin_score(df, self.clusters)
        self.metrics.at[k, "Calinski-Harabasz Score"] = calinski_harabasz_score(df, self.clusters)

    def optimize(self, df, ks, **kwargs):
        """Function to find the optimal cluster number k from a list of different ks.

        :param df:      pd.DataFrame containing the features for clustering.
        :param ks:      List of cluster numbers to check

        :return:        pd.DataFrame containing per-sample silhouette scores for all cluster numbers (to create silhouette plots).
        """

        silhouettes = pd.DataFrame()

        for k in ks:

            self.run(df, k=k, **kwargs)
            silhouettes[k] = silhouette_samples(df, self.clusters)

        silhouettes.set_index(df.index, inplace=True)

        return silhouettes

    def statistics(self, df, k, n, refs=None, **kwargs):
        """Function to investigate the dependency of the clustering results on the random state by performing the clustering n times with different random seeds.

        :param df:      pd.DataFrame containing the features for clustering.
        :param k:       Cluster number k.
        :param n:       Number of times the algorithm is run with different random states.
        :param refs:    Reference(s) for calculation of the scores., defaults to None
        :return: _description_
        """
        
        clusters = pd.DataFrame()

        for i in range(n):

            # Set a different random seed & run clustering
            np.random.seed(i)
            self.rng = np.random.RandomState(i)

            self.run(df=df, k=k, **kwargs)

            # Store resulting clusters in a df
            clusters = pd.concat([clusters, pd.Series(self.clusters.astype(np.int64), name=f"N({i})")], axis=1)

        if refs is not None:          
            # Ensure reference(s) are a list
            refs = refs if isinstance(refs, list) else [refs]
            
            # One-Hot-Encode clusters (1 = in the same cluster as refs, 0 = not in the same cluster)
            for col in clusters.columns:
                clusters[col] = clusters[col].isin(clusters[col][refs]).astype(int)

            # Return percentage of times each instance is clustered with the references
            return clusters.sum(axis=1) / len(clusters.columns)

        else:
            return clusters