from pandas import np
from sklearn import metrics
from sklearn.cluster import DBSCAN


def dbscan(df):
    print("***********************************************************************************")
    print("\n\nAnalysis of the dataset based on DBSCAN")
    db = DBSCAN(eps=0.3, min_samples=10).fit(df)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    print('Estimated number of clusters: %d' % n_clusters_)
    print('Estimated number of noise points: %d' % n_noise_)
    print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(df, labels))

    print("\n\n***********************************************************************************")