import numpy as np
from sklearn.manifold import trustworthiness
from sklearn.neighbors import NearestNeighbors

def trustworthiness_knn(X, emb, n_neighbors=10):
    tw = trustworthiness(X, emb, n_neighbors=n_neighbors)
    # KNN preservation: compare neighbor overlap
    nbrs_X = NearestNeighbors(n_neighbors=n_neighbors).fit(X)
    nbrs_E = NearestNeighbors(n_neighbors=n_neighbors).fit(emb)
    idx_X = nbrs_X.kneighbors(return_distance=False)
    idx_E = nbrs_E.kneighbors(return_distance=False)
    # Flatten intersections across all rows
    overlap = (np.intersect1d(idx_X, idx_E).size) / (idx_X.size)
    return float(tw), float(overlap)
