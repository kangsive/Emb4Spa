import numpy as np
from sklearn.neighbors import NearestNeighbors

def compute_NN(X, Y):
    X, Y = np.array(X), np.array(Y)
    neigh = NearestNeighbors(n_neighbors = 1)
    neigh.fit(X)

    knn_array = neigh.kneighbors(return_distance=False)
    knn_array = knn_array.reshape(1, -1)    # for k=1, we can flaten the array
    matches = Y[knn_array] == Y     # Match nns' labels with corresponding y
    avg_acc = matches.mean()

    return avg_acc

def compute_FT(X, Y, k=None):
    X, Y = np.array(X), np.array(Y)
    types, k_list = np.unique(Y, return_counts=True)
    neigh = NearestNeighbors(n_neighbors = k)
    neigh.fit(X)

    avg_acc = 0

    for i in range(len(types)):
        X_t, k = X[Y == types[i]], k_list[i]
        knn_array = neigh.kneighbors(X_t, n_neighbors=k, return_distance=False)     # Get knn for every instance
        Y_t = np.full(knn_array.shape, types[i])    # create ground truth for every nn
        matches = Y[knn_array] == Y_t    # Match knns' labels with corresponding y
        avg_acc += (matches.sum(axis=-1)/k).mean()   # Calulate acc of each row (instance) and average them

    return avg_acc/len(k_list)


def compute_ST(X, Y, k=None):
    X, Y = np.array(X), np.array(Y)
    types, k_list = np.unique(Y, return_counts=True)
    neigh = NearestNeighbors(n_neighbors = k)
    neigh.fit(X)

    avg_acc = 0

    for i in range(len(types)):
        X_t, k = X[Y == types[i]], k_list[i]
        knn_array = neigh.kneighbors(X_t, n_neighbors=2*k, return_distance=False)     # Now we look at top 2*k nns
        Y_t = np.full(knn_array.shape, types[i])    # create ground truth for every nn
        matches = Y[knn_array] == Y_t    # Match knns' labels with corresponding y
        avg_acc += (matches.sum(axis=-1)/k).mean()   # Calulate acc of each row (instance) and average them

    return avg_acc/len(k_list)