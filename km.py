import numpy as np
from numpy.linalg import norm
import pandas as pd
import itertools

class Kmeans:

    def __init__(self, n_clusters, max_iter, random_state = 1):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state

    def initialize_centroids(self, X):
        np.random.RandomState(self.random_state)
        random_idx = np.random.permutation(X.shape[0])
        centroids = X[random_idx[:self.n_clusters]]
        return centroids

    def compute_centroids(self, X, labels):
        centroids = np.zeros((self.n_clusters, X.shape[1]))
        for k in range(self.n_clusters):
            centroids[k, :] = np.mean(X[labels == k, :], axis=0)
            print(centroids)
        return centroids

    def compute_distance(self, X, centroids):
        distance = np.zeros((X.shape[0], self.n_clusters))
        for k in range(self.n_clusters):
            row_norm = norm(X - centroids[k, :], axis=1)
            distance[:, k] = np.square(row_norm)
        return distance

    def find_closest_cluster(self, distance):
        return np.argmin(distance, axis=1)

    def compute_sse(self, X, labels, centroids):
        distance = np.zeros(X.shape[0])
        for k in range(self.n_clusters):
            distance[labels == k] = norm(X[labels == k] - centroids[k], axis=1)
        return np.sum(np.square(distance))
    
    def fit(self, X):
        self.centroids = self.initialize_centroids(X)
        initial_clusters = self.centroids
        for i in range(self.max_iter):
            old_centroids = self.centroids
            distance = self.compute_distance(X, old_centroids)
            self.labels = self.find_closest_cluster(distance)
            self.centroids = self.compute_centroids(X, self.labels)
            if np.all(old_centroids == self.centroids):
                break
        self.error = self.compute_sse(X, self.labels, self.centroids)
        return initial_clusters
    
    def predict(self, X):
        old_centroids = self.centroids
        distance = self.compute_distance(X, old_centroids)
        return self.find_closest_cluster(distance)

def jaccard(labels1, labels2):
    n11 = n10 = n01 = 0
    n = len(labels1)
    for i, j in itertools.combinations(range(n), 2):
        comembership1 = labels1[i] == labels1[j]
        comembership2 = labels2[i] == labels2[j]
        if comembership1 and comembership2:
            n11 += 1
        elif comembership1 and not comembership2:
            n10 += 1
        elif not comembership1 and comembership2:
            n01 += 1
    return float(n11) / (n11 + n10 + n01)

if __name__=="__main__":

    df = pd.read_csv("iris.csv")
    X = df.iloc[:,1:5].values
    obj = Kmeans(3,10)
    initial_clusters = obj.fit(X)
    print(initial_clusters)
    y_pred = obj.predict(X)
    print(y_pred)

    y = df.iloc[:,5].values
    for i in range (len(y)):
        if (y[i] == "Iris-setosa"):
            y[i] = 0
        elif (y[i] == "Iris-versicolor"):
            y[i] = 1
        else:
            y[i] = 2
    print(y)
    print(1 - jaccard(y, y_pred))