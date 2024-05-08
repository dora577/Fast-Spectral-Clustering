    
import numpy as np

from utils import calcualte_inertia

class kMeans:

    def __init__(self, k, tolerance):
        self._k = k 
        self._tolerance = tolerance

        self._means = None
        self._memberships = None
        self._SSD = float('inf')
    
    def _initialize(self, X, k):

        n, d = X.shape

        means = []

        p = np.ones(n)/n

        while len(means) < k:

            index = np.random.choice(n, size = None, p = p)

            means.append(X[index, :])

            expanded_means = np.array(means)[:, np.newaxis, :]

            squared_distances = np.sum((X - expanded_means)**2, axis = -1).T

            min_distances = np.min(squared_distances, axis=1)

            p = min_distances/np.sum(min_distances)
        return np.array(means)[:, np.newaxis, :]


    def _train(self, X):
        
        n,d = X.shape

        memberships = np.zeros(n,dtype=int)

        means = self._initialize(X, self._k)

        SSD = float('inf')

        stop = False

        while not stop:

            distances = (np.sum((X - means)**2, axis=-1)).T

            memberships = np.argmin(distances, axis=1)

            for i in range(self._k):
                
                if len(np.where(memberships == i)[0]) > 0:

                    means[i]= np.mean(X[memberships == i,:],axis = 0)

            new_SSD = calcualte_inertia(X, memberships,means, self._k)

            if SSD - new_SSD  < self._tolerance:
                stop = True

            SSD = new_SSD

        return memberships, means, SSD

    def fit(self,X):

        for _ in range(10):

            memberhips, means, SSD = self._train(X)

            if SSD < self._SSD:
                
                self._memberships, self._centroids, self._inertia = memberhips, means, SSD

    @property
    def memberships(self):
        return self._memberships
    @property
    def cluster_centers(self):
        return self._centroids
    @property
    def intertia(self):
        return self._inertia

            
    def get_clusters(self,X):

        clusters = {}

        for i in range(self._k):

            clusters[i] = X[self._memberships == i,:]

        return clusters