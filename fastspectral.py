import numpy as np

from kmeans import kMeans

from math import log, ceil

from scipy import sparse 

class FastSpectralClustering:
    def __init__(self,k, tolerance):

        self.k = k
        self._tolerance = tolerance

        self._D =  None

        self._M, self._Y  = None, None
     
        self._model = kMeans(k, tolerance)

    def _count_degree(self, A):

        D = np.asarray(A.sum(axis = 0) - A.diagonal())

        return D.squeeze(0)
    
    def _power_method(self,M, Y, t):

        for _ in range(t-1):
            Y = M @ Y

        return Y        

    def _normalised_laplacian(self,A, D):

        n, _ = A.shape

        D_inv_sqrt = sparse.diags(np.sqrt(1/D))



        N = sparse.eye(n, format='csr') - D_inv_sqrt @ A @ D_inv_sqrt

        return N

    def _signless_laplacian(self, A, D ):
        
        N = self._normalised_laplacian(A,D)

        n,_ = N.shape

        M = sparse.eye(n, format='csr') - 0.5 * N

        return M
    
    def _create_embeddings(self, A, k):
 
        self._D = self._count_degree(A)

        self._M = self._signless_laplacian(A, self._D)

        n, _= self._M.shape

        l = ceil(log(k))

        t = ceil(10 * log(n/k))
         
        Y = self._power_method(self._M, np.random.randn(n,l), t)

        return Y

    def fit(self,A):

        self._Y = self._create_embeddings(A, self.k)

        D_inv_sqrt = sparse.diags(np.sqrt(1/self._D))

        X  = D_inv_sqrt @ self._Y

        self._model.fit(X)
    
    @property
    def memberships(self):
        return self._model.memberships
    @property
    def cluster_centers(self):
        return self._model.cluster_centers
    @property
    def intertia(self):
        return self._model.intertia

        





        

       

        

















        
