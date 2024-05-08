import numpy as np
from scipy.spatial import cKDTree

def mutual_knn_graph(X,k):

    n,d = X.shape

    A = np.zeros((n,n), dtype= np.bool_)

    squared_distances = np.sum((np.expand_dims(X, axis =1) - np.expand_dims(X, axis = 0))**2, axis = -1)

    for i in range(n):

        k_neighbors_i = np.argsort(squared_distances[i])[1:k+1]

        A[i, k_neighbors_i] = True
    
    mutual_A = A.T & A

    return mutual_A


# def knn_graph(X, k):

#     n,d = X.shape

#     A = np.zeros((n,n), dtype=np.bool_)
     
#     squared_distances = np.sum((np.expand_dims(X, axis =1) - np.expand_dims(X, axis = 0))**2, axis = -1)

#     for i in range(n):
    
#         k_neighbors_i = np.argsort(squared_distances[i])[1:k+1]
#         A[i, k_neighbors_i] = True
#         A[k_neighbors_i, i] = True

 
#     return A  
def knn_graph(X, k):

    n, _ = X.shape
    tree = cKDTree(X)
    A = np.zeros((n,n), dtype=np.bool_)

    for i in range(n):
        
        _, k_neighbors_i = tree.query(X[i], k+1)
        k_neighbors_i = k_neighbors_i[1:]  # skip the first index since it is the point itself
        A[i, k_neighbors_i] = True
        A[k_neighbors_i, i] = True
 
    return A  



def calcualte_inertia(X,memberships, means, k):

    SSD = 0.0
    for i in range(k):

        SSD += np.sum(np.sum((X[memberships == i,:]- means[i])**2,axis=-1))

    return SSD

if __name__ == "__main__":

    X = np.random.randn(50,3)

    A = np.random.randn(50,50)

    # knn_graph(X, k =5)
    breakpoint()
    x_0 = np.random.randn(50)
