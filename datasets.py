from utils import save_knn_graph

import numpy as np
from sklearn.datasets import fetch_openml


def fetch_mnist_data():

    mnist = fetch_openml('mnist_784', parser='auto')
    X = np.array(mnist['data'])

    n,_ = X.shape
    
    y = np.array(mnist['target'].astype(np.int8))

    return X,y


if __name__ == "__main__":

    X,y = fetch_mnist_data()

    for k in [7,10, 15, 20]:
        save_knn_graph(X, y, k, save_path= f"data/mnist_{k}_graph.pickle")