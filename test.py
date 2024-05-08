

from sklearn.datasets import fetch_openml
from sklearn.neighbors import kneighbors_graph

import numpy as np
import pickle

import matplotlib.pyplot as plt
from fastspectral import FastSpectralClustering
from utils import knn_graph, mutual_knn_graph

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def get_predictions(memberships, y, k):
    
    y_hat = np.zeros_like(y)

    for i in range(k):

        

        vals, counts = np.unique(y[memberships == i], return_counts=True)

        mode = vals[np.argmax(counts)]

        y_hat[memberships == i] = mode
    return y_hat

def save_knn_graph_mnist(k=10):

    mnist = fetch_openml('mnist_784', parser='auto')
    replace_dict = {chr(i): i-96 for i in range(97, 107)}
    X = np.array(mnist.data.replace(replace_dict))

    n,_ = X.shape
    
    y = np.array(mnist['target'].astype(np.int8))
    
    G = kneighbors_graph(X, n_neighbors=k, mode='connectivity')

    A = np.zeros((n,n), dtype=np.bool_)

    for i,j in zip(*G.nonzero()):
        A[i,j] = True
        A[j,i] = True
    
    with open(f'data/mnist_{k}graph.pickle', 'wb') as write_file:

       np.save(write_file, (A,y), allow_pickle=True)

if __name__ == "__main__":

    np.random.seed(37)

    # save_knn_graph_mnist(k =10)

    data = np.load('data/mnist_10graph.npz', allow_pickle=True)

    A = data['A']
    y = data['y']
    data.close()

    num_labels = len(np.unique(y))

    breakpoint()

    model = FastSpectralClustering(k = num_labels, tolerance=1e-5)

    model.fit(A)

    memberships = model.memberships

    y_hat = get_predictions(memberships, y, num_labels)

    cm = confusion_matrix(y, y_hat, )

    disp  = ConfusionMatrixDisplay(cm, display_labels=range(num_labels))

    disp.plot()

    plt.savefig('output_data/mnist_cm.png')

