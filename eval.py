


import numpy as np

import matplotlib.pyplot as plt
from fastspectral import FastSpectralClustering

from sklearn.metrics import confusion_matrix

from utils import compute_multiclass_metrics, plot_confusion_matrix_in_percent

import torch
from torch_geometric.datasets.wikics import WikiCS

from torch_geometric.utils import to_scipy_sparse_matrix
    
import time

def get_predictions(memberships, y, k):
    
    y_hat = np.zeros_like(y)

    for i in range(k):

        vals, counts = np.unique(y[memberships == i], return_counts=True)

        mode = vals[np.argmax(counts)]

        y_hat[memberships == i] = mode
    return y_hat

def run_mnist(load_path):

    A,y = np.load(load_path, allow_pickle=True)

    num_labels = len(np.unique(y))

    model = FastSpectralClustering(k = num_labels, tolerance=1e-5)

    start = time.time()
    model.fit(A)
    end = time.time()
    memberships = model.memberships

    y_hat = get_predictions(memberships, y, num_labels)

    cm = confusion_matrix(y, y_hat, labels=range(num_labels))

    metrics = compute_multiclass_metrics(y, y_hat, average='micro')

    return  end - start, metrics, cm 


def preprocess_graph(A,y):

    D = A.sum(axis=0).A1 - A.diagonal()  

    non_zero_degree_mask = np.where(D != 0)[0]


    A= A[non_zero_degree_mask, :]

    A = A[:, non_zero_degree_mask]

    y = y[non_zero_degree_mask]


    return A, y


def run_wikics(load_path):

    with open(load_path, 'rb') as read_file:
        dataset = torch.load(read_file)[0]

    num_nodes = dataset['y'].shape[0]

    A = to_scipy_sparse_matrix(dataset['edge_index'], num_nodes=num_nodes).tocsr()

    A,y = preprocess_graph(A, dataset['y'].numpy())

    num_labels = len(np.unique(dataset['y']))
    
    model = FastSpectralClustering(k = num_labels, tolerance=1e-5)

    start = time.time()
    model.fit(A)
    end = time.time()
    memberships = model.memberships

    y_hat = get_predictions(memberships, y, num_labels)

    cm = confusion_matrix(y, y_hat, labels=range(num_labels))

    metrics = compute_multiclass_metrics(y, y_hat, average='micro')

    return  end - start, metrics, cm 


def eval_mnist():

    for k in [7, 10, 15, 20]:

        runtimes = []

        f_1 = []
        accuracy  = []
        auc = []

        total_cm = None

        for _ in range(10):
            runtime, metrics, cm = run_mnist(f'data/mnist_{k}_graph.pickle')

            runtimes.append(runtime)

            f_1.append(metrics['F1 Score'])
            accuracy.append(metrics['Accuracy'])
            auc.append(metrics['AUC Score'])

            if total_cm is None:
                total_cm= cm

            else:
                total_cm += cm

        plot_confusion_matrix_in_percent(cm, labels=range(10))

        print("Run with neares neighbour k= ", k)
        
        print(f"Average Runtime + std :{np.mean(runtimes)} + {np.std(runtimes)}")

        print(f"F1 Score", np.mean(f_1))
        print(f"Accuracy", np.mean(accuracy))
        print(f"AUC Score", np.mean(auc))




def eval_wikics():

    runtimes = []

    f_1 = []
    accuracy  = []
    auc = []

    total_cm = None

      
    for _ in range(10):
        runtime, metrics, cm = run_wikics(f'data/wikics/processed/data_undirected.pt')
        
        runtimes.append(runtime)

        f_1.append(metrics['F1 Score'])
        accuracy.append(metrics['Accuracy'])
        auc.append(metrics['AUC Score'])

        if total_cm is None:
            total_cm= cm

        else:
            total_cm += cm

    plot_confusion_matrix_in_percent(cm, labels=range(10))
    
    print(f"Average Runtime + std :{np.mean(runtimes)} + {np.std(runtimes)}")

    print(f"F1 Score", np.mean(f_1))
    print(f"Accuracy", np.mean(accuracy))
    print(f"AUC Score", np.mean(auc))


if __name__ == "__main__":

    np.random.seed(37)

    eval_wikics()


