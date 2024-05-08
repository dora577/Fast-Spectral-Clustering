


import numpy as np
import pickle

import matplotlib.pyplot as plt
from fastspectral import FastSpectralClustering
from utils import save_knn_graph, fetch_mnist_data


from sklearn.metrics import confusion_matrix

from utils import compute_multiclass_metrics, plot_confusion_matrix_in_percent



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

if __name__ == "__main__":

    np.random.seed(37)

    for k in [7]:

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


