


import numpy as np
import pickle

import matplotlib.pyplot as plt
from fastspectral import FastSpectralClustering
from utils import save_knn_graph, fetch_mnist_data

from utils import compute_multiclass_metrics

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


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

    cm = confusion_matrix(y, y_hat, )

    disp  = ConfusionMatrixDisplay(cm, display_labels=range(num_labels))

    disp.plot()

    plt.savefig('output_data/mnist_cm.png')

    metrics = compute_multiclass_metrics(y, y_hat, average='micro')

    return metrics, time

if __name__ == "__main__":

    np.random.seed(37)

    runtimes = []

    # f_1 = []
    # accuracy  = []
    # auc = []

    for _ in range(10):
        runtime, metrics = run_mnist('data/mnist_10_graph.pickle')

        runtimes.append(runtime)

        # f_1.append(metrics['F1 Score'])
        # accuracy.append(metrics['Accuracy'])
        # auc.append(metrics['AUC Score'])

    
    print(f"Average Runtime + std :{np.mean(runtimes)} + {np.std(runtimes)}")

    # print(f"F1 Score", np.mean(f_1))
    # print(f"Accuracy", np.mean(accuracy))
    # print(f"AUC Score", np.mean(auc))


