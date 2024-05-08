import numpy as np


import pickle

from scipy import sparse

from sklearn.neighbors import kneighbors_graph

from sklearn.datasets import fetch_openml

from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
from sklearn.preprocessing import LabelBinarizer

from sklearn.metrics import ConfusionMatrixDisplay

import matplotlib.pyplot as plt


def plot_confusion_matrix_in_percent(cm, labels):
    
    cm_percent = np.round(100 * cm / np.sum(cm, axis= 1),1)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm_percent, display_labels=labels)

    fig, ax = plt.subplots(figsize=(10, 10))

    disp.plot(ax = ax)

    ax.set_xlabel('Predicted Label %')
    ax.set_ylabel('True Label %')

    # Display the plot
    plt.show()
    plt.close()


def compute_multiclass_metrics(y_true, y_pred, average='micro'):
    """
    Computes F1 score, AUC score, and accuracy for multiclass classification.

    Returns:
    - A dictionary with F1 score, AUC score, and accuracy
    """
    # Calculating F1 score

    f1 = f1_score(y_true, y_pred, average=average)

    # Calculating accuracy
    acc = accuracy_score(y_true, y_pred)

    # Calculating AUC for multiclass by treating it as a One-vs-Rest problem
    lb = LabelBinarizer()
    lb.fit(y_true)
    y_true_binarized = lb.transform(y_true)
    y_pred_binarized = lb.transform(y_pred)

    # Handling cases where a class label was never predicted or does not exist in y_true
    if y_true_binarized.shape[1] == 1:
        auc = 0.5  # Default to 0.5 (no discrimination) for a single class
    else:
        auc = roc_auc_score(y_true_binarized, y_pred_binarized, average=average, multi_class='ovr')

    return {
        'F1 Score': f1,
        'AUC Score': auc,
        'Accuracy': acc
    }


def mutual_knn_graph(X,k):

    n,d = X.shape

    A = np.zeros((n,n), dtype= np.bool_)

    squared_distances = np.sum((np.expand_dims(X, axis =1) - np.expand_dims(X, axis = 0))**2, axis = -1)

    for i in range(n):

        k_neighbors_i = np.argsort(squared_distances[i])[1:k+1]

        A[i, k_neighbors_i] = True
    
    mutual_A = A.T & A

    return mutual_A

def save_knn_graph(X,y, k, save_path):

    n, _ = X.shape

    G = kneighbors_graph(X, n_neighbors=k, mode='connectivity')


    A = sparse.lil_matrix((n,n), dtype=bool)

    for i,j in zip(*G.nonzero()):
        A[i,j] = True
        A[j,i] = True
    
    with open(save_path, 'wb') as write_file:
       
       pickle.dump((A, y), write_file)
    
def fetch_mnist_data():

    mnist = fetch_openml('mnist_784', parser='auto')
    X = np.array(mnist['data'])

    n,_ = X.shape
    
    y = np.array(mnist['target'].astype(np.int8))

    return X,y


def calcualte_inertia(X,memberships, means, k):

    SSD = 0.0
    for i in range(k):

        SSD += np.sum(np.sum((X[memberships == i,:]- means[i])**2,axis=-1))

    return SSD

if __name__ == "__main__":

    X = np.random.randn(50,3)

    A = np.random.randn(50,50)

    x_0 = np.random.randn(50)
