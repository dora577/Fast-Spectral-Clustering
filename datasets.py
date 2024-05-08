from utils import save_knn_graph, fetch_mnist_data



if __name__ == "__main__":

    X,y = fetch_mnist_data()

    for k in [7,10, 15, 20]:
        save_knn_graph(X, y, k, save_path= f"data/mnist_{k}_graph.pickle")