import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt


def lda(X, y, dims):
    """ 这是之前写的，出于时间原因就不重构了. """
    target_nums = np.unique(y)

    # calculate Sw and Sb
    Sw_list, Sb_list = [], []
    X_mean = X.mean(axis=0)
    for target in target_nums:
        Xi = X[y == target]
        Xi_nums = Xi.shape[0]
        Xi_mean = Xi.mean(axis=0)
        Sw_list.append((Xi-Xi_mean).T.dot(Xi-Xi_mean))
        Sb_list.append(Xi_nums*(Xi-X_mean).T.dot(Xi-X_mean))
    Sw_list, Sb_list = np.array(Sw_list), np.array(Sb_list)
    Sw, Sb = Sw_list.sum(axis=0), Sb_list.sum(axis=0)

    # calculate the dimensionality reduction matrix W
    S = np.linalg.inv(Sw) * Sb
    eig_vals, eig_vecs = np.linalg.eig(S)
    eig_vals_idx = np.argsort(eig_vals)
    eig_vals_idx = eig_vals_idx[::-1]
    eig_vals_idx = eig_vals_idx[:dims]
    W = eig_vecs[:, eig_vals_idx]

    # return the coordinates after dimensionality reduction
    return X.dot(W)


if __name__ == '__main__':
    iris_data = load_iris()
    iris_X = iris_data.data    # get the 150 samples, each sample has 4 features, thus the shape of X is (150, 4)
    iris_y = iris_data.target  # get the targets of those samples, the shape of y is (150,)
    res_data = lda(iris_X, iris_y, 2)
    plt.figure()
    plt.scatter(res_data[:, 0], res_data[:, 1], c=iris_y)
    plt.show()
