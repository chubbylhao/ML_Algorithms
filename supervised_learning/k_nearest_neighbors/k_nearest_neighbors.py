import numpy as np
from data import iris
from utils import accuracy_score, Plot


class KNN:
    """
    The implementation of k nearest neighbors algorithm.

    Attributes:
        k: the number of neighbors you want to calculate with.
    """
    def __init__(self, k=5):
        self.k = k

    @staticmethod
    def _vote(neighbor_labels):
        """ Get the labels of the category with the largest
        number of k nearest neighbors. """
        return np.argmax(np.bincount(neighbor_labels))

    @staticmethod
    def _euclidean_distance(x1, x2):
        """ Calculate the Euclidean distance between two vectors. """
        return np.linalg.norm(x1 - x2)

    # @staticmethod
    # def _feature_normalize(X, ax=0):
    #     """ Normalization of features. """
    #     margin = X.max(axis=ax) - X.min(axis=ax)
    #     margin = (margin == ax) * margin + margin
    #     X = (X - X.min(axis=ax)) / margin
    #     return X

    def predict(self, X_train, y_train, X_test):
        """ Prediction(lazy study, drop a point in and learn again). """
        # initialize the labels of the predicted results to -1
        y_pred = np.zeros(X_test.shape[0]) - 1
        # # normalization of features
        # X_train = KNN._feature_normalize(X_train)
        # X_test = KNN._feature_normalize(X_test)
        # predict a point and cycle once
        for i, one_sample in enumerate(X_test):
            # get the index of the nearest k points
            idx = np.argsort([KNN._euclidean_distance(x, one_sample) for x in X_train])[:self.k]
            # get the label of the nearest k points
            k_nearest_neighbors = np.array([y_train[j] for j in idx])
            # get the labels of the category with the largest number of k nearest neighbors
            y_pred[i] = KNN._vote(k_nearest_neighbors.astype(int))
        return y_pred


if __name__ == '__main__':
    def main():
        knn = KNN(k=5)
        y_pred = knn.predict(iris.X_train, iris.y_train, iris.X_test)
        accuracy = accuracy_score(iris.y_test, y_pred)
        # reduce the features to two dimensions
        Plot().plot_in_2d(iris.X_test, y_pred, title="K Nearest Neighbors",
                          accuracy=accuracy, legend_labels=iris.y_names)
    main()
