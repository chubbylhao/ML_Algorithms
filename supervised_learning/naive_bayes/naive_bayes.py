import numpy as np
from data import digits
from utils import accuracy_score, Plot


class NaiveBayes:
    """
    The implementation of naive bayes algorithm.

    Attributes:
        X_train: train data.
        y_train: train target.
    """
    def __init__(self, X_train, y_train):
        self.X_train, self.y_train = X_train, y_train
        self.classes = np.unique(self.y_train)
        self.means_vars = self._calculate_means_vars()

    def _calculate_means_vars(self):
        """ Calculate mean:μ(c,i) and var:σ²(c,i). """
        means_vars = []
        for i, class_ in enumerate(self.classes):
            X_class_ = self.X_train[np.where(self.y_train == class_)]
            means_vars.append([])
            for feature in X_class_.T:
                mean_var = {'mean': feature.mean(), 'var': feature.var()}
                means_vars[i].append(mean_var)
        return means_vars

    @staticmethod
    def _calculate_likelihood(mean, var, xi):
        """ Calculate P(xi|c). """
        eps = 1e-4    # maintain numerical stability
        coeff = 1 / np.sqrt(2 * np.pi * var + eps)
        exponent = np.exp(- np.power(xi - mean, 2) / (2 * var + eps))
        return coeff * exponent

    def _calculate_prior(self, c):
        """ Calculate P(c). """
        return np.mean(self.y_train == c)

    def _calculate_argmax(self, x):
        """ Calculate argmax{P(c)*[P(x1|c)*...*P(xd|c)]} for only one sample. """
        posteriors = []
        for i, c in enumerate(self.classes):
            # calculate P(c)
            posterior = self._calculate_prior(c)
            for xi, mean_var in zip(x, self.means_vars[i]):
                # calculate P(x1|c)*...*P(xd|c)
                likelihood = NaiveBayes._calculate_likelihood(
                    mean_var['mean'], mean_var['var'], xi)
                # calculate P(c)*[P(x1|c)*...*P(xd|c)]
                posterior *= likelihood
            posteriors.append(posterior)
        # calculate argmax{P(c)*[P(x1|c)*...*P(xd|c)]}
        return self.classes[np.argmax(posteriors)]

    def predict(self, X_test):
        """ Calculate argmax{P(c)*[P(x1|c)*...*P(xd|c)]} for all samples. """
        y_pred = np.array([self._calculate_argmax(x) for x in X_test])
        return y_pred


if __name__ == '__main__':
    def main():
        naivebayes = NaiveBayes(digits.X_train, digits.y_train)
        y_pred = naivebayes.predict(digits.X_test)
        accuracy = accuracy_score(digits.y_test, y_pred)
        # reduce the features to two dimensions
        Plot().plot_in_2d(digits.X_test, y_pred, title="Naive Bayes",
                          accuracy=accuracy, legend_labels=digits.y_names)
    main()
