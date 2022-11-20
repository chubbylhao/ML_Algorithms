import numpy as np
from unsupervised_learning.MDS.multiple_dimensional_scaling import MultipleDimensionalScaling


class LaplacianEigenmaps:
    """ 拉普拉斯特征映射，非线性降维.
    Attributes:
        d: 降维后的维数/特征数
        k: 邻域样本个数
        sigma: rbf距离的标准差
    """
    def __init__(self, d, k, sigma=10):
        self.d = d
        self.k = k
        self.sigma = sigma

    def _calculate_rbf_distance(self, x, neighbors):
        return np.exp(-1 * np.sum((x - neighbors)**2, axis=1) / self.sigma**2)

    def dimension_reduction(self, X):
        n_samples = X.shape[0]
        # 计算每个样本点的近邻点
        dist_matrix = MultipleDimensionalScaling.calculate_distance_matrix(X)
        k_neighbors_idx = np.argsort(dist_matrix)[:, 1:self.k + 1]
        # 计算权重矩阵，非近邻点的权重系数为0
        weight_matrix = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            weight_matrix[i, k_neighbors_idx[i]] = \
                self._calculate_rbf_distance(X[i], X[k_neighbors_idx[i]])
        # 进行特征分解，并得到降维后的坐标
        D = np.diag(weight_matrix.sum(axis=1))
        L = D - weight_matrix
        eigenvalues, eigenvectors = np.linalg.eig(np.dot(np.linalg.inv(D), L))
        picked_index = eigenvalues.argsort()[1:self.d + 1]
        return eigenvectors[:, picked_index].real


if __name__ == '__main__':
    def main():
        # from sklearn.datasets import load_iris
        # import matplotlib.pyplot as plt
        # iris_data = load_iris()
        # X, Y = iris_data.data, iris_data.target

        from sklearn.datasets import make_s_curve, make_swiss_roll
        import matplotlib.pyplot as plt
        # X, Y = make_s_curve(n_samples=500, noise=0.15, random_state=40)
        X, Y = make_swiss_roll(n_samples=500, noise=0.15, random_state=100)

        le = LaplacianEigenmaps(d=2, k=15)
        y = le.dimension_reduction(X)
        plt.scatter(y[:, 0], y[:, 1], c=Y)
        plt.show()
    main()
