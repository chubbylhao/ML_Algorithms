import numpy as np


class MultipleDimensionalScaling:
    """ 多尺度缩放，保持距离的线性降维方法.
    Attributes:
        d: 降维后的维数/特征数
    """

    def __init__(self, d):
        self.d = d

    @staticmethod
    def calculate_distance_matrix(X):
        """ 计算数据的距离平方矩阵 """
        # 利用 (a-b)^2=a^2+b^2-2ab 计算
        squared = np.sum(X ** 2, axis=1)
        dist = (squared - 2 * np.dot(X, X.T)).T + squared
        dist[dist < 0] = 0    # 避免数值不稳定
        return dist

    def dimension_reduction(self, X):
        """ 线性降维 """
        n_samples = X.shape[0]
        # 1) 计算距离的平方矩阵
        distance_squared_matrix = MultipleDimensionalScaling.calculate_distance_matrix(X)
        # 2) 计算降维后的内积矩阵
        dist_i_squared = np.expand_dims(distance_squared_matrix.sum(axis=1), axis=1) / n_samples
        dist_j_squared = distance_squared_matrix.sum(axis=0) / n_samples
        dist_squared = distance_squared_matrix.sum() / n_samples ** 2
        inner_product_matrix = -0.5 * (distance_squared_matrix - dist_i_squared -
                                       dist_j_squared + dist_squared)
        # 3) 对内积矩阵进行特征值分解以获得降维后的坐标
        eigenvalues, eigenvectors = np.linalg.eig(inner_product_matrix)
        picked_index = eigenvalues.argsort()[::-1][:self.d]
        picked_eigenvalues = np.sqrt(np.diag(eigenvalues[picked_index])).real
        picked_eigenvectors = eigenvectors.T[picked_index].real
        return np.dot(picked_eigenvectors.T, picked_eigenvalues)


if __name__ == '__main__':
    def main():
        # from sklearn.datasets import load_iris
        from sklearn.datasets import make_s_curve
        import matplotlib.pyplot as plt
        # iris_data = load_iris()
        # X, Y = iris_data.data, iris_data.target
        X, Y = make_s_curve(n_samples=1000, noise=0.15)
        mds = MultipleDimensionalScaling(d=2)
        y = mds.dimension_reduction(X)
        plt.scatter(y[:, 0], y[:, 1], c=Y)
        plt.show()
    main()
