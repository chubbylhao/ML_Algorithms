import numpy as np
from unsupervised_learning.MDS.multiple_dimensional_scaling import MultipleDimensionalScaling


class Isomap:
    """ Isomap，流形学习，非线性降维.
    Attributes:
        d: 降维后的维数/特征数
        k: 计算测地距离矩阵时的邻域样本个数
    """
    def __init__(self, d, k):
        self.d = d
        self.k = k

    def _floyd(self, euclidean_distance):
        """ 计算测地距离矩阵 """
        n_samples = euclidean_distance.shape[0]
        geodesic_distance = np.full_like(euclidean_distance, np.max(euclidean_distance) * 1e8)
        k_index = np.argsort(euclidean_distance, axis=1)
        # 构建迭代用的初始距离矩阵
        for i in range(n_samples):
            geodesic_distance[i, k_index[i, :self.k+1]] = euclidean_distance[i, k_index[i, :self.k+1]]
        # 使用Floyd算法进行迭代计算
        for step in range(n_samples):
            for i in range(n_samples):
                for j in range(n_samples):
                    if geodesic_distance[i, j] > geodesic_distance[i, step] + geodesic_distance[step, j]:
                        geodesic_distance[i, j] = geodesic_distance[i, step] + geodesic_distance[step, j]
        return geodesic_distance

    def dimension_reduction(self, X):
        """ 非线性降维 """
        n_samples = X.shape[0]
        # 以下是MDS算法（只不过用测地距离代替了欧氏距离）
        # -------------------------------------
        # 1) 计算距离的平方矩阵
        euclidean_distance = np.sqrt(MultipleDimensionalScaling.calculate_distance_matrix(X))
        distance_squared_matrix = np.power(self._floyd(euclidean_distance), 2)
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
        # import matplotlib.pyplot as plt
        # iris_data = load_iris()
        # X, Y = iris_data.data, iris_data.target

        from sklearn.datasets import make_s_curve, make_swiss_roll
        import matplotlib.pyplot as plt
        # X, Y = make_s_curve(n_samples=500, noise=0.15, random_state=40)
        X, Y = make_swiss_roll(n_samples=500, noise=0.15, random_state=100)

        isomap = Isomap(d=2, k=20)
        y = isomap.dimension_reduction(X)
        plt.scatter(y[:, 0], y[:, 1], c=Y)
        plt.show()
    main()
