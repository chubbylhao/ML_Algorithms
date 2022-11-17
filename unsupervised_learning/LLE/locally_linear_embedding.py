import numpy as np
from unsupervised_learning.MDS.multiple_dimensional_scaling import MultipleDimensionalScaling


class LLE:
    """ LLE，局部线性嵌入，非线性降维.
    Attributes:
        d: 降维后的维数/特征数
        k: 保持局部线性关系的邻域样本个数
    """
    def __init__(self, d, k):
        self.d = d
        self.k = k

    def dimension_reduction(self, X):
        """ 非线性降维 """
        n_samples, n_features = np.shape(X)
        # 获得每个样本点的k近邻（度量采用欧氏距离的平方）
        dist_matrix = MultipleDimensionalScaling.calculate_distance_matrix(X)
        k_neighbors_idx = np.argsort(dist_matrix)[:, 1:self.k+1]
        # 求解权重系数矩阵
        W = np.zeros((self.k, n_samples))
        for sample in range(n_samples):
            k_neighbors = X[sample] - X[k_neighbors_idx[sample]]
            temp_matrix = np.dot(k_neighbors, k_neighbors.T)
            # 作者给出的方法，原文写为：Constrained Least Squares Problem
            if self.k > n_features:
                temp_matrix += 1e-3 * np.trace(temp_matrix) * np.eye(self.k)
            pinv_matrix = np.linalg.pinv(temp_matrix)
            W[:, sample] = np.sum(pinv_matrix, axis=1) / np.sum(pinv_matrix)
        # 构造特征分解矩阵，并将非近邻点的权重系数置为0
        I_ = np.eye(n_samples)
        W_ = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(self.k):
                W_[k_neighbors_idx[i, j], i] = W[j, i]
        evd_matrix = np.dot((I_ - W_), (I_ - W_).T)
        # 进行特征值分解并获得样本数据降维后的坐标
        eigenvalues, eigenvectors = np.linalg.eig(evd_matrix)
        picked_index = eigenvalues.real.argsort()[1:self.d+1]
        return eigenvectors[:, picked_index].real


if __name__ == '__main__':
    def main():
        # from sklearn.datasets import load_iris
        # import matplotlib.pyplot as plt
        # iris_data = load_iris()
        # X, Y = iris_data.data, iris_data.target

        from sklearn.datasets import make_s_curve, make_swiss_roll
        import matplotlib.pyplot as plt
        X, Y = make_s_curve(n_samples=500, noise=0.15, random_state=40)
        # X, Y = make_swiss_roll(n_samples=500, noise=0.15, random_state=100)

        lle = LLE(d=2, k=30)
        y = lle.dimension_reduction(X)
        plt.scatter(y[:, 0], y[:, 1], c=Y)
        plt.show()
    main()
