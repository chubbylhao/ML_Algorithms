import numpy as np


class PrincipalComponentAnalysis:
    """ 主成分分析，线性降维方法.
    Attributes:
        d: 降维后的维数/特征数
    """
    def __init__(self, d):
        self.d = d

    def dimension_reduction(self, X):
        """ 线性降维 """
        # 1) 将样本数据中心化
        centered_X = X - np.mean(X, axis=0)
        # 2) 计算样本的的协方差矩阵并对其进行特征值分解得到投影矩阵
        eigenvalues, eigenvectors = np.linalg.eig(np.dot(centered_X.T, centered_X))
        picked_index = eigenvalues.argsort()[::-1][:self.d]
        picked_eigenvectors = eigenvectors.T[picked_index]
        return np.dot(X, picked_eigenvectors.T)


if __name__ == '__main__':
    def main():
        from sklearn.datasets import load_iris
        import matplotlib.pyplot as plt
        iris_data = load_iris()
        X, Y = iris_data.data, iris_data.target

        # from sklearn.datasets import make_s_curve, make_swiss_roll
        # import matplotlib.pyplot as plt
        # X, Y = make_s_curve(n_samples=500, noise=0.15, random_state=40)
        # X, Y = make_swiss_roll(n_samples=500, noise=0.15, random_state=100)

        pca = PrincipalComponentAnalysis(d=2)
        y = pca.dimension_reduction(X)
        plt.scatter(y[:, 0], y[:, 1], c=Y)
        plt.show()
    main()
