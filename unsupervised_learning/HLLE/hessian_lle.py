import numpy as np
from unsupervised_learning.MDS.multiple_dimensional_scaling import MultipleDimensionalScaling


class HessianLLE:
    """ HessianLLE，非线性降维.
    Attributes:
        d: 降维后的维数/特征数
        k: 保持局部Hessian矩阵的邻域样本个数
    """
    def __init__(self, d, k):
        self.k = k
        self.d = d

    def dimension_reduction(self, X):
        # 判断k是否满足最小要求
        if self.k <= self.d * (self.d + 3) / 2:
            raise ValueError('for hessian lle k must be greater than [d * (d + 3) / 2]')
        # 获得每个样本点的k近邻（度量采用欧氏距离的平方）
        dist_matrix = MultipleDimensionalScaling.calculate_distance_matrix(X)
        k_neighbors_idx = np.argsort(dist_matrix)[:, 1:self.k + 1]
        # 计算
        n_samples = X.shape[0]
        dp = self.d * (self.d + 1) // 2
        W = np.zeros((dp * n_samples, n_samples))
        # Yi: 1) 第1列全为1
        Yi = np.ones((self.k, 1 + self.d + dp))
        for i in range(n_samples):
            # Yi: 2) 第2~d+1列
            k_neighbors_idx_i = k_neighbors_idx[i, :]          # 第i个样本点的邻域点索引
            Mi = X[k_neighbors_idx_i]                          # 第i个样本点的邻域点坐标
            Mi -= np.mean(Mi, axis=0)                          # 邻域点特征中心化
            U = np.linalg.svd(Mi, full_matrices=False)[0]      # U的形状为k×min(k,d)，故full_matrices设置为False
            Yi[:, 1:self.d + 1] = U[:, :self.d]                # U的前d列给出了邻域点的切线坐标，而这也构成Yi的第2~d+1列
            # Yi: 3) 第d+2~1+d+d(d+1)/2列
            j = 1 + self.d
            for k in range(self.d):
                Yi[:, j:j + self.d - k] = U[:, k:k + 1] * U[:, k:self.d]    # 哈达玛积（对应元素相乘）
                j += self.d - k
            # 进行QR分解（下面开始懵逼...）
            Q, R = np.linalg.qr(Yi)
            w = np.array(Q[:, self.d + 1:])
            S = np.sum(w, axis=0)
            S[np.where(np.abs(S) < 1e-5)] = 1
            w /= S
            k_neighbors_idx_x, k_neighbors_idx_y = \
                np.meshgrid(k_neighbors_idx_i, k_neighbors_idx_i)
            W[k_neighbors_idx_x, k_neighbors_idx_y] += np.dot(w, w.T)

        _, sig, VT = np.linalg.svd(W, full_matrices=False)
        idx = sig.argsort()[1:self.d + 1]
        Y = VT[idx, :] * np.sqrt(n_samples)

        _, sig, VT = np.linalg.svd(Y, full_matrices=False)
        S = np.matrix(np.diag(sig ** (-1)))
        R = VT.T * S * VT

        return np.array(Y * R).T


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

        hlle = HessianLLE(d=2, k=15)
        y = hlle.dimension_reduction(X)
        plt.scatter(y[:, 0], y[:, 1], c=Y)
        plt.show()
    main()
