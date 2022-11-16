import numpy as np
from utils.kernels import rbf_kernel, polynomial_kernel, linear_kernel


class KernelizedPCA:
    """ KPCA，核化的PCA，非线性降维.
    Attributes:
        d: 降维后的维数/特征数
        kernel: 使用的核函数
        gamma: 高斯核参数
        power & coeff: 多项式核参数
    """
    def __init__(self, d, kernel=rbf_kernel, gamma=0.3, power=4, coeff=1):
        self.d = d
        self.kernel = kernel(gamma=gamma, power=power, coeff=coeff)

    def dimension_reduction(self, X):
        """ 核方法非线性降维 """
        n_samples, n_features = np.shape(X)
        # 生成核矩阵
        kernel_matrix = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                kernel_matrix[i, j] = self.kernel(X[i], X[j])
        kernel_matrix[kernel_matrix < 0] = 0    # 保证数值稳定性
        # 对核矩阵进行特征值分解（和普通PCA方法一致）
        eigenvalues, eigenvectors = np.linalg.eig(kernel_matrix)
        picked_index = eigenvalues.argsort()[::-1][:self.d].real    # 取real消除警告（有警告也无所谓）
        picked_eigenvectors = eigenvectors.T[picked_index].real     # 因为结果均是虚部为0的复数
        return np.dot(picked_eigenvectors, kernel_matrix).T


if __name__ == '__main__':
    def main():
        # from sklearn.datasets import load_iris
        from sklearn.datasets import make_s_curve
        import matplotlib.pyplot as plt
        # iris_data = load_iris()
        # X, Y = iris_data.data, iris_data.target
        X, Y = make_s_curve(n_samples=1000, noise=0.15)
        kpca = KernelizedPCA(d=2)
        y = kpca.dimension_reduction(X)
        plt.scatter(y[:, 0], y[:, 1], c=Y)
        plt.show()
    main()
