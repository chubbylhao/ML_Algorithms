import numpy as np
from utils import calculate_covariance_matrix


class GaussianMixtureModel:
    """ 高斯混合模型，使用EM算法进行参数估计.
    Attributes:
        k_clusters: 聚类的数量，也就是分模型的数量
        max_iterations: EM算法的最大迭代次数
        tolerance: EM算法停止迭代的判断收敛的阈值
    """
    def __init__(self, k_clusters=2, max_iterations=500, tolerance=1e-6):
        self.k_clusters = k_clusters
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        # 模型参数：权重系数，均值，协方差
        self.priors = None
        self.mean_cov = []
        # 模型对观测数据的响应度
        self.responsivity = None
        self.responsivities = []
        # 样本所属类别
        self.sample_assignments = None
        # 迭代的次数
        self.n_iterations = 0

    def _init_random_gaussian(self, X):
        """ 初始化模型参数 """
        # 1) 初始化各分模型的权重系数（常采用均匀初始化方法）
        self.priors = np.full(self.k_clusters, (1 / self.k_clusters))
        # 2) 初始化各分模型的均值和协方差
        for i in range(self.k_clusters):
            mean_cov = {'mean': X[np.random.choice(range(X.shape[0]))],
                        'cov': calculate_covariance_matrix(X)}
            self.mean_cov.append(mean_cov)

    @staticmethod
    def _multivariate_gaussian(X, mean_cov):
        """ 计算样本在单个”多元高斯模型“下的概率 """
        d = X.shape[1]    # 元数/变量数/特征数
        mean, cov = mean_cov['mean'], mean_cov['cov']    # 均值和协方差
        coeff = 1 / (np.power(2*np.pi, d/2) * np.sqrt(np.linalg.det(cov)))    # 系数项
        likelihoods = np.zeros(X.shape[0])
        # 遍历每一个样本
        for i, x in enumerate(X):
            exponent = np.exp(-0.5 * (x - mean).T.dot(np.linalg.pinv(cov)).dot((x - mean)))    # 指数项，求广义逆
            likelihoods[i] = coeff * exponent
        return likelihoods

    def _get_likelihoods(self, X):
        """ 计算样本在所有”多元高斯模型“下的概率 """
        likelihoods = np.zeros((X.shape[0], self.k_clusters))
        # 遍历每一个分模型
        for i in range(self.k_clusters):
            likelihoods[:, i] = GaussianMixtureModel._multivariate_gaussian(X, self.mean_cov[i])
        return likelihoods

    # 以下使用EM算法来估计GMM的参数
    # -------------------------

    # ----------E步------------
    def _expectation(self, X):
        """ 计算隐变量的概率分布/模型对观测数据的响应度 """
        priors_times_likelihoods = self.priors * self._get_likelihoods(X)
        sum_of_priors_times_likelihoods = np.expand_dims(priors_times_likelihoods.sum(axis=1), axis=1)
        self.responsivity = priors_times_likelihoods / sum_of_priors_times_likelihoods
        # 记录最大概率的类作为样本所属的类别
        self.sample_assignments = self.responsivity.argmax(axis=1)
        # 记录每次迭代每个样本的极大概率
        self.responsivities.append(self.responsivity.max(axis=1))

    # ----------M步------------
    def _maximization(self, X):
        """ 计算新一轮模型迭代的参数 """
        for i in range(self.k_clusters):
            responsivity = np.expand_dims(self.responsivity[:, i], axis=1)
            # 更新均值
            mean = (responsivity * X).sum(axis=0) / responsivity.sum()
            # 更新协方差
            cov = np.dot((X-mean).T, (X-mean)*responsivity) / responsivity.sum()
            # 更新权重系数
            self.priors[i] = responsivity.sum() / X.shape[0]
            # 更新
            self.mean_cov[i]['mean'], self.mean_cov[i]['cov'] = mean, cov

    # -------------------------
    # 以上使用EM算法来估计GMM的参数

    def predict(self, X):
        # 1) 初始化
        self._init_random_gaussian(X)
        # 2) EM
        for _ in range(self.max_iterations):
            self._expectation(X)     # E步
            self._maximization(X)    # M步
            # 迭代停止条件
            self.n_iterations += 1
            if self.n_iterations <= 1:
                continue
            if np.linalg.norm(self.responsivities[-1] -
                              self.responsivities[-2], 1) < self.tolerance:
                break
        return self.sample_assignments


if __name__ == '__main__':
    def main():
        from sklearn import datasets
        from utils import Plot
        X, y = datasets.make_blobs()
        clf = GaussianMixtureModel(k_clusters=3)
        y_pred = clf.predict(X)
        print('迭代次数:', clf.n_iterations)
        Plot().plot_in_2d(X, y_pred, title="GMM Clustering")
        Plot().plot_in_2d(X, y, title="Actual Clustering")
    main()
