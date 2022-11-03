import numpy as np
from utils.kernels import linear_kernel, polynomial_kernel, rbf_kernel
import cvxopt


class SupportVectorMachine:
    """ Uses cvxopt to solve the quadratic optimization problem.
    Attributes:
        C: 惩罚因子
        kernel: 核函数
        gamma: rbf核函数γ
        power: 多项式核函数的次数
        coeff: 多项式核函数的偏置
    """
    def __init__(self, C=1, kernel=rbf_kernel, gamma=1e2, power=4, coeff=1):
        self.C = C
        self.gamma = gamma
        self.power = power
        self.coeff = coeff
        self.kernel = kernel(gamma=self.gamma, power=self.power, coeff=self.coeff)
        self.lagrange_multipliers = None     # 非零拉格朗日乘子
        self.support_vectors = None          # 支持向量
        self.support_vector_labels = None    # 支持向量的标签
        self.bias = None                     # 决策函数的偏置

    def fit(self, X_train, y_train):
        """ 得到决策函数. """
        n_samples, n_features = np.shape(X_train)
        # 将所有样本点在高维空间内两两作内积（通过核函数实现）
        kernel_matrix = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                kernel_matrix[i, j] = self.kernel(X_train[i], X_train[j])
        # 定义二次规划问题
        P = cvxopt.matrix(np.outer(y_train, y_train) * kernel_matrix, tc='d')
        q = cvxopt.matrix(- np.ones(n_samples))
        A = cvxopt.matrix(y_train, (1, n_samples), tc='d')
        b = cvxopt.matrix(0, tc='d')
        if not self.C:
            G = cvxopt.matrix(- np.identity(n_samples))
            h = cvxopt.matrix(np.zeros(n_samples))
        else:
            G_max = - np.identity(n_samples)
            G_min = np.identity(n_samples)
            G = cvxopt.matrix(np.vstack((G_max, G_min)))
            h_max = cvxopt.matrix(np.zeros(n_samples))
            h_min = cvxopt.matrix(np.ones(n_samples) * self.C)
            h = cvxopt.matrix(np.vstack((h_max, h_min)))
        # 求解
        minimization = cvxopt.solvers.qp(P, q, G, h, A, b)
        # 取出拉格朗日乘子
        lagrange_multipliers = np.ravel(minimization['x'])
        # 取出支持向量及其对应的标签（由非零拉格朗日乘子获得）
        non_zero_idx = lagrange_multipliers > 1e-7
        self.lagrange_multipliers = lagrange_multipliers[non_zero_idx]
        self.support_vectors = X_train[non_zero_idx]
        self.support_vector_labels = y_train[non_zero_idx]
        # 计算决策函数的偏置
        self.bias = self.support_vector_labels[0]
        for i, non_zero_lagrange_multipliers in enumerate(self.lagrange_multipliers):
            self.bias -= non_zero_lagrange_multipliers * self.support_vector_labels[i] \
                         * self.kernel(self.support_vectors[i], self.support_vectors[0])

    def predict(self, X_test):
        """ 使用决策函数预测. """
        y_pred = []
        # 计算线性项
        for x in X_test:
            linear_term = 0
            for i, non_zero_lagrange_multipliers in enumerate(self.lagrange_multipliers):
                linear_term += non_zero_lagrange_multipliers * self.support_vector_labels[i] \
                                  * self.kernel(self.support_vectors[i], x)
            # 线性项+偏置项，再以符号函数预测
            y_pred.append(np.sign(linear_term + self.bias))
        return np.array(y_pred)


if __name__ == '__main__':
    def main():
        from sklearn import datasets
        from utils import train_test_split, normalize, accuracy_score, Plot
        cvxopt.solvers.options['show_progress'] = False
        data = datasets.load_iris()
        X = normalize(data.data[data.target != 0])
        y = data.target[data.target != 0]
        y[y == 1], y[y == 2] = -1, 1
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
        # 使用线性核
        # clf = SupportVectorMachine(kernel=linear_kernel)
        # 使用多项式核
        # clf = SupportVectorMachine(kernel=polynomial_kernel)
        # 使用rbf核
        clf = SupportVectorMachine(kernel=rbf_kernel)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        Plot().plot_in_2d(X_test, y_pred, title="Support Vector Machine", accuracy=accuracy)
    main()
