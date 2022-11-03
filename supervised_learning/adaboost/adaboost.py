import numpy as np


class DecisionStump:
    """ Use decision stump as weak classifier to
    implement adaboost algorithm. """
    def __init__(self):
        self.polarity = 1
        self.feature_idx = None   # 分类特征，用于构造决策树桩的判定属性
        self.threshold = None     # 分类阈值，小于阈值的样本判定为-1
        self.alpha = None         # 弱分类器的权值


class AdaBoost:
    """
    Attributes:
        n_clfs: The number of weak classifiers will be used.
    """
    def __init__(self, n_clfs=5):
        self.n_clfs = n_clfs
        self.clfs = []

    def fit(self, X_train, y_train):
        """ Fit each weak classifier. """
        n_samples, n_features = np.shape(X_train)

        # (1) 初始化数据的权值分布
        w = np.full(n_samples, (1 / n_samples))

        # (2) 选择“桩节点”的属性和阈值、计算弱分类器的权值、更新数据的权值
        for _ in range(self.n_clfs):
            clf = DecisionStump()    # 生成一棵“空白的”决策树桩
            min_error = np.inf
            # 1) 为决策树桩的“桩节点”选择属性
            for i in range(n_features):
                feature_values = np.unique(X_train[:, i])
                # 2) 为决策树桩的“桩节点”选择阈值
                for threshold in feature_values:
                    p = 1
                    prediction = np.ones(np.shape(y_train))
                    prediction[X_train[:, i] < threshold] = -1
                    error = np.sum(w[y_train != prediction])
                    if error > 0.5:
                        error = 1 - error
                        p = -1
                    if error < min_error:
                        clf.polarity = p
                        clf.feature_idx = i
                        clf.threshold = threshold
                        min_error = error
            # 3) 计算弱分类器的权值
            clf.alpha = 0.5 * np.log((1 - min_error) / (min_error + 1e-10))
            self.clfs.append(clf)
            # 4) 更新数据的权值
            predictions = np.ones(np.shape(y_train))
            predictions[clf.polarity * X_train[:, clf.feature_idx] <
                        clf.polarity * clf.threshold] = -1
            w *= np.exp(- clf.alpha * y_train * predictions)
            w /= np.sum(w)    # 归一化权值

    def predict(self, X_test):
        """ Linear combination of all weak classifiers. """
        y_pred = np.zeros(np.shape(X_test)[0])
        #  (3) 构建基本分类器的线性组合
        for clf in self.clfs:
            predictions = np.ones(np.shape(y_pred))
            predictions[clf.polarity * X_test[:, clf.feature_idx] <
                        clf.polarity * clf.threshold] = -1
            y_pred += clf.alpha * predictions
        return np.sign(y_pred)


if __name__ == '__main__':
    pass
