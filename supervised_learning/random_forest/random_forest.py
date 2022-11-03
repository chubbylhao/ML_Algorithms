import numpy as np
from utils import get_random_subsets
from supervised_learning.decision_tree import ClassificationTree


class RandomForest:
    """
    The implementation of random forest algorithm.

    Attributes:
        ...
    """
    def __init__(self, n_estimators=100, max_features=None,
                 min_samples_split=2, min_gain=0, max_depth=np.inf):
        self.n_estimators = n_estimators    # 森林中树的总棵树
        self.max_features = max_features    # 一棵树至多判定的特征数量
        self.min_samples_split = min_samples_split    # 构造一棵树至少需要2个样本
        self.min_gain = min_gain
        self.max_depth = max_depth    # 一棵树的最大深度
        # 初始化所有树
        self.trees = []
        for _ in range(n_estimators):
            self.trees.append(
                ClassificationTree(
                    min_samples_split=self.min_samples_split,
                    min_impurity=min_gain,
                    max_depth=self.max_depth))

    def fit(self, X_train, y_train):
        """ Fit all the trees. """
        n_features = np.shape(X_train)[1]
        # 未指定每棵树至多判定的特征数时采用以下算法
        if not self.max_features:
            self.max_features = int(np.sqrt(n_features))
        # 使用自助采样法构建所有树的训练集（有交叉重叠） ———— 随机性1
        subsets = get_random_subsets(X_train, y_train, self.n_estimators)
        # 按循环依次 训练/构建 所有树
        for i in range(self.n_estimators):
            # 使用自助采样法构建所有树（各自）的分类特征 ———— 随机性2
            idx = np.random.choice(range(n_features), self.max_features)
            X_subset, y_subset = subsets[i]
            X_subset = X_subset[:, idx]
            # 训练/构建 树
            self.trees[i].fit(X_subset, y_subset)
            self.trees[i].feature_indices = idx

    def predict(self, X_test):
        """ Use simple voting method to determinate class. """
        # 每一个样本有 n_estimators/len(self.trees) 个预测值
        y_preds = np.zeros((np.shape(X_test)[0], len(self.trees)))
        for i, tree in enumerate(self.trees):
            y_preds[:, i] = tree.predict(X_test[:, tree.feature_indices])
        y_pred = []
        for one_preds in y_preds:
            # 使用简单投票法（票数多的类获胜/少数服从多数）
            y_pred.append(np.bincount(one_preds.astype(int)).argmax())
        return y_pred


if __name__ == '__main__':
    pass
