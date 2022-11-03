import numpy as np
from abc import ABC, abstractmethod
from utils import divide_on_feature, calculate_entropy


class DecisionNode:
    """
    Class that represents a decision node or leaf in the decision tree.

    Attributes:
        feature_i: Feature index which we want to use as the threshold measure.
        threshold: The value that we will compare feature values at feature_i against to determine the prediction.
        value: The class prediction if classification tree, or float value if regression tree.
        true_branch: Next decision node for samples where features value met the threshold.
        false_branch: Next decision node for samples where features value did not meet the threshold.
    """
    def __init__(self, feature_i=None, threshold=None, value=None,
                 true_branch=None, false_branch=None):
        self.feature_i = feature_i          # Index for the feature that is tested
        self.threshold = threshold          # Threshold value for feature
        self.value = value                  # Value if the node is a leaf in the tree
        self.true_branch = true_branch      # 'Left' subtree
        self.false_branch = false_branch    # 'Right' subtree


# Super class of RegressionTree and ClassificationTree
class DecisionTree(ABC):
    """ 回归树和分类树的抽象基类.(仅研究分类树)
    Attributes:
        min_samples_split: 继续划分节点的最小样本数.
        min_impurity: 继续划分节点的最小熵.
        max_depth: 树的最大深度.
    """
    def __init__(self, min_samples_split=2, min_impurity=1e-7, max_depth=np.inf):
        self.min_samples_split = min_samples_split
        self.min_impurity = min_impurity
        self.max_depth = max_depth
        self.root = None

    @abstractmethod
    def _impurity_calculation(self, y, y1, y2): pass    # 计算信息增益

    @abstractmethod
    def _leaf_value_calculation(self, y): pass    # 返回实例数最多的类的标记

    def _build_tree(self, X, y, current_depth=0):
        """ Recursive method which builds out the decision tree and splits X and respective y
        on the feature of X which (based on impurity) best separates the data. """
        largest_impurity = 0
        best_criteria = None    # Feature index and threshold
        best_sets = None        # Subsets of the data
        # Check if expansion of y is needed
        if len(np.shape(y)) == 1:
            y = np.expand_dims(y, axis=1)
        # Add y as last column of X
        Xy = np.concatenate((X, y), axis=1)
        n_samples, n_features = np.shape(X)
        if n_samples >= self.min_samples_split and current_depth <= self.max_depth:
            # Calculate the impurity for each feature
            for feature_i in range(n_features):
                # All values of feature_i
                feature_values = np.expand_dims(X[:, feature_i], axis=1)
                unique_values = np.unique(feature_values)
                # Iterate through all unique values of feature column i and calculate the impurity
                for threshold in unique_values:
                    # Divide X and y depending on if the feature value of X at index feature_i
                    # meets the threshold
                    Xy1, Xy2 = divide_on_feature(Xy, feature_i, threshold)
                    if len(Xy1) > 0 and len(Xy2) > 0:
                        # Select the y-values of the two sets
                        y1 = Xy1[:, n_features:]
                        y2 = Xy2[:, n_features:]
                        # Calculate impurity
                        impurity = self._impurity_calculation(y, y1, y2)
                        # If this threshold resulted in a higher information gain than previously
                        # recorded save the threshold value and the feature index
                        if impurity > largest_impurity:
                            largest_impurity = impurity
                            best_criteria = {"feature_i": feature_i, "threshold": threshold}
                            best_sets = {
                                "leftX": Xy1[:, :n_features],   # X of left subtree
                                "lefty": Xy1[:, n_features:],   # y of left subtree
                                "rightX": Xy2[:, :n_features],  # X of right subtree
                                "righty": Xy2[:, n_features:]   # y of right subtree
                                }
        if largest_impurity > self.min_impurity:
            # Build subtrees for the right and left branches
            true_branch = self._build_tree(best_sets["leftX"], best_sets["lefty"], current_depth + 1)
            false_branch = self._build_tree(best_sets["rightX"], best_sets["righty"], current_depth + 1)
            return DecisionNode(feature_i=best_criteria["feature_i"], threshold=best_criteria["threshold"],
                                true_branch=true_branch, false_branch=false_branch)
        # We're at leaf => determine value
        leaf_value = self._leaf_value_calculation(y)
        return DecisionNode(value=leaf_value)

    def fit(self, X, y):
        """ Build decision tree. """
        self.root = self._build_tree(X, y)

    def predict_value(self, x, tree=None):
        """ Do a recursive search down the tree and make a prediction of the data sample by the
            value of the leaf that we end up at. """
        if tree is None:
            tree = self.root
        # If we have a value (i.e. we're at a leaf) => return value as the prediction
        if tree.value is not None:
            return tree.value
        # Choose the feature that we will test
        feature_value = x[tree.feature_i]
        # Determine if we will follow left or right branch
        branch = tree.false_branch if feature_value < tree.threshold else tree.true_branch
        # Test subtree
        return self.predict_value(x, branch)

    def predict(self, X):
        """ Classify samples one by one and return the set of labels. """
        y_pred = [self.predict_value(x) for x in X]
        return y_pred


class ClassificationTree(DecisionTree):
    """ 分类树，继承自抽象基类. """
    def _impurity_calculation(self, y, y1, y2):
        """ 计算信息增益. """
        p = len(y1) / len(y)
        info_gain = \
            calculate_entropy(y) - \
            p * calculate_entropy(y1) - \
            (1 - p) * calculate_entropy(y2)
        return info_gain

    def _leaf_value_calculation(self, y):
        """ 返回实例数最多的类的标记. """
        most_common, max_count = None, 0
        for label in np.unique(y):
            count = len(y[y == label])
            if count > max_count:
                most_common, max_count = label, count
        return most_common


if __name__ == '__main__':
    def main():
        from sklearn import datasets
        from utils import train_test_split, accuracy_score, Plot
        data = datasets.load_iris()
        X, y = data.data, data.target
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)
        clf = ClassificationTree()
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        Plot().plot_in_2d(X_test, y_pred, title="Decision Tree", accuracy=accuracy,
                          legend_labels=data.target_names)
    main()
