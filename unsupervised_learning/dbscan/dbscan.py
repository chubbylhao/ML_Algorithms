import numpy as np
from utils import euclidean_distance


class DBSCAN:
    """ 具有噪声的基于密度的聚类算法.
    Density-Based Spatial Clustering of Applications with Noise.

    Attributes:
        epsilon: 邻域半径
        min_pts: 邻域内样本数量的阈值
    """
    def __init__(self, epsilon, min_pts):
        self.epsilon = epsilon
        self.min_pts = min_pts
        self.X = None
        self.clusters = []
        self.visited_samples = []
        self.neighbors = {}

    def _get_neighbors(self, sample_i):
        """ 返回邻域内的所有点（不包括核心对象）"""
        neighbors = []
        # 迭代核心对象外的所有点
        for i, sample in enumerate(self.X):
            if i == sample_i:
                continue
            distance = euclidean_distance(self.X[sample_i], sample)
            if distance < self.epsilon:
                neighbors.append(i)
        return np.array(neighbors)

    def _expand_cluster(self, sample_i, neighbors):
        """ 以递归方式聚类 """
        cluster = [sample_i]
        for neighbor_i in neighbors:
            # 不再迭代已被访问过的点，以避免递归陷入无限循环
            if neighbor_i not in self.visited_samples:
                self.visited_samples.append(neighbor_i)
                self.neighbors[neighbor_i] = self._get_neighbors(neighbor_i)
                if len(self.neighbors[neighbor_i]) >= self.min_pts:
                    expanded_cluster = self._expand_cluster(neighbor_i, self.neighbors[neighbor_i])
                    cluster += expanded_cluster
                else:
                    cluster.append(neighbor_i)
        return cluster

    def _get_cluster_labels(self):
        """ 为每一个样本点打上分类标签 """
        labels = np.full(self.X.shape[0], len(self.clusters))
        for cluster_i, cluster in enumerate(self.clusters):
            for sample_i in cluster:
                labels[sample_i] = cluster_i
        return labels

    def predict(self, X):
        """ 聚类 """
        self.X = X
        n_samples = self.X.shape[0]
        for sample_i in range(n_samples):
            if sample_i in self.visited_samples:
                continue
            self.neighbors[sample_i] = self._get_neighbors(sample_i)
            if len(self.neighbors[sample_i]) >= self.min_pts:
                self.visited_samples.append(sample_i)
                new_cluster = self._expand_cluster(sample_i, self.neighbors[sample_i])
                self.clusters.append(new_cluster)
        return self._get_cluster_labels()


if __name__ == '__main__':
    def main():
        from sklearn import datasets
        from utils import Plot
        # 样本数量和噪声大小都会对聚类结果产生影响
        X, y = datasets.make_moons(n_samples=300, noise=0.08, shuffle=False)
        clf = DBSCAN(epsilon=0.2, min_pts=5)
        y_pred = clf.predict(X)
        # 对比原类和聚类
        Plot().plot_in_2d(X, y_pred, title="DBSCAN")
        Plot().plot_in_2d(X, y, title="Actual Clustering")
    main()
