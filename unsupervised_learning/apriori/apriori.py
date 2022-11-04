import numpy as np
import itertools


class Apriori:
    """ 在交易 列表/数据库 中寻找频繁项集，并求解它们间的关联规则.
    Attributes:
        support_thresh: 支持度阈值
        confidence_thresh: 置信度阈值
        transactions: 交易 列表/数据库
        frequent_item_sets: 频繁项集
    """
    def __init__(self, support_thresh, confidence_thresh):
        self.support_thresh = support_thresh
        self.confidence_thresh = confidence_thresh
        self.transactions = None
        self.frequent_item_sets = None

    """ 1) 寻找频繁项集 """
    @staticmethod
    def _transaction_contains_items(transaction, items):
        # 1项集不可迭代，故单独处理
        if isinstance(items, int):
            return items in transaction
        # >=2项集是个列表，故迭代处理
        for item in items:
            # 只要有一个元素未包含在原始的transaction中，即为假
            if item not in transaction:
                return False
        return True

    def _calculate_support(self, item_set):
        count = 0
        for transaction in self.transactions:
            if Apriori._transaction_contains_items(transaction, item_set):
                count += 1
        support = count / len(self.transactions)
        return support

    def _get_frequent_item_sets(self, candidates):
        frequent = []
        for item_set in candidates:
            support = self._calculate_support(item_set)
            if support >= self.support_thresh:
                frequent.append(item_set)
        return frequent

    def _has_infrequent_item_sets(self, candidate):
        k = len(candidate)
        subsets = list(itertools.combinations(candidate, k - 1))
        for t in subsets:
            subset = list(t) if len(t) > 1 else t[0]
            if subset not in self.frequent_item_sets[-1]:
                return True
        return False

    def _generate_candidates(self, frequent_item_set):
        candidates = []
        # 遍历两两组合的所有情况
        for item_set_1 in frequent_item_set:
            for item_set_2 in frequent_item_set:
                valid = False
                single_item = isinstance(item_set_1, int)
                # 针对1项集
                if single_item and item_set_1 < item_set_2:
                    valid = True
                # 针对>=2项集
                elif not single_item and np.array_equal(item_set_1[:-1], item_set_2[:-1]) and item_set_1[-1] < item_set_2[-1]:
                    valid = True
                if valid:
                    if single_item:
                        candidate = [item_set_1, item_set_2]
                    else:
                        candidate = item_set_1 + [item_set_2[-1]]
                    infrequent = self._has_infrequent_item_sets(candidate)
                    if not infrequent:
                        candidates.append(candidate)
        return candidates

    def find_frequent_item_sets(self, transactions):
        """ 寻找交易 列表/数据库 中的频繁项集 """
        self.transactions = transactions
        unique_items = set(item for transaction in transactions for item in transaction)
        # 1) 从1项集开始
        self.frequent_item_sets = [self._get_frequent_item_sets(unique_items)]
        # 2) 挖掘/搜索 >=2项集
        while True:
            candidates = self._generate_candidates(self.frequent_item_sets[-1])
            frequent_item_sets = self._get_frequent_item_sets(candidates)
            if not frequent_item_sets:
                break    # 终止条件
            self.frequent_item_sets.append(frequent_item_sets)
        return [item_set for sublist in self.frequent_item_sets for item_set in sublist]

    """ 2) 寻找频繁项集间的关联规则 """
    def find_association_rules(self, transactions):
        pass


if __name__ == '__main__':
    def main():
        transactions = np.array([[1, 2, 3, 4], [1, 2, 4], [1, 2], [2, 3, 4], [2, 3], [3, 4], [2, 4]])
        print("+-------------+")
        print("|   Apriori   |")
        print("+-------------+")
        support_thresh, confidence_thresh = 0.25, 0.8
        print("Minimum Support: %.2f" % support_thresh)
        print("Minimum Confidence: %s" % confidence_thresh)
        print("Transactions:")
        for transaction in transactions:
            print("\t%s" % transaction)
        apriori = Apriori(support_thresh=support_thresh, confidence_thresh=confidence_thresh)
        frequent_item_sets = apriori.find_frequent_item_sets(transactions)
        print("Frequent Item sets:\n\t%s" % frequent_item_sets)
    main()
