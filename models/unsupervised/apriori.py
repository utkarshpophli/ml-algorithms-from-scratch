from itertools import combinations


class Apriori:
    def __init__(self, min_support=0.5, min_confidence=0.7):
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.frequent_itemsets = []
        self.support_data = {}

    def fit(self, transactions):
        self.transactions = list(map(set, transactions))
        self.num_transactions = len(transactions)
        self.itemsets = self.create_initial_itemsets()
        self.frequent_itemsets = [self.find_frequent_itemsets(self.itemsets)]

        k = 2
        while True:
            candidate_itemsets = self.apriori_gen(self.frequent_itemsets[-1], k)
            frequent_itemsets_k = self.find_frequent_itemsets(candidate_itemsets)
            if not frequent_itemsets_k:
                break
            self.frequent_itemsets.append(frequent_itemsets_k)
            k += 1

        self.generate_association_rules()

    def create_initial_itemsets(self):
        itemsets = []
        for transaction in self.transactions:
            for item in transaction:
                if frozenset([item]) not in itemsets:
                    itemsets.append(frozenset([item]))
        itemsets.sort()
        return itemsets

    def find_frequent_itemsets(self, itemsets):
        itemset_counts = {}
        for transaction in self.transactions:
            for itemset in itemsets:
                if itemset.issubset(transaction):
                    if itemset not in itemset_counts:
                        itemset_counts[itemset] = 1
                    else:
                        itemset_counts[itemset] += 1

        num_transactions = float(len(self.transactions))
        frequent_itemsets = []
        for itemset, count in itemset_counts.items():
            support = count / num_transactions
            if support >= self.min_support:
                frequent_itemsets.append(itemset)
                self.support_data[itemset] = support
        return frequent_itemsets

    def apriori_gen(self, itemsets, k):
        candidates = []
        len_itemsets = len(itemsets)
        for i in range(len_itemsets):
            for j in range(i + 1, len_itemsets):
                L1 = list(itemsets[i])[:k-2]
                L2 = list(itemsets[j])[:k-2]
                L1.sort()
                L2.sort()
                if L1 == L2:
                    candidates.append(itemsets[i] | itemsets[j])
        return candidates

    def generate_association_rules(self):
        self.rules = []
        for itemsets in self.frequent_itemsets[1:]:
            for freq_set in itemsets:
                H1 = [frozenset([item]) for item in freq_set]
                if len(freq_set) > 1:
                    self.rules_from_conseq(freq_set, H1)

    def rules_from_conseq(self, freq_set, H):
        m = len(H[0])
        if len(freq_set) > (m + 1):
            Hmp1 = self.apriori_gen(H, m + 1)
            Hmp1 = self.calc_confidence(freq_set, Hmp1)
            if Hmp1:
                self.rules_from_conseq(freq_set, Hmp1)

    def calc_confidence(self, freq_set, H):
        pruned_H = []
        for conseq in H:
            if freq_set - conseq in self.support_data:
                conf = self.support_data[freq_set] / self.support_data[freq_set - conseq]
                if conf >= self.min_confidence:
                    self.rules.append((list(freq_set - conseq), list(conseq), conf))  # Fix here
                    pruned_H.append(conseq)
            else:
                print(f"Missing support data for {freq_set - conseq}")
        return pruned_H


    def get_frequent_itemsets(self):
        return [(list(itemset), self.support_data[itemset]) for itemsets in self.frequent_itemsets for itemset in itemsets]

    def get_rules(self):
        return [(list(rule[0]), list(rule[1]), rule[2]) for rule in self.rules]
