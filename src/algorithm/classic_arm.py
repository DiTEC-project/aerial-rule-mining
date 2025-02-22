import numpy as np
import time
from src.util.rule_quality import *
from src.util.ucimlrepo import *
from mlxtend.frequent_patterns import association_rules
from mlxtend.frequent_patterns import fpgrowth, hmine
from joblib import Parallel, delayed


class ClassicARM:
    """
    This class implements classical ARM approaches (FP-Growth and HMine) using the Mlxtend Python package
    """

    def __init__(self, min_support=0.5, min_confidence=0.8, algorithm="fpgrowth"):
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.algorithm = algorithm

    def mine_rules(self, dataset, antecedents=2, frequent_items=False, rule_stats=True):
        one_hot_encoded_input = one_hot_encoding(dataset)
        start = time.time()

        if self.algorithm == "fpgrowth":
            frq_items = fpgrowth(one_hot_encoded_input, self.min_support, use_colnames=True, max_len=antecedents + 1)
        else:
            frq_items = hmine(one_hot_encoded_input, self.min_support, use_colnames=True, max_len=antecedents + 1)
        if len(frq_items) == 0:
            return None, None

        if frequent_items:
            exec_time = (time.time() - start)
            return frq_items, exec_time

        rules = association_rules(frq_items, metric="confidence", min_threshold=self.min_confidence)
        exec_time = time.time() - start

        if len(rules) == 0:
            return None, None

        if not rule_stats:
            reformatted_rules = self.reformat_rules(rules)
            return reformatted_rules, exec_time

        coverage = self.calculate_dataset_coverage(rules, dataset)
        reformatted_rules = self.reformat_rules(rules)
        stats = calculate_average_rule_quality(reformatted_rules)
        return [len(rules), exec_time, stats['support'], stats["confidence"], coverage,
                stats["zhangs_metric"]], reformatted_rules

    def reformat_rules(self, rules):
        reformatted_rules = []
        for rule_index, rule in rules.iterrows():
            new_rule = {'antecedents': list(rule['antecedents']), 'consequent': list(rule['consequents']),
                        'support': rule['support'], 'confidence': rule['confidence'],
                        'zhangs_metric': rule["zhangs_metric"]}
            reformatted_rules.append(new_rule)
        return reformatted_rules

    @staticmethod
    def calculate_dataset_coverage(rules, dataset):
        dataset_sets = [set(transaction) for transaction in dataset]
        coverage_array = np.zeros(len(dataset), dtype=bool)

        for rule in rules.itertuples(index=False):
            antecedents = set(rule.antecedents)
            coverage_array |= np.array([antecedents.issubset(transaction) for transaction in dataset_sets])

        coverage = np.sum(coverage_array) / len(dataset)
        return coverage

    def calculate_stats(self, rules, exec_time, dataset):
        """
        Optimized function for calculating rule quality metrics including coverage,
        interestingness, and Yule's Q.
        """
        # Precompute dataset as a list of sets for efficient membership checking
        dataset_sets = [set(transaction) for transaction in dataset]
        num_transactions = len(dataset)

        # Function to process each rule and return its stats and local coverage
        def process_rule(index, row):
            local_coverage = np.zeros(num_transactions)

            antecedents = set(row['antecedents'])

            for transaction_index, transaction in enumerate(dataset_sets):
                if antecedents.issubset(transaction):
                    local_coverage[transaction_index] = 1  # Mark transaction as covered

            # Calculate metrics
            # row["interestingness"] = calculate_interestingness(
            #     row['confidence'], row['support'], row['consequent support'], num_transactions
            # )
            # row["yulesq"] = calculate_yulesq(
            #     ant_and_cons_count, no_ant_no_cons_count, cons_no_ant_count, ant_no_cons_count
            # )

            return row, local_coverage

        # Parallel processing of rules
        results = Parallel(n_jobs=-1)(
            delayed(process_rule)(index, row) for index, row in rules.iterrows()
        )

        # Extract rule stats and combine coverage
        rule_stats, coverages = zip(*results)
        rule_coverage = np.maximum.reduce(coverages)  # Combine local coverages using element-wise maximum

        # Calculate overall statistics
        stats = calculate_average_rule_quality(rule_stats)
        stats["coverage"] = sum(rule_coverage) / num_transactions

        return [len(rules), exec_time, stats['support'], stats["confidence"], stats["coverage"], stats["zhangs_metric"]]
