from niaarm import get_rules, Dataset
import numpy as np


class OptimizationARM:
    """
    This class implements the optimization-based ARM approaches (BAT, GWO, SC, FSS), using NiaARM and NiaPY packages
    """

    def __init__(self, algorithm, max_evals=50000):
        self.max_evals = max_evals
        self.algorithm = algorithm
        self.metrics = ['support', 'confidence']

    def learn_rules(self, transactions):
        dataset = Dataset(transactions)
        rules, run_time = get_rules(dataset, algorithm=self.algorithm, metrics=self.metrics,
                                    max_evals=self.max_evals, logging=False)
        if len(rules) == 0:
            return [0, run_time, 0, 0, 0, 0], rules
        coverage = self.calculate_coverage(rules, transactions)
        support, confidence = rules.mean("support"), rules.mean("confidence")
        rules = self.reformat_rules(rules)
        return [len(rules), run_time, support, confidence, coverage], rules

    @staticmethod
    def calculate_coverage(rules, dataset):
        rule_coverage = np.zeros(len(dataset))

        for index, row in dataset.iterrows():
            for rule in rules:
                covered = True
                for item in rule.antecedent:
                    if item.categories:
                        if item.categories[0] not in list(row):
                            covered = False
                            break
                    else:
                        covered = False
                        for key, value in row.items():
                            if item.name == key:
                                if item.min_val <= value <= item.max_val:
                                    covered = True
                                    break
                if covered:
                    rule_coverage[index] = 1
                    break

        return sum(rule_coverage) / len(dataset)

    def reformat_rules(self, rules):
        reformatted_rules = []
        for rule in rules:
            antecedents = []
            consequent = []
            for item in rule.antecedent:
                antecedents += item.categories
            for item in rule.consequent:
                consequent += item.categories
            reformatted_rules.append({"antecedents": antecedents, "consequent": consequent})
        return reformatted_rules
