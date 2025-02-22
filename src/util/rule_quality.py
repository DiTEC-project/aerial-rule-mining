"""
This script implements helper functions relevant to logical association rule quality metrics
"""

import pandas as pd
from statistics import mean


def calculate_average_support(itemsets, transactions):
    """
    Calculate average support value for the items in the given itemsets, in the transactions
    This was necessary, since the BRL algorithm works with itemsets rather than rules
    :param itemsets:
    :param transactions:
    :return:
    """
    if len(itemsets) == 0:
        return 0

    support = 0
    for itemset in itemsets:
        occurrences = 0
        for transaction_index in range(len(transactions)):
            exists = True
            for item in itemset:
                if transactions[item].iloc[transaction_index] != 1:
                    exists = False
                    break
            if exists:
                occurrences += 1
        support += (occurrences / len(transactions))

    support = support / len(itemsets)
    return support


def calculate_average_rule_quality(rules):
    stats = []
    for rule in rules:
        stats.append([
            rule["support"], rule["confidence"], rule["zhangs_metric"]
        ])

    stats = pd.DataFrame(stats).mean()
    stats = {
        "support": stats[0],
        "confidence": stats[1],
        "zhangs_metric": stats[2]
    }
    return stats


def calculate_interestingness(confidence, support, rhs_support, input_length):
    """
    calculate interestingness rule quality criterion for a single rule
    :param confidence:
    :param support:
    :param rhs_support: consequent support
    :param input_length: number of transactions
    :return:
    """
    # formula taken from NiaPy 'rule.py'
    return confidence * (support / rhs_support) * (1 - (support / input_length))


def calculate_yulesq(full_count, not_ant_not_con, con_not_ant, ant_not_con):
    """
    calculate yules'q rule quality criterion for a single rule
    :param full_count: number of transactions that contain both antecedent and consequent side of a rule
    :param not_ant_not_con: number of transactions that does not contain neither antecedent nor consequent
    :param con_not_ant: number of transactions that contain consequent side but not antecedent
    :param ant_not_con: number of transactions that contain antecedent side but not consequent
    :return:
    """
    # formula taken from NiaPy 'rule.py'
    ad = full_count * not_ant_not_con
    bc = con_not_ant * ant_not_con
    yulesq = (ad - bc) / (ad + bc + 2.220446049250313e-16)
    return yulesq


def calculate_lift(support, confidence):
    return confidence / support


def calculate_conviction(support, confidence):
    return (1 - support) / (1 - confidence + 2.220446049250313e-16)


def calculate_zhangs_metric(support, support_ant, support_cons):
    """
    Taken from NiaARM's rule.py
    :param support_cons:
    :param support_ant:
    :param support:
    :return:
    """
    numerator = support - support_ant * support_cons
    denominator = (
            max(support * (1 - support_ant), support_ant * (support_cons - support))
            + 2.220446049250313e-16
    )
    return numerator / denominator


def calculate_rule_overlap(results):
    overlap_list = {}
    for dataset in results:
        for algorithm in results[dataset]:
            overlap_list[algorithm] = {}
            for algorithm2 in results[dataset]:
                if algorithm == algorithm2:
                    continue
                match = 0
                if results[dataset][algorithm]['rules'] and len(results[dataset][algorithm]['rules']) > 0 and \
                        results[dataset][algorithm2]['rules'] and len(results[dataset][algorithm2]['rules']) > 0:
                    for rule in results[dataset][algorithm]['rules']:
                        for rule2 in results[dataset][algorithm2]['rules']:
                            if set(rule["antecedents"]) == set(rule2["antecedents"]) and \
                                    set(rule["consequent"]) == set(rule2["consequent"]):
                                match += 1
                    overlap_list[algorithm][algorithm2] = match / len(results[dataset][algorithm]['rules'])
    print("Rule overlaps:", overlap_list)


def evaluate_rules(association_rules, exec_time, training_time):
    """
    average support, confidence, lift, conviction, leverage and zhangs_metric value of the rules
    """
    support_list = []
    confidence_list = []
    # yulesq_metric_list = []
    # interestingness_score_list = []
    zhangs_metric_list = []
    coverage_list = []
    for rule in association_rules:
        support_list.append(rule['support'])
        confidence_list.append(rule['confidence'])
        # yulesq_metric_list.append(rule['yulesq'])
        # interestingness_score_list.append(rule['interestingness'])
        zhangs_metric_list.append(rule['zhangs_metric'])
        coverage_list.append(rule['coverage'])

    return [len(association_rules), training_time, exec_time, mean(support_list), mean(confidence_list),
            mean(coverage_list), mean(zhangs_metric_list)]
