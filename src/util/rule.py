from concurrent.futures import ThreadPoolExecutor
from functools import partial
from src.util.corels import *
from tqdm import tqdm


def aerial_plus_to_cba(aerial_plus_rules):
    """
    Convert the rules learned by our aerial_plus method to CBA format
    :param target_class:
    :param aerial_plus_rules:
    :return:
    """
    cba_rules = []
    for rule in aerial_plus_rules:
        consequent = rule["consequent"].replace("__", ":=:")
        antecedents = tuple([ant.replace("__", ":=:") for ant in rule["antecedents"]])
        cba_rules.append((consequent, antecedents, rule['support'], rule['confidence']))

    return cba_rules


def fpgrowth_to_cba(fpgrowth_rules):
    """
    Convert the rules learned by MLxtend's FP-Growth to CBA format
    :param fpgrowth_rules:
    :return:
    """
    cba_rules = []
    for rule in fpgrowth_rules:
        # rule["consequent"] should be filtered (constraint itemset mining) before this step, to have the class label
        consequent = rule["consequent"][0].replace("__", ":=:")
        antecedents = tuple([ant.replace("__", ":=:") for ant in rule["antecedents"]])
        cba_rules.append((consequent, antecedents, rule['support'], rule['confidence']))

    return cba_rules


def fpg_to_corels(freq_items, labels, transactions, label_descriptions):
    """
    convert the frequent itemsets to a format that CORELS can consume
    :param freq_items:
    :return:
    """
    start = time.time()
    labels_corels_format = {}

    for key in label_descriptions:
        labels_corels_format[key] = []

    label_descriptions = {v: k for k, v in label_descriptions.items()}
    for label in labels:
        for key in labels_corels_format:
            if key == label_descriptions[label]:
                labels_corels_format[key].append(1)
            else:
                labels_corels_format[key].append(0)

    formatted_freq_itemsets = []
    for itemset in freq_items:
        new_itemset = {}
        for item in itemset:
            key, value = item.split("__")
            new_itemset[key] = value
        formatted_freq_itemsets.append(new_itemset)

    # the function below suppresses the progress output by the partial function below
    partial_process_item = partial(create_corels_freq_items_input, transactions=transactions)

    with ThreadPoolExecutor(max_workers=10) as executor:
        # Use executor.map with tqdm for progress tracking
        freq_items_corels_format = list(
            tqdm(executor.map(partial_process_item, formatted_freq_itemsets), total=len(formatted_freq_itemsets)
                 , disable=True)
        )

    return freq_items_corels_format, labels_corels_format, time.time() - start
