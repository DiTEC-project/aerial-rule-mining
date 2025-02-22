import numpy as np
import pandas
import pandas as pd
from ucimlrepo import fetch_ucirepo

from src.algorithm.aerial.aerial import Aerial
from src.algorithm.brl.bayesian_rule_list import BayesianRuleListClassifier
from src.algorithm.classic_arm import ClassicARM
from src.util.data import encode_categories
from sklearn.model_selection import StratifiedKFold

from src.algorithm.cba.cba_main import CBA
from src.algorithm.cba.data_structures.transaction_db import TransactionDB
from src.util.ucimlrepo import *
from src.util.rule import *
from src.util.corels import *


def get_datasets():
    datasets = []

    print("Loading the datasets ...")
    breast_cancer = discretize_numerical_features(fetch_ucirepo(id=14))  # low accuracy
    # congress_voting_records = fetch_ucirepo(id=105)
    # mushroom = fetch_ucirepo(id=73)
    # chess_king_rook_vs_king_pawn = fetch_ucirepo(id=22)  # low accuracy
    # spambase = discretize_numerical_features(fetch_ucirepo(id=94))

    datasets += [
        # (congress_voting_records, "Class", {'democrat': 0, 'republican': 1}),
        # (mushroom, "poisonous", {'e': 0, 'p': 1}),
        (breast_cancer, "Class", {"recurrence-events": 0, "no-recurrence-events": 1}),
        # (chess_king_rook_vs_king_pawn, "wtoeg", {"won": 0, "nowin": 1}),
        # (spambase, "Class", {"0": 0, "1": 1})
    ]

    print("Following datasets are loaded:", [dataset.metadata.name for dataset, class_label, labels in datasets])

    return datasets


def print_stats(stats):
    averages = [sum(column) / len(stats) for column in zip(*stats)]

    print("# Rules:", averages[0])
    print("Support:", averages[1])
    print("Confidence:", averages[2])
    print("Accuracy:", averages[4])
    print("Duration:", averages[3])


def test_on_cba(datasets):
    dataset, class_label, labels = datasets[0]
    X = pandas.DataFrame(dataset.data.features).reset_index(drop=True)
    y = pandas.DataFrame(dataset.data.targets).reset_index(drop=True)
    X = X.dropna()
    y = y.loc[X.index]

    total_stats = []
    stratified_kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=2)
    for fold, (train_idx, test_idx) in enumerate(stratified_kfold.split(X, y)):
        txns_train = TransactionDB.from_DataFrame(pd.concat([X.iloc[train_idx], y.iloc[train_idx]], axis=1))
        txns_test = TransactionDB.from_DataFrame(pd.concat([X.iloc[test_idx], y.iloc[test_idx]], axis=1))
        cba = CBA(maxlen=3, algorithm="m2")
        stats = cba.fit(txns_train, algorithm="aerial", target_class=class_label)
        accuracy = cba.rule_model_accuracy(txns_test)
        stats.append(accuracy)
        print(stats)
        total_stats.append(stats)

    average_values = np.array(total_stats).mean(axis=0)
    print("Average values after 10-fold cross validation:")
    print("# Items:", average_values[0])
    print("Support:", average_values[1])
    print("Confidence:", average_values[2])
    print("Accuracy:", average_values[5])
    print("Rule mining time (s):", average_values[3])
    print("Rule list learning time (s):", average_values[4])


def test_on_brl(datasets):
    dataset, class_label, labels = datasets[0]
    X = pandas.DataFrame(dataset.data.features).reset_index(drop=True)
    y = pandas.DataFrame(dataset.data.targets).reset_index(drop=True)
    X = X.dropna()
    y = y.loc[X.index]

    y = y[class_label].replace(labels)

    X_encoded = encode_categories(X, X.columns)
    feature_list = X_encoded.columns.to_list()

    stats = []
    stratified_kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=2)
    for fold, (train_idx, test_idx) in enumerate(stratified_kfold.split(X_encoded, y)):
        X_train = X_encoded.iloc[train_idx]
        y_train = y.iloc[train_idx]
        X_test = X_encoded.iloc[test_idx]
        y_test = y.iloc[test_idx]

        model = BayesianRuleListClassifier(minsupport=0.4, maxcardinality=3)
        model.fit(X_train, X.iloc[train_idx], y_train, feature_names=feature_list, algorithm="aerial")  # fit model
        preds = model.predict(X_test)  # discrete predictions: shape is (n_test, 1)
        # preds_proba = model.predict_proba(X_test)  # predicted probabilities: shape is (n_test, n_classes)
        accuracy = np.sum(preds == y_test) / len(y_test)
        stats.append(
            [model.freq_itemsets_count, model.average_itemset_support, accuracy, model.rule_mining_time,
             model.rule_list_building_time])

        print(stats[-1])
    average_values = np.array(stats).mean(axis=0)
    print("Average values after 10-fold cross validation:")
    print("# Items:", average_values[0])
    print("Support:", average_values[1])
    print("Accuracy:", average_values[2])
    print("Freq. item learning time (s):", average_values[3])
    print("Rule list learning time (s):", average_values[4])


def test_on_corels(datasets):
    """
    CORELS' source code is written in C++, and the C++ program accepts data and the labels
    as parameters. This function first learns the parameters, put them in a format that is consumable
    by CORELS, and then calls CORELS.
    :param datasets:
    :return:
    """
    dataset, class_label, labels = datasets[0]
    X = pandas.DataFrame(dataset.data.features).reset_index(drop=True)
    y = pandas.DataFrame(dataset.data.targets).reset_index(drop=True)
    X = X.dropna()
    y = y.loc[X.index]

    # X_encoded = encode_categories(X, X.columns)
    y[class_label] = y[class_label].replace(labels)

    fpgrowth = ClassicARM(min_support=0.3, algorithm="fpgrowth")
    aerial = Aerial(ant_similarity=0.35, cons_similarity=0.8, max_antecedents=2)
    algorithm = "aerial"

    stats = []
    stratified_kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=2)
    for fold, (train_idx, test_idx) in enumerate(stratified_kfold.split(X, y)):
        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_test = y.iloc[test_idx]

        training_dataset_name = str(dataset.metadata.name) + "_" + str(fold)

        if algorithm == "aerial":
            aerial.create_input_vectors(X_train)
            aerial_training_time = aerial.train()
            freq_items, ae_exec_time = aerial.generate_frequent_itemsets()
            # rules, ae_exec_time = aerial.generate_rules(target_class=class_label)
            # freq_items = [entry['antecedents'] for entry in rules]
            freq_items_time = aerial_training_time + ae_exec_time
            freq_items, mean_support = aerial.calculate_freq_item_support(freq_items, X_train)
        else:
            # fpgrowth.mine_rules() would normally return stats as well, but for not skip that part
            # comment out the stats calculation in mine_rules()
            fpgrowth_input = prepare_classic_arm_input(X_train)
            freq_items, freq_items_time = fpgrowth.mine_rules(fpgrowth_input, antecedents=2, frequent_items=True)
            mean_support = freq_items["support"].mean()
            freq_items = freq_items["itemsets"]

        print(len(freq_items), " - ", freq_items_time)
        corels_train, corels_train_labels, conv_time = fpg_to_corels(freq_items, y_train[class_label], X_train, labels)
        create_corels_input_files(corels_train, corels_train_labels, training_dataset_name)
        rule_list_model, rule_list_learning_time = run_corels(training_dataset_name)
        accuracy = test_corels_model(rule_list_model, pd.DataFrame(X_test), pd.DataFrame(y_test))

        stats.append(
            [len(freq_items), mean_support, accuracy, freq_items_time, rule_list_learning_time, conv_time])
        print(stats[-1])

    average_values = np.array(stats).mean(axis=0)
    print("Average values after 10-fold cross validation:")
    print("# Items:", average_values[0])
    print("Support:", average_values[1])
    print("Accuracy:", average_values[2])
    print("Freq. item learning time (s):", average_values[3])
    print("Rule list learning time (s):", average_values[4])
    print("Data preparation time (s):", average_values[5])


if __name__ == '__main__':
    datasets = get_datasets()
    test_on_brl(datasets)
