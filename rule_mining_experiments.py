import os
from multiprocessing import Pool

import pandas as pd
import csv
import warnings
import tracemalloc
import config

from datetime import datetime
from niapy.algorithms.basic import DifferentialEvolution, GeneticAlgorithm, ParticleSwarmOptimization, \
    SineCosineAlgorithm, GreyWolfOptimizer, BatAlgorithm, FishSchoolSearch
from niapy.algorithms.modified import SuccessHistoryAdaptiveDifferentialEvolution, SelfAdaptiveDifferentialEvolution

from ucimlrepo import fetch_ucirepo

from src.algorithm.classic_arm import ClassicARM
from src.algorithm.aerial.aerial import Aerial
from src.algorithm.arm_ae.armae import ARMAE
from src.algorithm.optimization_arm import OptimizationARM
from src.util.rule_quality import *
from src.util.ucimlrepo import *

# todo: resolve the warnings
warnings.filterwarnings("ignore")


def get_datasets():
    print("Loading the datasets ...")
    # breast_cancer = discretize_numerical_features(fetch_ucirepo(id=14))  # low accuracy
    # congress_voting_records = fetch_ucirepo(id=105)
    # mushroom = fetch_ucirepo(id=73)
    # chess_king_rook_vs_king_pawn = fetch_ucirepo(id=22)  # low accuracy
    spambase = discretize_numerical_features(fetch_ucirepo(id=94))

    return [spambase]


def print_parameters():
    print("----------------------------------------------------")
    print("Algorithm parameters:\n")
    print("Aerial - Antecedent similarity:", config.ANTECEDENT_SIMILARITY)
    print("Aerial - Consequent similarity:", config.CONSEQUENT_SIMILARITY)
    print("Exhaustive - Minimum support:", config.MIN_SUPPORT)
    print("Exhaustive - Minimum confidence:", config.MIN_CONFIDENCE)
    print("Optimization-based - Population size:", config.POPULATION_SIZE)
    print("Optimization-based - Maximum evaluations:", config.SIMILARITY_THRESHOLD)
    print("ARM-AE - Similarity threshold:", config.SIMILARITY_THRESHOLD)
    print("Generic - Number of antecedents:", config.MAX_ANTECEDENT)
    print("Generic - Number of bins for discretization:", config.NUM_BINS)
    print("----------------------------------------------------\n")


def execute(parameters):
    """
    this is to run multiple optimization methods in parallel (separately), to speed up the experiment stage
    :return:
    """
    dataset, algorithm = parameters
    opt_stats, opt_rules = algorithm.learn_rules(dataset)
    return opt_stats, opt_rules


def save_results(result_list):
    timestamp = datetime.now().strftime("%m-%d-%Y_%H:%M:%S")
    for dataset, results in result_list.items():
        print("\nRESULTS: Rule quality evaluation results for the dataset", dataset)
        with open(dataset + "_" + timestamp + '.csv', 'w+', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(["Dataset: " + dataset])
            columns = ["Algorithm", "Rule Count", "Rule Mining Time (s)", "Support",
                       "Confidence", "Data Coverage", "Zhang"]
            writer.writerow(columns)
            print(columns)
            for algorithm in results:
                if len(results[algorithm]['stats']) > 0:
                    alg_stats = [round(float_value, 2) for float_value in
                                 list(pd.DataFrame(results[algorithm]["stats"]).mean())]
                    row = [algorithm] + alg_stats
                    print(row)
                else:
                    row = [algorithm, "No rules found!"]
                writer.writerow(row)
        print("\nSAVED: The results are saved into '", dataset + "_" + timestamp + ".csv' file.")


if __name__ == "__main__":
    print_parameters()

    dataset_list = get_datasets()
    results = {}

    fpgrowth = ClassicARM(min_support=config.MIN_SUPPORT, min_confidence=config.MIN_CONFIDENCE,
                          algorithm="fpgrowth")
    hmine = ClassicARM(min_support=config.MIN_SUPPORT, min_confidence=config.MIN_CONFIDENCE, algorithm="hmine")

    aerial = Aerial(max_antecedents=config.MAX_ANTECEDENT, ant_similarity=config.ANTECEDENT_SIMILARITY,
                    cons_similarity=config.CONSEQUENT_SIMILARITY)

    for dataset in dataset_list:
        print("Learning rules from dataset:", dataset.metadata.name, "...")
        classical_arm_input = prepare_classic_arm_input(dataset.data.features)
        optimization_based_arm_input = prepare_opt_arm_input(dataset)

        results[dataset.metadata.name] = {}
        results[dataset.metadata.name]["bat"] = {"stats": [], "rules": None}
        results[dataset.metadata.name]["gwo"] = {"stats": [], "rules": None}
        results[dataset.metadata.name]["sc"] = {"stats": [], "rules": None}
        results[dataset.metadata.name]["fss"] = {"stats": [], "rules": None}
        results[dataset.metadata.name]["fpgrowth"] = {"stats": [], "rules": None}
        results[dataset.metadata.name]["hmine"] = {"stats": [], "rules": None}
        results[dataset.metadata.name]["arm-ae"] = {"stats": [], "rules": None}
        results[dataset.metadata.name]["aerial"] = {"stats": [], "rules": None}

        # classical ARM
        # fpgrowth_stats, fpgrowth_rules = fpgrowth.mine_rules(classical_arm_input)
        # if fpgrowth_stats:
        #     results[dataset.metadata.name]["fpgrowth"]["stats"].append(fpgrowth_stats)
        #     results[dataset.metadata.name]["fpgrowth"]["rules"] = fpgrowth_rules

        # hmine_stats, hmine_rules = hmine.mine_rules(classical_arm_input)
        # if hmine_stats:
        #     results[dataset.metadata.name]["hmine"]["stats"].append(hmine_stats)
        #     results[dataset.metadata.name]["hmine"]["rules"] = hmine_rules

        # optimization-based ARM
        # Run all 4 optimization-based ARM algorithms in parallel. Running one algorithm 'NUM_OF_RUNS' times in
        # parallel results in the same rules, therefore, we parallelize running different algorithms at the same
        # time. Also, we need to re-create algorithm instances to avoid having the same results in each run
        pool = Pool(10)
        tasks = []
        for index in range(config.NUM_RUNS):
            tasks.append(
                (optimization_based_arm_input, OptimizationARM(SineCosineAlgorithm(config.POPULATION_SIZE),
                                                               max_evals=config.MAX_EVALS)))
            tasks.append(
                (optimization_based_arm_input, OptimizationARM(GreyWolfOptimizer(config.POPULATION_SIZE),
                                                               max_evals=config.MAX_EVALS)))
            tasks.append(
                (optimization_based_arm_input, OptimizationARM(BatAlgorithm(config.POPULATION_SIZE),
                                                               max_evals=config.MAX_EVALS)))
            tasks.append(
                (optimization_based_arm_input, OptimizationARM(FishSchoolSearch(config.POPULATION_SIZE),
                                                               max_evals=config.MAX_EVALS)))

        stats = pool.map(execute, tasks)
        for index in range(len(stats)):
            if stats[index][0]:
                if index % 4 == 0:
                    results[dataset.metadata.name]["sc"]["stats"].append(stats[index][0])
                    results[dataset.metadata.name]["sc"]["rules"] = stats[index][1]
                if index % 4 == 1:
                    results[dataset.metadata.name]["gwo"]["stats"].append(stats[index][0])
                    results[dataset.metadata.name]["gwo"]["rules"] = stats[index][1]
                if index % 4 == 2:
                    results[dataset.metadata.name]["bat"]["stats"].append(stats[index][0])
                    results[dataset.metadata.name]["bat"]["rules"] = stats[index][1]
                if index % 4 == 3:
                    results[dataset.metadata.name]["fss"]["stats"].append(stats[index][0])
                    results[dataset.metadata.name]["fss"]["rules"] = stats[index][1]

        pool.close()
        pool.join()

        # Aerial (2024)
        # aerial.create_input_vectors(dataset.data.features)
        # aerial_training_time = aerial.train()
        # aerial_association_rules, ae_exec_time = aerial.generate_rules()
        # aerial_stats, aerial_rules = aerial.calculate_stats(aerial_association_rules, classical_arm_input,
        #                                                     aerial_training_time + ae_exec_time)
        # if aerial_stats:
        #     results[dataset.metadata.name]["aerial"]["stats"].append(aerial_stats)
        #     results[dataset.metadata.name]["aerial"]["rules"] = aerial_association_rules

        # ARM-AE from Berteloot et al. (2023)
        # one_hot_encoded = one_hot_encoding(classical_arm_input)
        # arm_ae = ARMAE(len(one_hot_encoded.loc[0]), maxEpoch=1, batchSize=2, learningRate=1e-2, likeness=0.5)
        # dataLoader = arm_ae.dataPreprocessing(one_hot_encoded)
        # arm_ae_training_time = arm_ae.train(dataLoader)
        # arm_ae.generateRules(one_hot_encoded,
        #                      numberOfRules=max(int(len(aerial_association_rules) / one_hot_encoded.shape[1]), 2),
        #                      nbAntecedent=config.MAX_ANTECEDENT)
        # arm_ae_stats, arm_ae_rules = arm_ae.reformat_rules(classical_arm_input, list(one_hot_encoded.columns))
        # if arm_ae_stats:
        #     results[dataset.metadata.name]["arm-ae"]["stats"].append(arm_ae_stats)
        #     results[dataset.metadata.name]["arm-ae"]["rules"] = arm_ae_rules

    save_results(results)
