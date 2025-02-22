import time
import fim
import logging

from src.algorithm.cba.data_structures.car import ClassAssocationRule
from src.algorithm.cba.data_structures.consequent import Consequent
from src.algorithm.cba.data_structures.antecedent import Antecedent
from src.algorithm.cba.data_structures.item import Item
from src.algorithm.cba.util.transactions import *

from src.algorithm.aerial.aerial import Aerial
from src.algorithm.classic_arm import ClassicARM
from src.util.rule import *
from src.util.ucimlrepo import *


def createCARs(rules):
    """Function for converting output from fim.arules or fim.apriori
    to a list of ClassAssociationRules

    Parameters
    ----------
    rules : output from fim.arules or from generateCARs


    Returns
    -------
    list of CARs

    """
    CARs = []

    for rule in rules:
        con_tmp, ant_tmp, support, confidence = rule

        con = Consequent(*con_tmp.split(":=:"))

        # so that the order of items in antecedent is always the same
        ant_tmp = sorted(list(ant_tmp))
        ant_items = [Item(*i.split(":=:")) for i in ant_tmp]
        ant = Antecedent(ant_items)

        CAR = ClassAssocationRule(ant, con, support=support, confidence=confidence)
        CARs.append(CAR)

    CARs.sort(reverse=True)

    return CARs


def generateCARs(transactionDB, algorithm="aerial", target_class=None, support=1, confidence=50, maxlen=10, **kwargs):
    """Function for generating ClassAssociationRules from a TransactionDB

    Parameters
    ----------
    transactionDB : TransactionDB

    support : float
        minimum support in percents if positive
        absolute minimum support if negative

    confidence : float
        minimum confidence in percents if positive
        absolute minimum confidence if negative

    maxlen : int
        maximum length of mined rules

    **kwargs : 
        arbitrary number of arguments that will be 
        provided to the fim.apriori function

    Returns
    -------
    list of CARs

    """
    if algorithm == "aerial":
        aerial_input = transactiondb_to_dataframe(transactionDB)
        aerial = Aerial(ant_similarity=0.5, cons_similarity=0.8, max_antecedents=maxlen - 1)
        aerial.create_input_vectors(aerial_input)
        # MLxtend's FP-Growth does not support constraint itemset mining while Aerial does (with target_class
        # parameter. For fairness, we run both method without itemset constraint mining option and compare the
        # execution times.
        # Note that a set of comprehensive rule mining time experiments are conducted as part of rule quality
        # experiments, and this is not the main experiment for execution time between FP-Growth and Aerial
        aerial_training_time = aerial.train()
        rules, ae_exec_time = aerial.generate_rules()
        filtered_rules = [
            rule for rule in rules
            if target_class in rule['consequent']
        ]
        rules = aerial.calculate_basic_stats(filtered_rules, prepare_classic_arm_input(aerial_input))
        exec_time = aerial_training_time + ae_exec_time
        if rules:
            rules = aerial_to_cba(rules)
    else:
        fpgrowth = ClassicARM(min_support=0.3, min_confidence=0.8, algorithm="fpgrowth")
        fpgrowth_input = prepare_classic_arm_input(transactiondb_to_dataframe(transactionDB))
        fpg_rules, exec_time = fpgrowth.mine_rules(fpgrowth_input, antecedents=maxlen - 1, rule_stats=False)
        filtered_rules = [
            rule for rule in fpg_rules
            if len(rule['consequent']) == 1 and any(target_class in item for item in rule['consequent'])
        ]
        rules = fpgrowth_to_cba(filtered_rules)

    average_support = sum(rule[2] for rule in rules) / len(rules) if rules else 0
    average_confidence = sum(rule[3] for rule in rules) / len(rules) if rules else 0

    return createCARs(rules), [len(rules), average_support, average_confidence, exec_time]


def top_rules(transactions,
              algorithm="aerial",
              target_class=None,
              appearance={},
              target_rule_count=1000,
              init_support=0.,
              init_conf=0.5,
              conf_step=0.05,
              supp_step=0.05,
              minlen=2,
              init_maxlen=3,
              total_timeout=100.,
              max_iterations=30):
    """Function for finding the best n (target_rule_count)
    rules from transaction list

    Parameters
    ----------
    transactions : 2D array of strings
        e.g. [["a:=:1", "b:=:3"], ["a:=:4", "b:=:2"]]

    appearance : dictionary
        dictionary specifying rule appearance

    targent_rule_count : int
        target number of rules to mine

    init_conf : float
        confidence from which to start mining

    conf_step : float

    supp_step : float

    minen : int
        minimum len of rules to mine

    init_maxlen : int
        maxlen from which to start mining

    total_timeout : float
        maximum execution time of the function

    max_iterations : int
        maximum iterations to try before stopping
        execution


    Returns
    -------
    list of mined rules. The rules are not ordered.

    """

    starttime = time.time()

    MAX_RULE_LEN = len(transactions[0])

    support = init_support
    conf = init_conf

    maxlen = init_maxlen

    flag = True
    lastrulecount = -1
    maxlendecreased_due_timeout = False
    iterations = 0

    rules = None

    while flag:
        iterations += 1

        if iterations == max_iterations:
            logging.debug("Max iterations reached")
            break

        logging.debug(
            "Running apriori with setting: confidence={}, support={}, minlen={}, maxlen={}, MAX_RULE_LEN={}".format(
                conf, support, minlen, maxlen, MAX_RULE_LEN))

        if algorithm == "aerial":
            aerial_input = transactiondb_to_dataframe(transactions)
            aerial = Aerial(noise_factor=0.5, max_antecedents=maxlen - 1)
            aerial.create_input_vectors(aerial_input)
            aerial_training_time = aerial.train()
            rules_current, ae_exec_time = aerial.generate_rules(target_class=target_class)
            rules_current = aerial.calculate_basic_stats(rules_current,
                                                         prepare_classic_arm_input(aerial_input))
            rules_current = aerial_to_cba(rules_current)
        else:
            rules_current = fim.fpgrowth(transactions, supp=support, conf=conf, mode="o", report="sc",
                                         appear=appearance, zmax=maxlen, zmin=minlen)

        rules = rules_current

        rule_count = len(rules)

        logging.debug("Rule count: {}, Iteration: {}".format(rule_count, iterations))

        if (rule_count >= target_rule_count):
            flag = False
            logging.debug(f"Target rule count satisfied: {target_rule_count}")
        else:
            exectime = time.time() - starttime

            if exectime > total_timeout:
                logging.debug(f"Execution time exceeded: {total_timeout}")
                flag = False

            elif maxlen < MAX_RULE_LEN and lastrulecount != rule_count and not maxlendecreased_due_timeout:
                maxlen += 1
                lastrulecount = rule_count
                logging.debug(f"Increasing maxlen {maxlen}")

            elif maxlen < MAX_RULE_LEN and maxlendecreased_due_timeout and support <= 1 - supp_step:
                support += supp_step
                maxlen += 1
                lastrulecount = rule_count

                logging.debug(f"Increasing maxlen to {maxlen}")
                logging.debug(f"Increasing minsup to {support}")

                maxlendecreased_due_timeout = False

            elif conf > conf_step:
                conf -= conf_step
                logging.debug(f"Decreasing confidence to {conf}")

            else:
                logging.debug("All options exhausted")
                flag = False

    average_support = sum(rule[2] for rule in rules) / len(rules)
    average_confidence = sum(rule[3] for rule in rules) / len(rules)

    return rules, [len(rules), average_support, average_confidence]
