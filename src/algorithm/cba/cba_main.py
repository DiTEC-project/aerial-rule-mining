import time

from src.algorithm.cba.algorithms.m1algorithm import M1Algorithm
from src.algorithm.cba.algorithms.m2algorithm import M2Algorithm
from src.algorithm.cba.algorithms.rule_generation import generateCARs, createCARs, top_rules
from src.algorithm.cba.data_structures.transaction_db import TransactionDB


class CBA:
    """Class for training a testing the
    CBA Algorithm.

    Parameters:
    -----------
    support : float
    confidence : float
    algorithm : string
        Algorithm for building a classifier.
    maxlen : int
        maximum length of mined rules
    """

    def __init__(self, support=0.10, confidence=0.5, maxlen=10, algorithm="m1"):
        if algorithm not in ["m1", "m2"]:
            raise Exception("algorithm parameter must be either 'm1' or 'm2'")
        if 0 > support or support > 1:
            raise Exception("support must be on the interval <0;1>")
        if 0 > confidence or confidence > 1:
            raise Exception("confidence must be on the interval <0;1>")
        if maxlen < 1:
            raise Exception("maxlen cannot be negative or 0")

        self.support = support * 100
        self.confidence = confidence * 100
        self.algorithm = algorithm
        self.maxlen = maxlen
        self.clf = None
        self.target_class = None

        self.available_algorithms = {
            "m1": M1Algorithm,
            "m2": M2Algorithm
        }

    def rule_model_accuracy(self, txns):
        """Takes a TransactionDB and outputs
        accuracy of the classifier
        """
        if not self.clf:
            raise Exception("CBA must be trained using fit method first")
        if not isinstance(txns, TransactionDB):
            raise Exception("txns must be of type TransactionDB")

        return self.clf.test_transactions(txns)

    def fit(self, transactions, target_class, top_rules_args={}, algorithm="fpgrowth"):
        """Trains the model based on input transaction
        and returns self.
        """
        if not isinstance(transactions, TransactionDB):
            raise Exception("transactions must be of type TransactionDB")

        self.target_class = transactions.header[-1]

        used_algorithm = self.available_algorithms[self.algorithm]

        if not top_rules_args:
            cars, stats = generateCARs(transactions, algorithm=algorithm, target_class=target_class,
                                       support=self.support, confidence=self.confidence, maxlen=self.maxlen)
        else:
            rules, stats = top_rules(transactions.string_representation, algorithm=algorithm, target_class=target_class,
                                     init_maxlen=self.maxlen, appearance=transactions.appeardict, **top_rules_args)
            cars = createCARs(rules)

        start_time = time.time()
        self.clf = used_algorithm(cars, transactions).build()
        duration = time.time() - start_time
        stats.append(duration)
        return stats

    def predict(self, X):
        """Method that can be used for predicting
        classes of unseen cases.

        CBA.fit must be used before predicting.
        """
        if not self.clf:
            raise Exception("CBA must be train using fit method first")

        if not isinstance(X, TransactionDB):
            raise Exception("X must be of type TransactionDB")

        return self.clf.predict_all(X)

    def predict_probability(self, X):
        """Method for predicting probablity of
        given classification
¨
        CBA.fit must be used before predicting probablity.
        """

        return self.clf.predict_probability_all(X)

    def predict_matched_rules(self, X):
        """for each data instance, returns a rule that
        matched it according to the CBA order (sorted by
        confidence, support and length)
        """

        return self.clf.predict_matched_rule_all(X)
