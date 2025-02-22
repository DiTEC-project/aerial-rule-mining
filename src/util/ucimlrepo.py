"""
This script includes helper functions for processing UCI ML repository datasets
"""
import pandas
import pandas as pd
import numpy as np

from mlxtend.preprocessing import TransactionEncoder


def prepare_classic_arm_input(dataset):
    """
    MLxtend package accepts only array of arrays that is converted to dataframes
    This method converts a given dataset to categorical form by parsing data to string
    and coupling it with column name (this is done for classical ARM approaches only)
    :param dataset:
    :return:
    """
    return [
        [f"{column}__{value}" for column, value in row.items()]
        for _, row in dataset.iterrows()
    ]


def prepare_opt_arm_input(dataset):
    opt_arm_input = []
    for index, row in dataset.data.features.iterrows():
        new_row = {}
        for row_index, item in row.items():
            new_row[row_index] = row_index + "__" + str(item)
        opt_arm_input.append(pd.Series(new_row))
    new_input = pd.DataFrame(opt_arm_input)
    return new_input


def one_hot_encoding(categorical_dataset):
    te = TransactionEncoder()
    te_ary = te.fit(categorical_dataset).transform(categorical_dataset)
    df = pd.DataFrame(te_ary, columns=te.columns_)
    return df


def get_unique_values_per_column(transactions):
    columns = list(transactions.columns)
    value_count = 0

    unique_values = {}
    for column in columns:
        unique_values[column] = []

    for transaction_index, transaction in transactions.iterrows():
        for index in range(len(list(transaction))):
            if transaction.iloc[index] not in unique_values[columns[index]]:
                unique_values[columns[index]].append(transaction.iloc[index])
                value_count += 1

    return unique_values, value_count


def discretize_numerical_features(dataset, columns_to_drop=[]):
    X = pd.DataFrame(dataset.data.features).drop(columns=columns_to_drop)
    numerical_columns = X.select_dtypes(include=[np.number]).columns

    binary_columns = [
        col for col in numerical_columns
        if set(X[col].dropna().unique()).issubset({0, 1})
    ]

    non_binary_columns = [col for col in numerical_columns if col not in binary_columns]
    discretized_columns = {
        col + "_discretized": pd.qcut(X[col], q=10, duplicates='drop')
        .astype(str)
        .str.replace(" ", "", regex=False)
        .str.replace(",", "-", regex=False)
        for col in non_binary_columns
    }

    X = pd.concat([X, pd.DataFrame(discretized_columns)], axis=1)
    dataset.data.features = X.drop(columns=non_binary_columns)

    return dataset