import subprocess
import time
import numpy as np
import re


def create_corels_input_files(rules, labels, dataset_name):
    """
    CORELS expect frequent itemsets and their occurrences as an input file.
    This function creates those files.
    """
    corels_train_dataset_name = dataset_name.lower().replace(" ", "_")
    with open("src/algorithm/corels/data/" + corels_train_dataset_name + ".out", "w") as file:
        for row in rules:
            file.write(" ".join(map(str, row)) + "\n")

    with open("src/algorithm/corels/data/" + corels_train_dataset_name + ".label", "w") as file:
        for key in labels:
            file.write("{" + key + ":Yes} ")
            file.write(" ".join(map(str, labels[key])) + "\n")


def run_corels(dataset_name):
    """
    Run CORELS' C++ Code
    :param dataset_name:
    :return:
    """
    dataset_name = dataset_name.lower().replace(" ", "_")
    command = "./corels"
    parameters = ["-r", "0.015", "-c", "2", "-p", "1", "../data/" + dataset_name + ".out",
                  "../data/" + dataset_name + ".label"]

    try:
        start = time.time()
        result = subprocess.run([command] + parameters, capture_output=True, text=True, check=True,
                                cwd="./src/algorithm/corels/src")
        exec_time = time.time() - start
        optimal_rule_list_file = None
        time.sleep(1)
        # find where the optimal rule is stored by looking at CORELS' stdout
        for line in result.stdout.splitlines():
            if line.startswith("writing optimal rule list to: "):
                optimal_rule_list_file = "src/algorithm/corels/src/" + line[len("writing optimal rule list to: "):]
                break
        if optimal_rule_list_file:
            with open(optimal_rule_list_file, "r") as file:
                optimal_rule_list = file.read()
                corel_rule_list_model = parse_corels_rule_lists(optimal_rule_list)
            return corel_rule_list_model, exec_time

    except subprocess.CalledProcessError as e:
        # Handle errors
        print(f"Error occurred: {e}")
        print("Standard Error:")
        print(e.stderr)

    return None, None


def create_corels_freq_items_input(itemset, transactions):
    corels_freq_item_string = "{" + ",".join(
        f"{key.replace(' ', '-')}:={value}" for key, value in itemset.items()) + "}"

    # Vectorized matching of rows that satisfy the itemset
    conditions = np.all([transactions[key].astype(str) == value for key, value in itemset.items()], axis=0)

    corels_format_row = [corels_freq_item_string] + conditions.astype(int).tolist()

    return corels_format_row


def parse_corels_rule_lists(rule_list_model_in_text):
    """
    Parse the output of CORELS (rule list models) into Python objects
    Example CORELS output: {spore-print-color_h:=0.0,gill-size_b:=1.0}~0;default~1
    :param rule_list_model_in_text:
    :return:
    """
    condition_blocks = rule_list_model_in_text.split(';')

    result = []

    for block in condition_blocks:
        # Check if the block contains conditions in curly braces
        if '{' in block and '}' in block:
            # Separate the condition and the "then" value using the '~' symbol
            condition_part, then_part = block.split('~')

            # Parse the conditions into key-value pairs by splitting at ':='
            condition_list = []
            for cond in condition_part.strip('{}').split(','):
                key, value = cond.split(':=')
                condition_list.append((key.strip(), value.strip()))

            # Append the parsed condition list and the "then" part to the result
            result.append(condition_list + [int(then_part.strip())])
        else:
            # Handle the default case
            print(block)
            _, default_value = block.split('~')
            result.append([("default", int(default_value.strip()))])

    return result


def test_corels_model(model, test_X, test_y):
    test_X.reset_index(drop=True, inplace=True)
    test_y.reset_index(drop=True, inplace=True)

    default_rule = next((rule[-1][1] for rule in model if rule[0][0] == 'default'), None)

    # Initialize a list to store whether the model holds for each row
    model_holds = []

    # Iterate through each row in the features dataframe
    for i, feature_row in test_X.iterrows():
        actual_label = test_y.iloc[i, 0]  # Get the actual label for the current row
        predicted_label = None

        # Iterate through the conditions in the model (result)
        for condition in model:
            # Skip the default rule for now
            if condition[0][0] == 'default':
                continue

            # Extract conditions and the expected label
            conditions = condition[:-1]
            expected_label = condition[-1]

            # Check if all conditions are satisfied for the current row
            if all(feature_row[key] == value for key, value in conditions):
                predicted_label = expected_label
                break  # Stop checking further rules once a match is found

        # If no conditions match, use the default rule
        if predicted_label is None and default_rule is not None:
            predicted_label = default_rule

        # Compare predicted label with the actual label
        model_holds.append(predicted_label == actual_label)

    # Calculate the percentage of correct predictions
    accuracy = (sum(model_holds) / len(model_holds)) * 100
    return accuracy
