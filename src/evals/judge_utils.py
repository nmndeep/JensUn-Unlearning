import json
import numpy as np
import os

def compute_any_one_average(data:dict, prefix:str="outs_"):
    relevant_keys = [k for k in data if k.startswith(prefix)]
    data = [data[key] for key in relevant_keys if key in data]
    N = len(data[0])
    coverage = [0] * N
    accuracies = []
    row_means = [sum(row) / len(row) for row in data]

    for row in data:
        for i in range(N):
            if row[i] == 1:
                coverage[i] = 1
        accuracy = sum(coverage) / N
        accuracies.append(accuracy)

    return accuracies, row_means


def check_yes_no(input_str:str, inside:int=0):
    """
    Returns 'yes' if input is a variation of yes,
    'no' if a variation of no,
    and None if it's neither.
    """
    yes_variants = {"yes", "yeah"}
    no_variants = {"no", "nope", "not sure"}

    normalized_input = input_str.strip().lower()

    if not normalized_input:
        raise ValueError("Input cannot be empty or just whitespace.")

    if inside:
        return 1

    if normalized_input in yes_variants:
        return 1
    if normalized_input in no_variants:
        return 0
    else:
        raise ValueError("Input not recognized as 'yes' or 'no'.")

def average_case_acc(data:dict, prefix:str="outs_"):
    relevant_keys = [k for k in data if k.startswith(prefix)]
    data = [data[key] for key in relevant_keys if key in data]
    N = len(data[0])
    coverage = [0] * N
    accuracies = []

    if isinstance(data[0], list):

        arr = np.array(data)

        # Step 1: Mean across the 3 rows (axis=0), result is shape (100,)
        mean_across = np.mean(arr, axis=0)

        # Step 2: Mean across the 100 values (axis=0, since it's 1D now)
        final_mean = np.mean(mean_across)
    else:
        arr = np.array(data)
        final_mean = np.mean(arr, axis=0)

    return final_mean



