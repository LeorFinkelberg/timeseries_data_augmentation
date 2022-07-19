import os
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from data_generators import get_data_from_file
from tabulate import tabulate
from tools import experiment


def get_N_K_best_params(nums, model):
    """
    Returns best N, K hyperparameters based on some experiments (nums)

    Args:
        nums (int): number of experiments
        model (sklearn.model): sklearn model

    Returns:
        tuple(int, int, impr): N_best, K_best, impr
    """
    experiments = []

    for num in nums:
        filename = os.path.join("examples", "data", f"df_{num}.csv")
        exp = get_data_from_file(filename)
        experiments.append(exp)

    experiments = pd.DataFrame(experiments, columns=["df", "train_test_split"])

    N_possible_values = range(6, 30, 4)
    K_possible_values = range(6, 22, 2)

    res = []
    for i, row in experiments.iterrows():
        df, train_test_split = row["df"], row["train_test_split"]
        print(f"processing experiment {i}...")
        pivot_result_table = []
        for n in N_possible_values:
            for k in K_possible_values:
                result_raw_data, result_augmented_data = experiment(
                    model=model,
                    df=df,
                    train_test_split=train_test_split,
                    N=n,
                    K=k,
                    tabgan=False,
                )

                pivot_result_table.append(
                    [i, n, k, result_raw_data, result_augmented_data]
                )

        pivot_result_table = pd.DataFrame(
            data=pivot_result_table,
            columns=["experiment", "N", "K", "raw_data_smape", "augmented_data_smape"],
        )

        pivot_result_table[
            ["raw_data_smape", "augmented_data_smape"]
        ] /= pivot_result_table[["raw_data_smape", "augmented_data_smape"]].mean()

        pivot_result_table = pivot_result_table.sort_values("augmented_data_smape")
        pivot_result_table["exp rang"] = range(len(pivot_result_table))
        res.append(pivot_result_table)

    res = pd.concat(res)

    res = res.groupby(["N", "K"]).mean()
    impr = res.sort_values("exp rang").iloc[0]["augmented_data_smape"]
    ind = res.sort_values("exp rang").index[0]
    N_best, K_best = ind[0], ind[1]
    return N_best, K_best, impr


if __name__ == "__main__":
    """
    Running a set of experiments to find the optimum parameters
    """
    default_model = RandomForestRegressor(n_estimators=200, random_state=42)

    nums = ["01", "02", "03", "04", "05", "06", "07", "08", "09"]
    N_best, K_best, impr = get_N_K_best_params(nums, default_model)
    print(
        f"with hyperparams tuning, prediction result was improved by {(1-impr) * 100:.1f}%"
    )
    print(f"best params: N: {N_best}     K: {K_best}")

    # try to implement such params on test datasets
    experiments = []
    for num in ["07", "08", "09"]:
        filename = os.path.join("examples", "data", f"df_{num}.csv")
        exp = get_data_from_file(filename)
        experiments.append(exp)

    experiments = pd.DataFrame(experiments, columns=["df", "train_test_split"])

    res = []
    for i, row in experiments.iterrows():
        df, train_test_split = row["df"], row["train_test_split"]

        result_raw_data, result_augmented_data = experiment(
            model=default_model,
            df=df,
            train_test_split=train_test_split,
            N=N_best,
            K=K_best,
            tabgan=False,
        )

        res.append([N_best, K_best, result_raw_data, result_augmented_data])

    res = pd.DataFrame(
        data=res,
        columns=["N", "K", "raw_data_smape", "augmented_data_smape"],
    )

    print(tabulate(res, headers=res.columns, tablefmt="github"))
