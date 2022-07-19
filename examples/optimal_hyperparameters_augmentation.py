import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from data_generators import generate_test_data_4
from tabulate import tabulate
from tools import experiment


if __name__ == "__main__":
    """
    Running a set of experiments to find the optimum parameters
    """

    default_model = RandomForestRegressor(n_estimators=200, random_state=42)
    N_possible_values = range(10, 22, 2)
    K_possible_values = range(6, 22, 2)

    df, train_test_split = generate_test_data_4()

    pivot_result_table = []
    for n in N_possible_values:
        for k in K_possible_values:

            result_raw_data, result_augmented_data = experiment(
                model=default_model,
                df=df,
                train_test_split=train_test_split,
                N=n,
                K=k,
                tabgan=False,
            )

            pivot_result_table.append([n, k, result_raw_data, result_augmented_data])

    pivot_result_table = pd.DataFrame(
        data=pivot_result_table,
        columns=["N", "K", "raw_data_smape", "augmented_data_smape"],
    )

    print(
        tabulate(
            pivot_result_table, headers=pivot_result_table.columns, tablefmt="github"
        )
    )

    idx = pivot_result_table.sort_values("augmented_data_smape").iloc[:1].index
    N = pivot_result_table.loc[idx, "N"].values[0]
    K = pivot_result_table.loc[idx, "K"].values[0]
    print(f"\n\n the best solution was found with K:{K},  N:{N}")
