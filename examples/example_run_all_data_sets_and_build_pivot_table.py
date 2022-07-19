import os
import sys
import warnings

import numpy as np
import pandas as pd


sys.path.append("src")

from data_generators import generate_test_data_1
from data_generators import generate_test_data_2
from data_generators import generate_test_data_3
from data_generators import generate_test_data_4
from data_generators import generate_test_data_5
from data_generators import generate_test_data_6
from data_generators import get_data_from_file
from sklearn.ensemble import RandomForestRegressor
from tabulate import tabulate
from tools import run_model_for_raw_and_augmented_data
from tools import smape


warnings.filterwarnings("ignore")


if __name__ == "__main__":
    """
    run butch of experiments, collect results into pivot_result_table dataframe and print it to console
    """

    model = RandomForestRegressor(n_estimators=200, random_state=42)

    experiments = [
        generate_test_data_1(),
        generate_test_data_2(),
        generate_test_data_3(),
        generate_test_data_4(),
        generate_test_data_5(),
        generate_test_data_6(),
    ]

    for num in [
        "01",
        "02",
        "03",
        "04",
        "05",
        "06",
        "07",
        "08",
        "09",
        "10",
        "11",
        "12",
    ]:
        filename = os.path.join("examples", "data", f"df_{num}.csv")
        exp = get_data_from_file(filename)
        experiments.append(exp)

    experiments = pd.DataFrame(experiments, columns=["df", "train_test_split"])

    pivot_result_table = []
    all_result_together = []

    tabgan_params = None

    for i, row in experiments.iterrows():
        df, train_test_split, num = row.df, row.train_test_split, i
        print(f"experiment {num}...")
        print(f"experiment {num}...")
        experiment_name = f"experiment_{num}"
        experiment_result = run_model_for_raw_and_augmented_data(
            model, df, train_test_split, tabgan_params=tabgan_params, tabgan=True
        )
        print(f"experiment_result={experiment_result}")
        e = experiment_result[~np.isnan(experiment_result.y)]
        e = e[e.y > 1]
        result_raw_data = smape(e.y, e.pred_raw)
        result_augmented_data = smape(e.y, e.pred_augm)
        result_tabgan_data = smape(e.y, e.pred_gan)

        experiment_result["experiment_name"] = num
        all_result_together.append(experiment_result)

        pivot_result_table.append(
            [
                num,
                result_raw_data,
                result_augmented_data,
                result_tabgan_data,
                (result_raw_data - result_augmented_data) / result_raw_data,
                (result_raw_data - result_tabgan_data) / result_raw_data,
            ]
        )

    pivot_result_table = pd.DataFrame(
        data=pivot_result_table,
        columns=[
            "experiment name",
            "raw data smape",
            "augmented data smape",
            "tabgan smape",
            "raw - augm / raw",
            "raw - tabgan / raw",
        ],
    )

    all_result_together = pd.concat(all_result_together)

    print(
        tabulate(
            pivot_result_table.round(
                {
                    "augm / raw": 1,
                    "tabgan / raw": 1,
                    "raw data smape": 4,
                    "augmented data smape": 4,
                }
            ),
            headers=pivot_result_table.columns,
            tablefmt="github",
        )
    )
