import sys
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import warnings

warnings.filterwarnings("ignore")

sys.path.append("src")
from data_generators import generate_test_data_3
from tools import run_model_for_raw_and_augmented_data
from tools import smape

if __name__ == "__main__":
    """
    Run only one experiment using generate_test_data_1 dataset, 
    collect result into res DataFrame and print it
    The method generate_test_data_1 could be replaced by 
    generate_test_data_2, generate_test_data_3 or you own data set
    """

    df, train_test_split = generate_test_data_3()
    res = run_model_for_raw_and_augmented_data(
        model=RandomForestRegressor(n_estimators=100, random_state=42),
        df=df,
        train_test_split=train_test_split,
        tabgan=True,
    )

    # visualize the result in the console
    r = res[~np.isnan(res.y)]

    print("SMAPE for raw data:", smape(r.y, r.pred_raw))
    print("SMAPE for augmented data:", smape(r.y, r.pred_augm))
    print("SMAPE for tabgan data:", smape(r.y, r.pred_gan))
