import datetime
import os
import sys
import warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from tabgan.sampler import GANGenerator

sys.path.append("src")
warnings.filterwarnings("ignore")

from data_generators import get_data_from_file
from tools import run_model_for_raw_and_augmented_data
from tools import smape
from augmentation import TransformData


if __name__ == "__main__":
    """
    The comparison consumption time of Tabgan and augmentation
    """

    df, train_test_split = get_data_from_file(
        os.path.join(os.path.join("data", "df_01.csv"))
    )
    d = df

    train_df = d[(d.time < train_test_split) & (~np.isnan(d.y))]
    test_df = d[d.time >= train_test_split]
    train_df.fillna(train_df.mean(), inplace=True)
    train_df.fillna(0, inplace=True)
    q_liq_original_col = test_df.y.copy()
    test_df.fillna(train_df.mean(), inplace=True)
    test_df.fillna(0, inplace=True)
    test_df.y = q_liq_original_col

    model = RandomForestRegressor(n_estimators=200)
    model.fit(train_df.drop(columns=["y", "time"]), train_df.y)
    res1 = model.predict(test_df.drop(columns=["y", "time"]))

    r = pd.DataFrame({"time": test_df.time, "y": test_df.y, "pred_raw": res1})
    n = 1
    tm1 = datetime.datetime.now()
    for i in range(n):
        tr = TransformData(target_column="y", time_column="time", N=15, k=10)
        augmented_data = tr.augment_data(train_df)

    tm2 = datetime.datetime.now()
    augm_time = tm2 - tm1

    model.fit(augmented_data.drop(columns=["time", "y"]), augmented_data.y)
    res2 = model.predict(test_df.drop(columns=["time", "y"]))
    r["pred_augm"] = res2

    tm1 = datetime.datetime.now()
    for i in range(n):
        tg = GANGenerator()

        gan_train, gan_y = tg.generate_data_pipe(
            train_df=train_df.drop(columns=["y", "time"]),
            target=train_df[["y"]],
            test_df=test_df.drop(columns=["y", "time"]),
        )
    tm2 = datetime.datetime.now()
    tabgan_time = tm2 - tm1

    model.fit(gan_train, gan_y)
    gan_pred = model.predict(test_df.drop(columns=["y", "time"]))
    r["pred_gan"] = gan_pred

    res = run_model_for_raw_and_augmented_data(
        model=model,
        df=df,
        train_test_split=train_test_split,
        tabgan=True,
    )

    # visualize the result in the console
    r = res[~np.isnan(res.y)]
    print("SMAPE for raw data:", smape(r.y, r.pred_raw))
    print("SMAPE for augmented data:", smape(r.y, r.pred_augm))
    print("SMAPE for tabgan data:   ", smape(r.y, r.pred_gan))
    print()
    print("augmentation time consumption:", augm_time / n)
    print("tabgan time consumption:   ", tabgan_time / n)
