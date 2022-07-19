import math

from datetime import datetime
from datetime import timedelta

import numpy as np
import pandas as pd


def generate_test_data_1() -> (pd.DataFrame, datetime):
    """
    Generating dataset and datetime for splitting DataFrame into train and test

    Returns:
        tuple: (pd.DataFrame, datetime) : tuple of DataFrame and datetime

        pd.DataFrame: DataFrame of stationary  process with few "peaks"
        in train and test parts like this:
        time                    x             y
        2021-01-01 00:00:00,    39.59960,     96.44521
        2021-01-01 08:00:00,    40.04312,     nan
        2021-01-01 09:00:00,    39.78953,     nan
        2021-01-01 10:00:00,    42.13245,     95.08260
        2021-01-01 11:00:00,    37.89919,     nan

        datetime: datetime for splitting DataFrame into train and test
    """

    changes = {100: 8, 200: 8, 500: 2, 550: 3, 600: 4}

    rnd_koeff = 5
    ln = 24 * 30 * 1  # 1 month by hours

    np.random.seed(2021)
    x = 40 + rnd_koeff * (np.random.rand(ln) - 0.5)
    np.random.seed(2022)
    y = 100 + rnd_koeff * 2 * (np.random.rand(ln) - 0.5)

    for i in range(ln):
        if i % 10 != 0:
            y[i] = np.nan

    for n, multy in changes.items():
        x[n] *= multy
        y[n] *= multy

    dt = [datetime(2021, 1, 1) + timedelta(hours=x) for x in range(ln)]

    res = pd.DataFrame({"time": dt, "x": x, "y": y})

    return res, datetime(2021, 1, 20)


def generate_test_data_2() -> (pd.DataFrame, datetime):
    """
    Generating dataset and datetime for splitting DataFrame into train and test

    Returns:
        tuple: (pd.DataFrame, datetime) : tuple of DataFrame and datetime

        pd.DataFrame: DataFrame of stationary  process with few "peaks"
        in train and test parts like this:
        time                    x             y
        2021-01-01 00:00:00,    39.59960,     96.44521
        2021-01-01 08:00:00,    40.04312,     nan
        2021-01-01 09:00:00,    39.78953,     nan
        2021-01-01 10:00:00,    42.13245,     95.08260
        2021-01-01 11:00:00,    37.89919,     nan

        datetime: datetime for splitting DataFrame into train and test
    """

    changes = {100: 8, 200: 8, 500: 2, 550: 3, 600: 4}

    rnd_koeff = 10
    ln = 24 * 30 * 1

    np.random.seed(2021)
    x = [
        (40 + 40 * math.sin(x / 10) + rnd_koeff * (np.random.rand() - 0.5))
        for x in range(ln)
    ]
    np.random.seed(2022)
    y = [
        (100 + 40 * math.sin(x / 10) + 2 * rnd_koeff * (np.random.rand() - 0.5))
        for x in range(ln)
    ]

    for i in range(ln):
        if i % 10 != 0:
            y[i] = np.nan

    for n, multy in changes.items():
        x[n] *= multy
        y[n] *= multy

    dt = [datetime(2021, 1, 1) + timedelta(hours=x) for x in range(ln)]

    res = pd.DataFrame({"time": dt, "x": x, "y": y})

    return res, datetime(2021, 1, 20)


def generate_test_data_3() -> (pd.DataFrame, datetime):
    """
    Generating dataset and datetime for splitting DataFrame into train and test

    Returns:
        tuple: (pd.DataFrame, datetime) : tuple of DataFrame and datetime

        pd.DataFrame: DataFrame of stationary  process with few "peaks"
        in train and test parts like this:
        time                    x             y
        2021-01-01 00:00:00,    39.59960,     96.44521
        2021-01-01 08:00:00,    40.04312,     nan
        2021-01-01 09:00:00,    39.78953,     nan
        2021-01-01 10:00:00,    42.13245,     95.08260
        2021-01-01 11:00:00,    37.89919,     nan

        datetime: datetime for splitting DataFrame into train and test
    """

    changes = {200: 8, 520: 2, 580: 3, 660: 5}

    rnd_koeff = 10
    ln = 24 * 30 * 1

    np.random.seed(2021)
    x = [
        (40 + 40 * math.sin(x / 10) + rnd_koeff * (np.random.rand() - 0.5))
        for x in range(ln)
    ]
    np.random.seed(2022)
    y = [
        (100 + 40 * math.sin(x / 10) + 2 * rnd_koeff * (np.random.rand() - 0.5))
        for x in range(ln)
    ]

    for i in range(ln):
        if i % 10 != 0:
            y[i] = np.nan

    for n, multy in changes.items():
        x[n] *= multy
        y[n] *= multy

    dt = [datetime(2021, 1, 1) + timedelta(hours=x) for x in range(ln)]

    res = pd.DataFrame({"time": dt, "x": x, "y": y})

    return res, datetime(2021, 1, 20)


def generate_test_data_4() -> (pd.DataFrame, datetime):
    """
    Generating dataset and datetime for splitting DataFrame into train and test

    Returns:
        tuple: (pd.DataFrame, datetime) : tuple of DataFrame and datetime

        pd.DataFrame: DataFrame of stationary  process with few "peaks"
        in train and test parts like this:
        time                    x             y
        2021-01-01 00:00:00,    39.59960,     96.44521
        2021-01-01 08:00:00,    40.04312,     nan
        2021-01-01 09:00:00,    39.78953,     nan
        2021-01-01 10:00:00,    42.13245,     95.08260
        2021-01-01 11:00:00,    37.89919,     nan

        datetime: datetime for splitting DataFrame into train and test
    """

    changes = {200: 8, 520: 2, 580: 3, 660: 5}

    rnd_koeff = 10
    ln = 24 * 30 * 1

    np.random.seed(2021)
    x1 = 40 + rnd_koeff * (np.random.rand(ln) - 0.5)
    np.random.seed(2022)
    x2 = [
        (40 + 40 * math.sin(x / 10) + rnd_koeff * (np.random.rand() - 0.5))
        for x in range(ln)
    ]
    np.random.seed(2023)
    x3 = [
        (60 + 20 * math.sin(x / 20) + rnd_koeff * 2 * (np.random.rand() - 0.5))
        for x in range(ln)
    ]
    np.random.seed(2024)
    x4 = [(x + rnd_koeff * (np.random.rand() - 0.5)) for x in range(ln)]

    np.random.seed(2025)
    y = [
        (150 + 20 * math.sin(x / 10) + 2 * rnd_koeff * (np.random.rand() - 0.5))
        for x in range(ln)
    ]

    for i in range(ln):
        if i % 10 != 0:
            y[i] = np.nan

    for n, multy in changes.items():
        x1[n] /= multy
        x2[n] *= multy
        y[n] *= multy

    dt = [datetime(2021, 1, 1) + timedelta(hours=x) for x in range(ln)]

    res = pd.DataFrame({"time": dt, "x1": x1, "x2": x2, "x3": x3, "x4": x4, "y": y})

    return res, datetime(2021, 1, 20)


def gauss_with_exp_acf_gen(
    *,
    sigma: float = 2,
    w_star: float = 1.25,
    delta_t: float = 0.05,
    N: int = 1000,
    seed=2021,
) -> np.array:
    """
    Generates a discrete realization of a stationary Gaussian pseudo-random process
    with a correlation function of the exponential cosine type

    Args:
        sigma : standard deviation
        w_star : model parameter
        w0 : model parameter
        delta_t : time step
        N : number of samples

    Returns:
        xi : np.array of elements of a pseudo-random process with a given correlation function

    """
    np.random.seed(seed)
    gamma_star = w_star * delta_t
    rho = math.exp(-gamma_star)
    b1 = rho
    a0 = sigma * math.sqrt(1 - rho**2)

    xi = np.zeros(N)
    xi[0] = np.random.rand()
    x = np.random.randn(N)

    for n in range(1, N):
        xi[n] = a0 * x[n] + b1 * xi[n - 1]

    return xi


def generate_test_data_5() -> (pd.DataFrame, datetime):
    """
    Generating dataset and datetime for splitting DataFrame into train and test

    Returns:
        tuple: (pd.DataFrame, datetime) : tuple of DataFrame and datetime

        pd.DataFrame: DataFrame of stationary  process with few "peaks"
        in train and test parts like this:
        time                    x             y
        2021-01-01 00:00:00,    39.59960,     96.44521
        2021-01-01 08:00:00,    40.04312,     nan
        2021-01-01 09:00:00,    39.78953,     nan
        2021-01-01 10:00:00,    42.13245,     95.08260
        2021-01-01 11:00:00,    37.89919,     nan

        datetime: datetime for splitting DataFrame into train and test
    """

    xi_1 = gauss_with_exp_acf_gen(N=500, seed=2021) * 2 + 25
    xi_2 = gauss_with_exp_acf_gen(N=500, seed=2022) + 35
    xi_3 = gauss_with_exp_acf_gen(N=500, seed=2023) * 2.5 + 10
    xi_4 = gauss_with_exp_acf_gen(N=500, seed=2024) * 1.5 + 10

    y = xi_1 + xi_2 - xi_3 - xi_4

    changes = {50: 2, 150: 3, 200: 4, 440: 1.5, 450: 2, 460: 2.5, 480: 3}

    for n, multy in changes.items():
        y[n] *= multy
        xi_1[n] *= multy
        xi_3[n] *= multy

    process = pd.DataFrame(
        {
            "time": pd.date_range("2022-01-01", periods=len(y), freq="H"),
            "y": y,
            "xi_1": xi_1,
            "xi_2": xi_2,
            "xi_3": xi_3,
            "xi_4": xi_4,
        }
    )

    return process, process.loc[400, "time"]


def gauss_with_expcos_family_acf_base(
    *,
    a0: float,
    a1: float,
    b1: float,
    b2: float,
    N: int,
    seed: int = 2021,
) -> np.array:
    """
    Reference algorithm for constructing a discrete realisation of a pseudo-random process
    with a correlation function of the exponential-cosine family

    Args:
        a0 : model parameter
        a1 : model parameter
        b1 : model parameter
        b2 : model parameter
        N : number of samples

    Returns:
        xi : np.array of elements of a pseudo-random process with a given correlation function

    """
    np.random.seed(seed)
    xi = np.zeros(N)
    for i in range(2):
        xi[i] = np.random.rand()

    x = np.random.randn(N)

    for n in range(1, N):
        xi[n] = a0 * x[n] + a1 * x[n - 1] + b1 * xi[n - 1] + b2 * xi[n - 2]

    return xi


def gauss_with_expcos_acf_gen(
    *,
    sigma: float = 2,
    w_star: float = 1.25,
    w0: float = 3,
    delta_t: float = 0.05,
    N: int = 10000,
    seed=2021,
) -> np.array:
    """
    Generates a discrete realization of a stationary Gaussian pseudo-random process
    with a correlation function of the exponential type

    Args:
        sigma : standard deviation
        w_star : model parameter
        delta_t : time step
        N : number of samples

    Returns:
        xi : np.array of elements of a pseudo-random process with a given correlation function
    """

    gamma_star = w_star * delta_t
    gamma0 = w0 * delta_t
    rho = math.exp(-gamma_star)
    alpha0 = rho * (rho**2 - 1) * math.cos(gamma0)
    alpha1 = 1 - rho**4
    alpha = math.sqrt((alpha1 + math.sqrt(alpha1**2 - 4 * alpha0**2)) / 2)
    a0 = sigma * alpha
    a1 = sigma * alpha0 / alpha
    b1 = 2 * rho * math.cos(gamma0)
    b2 = -(rho**2)

    params = dict(
        a0=a0,
        a1=a1,
        b1=b1,
        b2=b2,
        N=N,
        seed=seed,
    )
    xi = gauss_with_expcos_family_acf_base(**params)
    return xi


def generate_test_data_6() -> (pd.DataFrame, datetime):
    """
    Reference algorithm for constructing a discrete realisation of a pseudo-random process
    with a correlation function of the exponential-cosine family

    Returns:
        tuple: (pd.DataFrame, datetime) : tuple of DataFrame and datetime

        pd.DataFrame: DataFrame of stationary  process with few "peaks"
        in train and test parts like this:
        time                    x             y
        2021-01-01 00:00:00,    39.59960,     96.44521
        2021-01-01 08:00:00,    40.04312,     nan
        2021-01-01 09:00:00,    39.78953,     nan
        2021-01-01 10:00:00,    42.13245,     95.08260
        2021-01-01 11:00:00,    37.89919,     nan

        datetime: datetime for splitting DataFrame into train and test

    """

    xi_1 = (
        gauss_with_expcos_acf_gen(
            N=500, sigma=2, w_star=1.25, w0=1, delta_t=0.05, seed=2021
        )
        * 2
        + 25
    )
    xi_2 = (
        gauss_with_expcos_acf_gen(
            N=500, sigma=1.5, w_star=1.5, w0=2, delta_t=0.05, seed=2022
        )
        * 1.5
        + 35
    )
    xi_3 = (
        gauss_with_expcos_acf_gen(
            N=500, sigma=1.7, w_star=1.75, w0=3, delta_t=0.06, seed=2023
        )
        * 1.5
        + 50
    )
    xi_4 = (
        gauss_with_expcos_acf_gen(
            N=500, sigma=3, w_star=1.05, w0=4, delta_t=0.07, seed=2024
        )
        * 1.5
        + 10
    )

    y = -1.5 * xi_1 + xi_2 - xi_3 / 2 + xi_4 / 2 + 100

    changes = {50: 2, 150: 3, 200: 2.5, 440: 1.5, 450: 2, 460: 2.5, 480: 3}

    for n, multy in changes.items():
        y[n] *= multy
        xi_3[n] *= multy
        xi_4[n] *= multy

    process = pd.DataFrame(
        {
            "time": pd.date_range("2022-01-01", periods=len(y), freq="H"),
            "y": y,
            "xi_1": xi_1,
            "xi_2": xi_2,
            "xi_3": xi_3,
            "xi_4": xi_4,
        }
    )

    return process[:500], process.loc[400, "time"]


def get_data_from_file(filename):
    """
    Read data from file

    Returns:
         tuple: (pd.DataFrame, datetime) : tuple of DataFrame and datetime
            - pd.Dataset: generated data
            - datetime: datetime for splitting DataFrame into train and test
    """
    dates_dict = {
        "01": datetime(2017, 10, 19, 13, 30),
        "02": datetime(2017, 8, 19, 13, 30),
        "03": datetime(2017, 5, 19, 13, 30),
        "04": datetime(2017, 11, 2, 13, 30),
        "05": datetime(2017, 4, 18, 13, 30),
        "06": datetime(2017, 5, 19, 13, 30),
        "07": datetime(2017, 8, 28, 13, 30),
        "08": datetime(2017, 11, 19, 13, 30),
        "09": datetime(2017, 11, 18, 13, 30),
        "10": datetime(2017, 7, 7, 13, 30),
        "11": datetime(2017, 6, 2, 13, 30),
        "12": datetime(2017, 6, 18, 13, 30),
        "13": datetime(2013, 10, 19, 13, 30),
    }

    df = pd.read_csv(filename)
    df["time"] = pd.to_datetime(df["time"], format="%Y-%m-%d %H:%M:%S")
    id = filename.split("_")[1].split(".")[0]
    return df, dates_dict.get(id)
