"""
    Tools for loading a synthetic or real dataset.
"""
import numpy as onp
import math

from pathlib import Path
from sklearn.model_selection import train_test_split
import scipy.stats as stats
from .preprocessing import prepare_data
from .generate_data import synthetic_dataset

data_dir = Path(__file__).parent.parent.parent.joinpath("data")


def load_dataset(
    site="AT-Neu",
    target="syn",
    frac=0.2,
    years=[2003, 2004, 2005, 2006, 2007],
    noise=0.2,
    seed=33,
):
    data = prepare_data(site)
    if target == "syn":
        data = synthetic_dataset(data, Q10=1.5)
        data = impose_noise(data, "RECO_syn", noise)

    data = data.sort_index()
    data = data[data.Year.isin(years)]

    # Split into train & test datasets
    if frac > 0:
        if target == "measured":
            data["NIGHT"] = 0
            data.loc[(data["SW_IN_POT"] == 0), "NIGHT"] = 1
            data = data[(data["NIGHT"] == 1)]
            data = data[(data["NEE_QC"] == 0)]
            data = data[(data["NEE"] > 0)]

        train, test = train_test_split(
            data, test_size=frac, random_state=seed, shuffle=True
        )
        train["train_label"] = "Training set"
        test["train_label"] = "Test set"

        # Define target and explanatory variables
        if target == "syn":
            var_RECO = "RECO_obs"
            var_RECO_GT = "RECO_syn"
            var_temp = "TA"
            EV_label = ["SW_POT_sm", "SW_POT_sm_diff"]
        elif target == "measured":
            var_RECO = "NEE"
            var_RECO_GT = "NEE"
            var_temp = "TA"
            EV_label = ["SW_POT_sm", "SW_POT_sm_diff", "VPD"]

        EV_train = train[EV_label].astype("float32")
        RECO_train = train[var_RECO].astype("float32")
        RECO_train_GT = train[var_RECO_GT].astype("float32")
        driver_train = train[var_temp].astype("float32")

        EV_test = test[EV_label].astype("float32")
        RECO_test = test[var_RECO].astype("float32")
        RECO_test_GT = test[var_RECO_GT].astype("float32")
        driver_test = test[var_temp].astype("float32")

        # Y_data Normalization
        RECO_max_abs = (onp.abs(RECO_train.values)).max()

        RECO_train = RECO_train.values
        RECO_test = RECO_test.values
        RECO_train_GT = RECO_train_GT.values
        RECO_test_GT = RECO_test_GT.values

        EV_train, EV_test = EV_train.values, EV_test.values
        driver_train, driver_test = driver_train.values, driver_test.values
        out = [
            EV_train,
            RECO_train,
            RECO_train_GT,
            driver_train,
            EV_test,
            RECO_test,
            RECO_test_GT,
            driver_test,
            RECO_max_abs,
        ]
        return train, test, out
    else:
        if target == "measured":
            data["NIGHT"] = 0
            data.loc[(data["SW_IN_POT"] == 0), "NIGHT"] = 1
            data = data[(data["NIGHT"] == 1)]
            data = data[(data["NEE_QC"] == 0)]
            data = data[(data["NEE"] > 0)]

        train = data
        train["train_label"] = "Training set"

        # Define target and explanatory variables
        if target == "syn":
            var_RECO = "RECO_obs"
            var_RECO_GT = "RECO_syn"
            var_temp = "TA"
            EV_label = ["SW_POT_sm", "SW_POT_sm_diff"]
        elif target == "measured":
            var_RECO = "NEE"
            var_RECO_GT = "NEE"
            var_temp = "TA"
            EV_label = ["SW_POT_sm", "SW_POT_sm_diff", "VPD"]

        EV_train = train[EV_label].astype("float32")
        RECO_train = train[var_RECO].astype("float32")
        RECO_train_GT = train[var_RECO_GT].astype("float32")
        driver_train = train[var_temp].astype("float32")

        # Y_data Normalization
        RECO_max_abs = (onp.abs(RECO_train.values)).max()
        RECO_train = RECO_train.values
        RECO_train_GT = RECO_train_GT.values
        EV_train = EV_train.values
        driver_train = driver_train.values
    out = [
        EV_train,
        RECO_train,
        RECO_train_GT,
        driver_train,
        RECO_max_abs,
    ]
    return train, out


def make_cyclic(x):
    """
    Computes the cyclic representation of a variables.

    Args:
        x (array_like): Input array to be transformed

    Returns:
        (array_like): x axis of transform
        (array_like): y axis of transform
    """

    x_norm = 2 * math.pi * x / x.max()
    return onp.sin(x_norm), onp.cos(x_norm)


def impose_noise(data, RECO_var, RECOnoise_std=0.3):
    """
        Function that computes heteroschedastic noisy RECO

    Args:
        data (pd.Dataframe): Dataframe with meteorological drivers and computed RECO
        RECOnoise_std (float, optional): Heteroschedastic noise that scales for RECO magnitude.
        Defaults to 0.3.


    Returns:
        pd.Dataframe: Dataframe with additional noisy RECO under column name RECO_obs.
    """
    onp.random.seed(42)
    if RECOnoise_std == 0:
        data.loc[:, "RECO_obs"] = data[RECO_var]
        return data

    # compute noise
    noise_RECO = stats.truncnorm(
        -0.95 / RECOnoise_std, 0.95 / RECOnoise_std, loc=0, scale=RECOnoise_std
    ).rvs(data[RECO_var].shape)
    # add RECO noise
    data.loc[:, "RECO_obs"] = data[RECO_var] * (1 + noise_RECO)
    return data
