"""
    Tools for loading a synthetic or real dataset.
"""
import numpy as onp
import math

from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils import data
from jax import grad, vmap, random, jit
from jax import numpy as jnp
from functools import partial
import torch.nn as nn
import torch
import random as orandom
import scipy.stats as stats

data_dir = Path(__file__).parent.parent.parent.joinpath("data")


def load_dataset(
    filename,
    normalize=True,
    years=[2003, 2004, 2005, 2006, 2007],
    frac=0.2,
    suffix="syn",
    add_EX="None",
    noise=0.2,
    inputs=1,
    TA_syn=None,
    IV=False,
):
    data = pd.read_csv(data_dir / filename, index_col=1)
    # data = pd.read_csv(data_dir / filename, index_col=0)

    data.index = pd.to_datetime(data.index)
    data["Date"] = data.index.date
    data["Time"] = data.index.time
    data["Year"] = data.index.year
    data["doy"] = data.index.dayofyear
    data["doy_sin"], data["doy_cos"] = make_cyclic(data["doy"])
    hour_of_day = data.index.hour
    minute_of_hour = data.index.minute
    data["tod"] = hour_of_day * 2 + minute_of_hour // 30
    data["tod_sin"], data["tod_cos"] = make_cyclic(data["tod"])

    data = data.sort_index()
    data = data[data.Year.isin(years)]

    # Add noise to RECO simulations
    data = impose_noise(data, "RECO_syn", noise)
    IV_label = None

    # Split into train & test datasets
    if frac > 0:
        if suffix == "measured":
            data["NIGHT"] = 0
            data.loc[(data["SW_IN_POT"] == 0), "NIGHT"] = 1
            data = data[(data["NIGHT"] == 1)]
            data = data[(data["NEE_QC"] == 0)]
            data = data[(data["NEE"] > 0)]

        train, test = train_test_split(
            data, test_size=frac, random_state=31, shuffle=True
        )
        train["train_label"] = "Training set"
        test["train_label"] = "Test set"

        # Define target and explanatory variables
        if suffix == "syn":
            var_RECO = "RECO_obs"
            var_RECO_GT = "RECO_syn"
            var_temp = "TA"
            EV_label = ["SW_POT_sm", "SW_POT_sm_diff", add_EX]

        elif suffix == "measured":
            var_RECO = "NEE"
            var_RECO_GT = "NEE"
            var_temp = "TA"
            if inputs == 1:
                EV_label = [
                    "SW_POT_sm",
                    "SW_POT_sm_diff",
                ]  # ['doy_sin', 'doy_cos', 'VPD', 'SWC_1', 'SWC_2']
                EV_label = ["SW_POT_sm", "SW_POT_sm_diff", "VPD"]
        else:
            var_RECO = f"RECO_{suffix}"
            var_RECO_GT = f"RECO_{suffix}"
            var_temp = "TA"
            #'doy_sin', 'doy_cos'
            EV_label = ["doy_sin", "doy_cos", "VPD", "SWC_1", "SWC_2"]

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
        if normalize:
            RECO_train = RECO_train.values / RECO_max_abs
            RECO_test = RECO_test.values / RECO_max_abs
            RECO_train_GT = RECO_train_GT.values / RECO_max_abs
            RECO_test_GT = RECO_test_GT.values / RECO_max_abs
            EV_train, EV_test = standard_x(EV_train, EV_test)
        else:
            RECO_train = RECO_train.values
            RECO_test = RECO_test.values
            RECO_train_GT = RECO_train_GT.values
            RECO_test_GT = RECO_test_GT.values

            EV_train, EV_test = EV_train.values, EV_test.values
            driver_train, driver_test = driver_train.values, driver_test.values

        if IV_label:
            IV_train = train[IV_label].astype("float32")
            IV_test = test[IV_label].astype("float32")
            if normalize:
                IV_train, IV_test = standard_x(IV_train, IV_test)
                IV_train, IV_test = IV_train.values, IV_test.values
            else:
                IV_train, IV_test = IV_train.values, IV_test.values
            out = [
                EV_train,
                None,
                RECO_train,
                RECO_train_GT,
                driver_train,
                EV_test,
                None,
                RECO_test,
                RECO_test_GT,
                driver_test,
                RECO_max_abs,
            ]
            return train, test, out
    else:
        if suffix == "measured":
            data["NIGHT"] = 0
            data.loc[(data["SW_IN_POT"] == 0), "NIGHT"] = 1
            data = data[(data["NIGHT"] == 1)]
            data = data[(data["NEE_QC"] == 0)]
            data = data[(data["NEE"] > 0)]

        train = data
        train["train_label"] = "Training set"

        # Define target and explanatory variables
        if suffix == "syn":
            var_RECO = "RECO_obs"
            var_RECO_GT = "RECO_syn"
            var_temp = "TA"
            if add_EX == "None":
                EV_label = ["SW_POT_sm", "SW_POT_sm_diff"]
            elif IV:
                EV_label = ["SW_POT_sm", "SW_POT_sm_diff"]
                IV_label = [add_EX]
            else:
                EV_label = ["SW_POT_sm", "SW_POT_sm_diff", add_EX]
        elif suffix == "measured":
            var_RECO = "NEE"
            var_RECO_GT = "NEE"
            var_temp = "TA"
            if inputs == 1:
                EV_label = [
                    "SW_POT_sm",
                    "SW_POT_sm_diff",
                ]  # ['doy_sin', 'doy_cos', 'VPD', 'SWC_1', 'SWC_2']
            elif inputs == 2:
                EV_label = ["SW_POT_sm", "SW_POT_sm_diff", "SWC_1"]
            elif inputs == 3:
                EV_label = ["SW_POT_sm", "SW_POT_sm_diff", "SWC_1", "SWC_2"]
            elif inputs == 4:
                EV_label = ["SW_POT_sm", "SW_POT_sm_diff", "SWC_1", "SWC_2", "VPD"]
            elif inputs == 5:
                EV_label = ["SW_POT_sm", "SW_POT_sm_diff", "VPD"]
        else:
            var_RECO = f"RECO_{suffix}"
            var_RECO_GT = f"RECO_{suffix}"
            var_temp = "TA"
            EV_label = ["doy_sin", "doy_cos", "VPD", "SWC_1", "SWC_2"]
            # EV_label = ['SW_POT_sm', 'SW_POT_sm_diff','SWC_1', 'SWC_2']

        # var_RECO = 'Respiration_heterotrophic'
        # var_temp = 'T'
        # EV_label = ['Moist', 'Rgpot', 'doy_sin', 'doy_cos']

        EV_train = train[EV_label].astype("float32")
        RECO_train = train[var_RECO].astype("float32")
        RECO_train_GT = train[var_RECO_GT].astype("float32")
        driver_train = train[var_temp].astype("float32")

        # Y_data Normalization
        RECO_max_abs = (onp.abs(RECO_train.values)).max()
        if normalize:
            RECO_train = RECO_train.values / RECO_max_abs
            RECO_train_GT = RECO_train_GT.values / RECO_max_abs
            EV_train = standard_x(EV_train)
        else:
            RECO_train = RECO_train.values
            RECO_train_GT = RECO_train_GT.values
            EV_train = EV_train.values
            driver_train = driver_train.values
        if IV_label:
            IV_train = train[IV_label].astype("float32")
            if normalize:
                IV_train = standard_x(IV_train)
            else:
                IV_train = IV_train.values
        if IV:
            out = [
                EV_train,
                IV_train,
                RECO_train,
                RECO_train_GT,
                driver_train,
                RECO_max_abs,
                data["Year"],
            ]
            return train, out
        else:
            out = [
                EV_train,
                None,
                RECO_train,
                RECO_train_GT,
                driver_train,
                RECO_max_abs,
                data["Year"],
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
    noise_RECO = stats.truncnorm(-0.95/RECOnoise_std, 0.95/RECOnoise_std, loc=0, scale=RECOnoise_std).rvs(data[RECO_var].shape)
    # add RECO noise
    data.loc[:, 'RECO_obs'] = data[RECO_var] * (1 + noise_RECO)
    return data


def standard_x(x_train, x_test=None):
    """
        Function that stanrdizes along all columns of given data and
        applies the same transformation also to a test set.

    Args:
        x_train (ndarray): training data
        x_test (ndarray, optional): test data. Defaults to None.

    Returns:
        ndarray: standardized training (and if provided test) set
    """
    # the mean and std values are only calculated by training set
    x_mean = x_train.mean(axis=0)
    x_std = x_train.std(axis=0)
    x_train1 = ((x_train - x_mean) / x_std).values
    if x_test is not None:
        x_test1 = ((x_test - x_mean) / x_std).values
        return x_train1, x_test1
    return x_train1