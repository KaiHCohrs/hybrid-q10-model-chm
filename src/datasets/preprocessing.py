"""Functions for preparing the fluxnet dataset."""
import pathlib

import pandas as pd
import numpy as np
import src.datasets.relevant_variables as relevant_variables


def prepare_data(site, path=None):
    site = site

    # Run a data preprocessing pipeline
    data = load_data(site, path=path)
    data = unwrap_time(data)
    data = data.set_index("DateTime")
    data["DateTime"] = data.index
    data = standardize_column_names(data)

    #### Compute additional variables ####
    data["SW_POT_sm"] = sw_pot_sm(data)
    data["SW_POT_sm_diff"] = sw_pot_sm_diff(data)
    data["CWD"] = wdefcum(data)
    data = data.set_index("DateTime")

    # Drop problematic rows
    data = data.replace(-9999, np.nan)
    return data


def load_data(site_name, path=None):
    """
    Loads data flux tower data from different sources for flux partitioning experiments.

    Args:
        site_name (str): Flux site in the format of the site code , e.g. "DE-Hai".

    Returns:
        data (pd.DataFrame): Flux data including all available meterological data.
    """

    if path is None:
        data_folder = pathlib.Path(__file__).parent.parent.parent.joinpath("data")
    else:
        data_folder = pathlib.Path(path).expanduser()

    filename = None
    for file in data_folder.glob("*"):
        # Most sites have half-hourly data
        if file.name.startswith(f"FLX_{site_name}_FLUXNET2015_FULLSET_HH"):
            filename = file.name
        # A few have hourly data
        elif file.name.startswith(f"FLX_{site_name}_FLUXNET2015_FULLSET_HR"):
            filename = file.name

    if filename is None:
        for file in data_folder.glob("*"):
            # Most sites have half-hourly data
            if file.name.find(site_name) != -1:
                filename = file.name

    # Write exception for when filename is None
    data = pd.read_csv(data_folder.joinpath(filename))
    data["site"] = site_name
    return data


def unwrap_time(data):
    """
    Takes a TIMESTAMP column of format 20040102 in generates a column for data,
    time, month, year, doy

    Args:
        data (pd.DataFrame): Dataframe with all the FLUXNET data including the time stamp.

    Returns:
        df (pd.DataFrame): Copy of dataframe with standardized date and time columns.
    """

    df = data.copy()

    # Compute DateTime column depending on different formatings of the data
    if "datetime" in df.columns:
        df = df.rename(columns={"datetime": "DateTime"})
    if "TIMESTAMP_START" in df.columns:
        # Choose the middle of the timewindow as reference
        df["DateTime"] = pd.to_datetime(df["TIMESTAMP_START"] + 15, format="%Y%m%d%H%M")
    if all(col in df.columns for col in ["Date", "Time"]):
        df["DateTime"] = df["Date"] + "T" + df["Time"]
    if pd.api.types.is_object_dtype(df["DateTime"]):
        df["DateTime"] = pd.to_datetime(df["DateTime"], format="%Y-%m-%dT%H:%M:%S")

    df["Date"] = pd.to_datetime(df["DateTime"]).dt.date
    df["Time"] = pd.to_datetime(df["DateTime"]).dt.time
    df["Month"] = pd.to_datetime(df["Date"]).dt.month
    df["Year"] = pd.to_datetime(df["Date"]).dt.year
    df["doy"] = pd.to_datetime(df["DateTime"]).dt.dayofyear
    return df


def standardize_column_names(data):
    """
    Changes column names of a dataset according to the relevant_variables file.

    Args:
        data (pd.DataFrame): Dataframe with flux data.

    Returns:
        df (pd.DataFrame): Copy of dataframe with standardized column names.
    """

    df = data.copy()
    for old, new in relevant_variables.mappings.items():
        # if old in data.columns and new not in data.columns:
        if old in df.columns:
            df[new] = df[old]
    return df


def wdefcum(data):
    """
    Function to compute cumulative water deficite from precipitation P and latent heat flux LE.
    Equation obtained from Nuno and Markus. This is its simplest form the LE to ET function can be
    made more complex as a next step.

    The names and units of the variables are
        P (unit): precipitation
        LE (unit): latent heat flux
        ET (unit): evapotranspiration
        CWD (unit): cumulative water deficit


    Args:
        data (pd.DataFrame): Dataframe with flux data.


    Returns:
        CWD (float64): cumulative water deficit
    """
    # TODO: Discuss. Is this a proper way to compute it and what does it mean.
    # TODO: Figure out what to do with naming conventions in scientific context.
    P = data["P"].values
    LE = data["LE"].values
    if "P" in data.columns and "LE" in data.columns:
        n = len(LE)
        # TODO: Figure out the meaning of the magic number here.
        ET = LE / 2.45e6 * 1800
        CWD = np.zeros(n)
        CWD[1:] = np.nan

        for i in range(1, n):
            CWD[i] = np.minimum(CWD[i - 1] + P[i - 1] - ET[i - 1], 0)

        if np.isnan(CWD[i]):
            CWD[i] = CWD[i - 1]
    else:
        print("You are missing either P or LE to compute CWD")
        CWD = None
    return CWD


def moving_average(x, w):
    """
    Computes the moving average of window size w over array x

    Args:
        x (float64): array that is convolved over
        w (int64): window size

    Returns:
        float64: moving averages of x
    """
    return np.convolve(x, np.ones(w), "same") / w


def sw_pot_sm(data):
    """
    Smooth curve of potential incoming radiation computed as 10 day movering averages
    over SW_IN_POT.

    Args:
        data (pd.DataFrame): Dataframe with all the flux data. Needs 'SW_IN_POT' as column.

    Returns:
        float64: smooth cycle of potential incoming radiation
    """
    SW_POT_sm = moving_average(data["SW_IN_POT"], 480)
    return SW_POT_sm


def sw_pot_sm_diff(data):
    """
    Smooth derivative of the smooth cycle of potential incoming radiation.

    Args:
        data (pd.DataFrame): Dataframe with all the flux data. Needs 'SW_POT_sm' as column.

    Returns:
        SW_POT_sm_diff (float64): smooth derivative of smooth potential incoming radiation
    """

    SW_POT_sm = data["SW_POT_sm"].values
    SW_POT_sm_diff = np.hstack(
        (
            np.array(SW_POT_sm[1] - SW_POT_sm[0]),
            (np.roll(SW_POT_sm, -1) - SW_POT_sm)[1:],
        )
    )
    SW_POT_sm_diff = moving_average(10000 * SW_POT_sm_diff, 480)
    return SW_POT_sm_diff


def sw_pot_diff(data):
    """
    Smooth derivative of the smooth cycle of potential incoming radiation.

    Args:
        data (pd.DataFrame): Dataframe with all the flux data. Needs 'SW_POT_sm' as column.

    Returns:
        SW_POT_sm_diff (float64): smooth derivative of smooth potential incoming radiation
    """

    SW_IN_POT = data["SW_IN_POT"]
    SW_POT_diff = np.hstack(
        (
            np.array(SW_IN_POT[1] - SW_IN_POT[0]),
            (np.roll(SW_IN_POT, -1) - SW_IN_POT)[1:],
        )
    )
    # SW_POT_diff = moving_average(10000*SW_POT_diff, 480)
    return SW_POT_diff
