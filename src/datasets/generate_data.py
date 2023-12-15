""" Functions for generating synthetic datasets """
import os
import random

import pandas as pd
import numpy as np
from pathlib import Path

import dml4fluxes.datasets.relevant_variables as relevant_variables


def synthetic_dataset(data, Q10):
    """
    Generates or loads a precomputed dataset based on the Q10 model for RECO and LUE model for GPP.
    In its simplest form it resembles the model from the book chapter.
    
    Args:
        data (pd.DataFrame): FLUXNET dataset
        Q10 (float): Q10 specifying the value of the Q10 model based simulation
        relnoise (float): non-negative sd of noise applied as a factor (1+noise) to the final NEE

    Returns:
        data (pd.DataFrame): Dataset with additional columns with intermediate values of the data
        generation process. In particular RECO_syn, GPP_syn, NEE_syn_clean (noise free), NEE_syn
    """
    SW_IN = data['SW_IN']
    SW_POT_sm = data['SW_POT_sm']
    SW_POT_sm_diff = data['SW_POT_sm_diff']
    TA = data['TA']
    VPD = data['VPD']
        
    RUE_syn = 0.5 * np.exp(-(0.1*(TA-20))**2) * np.minimum(1, np.exp(-0.1 * (VPD-10)))
        
    GPP_syn = RUE_syn * SW_IN / 12.011
    Rb_syn = SW_POT_sm * 0.01 - SW_POT_sm_diff * 0.005
    Rb_syn = 0.75 * (Rb_syn - np.nanmin(Rb_syn) + 0.1*np.pi)
    RECO_syn = Rb_syn * Q10 ** (0.1*(TA-15.0))
    NEE_syn = RECO_syn - GPP_syn 
    
    data['RUE_syn'] = RUE_syn
    data['GPP_syn'] = GPP_syn
    data['Rb_syn'] = Rb_syn
    data['RECO_syn'] = RECO_syn
    data['NEE_syn'] = NEE_syn
        
    return data