"""File containing a mapping of old to new column names and relevant variables."""

mappings = {
    "TA_F": "TA",  # Fluxnet mappings
    "TA_F_QC": "TA_QC",
    "SW_IN_F": "SW_IN",
    "SW_IN_F_QC": "SW_IN_QC",
    "VPD_F": "VPD",
    "VPD_F_QC": "VPD_QC",
    'P_F': 'P',
    'P_F_QC': 'P_QC',
    'P_ERA': 'P_ERA',
    "TS_F_MDS_1": "TS_1",
    "TS_F_MDS_1_QC": "TS_1_QC",
    "TS_F_MDS_2": "TS_2",
    "TS_F_MDS_2_QC": "TS_2_QC",
    'LE_F_MDS': 'LE',
    'LE_F_MDS_QC': 'LE_QC',
    "SWC_F_MDS_1": "SWC_1",
    "SWC_F_MDS_1_QC": "SWC_1_QC",
    "SWC_F_MDS_2": "SWC_2",
    "SWC_F_MDS_2_QC": "SWC_2_QC",
    "NEE_CUT_USTAR50": "NEE",
    "NEE_CUT_USTAR50_QC": "NEE_QC",
    "RECO_NT_CUT_USTAR50": "RECO_NT",
    "GPP_NT_CUT_USTAR50": "GPP_NT",
    "RECO_DT_CUT_USTAR50": "RECO_DT",
    "GPP_DT_CUT_USTAR50": "GPP_DT",
}

variables = [
    "TA",
    "TA_QC",
    "SW_IN",
    "SW_IN_QC",
    "SW_IN_POT",
    "VPD",
    "VPD_QC",
    'LE',
    'LE_QC',
    'P',
    'P_QC',
    "TS_1",
    "TS_1_QC",
    "TS_2",
    "TS_2_QC",
    "SWC_1",
    "SWC_1_QC",
    "SWC_2",
    "SWC_2_QC",
    "NEE",
    "NEE_QC",
    "NEE_RANDUNC",
    "RECO_NT",
    "GPP_NT",
    "RECO_DT",
    "GPP_DT",
    "GPP_syn",
    "RECO_syn",
    "wdefCum",
    "CWD",
    "doy",
    "SW_IN",
    "Year",
    "Month",
    "doy",
    "site",
    "DateTime",
    "NIGHT"]
