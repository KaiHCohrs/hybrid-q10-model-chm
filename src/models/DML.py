import sys
import os

import numpy as np
import random as orandom

import doubleml as dml
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    StackingRegressor,
)
from src.models.NN_regressor import EnsembleCustomJNN


os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 0 1 7
# os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"]=".50"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"


class DML:
    def __init__(self, config):
        # Define the learners:
        self.seed = config["seed"]
        self.config = config

        ml_m = getattr(sys.modules[__name__], config["ml_m"])
        self.ml_m_config = config["ml_m_config"]
        self.ml_m = ml_m(**config["ml_m_config"])

        ml_l = getattr(sys.modules[__name__], config["ml_l"])
        self.ml_l_config = config["ml_l_config"]
        self.ml_l = ml_l(**config["ml_l_config"])

        # Set DML configs
        self.dml_config = config["dml_config"]

        ml_Rb = getattr(sys.modules[__name__], config["ml_Rb"])
        self.ml_Rb_config = config["ml_Rb_config"]
        self.ml_Rb = ml_Rb(**config["ml_Rb_config"])

        # Further attributes
        self.Q10_mean = None
        self.Q10_std = None
        self.Rb_mean = None
        self.Rb_std = None
        self.Rb = None

    def init_Rb(self, config):
        ml_Rb = getattr(sys.modules[__name__], config["ml_Rb"])
        self.ml_Rb_config = config["ml_Rb_config"]
        self.ml_Rb = ml_Rb(**config["ml_Rb_config"])

    def fit(self, X, y):
        # Requires to apply log transform first
        y_log = np.log(y)

        # Scale T for the Q10 model
        d = 0.1 * (X["T"] - 15)

        # Normalize the input
        self.means = X["EV"].mean(axis=0)
        self.std = X["EV"].std(axis=0)

        X["EV"] = (X["EV"] - X["EV"].mean(axis=0)) / X["EV"].std(0)

        # from seed generate twice the amount of seeds according to the split
        # print(self.seed)
        # orandom.seed(self.seed)
        ml_l_seeds = orandom.sample(range(10000), self.dml_config["n_folds"])
        ml_m_seeds = orandom.sample(range(10000), self.dml_config["n_folds"])

        self.obj_dml_data = dml.DoubleMLData.from_arrays(
            x=X["EV"], y=y_log, d=d, use_other_treat_as_covariate=False
        )
        self.dml_plr_obj = dml.DoubleMLPLR(
            self.obj_dml_data, self.ml_m, self.ml_l, **self.dml_config
        )

        if self.config["ml_l"] == "EnsembleCustomJNN":
            parameters_ml_l = list()
            for seed in ml_l_seeds:
                parameters_ml_l.append(
                    {
                        "model_config": self.ml_l_config["model_config"],
                        "trainer_config": self.ml_l_config["trainer_config"],
                        "seed": seed,
                    }
                )
            self.dml_plr_obj.set_ml_nuisance_params("ml_l", "d", [parameters_ml_l])

        if self.config["ml_m"] == "EnsembleCustomJNN":
            parameters_ml_m = list()
            for seed in ml_m_seeds:
                parameters_ml_m.append(
                    {
                        "model_config": self.ml_m_config["model_config"],
                        "trainer_config": self.ml_m_config["trainer_config"],
                        "seed": seed,
                    }
                )
            self.dml_plr_obj.set_ml_nuisance_params("ml_m", "d", [parameters_ml_m])

        self.dml_plr_obj.fit(store_models=True)

        self.Q10_mean = np.exp(self.dml_plr_obj.coef)[0]
        self.Q10_std_upper = (
            np.exp(self.dml_plr_obj.coef + 1.96 * self.dml_plr_obj.se)[0]
            - self.Q10_mean
        )
        self.Q10_std_lower = (
            self.Q10_mean
            - np.exp(self.dml_plr_obj.coef - 1.96 * self.dml_plr_obj.se)[0]
        )

        def Rb_mean(x):
            Reco_mean = np.mean(
                [ml_l.predict(x) for ml_l in self.dml_plr_obj._models["ml_l"]["d"][0]],
                axis=0,
            )
            T_mean = np.mean(
                [ml_m.predict(x) for ml_m in self.dml_plr_obj._models["ml_m"]["d"][0]],
                axis=0,
            )

            return np.exp(Reco_mean - self.dml_plr_obj.coef * T_mean)

        def Rb_std(x):
            return np.std(
                [ml_l.predict(x) for ml_l in self.dml_plr_obj._models["ml_l"]["d"][0]],
                axis=1,
            )

        self.Rb_mean = Rb_mean
        self.Rb_std = Rb_std

    def fit_Rb(self, X, y):
        # Scale T for the Q10 model
        d = 0.1 * (X["T"] - 15)

        ## In actual space
        res = y / (self.Q10_mean**d)
        self.ml_Rb.fit(X["Rb"], res)

        def Rb(x):
            return self.ml_Rb.predict(x)

        self.Rb = Rb

    def nuisance_score(self, X, y):
        # inputs need to be unnormalized

        # Requires to apply log transform first
        y_log = np.log(y)

        # Scale T for the Q10 model
        d = 0.1 * (X["T"] - 15)

        # Normalize the input
        X["EV"] = (X["EV"] - self.means) / self.std
        # y_pred = np.mean([ml_l.predict(X['EV']) for ml_l in self.dml_plr_obj._models['ml_l']['d'][0]], axis=0)
        # d_pred = np.mean([ml_m.predict(X['EV']) for ml_m in self.dml_plr_obj._models['ml_m']['d'][0]], axis=0)
        ml_l_score = np.sqrt(
            np.mean(
                [
                    (ml_l.predict(X["EV"]) - y_log) ** 2
                    for ml_l in self.dml_plr_obj._models["ml_l"]["d"][0]
                ]
            )
        )
        ml_m_score = np.sqrt(
            np.mean(
                [
                    (ml_m.predict(X["EV"]) - d) ** 2
                    for ml_m in self.dml_plr_obj._models["ml_m"]["d"][0]
                ]
            )
        )

        return {"rmse_ml_l": ml_l_score, "rmse_ml_m": ml_m_score}

    def predict(self, X, mean=False):
        d = 0.1 * (X["T"] - 15)
        if mean:
            return self.Rb_mean(X["EV"]) * self.Q10_mean**d
        else:
            return self.Rb(X["EV"]) * self.Q10_mean**d

    def score(self, X, y, mean=False):
        y_pred = self.predict(X, mean)
        return mean_squared_error(y, y_pred, squared=False), r2_score(y, y_pred)

    def set_params(self, Q10_mean, Rb_mean):
        self.Q10_mean = Q10_mean
        self.Rb_mean = Rb_mean
