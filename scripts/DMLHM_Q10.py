import os
import time
from argparse import ArgumentParser
import random as orandom
from pathlib import Path
import pandas as pd

# os.environ["TF_CPP_MIN_LOG_LEVEL"]="0"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 0 1 7
# os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".70"

import numpy as np
from src.datasets.loaders import load_dataset

from src.models.DML import (
    DML,
)
from jax import random
from jax import numpy as jnp

from sklearn.metrics import r2_score, mean_squared_error
import jax

# jax.devices("gpu")[0]


def main(parser: ArgumentParser = None, **kwargs):
    if parser is None:
        parser = ArgumentParser()
    parser.add_argument(
        "-samples", type=int, default=25600, help="number of samples to train with"
    )
    parser.add_argument(
        "-ml",
        type=str,
        choices=("nn", "rf"),
        default="rf",
        help="Choice of nuisance estimators",
    )
    parser.add_argument("--reg", action="store_true")
    parser.add_argument("--no-reg", dest="reg", action="store_false")
    parser.add_argument("--T", action="store_true")
    parser.add_argument("--no-T", dest="T", action="store_false")
    parser.add_argument("--seed", dest="seed", default=33, help="random seed")
    parser.add_argument("--data", dest="data", default="syn", help="syn or real data")

    args = parser.parse_args()
    for k, v in kwargs.items():
        setattr(args, k, v)

    print(">>> Starting experiment.")
    if args.reg:
        drop_p = 0.2
    else:
        drop_p = 0.0

    if args.data == "syn":
        train, out = load_dataset(
        site='AT-Neu',
        option="syn",
        frac=0,
        years=[2003, 2004, 2005, 2006, 2007],
        noise=0.2,
        seed=33
        )
    elif args.data == "real":
        train, out = load_dataset(
        site='AT-Neu',
        option="real",
        frac=0,
        years=[2003, 2004, 2005, 2006, 2007],
        noise=0,
        seed=33
        )
    
    (
        EV_train,
        RECO_train,
        RECO_train_GT,
        driver_train,
        RECO_max_abs,
    ) = out
    indices = np.array(
        ~np.isnan(RECO_train)
        * np.prod(~np.isnan(EV_train), axis=1)
        * ~np.isnan(driver_train),
        dtype=bool,
    )

    orandom.seed(args.seed)
    rng_keys1 = orandom.sample(range(1000000), 1)
    rng_keys2 = orandom.sample(range(1000000), 1)

    if args.ml == "nn":
        trainer_config = {
            "weight_decay": 0,
            "iterations": 4000,
            "split": 1.0,
        }
        model_config = {
            "layers": [2, 16, 16, 1],
            "final_nonlin": False,
            "dropout_p": drop_p,
            "ensemble_size": 1,
        }
        config = {
            "ml_m": "EnsembleCustomJNN",
            "ml_m_config": {
                "model_config": model_config,
                "trainer_config": trainer_config,
            },
            "ml_l": "EnsembleCustomJNN",
            "ml_l_config": {
                "model_config": model_config,
                "trainer_config": trainer_config,
            },
        }
    elif args.ml == "rf":
        config = {
            "ml_m": "RandomForestRegressor",
            "ml_m_config": {"n_estimators": 100, "min_samples_leaf": 5},
            "ml_l": "RandomForestRegressor",
            "ml_l_config": {"n_estimators": 100, "min_samples_leaf": 5},
        }

    config["dml_config"] = {
                "dml_procedure": "dml2",
                "score": "partialling out",
                "n_folds": 5,
                "n_rep": 1,
            }

    # Define final estimator for Rb on the residuals. Potentially include T
    if args.T:
        T = 1
    else:
        T = 0

    config["ml_Rb_config"] = {
        "model_config": {
            "layers": [2 + T, 16, 16, 1],
            "final_nonlin": True,
            "dropout_p": drop_p,
            "ensemble_size": 1,
        },
        "trainer_config": {
            "weight_decay": 0,
            "iterations": 10000,
            "split": 1.0,
        },
    }
    config["ml_Rb"] = "EnsembleCustomJNN"

    seed = rng_keys1[0]
    orandom.seed(seed)
    np.random.seed(seed)
    bootstrapsample = np.random.choice(
        np.arange(len(indices))[indices], size=args.samples, replace=False
    )
    bootstrapsample.sort()

    X = EV_train[bootstrapsample]
    T = driver_train[bootstrapsample]
    y = RECO_train[bootstrapsample]

    X_T = np.c_[X, T]

    config["seed"] = rng_keys2
    inputs = {"T": T, "EV": X}
    dml = DML(config)
    dml.fit(inputs, y)
    
    if args.T:
        X_Rb = X_T
    else:
        X_Rb = X
        
    inputs = {"T": T, "EV": X, "Rb": X_Rb}
    dml.init_Rb(config)

    dml.fit_Rb(inputs, y)
    dml.Q10

if __name__ == "__main__":
    main()
