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
from src.datasets.loaders import load_dataset, BootstrapLoader
from src.models.DML import (
    EnsembleRegressor,
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
    parser.add_argument("--reg", action="store_true")
    parser.add_argument("--no-reg", dest="reg", action="store_false")
    parser.add_argument("--T", action="store_true")
    parser.add_argument("--no-T", dest="T", action="store_false")
    parser.add_argument("--seed", dest="seed", default=33, action="random seed")

    args = parser.parse_args()
    for k, v in kwargs.items():
        setattr(args, k, v)

    print(">>> Starting experiment.")
    if args.reg:
        drop_p = 0.2
    else:
        drop_p = 0.0
    if args.data == "syn":
        train, val, out = load_dataset(
            "Synthetic4BookChap.csv",
            frac=0.2,
            years=[2003, 2004, 2005, 2006, 2007],
        )

    elif args.data == "real":
        train, val, out = load_dataset(
            "Synthetic4BookChap.csv",
            frac=0.2,
            years=[2003, 2004, 2005, 2006, 2007],
        )

        (
            EV_train,
            RECO_train,
            RECO_train_GT,
            driver_train,
            EV_val,
            RECO_val,
            RECO_val_GT,
            driver_val,
            RECO_max_abs,
        ) = out
        indices = np.array(
            ~np.isnan(RECO_train)
            * np.prod(~np.isnan(EV_train), axis=1)
            * ~np.isnan(driver_train),
            dtype=bool,
        )
        indices_val = np.array(
            ~np.isnan(RECO_val)
            * np.prod(~np.isnan(EV_val), axis=1)
            * ~np.isnan(driver_val),
            dtype=bool,
        )

        X = EV_train[indices]
        T = driver_train[indices][:, None]
        y = RECO_train[indices][:, None]

        if args.T:
            X_Rb = np.c_[X, T]
        else:
            X_Rb = X

        X_val = EV_val[indices_val]
        T_val = driver_val[indices_val][:, None]
        y_val = RECO_val[indices_val][:, None]

        if args.T:
            X_Rb_val = np.c_[X_val, T_val]
        else:
            X_Rb_val = X_val

        data_val = [X_Rb_val, T_val, y_val]

        results = pd.DataFrame(columns=["Q"])

        rng_key = random.PRNGKey(args.seed)
        rng_key, rng_key2, rng_key3 = random.split(rng_key, 3)

        dataset = BootstrapLoader(
            X_Rb, T, y, 256, 100, n_samples=args.samples, rng_key=rng_key, replace=False
        )

        regressor = EnsembleRegressor(
            [X_Rb.shape[1], 16, 16, 1],
            100,
            p=drop_p,
            weight_decay=0,
            rng_key=rng_key2,
            Q10_mean_guess=1.5,
        )
        regressor.fit(dataset, data_val, nIter=10000, rng_key=rng_key3)

        sample, Rb_pred, Q10 = regressor.posterior(X_Rb, T)
        sample_val, Rb_pred_val, Q10 = regressor.posterior(X_Rb_val, T_val)

        results["Q"] = Q10[:, 0]


if __name__ == "__main__":
    main()
