import os
from argparse import ArgumentParser
import pandas as pd

# os.environ["TF_CPP_MIN_LOG_LEVEL"]="0"
# os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 0 1 7
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".70"

import numpy as np
from src.datasets.loaders import load_dataset
from src.models.NN import (
    Q10Regressor,
)
from src.datasets.utility import BootstrapLoader
from jax import random
import pathlib
from src.utility.experiments import create_experiment_folder

# jax.devices("gpu")[0]


def main(args):
    ################ Define the experiment  ################
    # Data
    dataset_config = {
        "site": "AT-Neu",
        "target": args.target,
        "frac": 0.2,
        "years": [2003, 2004, 2005, 2006, 2007],
        "noise": 0.2,
        "seed": 33,
    }
    
    train, val, out = load_dataset(**dataset_config)
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

    X_Rb = np.c_[X, T] if args.T else X

    X_val = EV_val[indices_val]
    T_val = driver_val[indices_val][:, None]
    y_val = RECO_val[indices_val][:, None]

    X_Rb_val = np.c_[X_val, T_val] if args.T else X_val
    data_val = [X_Rb_val, T_val, y_val]

    drop_p = 0.2 if args.reg else 0.0

    model_config = {
        "layers": [X_Rb.shape[1], 16, 16, 1],
        "ensemble_size": 100,
        "p": drop_p,
        "weight_decay": 0,
        "Q10_mean_guess":1.5,
    }


    ################ Save the experiment setup  ################    
    experiment_dict = {'model_config': model_config, 'data_config': dataset_config}

    #### Create the experiment folder ####
    print("Creating the experiment folder...")
    reg_str = "_reg" if args.reg else ""
    T_str = "_T" if args.T else ""
    experiment_path = create_experiment_folder(f"output_NN{reg_str}{T_str}_{args.target}", experiment_dict, path=args.results_folder)

    results = pd.DataFrame(columns=["Q"])

    rng_key = random.PRNGKey(args.seed)
    rng_key, rng_key2, rng_key3 = random.split(rng_key, 3)

    if args.target == "syn":
        dataset = BootstrapLoader(
            X_Rb, T, y, 256, 100, n_samples=args.samples, rng_key=rng_key, replace=False
        )
    else:
        dataset = BootstrapLoader(
            X_Rb, T, y, 256, 100, n_samples=sum(indices), rng_key=rng_key, replace=False
        )
    regressor = Q10Regressor(
        [X_Rb.shape[1], 16, 16, 1],
        100,
        p=drop_p,
        weight_decay=0,
        rng_key=rng_key2,
        Q10_mean_guess=1.5,
    )
    regressor.fit(dataset, data_val, nIter=10000, rng_key=rng_key3)

    sample, Rb_pred, Q10 = regressor.posterior(X_Rb, T)
    results["Q10"] = Q10[:, 0]    
    results.to_csv(experiment_path.joinpath("Q10s.csv"))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "-samples", type=int, default=25600, help="number of samples to train with"
    )
    parser.add_argument("--reg", action="store_true")
    parser.add_argument("--no-reg", dest="reg", action="store_false")
    parser.add_argument("--T", action="store_true")
    parser.add_argument("--no-T", dest="T", action="store_false")
    parser.add_argument(
        "--target", dest="target", default="syn", help="syn or measured data"
    )
    parser.add_argument("--seed", dest="seed", default=33, help="random seed")
    parser.add_argument("--results_folder", type=str, default=None, help="Folder to save results")
    parser.add_argument("--data_folder", type=str, default=None, help="Folder to load data from")

    args = parser.parse_args()
    if args.data_folder is None:
        args.data_folder = pathlib.Path(__file__).parent.parent.joinpath('data')
    if args.results_folder is None:
        args.results_folder = pathlib.Path(__file__).parent.parent.joinpath('results')

    print(args.data_folder)
    print(args.results_folder)
    main(args)
