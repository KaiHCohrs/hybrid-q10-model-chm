import os
from argparse import ArgumentParser
import random as orandom

# os.environ["TF_CPP_MIN_LOG_LEVEL"]="0"
# os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 0 1 7
#os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".70"

import numpy as np
from src.datasets.loaders import load_dataset
import pathlib
from src.utility.experiments import create_experiment_folder
import pandas as pd

from src.models.DML import (
    DML
)

def main(args):
    ################ Define the experiment  ################
    # Data
    dataset_config = {
        "site": "AT-Neu",
        "target": args.target,
        "frac": 0,
        "years": [2003, 2004, 2005, 2006, 2007],
        "noise": 0.2,
        "seed": 33,
    }

    train, out = load_dataset(**dataset_config)
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

    # Model
    drop_p = 0.2 if args.reg else 0.0

    if args.ml == "nn":
        trainer_config = {
            "weight_decay": 0,
            "iterations": 4000,
            "split": 1.0,
        }
        model_config = {
            "layers": [EV_train.shape[1], 16, 16, 1],
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
    config["ml_Rb"] = "EnsembleCustomJNN"
    config["ml_Rb_config"] = {
        "model_config": {
            "layers": [EV_train.shape[1] + T, 16, 16, 1],
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
    
    
    ################ Save the experiment setup  ################    
    experiment_dict = {'model_config': config, 'data_config': dataset_config}

    #### Create the experiment folder ####
    print("Creating the experiment folder...")
    reg_str = "_reg" if args.reg else ""
    T_str = "_T" if args.T else ""
    experiment_path = create_experiment_folder(f"output_DML{reg_str}{T_str}_{args.target}", experiment_dict, path=args.results_folder)


    orandom.seed(args.seed)
    Q10s = list()
    for i in range(100):
        rng_keys1 = orandom.sample(range(1000000), 1)
        rng_keys2 = orandom.sample(range(1000000), 1)
        
        if args.target == "syn":
            seed = rng_keys1[0]
            orandom.seed(seed)
            np.random.seed(seed)
            bootstrapsample = np.random.choice(
                np.arange(len(indices))[indices], size=args.samples, replace=False
            )
            bootstrapsample.sort()
        else:
            bootstrapsample = indices

        X = EV_train[bootstrapsample]
        T = driver_train[bootstrapsample]
        y = RECO_train[bootstrapsample]


        config["seed"] = rng_keys2
        inputs = {"T": T, "EV": X}
        dml = DML(config)
        dml.fit(inputs, y)


        X_Rb = np.c_[X, T] if args.T else X

        inputs = {"T": T, "EV": X, "Rb": X_Rb}
        dml.init_Rb(config)

        #dml.fit_Rb(inputs, y)
        Q10s.append(dml.Q10_mean)

        pd.DataFrame(Q10s, columns=['Q10']).to_csv(experiment_path.joinpath("Q10s.csv"))
        print(f"Finish estimating model {i+1}/100...")
    

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "-samples", type=int, default=3200, help="number of samples to train with"
    )
    parser.add_argument(
        "--ml",
        type=str,
        choices=("nn", "rf"),
        default="rf",
        help="Choice of nuisance estimators",
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