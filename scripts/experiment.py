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
    DML,
    EnsembleRegressor,
    RPNEnsembleRegressor,
    EnsembleCustomJNN,
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
        "-method",
        type=str,
        choices=("DML", "GD"),
        default="DML",
        help="GD or DML-based method",
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
    parser.add_argument("--seed", dest="seed", default=33, action="random seed")

    args = parser.parse_args()
    for k, v in kwargs.items():
        setattr(args, k, v)

    weight_decay = 0.2

    print(">>> Starting experiment.")
    if args.reg:
        drop_p = 0.2
    else:
        drop_p = 0.0
        
    n_folds = 5
    
    if args.method == "DML":
        for T in ["no_T", "T"]:
            results = pd.DataFrame(
                columns=[
                    "i",
                    "Q",
                ]
            )

            train, out = load_dataset(
                "Synthetic4BookChap.csv",
                normalize=False,
                frac=0,
                years=[2003, 2004, 2005, 2006, 2007],
                noise=args.noise,
                IV=args.IV,
            )
            (
                EV_train,
                RECO_train,
                RECO_train_GT,
                driver_train,
                RECO_max_abs,
                years,
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
            model_config = {
                "layers": [2 + ex, 16, 16, 1],
                "final_nonlin": False,
                "dropout_p": drop_p,
                "ensemble_size": 1,
            }

            if args.ml == "jnn":
                trainer_config = {
                    "weight_decay": weight_decay,
                    "iterations": int(args.training_iter),
                    "split": 1.0,
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
                    "dml_config": dict(
                        dml_procedure="dml2",  #'dml2',
                        score="partialling out",
                        n_folds=n_folds,
                        n_rep=1,
                    ),
                }
            elif args.ml == "rf":
                config = {
                    "ml_m": "RandomForestRegressor",
                    "ml_m_config": {"n_estimators": 100, "min_samples_leaf": 5},
                    "ml_l": "RandomForestRegressor",
                    "ml_l_config": {"n_estimators": 100, "min_samples_leaf": 5},
                    "dml_config": dict(
                        dml_procedure="dml2",
                        score="partialling out",
                        n_folds=n_folds,
                        n_rep=1,
                    )
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
                    "weight_decay": weight_decay,
                    "iterations": 2 * args.training_iter,
                    "split": 1.0,
                },
            }
            config["ml_Rb"] = "EnsembleCustomJNN"

            seed = rng_keys1
            orandom.seed(seed)
            np.random.seed(seed)
            bootstrapsample = np.random.choice(
                np.arange(len(indices))[indices], size=args.samples, replace=False
            )
            bootstrapsample.sort()

            X = EV_train[bootstrapsample]
            T = driver_train[bootstrapsample]
            y = RECO_train[bootstrapsample]
            y_GT = RECO_train_GT[bootstrapsample]
            Rb = train.Rb_syn[bootstrapsample]

            X_T = np.c_[X, T]

            config["seed"] = rng_keys2
            inputs = {"T": T, "EV": X}
            dml = DML(config)
            dml.fit(inputs, y)

            #### Scores on hold out data
            test, out = load_dataset(
                "Synthetic4BookChap.csv",
                normalize=False,
                frac=0,
                years=[2008, 2009],
                noise=args.noise,
            )
            (
                EV_test,
                RECO_test,
                RECO_test_GT,
                driver_test,
                RECO_max_abs_test,
                years,
            ) = out
            indices_test = np.array(
                ~np.isnan(RECO_test)
                * np.prod(~np.isnan(EV_test), axis=1)
                * ~np.isnan(driver_test),
                dtype=bool,
            )

            X_test = EV_test[indices_test]
            T_test = driver_test[indices_test]
            y_test = RECO_test[indices_test]
            y_test_GT = RECO_test_GT[indices_test]
            Rb_test = test.Rb_syn[indices_test]

            X_T_test = np.c_[X_test, T_test]
            for k, [X_Rb, X_Rb_test] in enumerate([[X, X_test], [X_T, X_T_test]]):
                config["ml_Rb_config"] = {
                    "model_config": {
                        "layers": [2 + k, 16, 16, 1],
                        "final_nonlin": True,
                        "dropout_p": drop_p,
                        "ensemble_size": 1,
                    },
                    "trainer_config": {
                        "weight_decay": weight_decay,
                        "iterations": 2 * args.training_iter,
                        "split": 1.0,
                    },
                }
                config["ml_Rb"] = "EnsembleCustomJNN"

                inputs = {"T": T, "EV": X, "Rb": X_Rb}
                dml.init_Rb(config)

                dml.fit_Rb(inputs, y)

                rmse_out = dml.nuisance_score(
                    {"T": T_test, "EV": X_test}, y_test_GT
                )
                Rb_pred = dml.Rb(X_Rb)
                Rb_mean_pred = dml.Rb_mean(X)
                Rb_pred_test = dml.Rb(X_Rb_test)
                Rb_mean_pred_test = dml.Rb_mean(X_test)

                Reco = dml.predict({"T": T, "EV": X_Rb})
                Reco_mean = dml.predict({"T": T, "EV": X}, mean=True)
                Reco_test = dml.predict({"T": T_test, "EV": X_Rb_test})
                Reco_mean_test = dml.predict({"T": T_test, "EV": X_test}, mean=True)

                results = pd.read_csv(results_file, index_col=0)

                results.loc["rmse_Rb"] = mean_squared_error(
                    Rb, Rb_pred, squared=False
                )
                results.loc["rmse_Rb_test"] = mean_squared_error(
                    Rb_test, Rb_pred_test, squared=False
                )
                results.loc["rmse_Reco"] = mean_squared_error(
                    y_GT, Reco, squared=False
                )
                results.loc["rmse_Reco_test"] = mean_squared_error(
                    y_test_GT, Reco_test, squared=False
                )
                results.loc["rmse_Rb_mean"] = mean_squared_error(
                    Rb, Rb_mean_pred, squared=False
                )
                results.loc["rmse_Rb_mean_test"] = mean_squared_error(
                    Rb_test, Rb_mean_pred_test, squared=False
                )
                results.loc["rmse_Reco_mean"] = mean_squared_error(
                    y_GT, Reco_mean, squared=False
                )
                results.loc["rmse_Reco_mean_test"] = mean_squared_error(
                    y_test_GT, Reco_mean_test, squared=False
                )
                results.loc["R2_Rb"] = r2_score(Rb, Rb_pred)
                results.loc["R2_Rb_test"] = r2_score(Rb_test, Rb_pred_test)
                results.loc["R2_Reco"] = r2_score(y_GT, Reco)
                results.loc["R2_Reco_test"] = r2_score(y_test_GT, Reco_test)
                results.loc["R2_Rb_mean"] = r2_score(Rb, Rb_mean_pred)
                results.loc["R2_Rb_mean_test"] = r2_score(Rb_test, Rb_mean_pred_test)
                results.loc["R2_Reco_mean"] = r2_score(y_GT, Reco_mean)
                results.loc["R2_Reco_mean_test"] = r2_score(
                    y_test_GT, Reco_mean_test
                )
                results.loc["rmse_ml_l"] = dml.dml_plr_obj.rmses["ml_l"][0]
                results.loc["rmse_ml_m"] = dml.dml_plr_obj.rmses["ml_m"][0]
                results.loc["rmse_ml_l_test"] = rmse_out["rmse_ml_l"]
                results.loc["rmse_ml_m_test"] = rmse_out["rmse_ml_m"]
                results.to_csv(results_file)

                dml.Q10

            del dml, out, train, bootstrapsample, X, T, y, inputs

    elif args.method == "GD":
        train, val, out = load_dataset(
            "Synthetic4BookChap.csv",
            normalize=False,
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
        y_GT = RECO_train_GT[indices][:, None]
        Rb = train.Rb_syn[indices].values[:, None]

        if args.T:
            X_Rb = np.c_[X, T]
        else:
            X_Rb = X

        X_val = EV_val[indices_val]
        T_val = driver_val[indices_val][:, None]
        y_val = RECO_val[indices_val][:, None]
        y_val_GT = RECO_val_GT[indices_val][:, None]
        Rb_val = val.Rb_syn[indices_val].values[:, None]

        if args.T:
            X_Rb_val = np.c_[X_val, T_val]
        else:
            X_Rb_val = X_val

        data_val = [X_Rb_val, T_val, y_val]

        results = pd.DataFrame(columns=["Q"])

        rng_key = random.PRNGKey(int(args.samples * args.split))
        rng_key, rng_key2, rng_key3 = random.split(rng_key, 3)

        dataset = BootstrapLoader(
            X_Rb, T, y, 256, 100, n_samples=args.samples, rng_key=rng_key, replace=False
        )

        regressor = EnsembleRegressor(
            [X_Rb.shape[1], 16, 16, 1],
            100,
            p=drop_p,
            weight_decay=weight_decay,
            rng_key=rng_key2,
            Q10_mean_guess=1.5,
        )
        regressor.fit(dataset, data_val, nIter=2 * args.training_iter, rng_key=rng_key3)

        #### Evaluate on hold-out data
        test, out = load_dataset(
            "Synthetic4BookChap.csv",
            normalize=False,
            frac=0,
            years=[2008, 2009],
        )
        (
            EV_test,
            RECO_test,
            RECO_test_GT,
            driver_test,
            RECO_max_abs_test,
            _,
        ) = out
        indices_test = np.array(
            ~np.isnan(RECO_test)
            * np.prod(~np.isnan(EV_test), axis=1)
            * ~np.isnan(driver_test),
            dtype=bool,
        )

        X_test = EV_test[indices_test]
        T_test = driver_test[indices_test][:, None]
        y_test = RECO_test[indices_test][:, None]
        y_test_GT = RECO_test_GT[indices_test][:, None]
        Rb_test = test.Rb_syn[indices_test][:, None]

        if args.T:
            X_Rb_test = np.c_[X_test, T_test]
        else:
            X_Rb_test = X_test

        # Evaluation
        sample, Rb_pred, Q10 = regressor.posterior(X_Rb, T)
        sample_val, Rb_pred_val, Q10 = regressor.posterior(X_Rb_val, T_val)
        sample_test, Rb_pred_test, Q10 = regressor.posterior(X_Rb_test, T_test)

        rmse_Rb = list()
        rmse_Rb_val = list()
        rmse_Rb_test = list()
        rmse_Reco = list()
        rmse_Reco_val = list()
        rmse_Reco_test = list()
        R2_Rb = list()
        R2_Rb_val = list()
        R2_Rb_test = list()
        R2_Reco = list()
        R2_Reco_val = list()
        R2_Reco_test = list()
        results["Q"] = Q10[:, 0]

        for i in range(100):
            rmse_Rb.append(mean_squared_error(Rb_pred[i, :, 0], Rb[:], squared=False))
            rmse_Rb_val.append(
                mean_squared_error(Rb_pred_val[i, :, 0], Rb_val[:], squared=False)
            )
            rmse_Rb_test.append(
                mean_squared_error(Rb_pred_test[i, :, 0], Rb_test[:], squared=False)
            )
            rmse_Reco.append(
                mean_squared_error(sample[i, :, 0], y_GT[:], squared=False)
            )
            rmse_Reco_val.append(
                mean_squared_error(sample_val[i, :, 0], y_val_GT[:], squared=False)
            )
            rmse_Reco_test.append(
                mean_squared_error(sample_test[i, :, 0], y_test_GT[:], squared=False)
            )
            R2_Rb.append(r2_score(Rb_pred[i, :, 0], Rb[:]))
            R2_Rb_val.append(r2_score(Rb_pred_val[i, :, 0], Rb_val[:]))
            R2_Rb_test.append(r2_score(Rb_pred_test[i, :, 0], Rb_test[:]))
            R2_Reco.append(r2_score(sample[i, :, 0], y_GT[:]))
            R2_Reco_val.append(r2_score(sample_val[i, :, 0], y_val_GT[:]))
            R2_Reco_test.append(r2_score(sample_test[i, :, 0], y_test_GT[:]))

        results["rmse_Rb"] = rmse_Rb
        results["rmse_Rb_val"] = rmse_Rb_val
        results["rmse_Rb_test"] = rmse_Rb_test
        results["rmse_Reco"] = rmse_Reco
        results["rmse_Reco_val"] = rmse_Reco_val
        results["rmse_Reco_test"] = rmse_Reco_test
        results["R2_Rb"] = R2_Rb
        results["R2_Rb_val"] = R2_Rb_val
        results["R2_Rb_test"] = R2_Rb_test
        results["R2_Reco"] = R2_Reco
        results["R2_Reco_val"] = R2_Reco_val
        results["R2_Reco_test"] = R2_Reco_test

        csv_name = f"synQ10_{args.samples}_{noise_string}_{args.method}_{args.ml}_{dropout}_{add_T}{split_string}_{args.cycle}_{args.number}.csv"
        results_file = results_path.joinpath(csv_name)
        results.to_csv(results_file)


if __name__ == "__main__":
    main()
