import os
import itertools
from functools import partial

from sklearn.metrics import mean_squared_error

import jax
from jax import vmap, random, jit
from jax import numpy as jnp
from tqdm import trange
import optax


from sklearn.base import BaseEstimator, RegressorMixin
from src.datasets.utility import (
    CustomBootstrapLoader,
)

from .building_blocks import MLPDropout
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 0 1 7
# os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"]=".50"
# os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

class EnsembleCustomJNN(BaseEstimator, RegressorMixin):
    def __init__(self, model_config, trainer_config, seed=1):
        self.model_config = model_config
        self.trainer_config = trainer_config
        self.seed = seed

        self.layers = self.model_config["layers"]
        self.final_nonlin = self.model_config["final_nonlin"]
        self.p = self.model_config["dropout_p"]
        self.weight_decay = self.trainer_config["weight_decay"]
        self.split = self.trainer_config["split"]
        self.ensemble_size = self.model_config["ensemble_size"]
        if "iterations" in self.trainer_config.keys():
            self.iterations = self.trainer_config["iterations"]
        else:
            self.iterations = 4000

        # Logger
        self.itercount = itertools.count()
        self.loss_log = []
        self.train_log = []
        self.val_log = []
        self.loss_test_log = [jnp.array(self.ensemble_size * [jnp.inf])]

    # Define the forward pass
    def net_forward(self, params, inputs, p, rng_key):
        Y_pred = self.apply(params, inputs, p, rng_key)
        return Y_pred

    def net_forward_test(self, params, inputs):
        Y_pred = self.apply_eval(params, inputs)
        return Y_pred

    def loss(self, params, batch, p, rng_key):
        inputs, targets = batch
        # Compute forward pass
        outputs = vmap(self.net_forward, (None, 0, None, None))(
            params, inputs, p, rng_key
        )
        # Compute loss
        loss = jnp.mean((targets - outputs) ** 2)
        return loss

    def monitor_loss(self, params, batch, p, rng_key):
        loss_value = self.loss(params, batch, p, rng_key)
        return loss_value

    def monitor_loss_test(self, params, batch):
        inputs, targets = batch
        outputs = vmap(self.net_forward_test, (None, 0))(params, inputs)
        loss = jnp.mean((targets - outputs) ** 2)
        return loss

    # Define the update step
    @partial(jit, static_argnums=(0,))
    def step(self, i, params, opt_state, batch, p, rng_key):
        grads = jax.grad(self.loss, argnums=0)(params, batch, p, rng_key)
        updates, opt_state = self.optimizer.update(grads, opt_state, params)

        params = optax.apply_updates(params, updates)

        return params, opt_state

    def update_weights(self, params, params_best):
        return params

    def keep_weights(self, params, params_best):
        return params_best

    def early_stopping(self, update, params, params_best):
        return jax.lax.cond(
            update, self.update_weights, self.keep_weights, params, params_best
        )

    def batch_normalize(self, data_val, norm_const):
        X, y = data_val

        (mu_X, sigma_X), (mu_y, sigma_y) = norm_const
        X = (X - mu_X) / sigma_X
        y = (y - mu_y) / sigma_y

        return [X, y]

    # Optimize parameters in a loop
    def fit(self, X, y):
        rng_key = random.PRNGKey(self.seed)
        nIter = self.iterations

        self.init, self.apply, self.apply_eval = MLPDropout(
            self.layers, self.final_nonlin
        )

        # Random keys
        rng_key1, rng_key2, self.rng_key_fit = random.split(rng_key, 3)
        (
            k1,
            k2,
        ) = random.split(rng_key1, 2)
        keys_1 = random.split(k1, self.ensemble_size)

        # Initialize
        self.params = vmap(self.init)(keys_1)
        schedule = optax.exponential_decay(
            init_value=0.01, transition_steps=500, decay_rate=0.95
        )
        self.optimizer = optax.chain(
            optax.adamw(
                learning_rate=schedule,
                weight_decay=self.weight_decay,
            )
        )
        self.opt_state = vmap(self.optimizer.init)(self.params)

        # Potentially reformat y
        if len(y.shape) == 1:
            y = y[:, None]
        X, y = jnp.array(X), jnp.array(y)

        rng_key, rng_key_loader = random.split(self.rng_key_fit, 2)

        dataset = CustomBootstrapLoader(
            X, y, 256, self.ensemble_size, split=self.split, rng_key=rng_key_loader
        )
        self.params_best = self.params

        data = iter(dataset)
        self.norm_const = dataset.norm_const
        (self.mu_X, self.sigma_X), (self.mu_y, self.sigma_y) = self.norm_const

        pbar = trange(nIter)
        # Define vectorized SGD step across the entire ensemble
        # jitted
        v_step = jit(vmap(self.step, in_axes=(None, 0, 0, 0, None, 0)))
        v_monitor_loss = jit(vmap(self.monitor_loss, in_axes=(0, 0, None, 0)))
        v_monitor_loss_test = jit(vmap(self.monitor_loss_test, in_axes=(0, 0)))
        v_early_stopping = vmap(self.early_stopping, in_axes=(0, 0, 0))
        v_batch_normalize = vmap(self.batch_normalize, in_axes=(0, 0))

        data_train = v_batch_normalize(dataset.data_train, dataset.norm_const)
        data_val = v_batch_normalize(dataset.data_val, dataset.norm_const)

        # Main training loop
        for it in pbar:
            rng_key, *rng_keys = random.split(rng_key, self.ensemble_size + 1)
            batch = next(data)
            self.params, self.opt_state = v_step(
                it, self.params, self.opt_state, batch, self.p, jnp.array(rng_keys)
            )
            # Logger
            if it % 100 == 0:
                loss_value = v_monitor_loss(
                    self.params, batch, self.p, jnp.array(rng_keys)
                )
                self.loss_log.append(loss_value)

                loss_test_value = v_monitor_loss_test(self.params, data_val)
                update = jnp.array(self.loss_test_log).min(axis=0) > loss_test_value
                self.loss_test_log.append(loss_test_value)

                loss_train_value = v_monitor_loss_test(self.params, data_train)
                self.train_log.append(loss_train_value)
                loss_val_value = v_monitor_loss_test(self.params, data_val)
                self.val_log.append(loss_val_value)

                self.params_best = v_early_stopping(
                    update, self.params, self.params_best
                )
                best_test_loss = v_monitor_loss_test(self.params_best, data_val)
                pbar.set_postfix(
                    {
                        "Max loss": loss_value.max(),
                        "Max test loss": loss_test_value.max(),
                        "Best test loss": best_test_loss.max(),
                    }
                )

    # Evaluates predictions at test points
    @partial(jit, static_argnums=(0,), device=jax.devices("cpu")[0])
    def posterior(self, x):
        normalize = vmap(lambda x, mu, std: (x - mu) / std, in_axes=(0, 0, 0))
        denormalize = vmap(lambda x, mu, std: x * std + mu, in_axes=(0, 0, 0))

        x = jnp.tile(x[jnp.newaxis, :, :], (self.ensemble_size, 1, 1))
        inputs = normalize(x, self.mu_X, self.sigma_X)

        samples = vmap(self.net_forward_test, (0, 0))(self.params_best, inputs)
        samples = denormalize(samples, self.mu_y, self.sigma_y)

        return samples

    @partial(jit, static_argnums=(0,), device=jax.devices("cpu")[0])
    def predict(self, x):
        # accepts and returns un-normalized data
        samples = self.posterior(x)
        return samples.mean(0).reshape(-1)

    def score(self, x, y):
        y_pred = self.predict(x)
        return mean_squared_error(y, y_pred, squared=False)
