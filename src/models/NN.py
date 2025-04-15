import os
import itertools
from functools import partial

import numpy as np
from sklearn.metrics import mean_squared_error

import jax
from jax import grad, vmap, random, jit
from jax import numpy as jnp

from tqdm import trange
import optax

from .building_blocks import MLPDropout  # , RespirationModel


# os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 0 1 7
# os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"]=".50"
# os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"


class Q10Regressor:
    def __init__(
        self,
        layers,
        ensemble_size,
        p=0.01,
        weight_decay=0,
        rng_key=random.PRNGKey(0),
        Q10_mean_guess=1.5,
    ):
        self.init, self.apply, self.apply_eval = MLPDropout(layers)
        self.Q10_mean_guess = Q10_mean_guess
        self.Q10_std_guess = 0.1
        self.ensemble_size = ensemble_size
        self.p = p

        # Random keys
        rng_key1, rng_key2 = random.split(rng_key, 2)

        k1, k2, k3, k4, k5 = random.split(rng_key1, 5)
        keys_1 = random.split(k1, ensemble_size)
        keys_2 = random.split(k2, ensemble_size)
        keys_3 = random.split(k3, ensemble_size)
        keys_4 = random.split(k5, ensemble_size)

        # Initialize
        self.params = vmap(self.init)(keys_1)
        self.Q10 = self.Q10_mean_guess + self.Q10_std_guess * random.normal(
            k4, (self.ensemble_size, 1)
        )
        self.Q10_init = self.Q10

        schedule = optax.exponential_decay(
            init_value=0.01, transition_steps=500, decay_rate=0.95
        )

        self.optimizer = optax.chain(
            optax.adamw(
                learning_rate=schedule,
                weight_decay=weight_decay,
            ),
        )
        self.opt_state = vmap(self.optimizer.init)(self.params)

        schedule_Q10 = optax.exponential_decay(
            init_value=0.1, transition_steps=500, decay_rate=0.95
        )

        self.optimizer_Q10 = optax.chain(
            optax.adamw(learning_rate=schedule_Q10, weight_decay=0)
        )

        self.opt_state_Q10 = vmap(self.optimizer_Q10.init)(self.Q10)

        # Logger
        self.itercount = itertools.count()
        self.loss_log = []

        self.train_log = []
        self.val_log = []
        self.test_log = []
        self.train_GT_log = []
        self.val_GT_log = []
        self.test_GT_log = []

        self.loss_test_log = [jnp.array(self.ensemble_size * [jnp.inf])]
        self.Q10_log = []

    # Define the forward pass
    def net_forward(self, params, Q10, inputs, T, p, rng_key):
        Rb = jax.nn.softplus(self.apply(params, inputs, p, rng_key))
        Y_pred = Rb * Q10 ** (0.1 * (T - 15))
        return Y_pred

    def loss(self, params, Q10, batch, p, rng_key):
        inputs, T, targets = batch
        # Compute forward pass
        outputs = vmap(self.net_forward, (None, None, 0, 0, None, None))(
            params, Q10, inputs, T, p, rng_key
        )
        # Compute loss
        loss = jnp.mean((targets - outputs) ** 2)
        return loss

    # Define the update step
    @partial(jit, static_argnums=(0,))
    def step(self, i, params, Q10, opt_state, opt_state_Q10, batch, p, rng_key):
        grads = jax.grad(self.loss, argnums=0)(params, Q10, batch, p, rng_key)
        grads_Q10 = jax.grad(self.loss, argnums=1)(params, Q10, batch, p, rng_key)
        updates, opt_state = self.optimizer.update(grads, opt_state, params)
        updates_Q10, opt_state_Q10 = self.optimizer_Q10.update(
            grads_Q10, opt_state_Q10, Q10
        )

        params = optax.apply_updates(params, updates)
        Q10 = optax.apply_updates(Q10, updates_Q10)

        return params, Q10, opt_state, opt_state_Q10

    def monitor_loss(self, params, Q10, batch, p, rng_key):
        loss_value = self.loss(params, Q10, batch, p, rng_key)
        return loss_value

    def net_forward_test(self, params, Q10, inputs, T):
        Rb = jax.nn.softplus(self.apply_eval(params, inputs))
        Y_pred = Rb * Q10 ** (0.1 * (T - 15))
        return Y_pred, Rb, Q10

    def monitor_loss_test(self, params, Q10, batch):
        inputs, T, targets = batch
        outputs, _, _ = vmap(self.net_forward_test, (None, None, 0, 0))(
            params, Q10, inputs, T
        )
        loss = np.mean((targets - outputs) ** 2)
        return loss

    def update_weights(self, params, Q10, params_best, Q10_best):
        return params, Q10

    def keep_weights(self, params, Q10, params_best, Q10_best):
        return params_best, Q10_best

    def early_stopping(self, update, params, Q10, params_best, Q10_best):
        return jax.lax.cond(
            update,
            self.update_weights,
            self.keep_weights,
            params,
            Q10,
            params_best,
            Q10_best,
        )

    def batch_normalize(self, data_val, norm_const):
        X, T, y = data_val

        (mu_X, sigma_X), (mu_y, sigma_y) = norm_const
        X = (X - mu_X) / sigma_X
        y = (y - mu_y) / sigma_y

        return [X, T, y]

    # Optimize parameters in a loop
    def fit(self, dataset, data_val, rng_key, nIter=1000):
        self.params_best = self.params
        self.Q10_best = self.Q10

        data = iter(dataset)
        (self.mu_X, self.sigma_X), (self.mu_y, self.sigma_y) = dataset.norm_const

        pbar = trange(nIter)
        # Define vectorized SGD step across the entire ensemble
        # jitted
        v_step = jit(vmap(self.step, in_axes=(None, 0, 0, 0, 0, 0, None, 0)))
        v_monitor_loss = jit(vmap(self.monitor_loss, in_axes=(0, 0, 0, None, 0)))
        v_monitor_loss_test = jit(vmap(self.monitor_loss_test, in_axes=(0, 0, 0)))
        v_early_stopping = vmap(self.early_stopping, in_axes=(0, 0, 0, 0, 0))
        v_batch_normalize = vmap(self.batch_normalize, in_axes=(None, 0))

        data_val = v_batch_normalize(data_val, dataset.norm_const)

        # Main training loop
        for it in pbar:
            rng_key, *rng_keys = random.split(rng_key, self.ensemble_size + 1)
            batch = next(data)
            self.params, self.Q10, self.opt_state, self.opt_state_Q10 = v_step(
                it,
                self.params,
                self.Q10,
                self.opt_state,
                self.opt_state_Q10,
                batch,
                self.p,
                jnp.array(rng_keys),
            )
            # Logger
            if it % 100 == 0:
                self.Q10_log.append(self.Q10)
                loss_value = v_monitor_loss(
                    self.params, self.Q10, batch, self.p, jnp.array(rng_keys)
                )
                self.loss_log.append(loss_value)

                loss_test_value = v_monitor_loss_test(self.params, self.Q10, data_val)
                update = jnp.array(self.loss_test_log).min(axis=0) > loss_test_value
                self.loss_test_log.append(loss_test_value)

                loss_val_value = v_monitor_loss_test(self.params, self.Q10, data_val)
                self.val_log.append(loss_val_value)

                pbar.set_postfix(
                    {
                        "Max loss": loss_value.max(),
                        "Mean test loss": loss_test_value.mean(),
                    }
                )

                self.params_best, self.Q10_best = v_early_stopping(
                    update, self.params, self.Q10, self.params_best, self.Q10_best
                )

    # Evaluates predictions at test points
    # @partial(jit, static_argnums=(0,))
    def posterior(self, x, t):
        normalize = vmap(lambda x, mu, std: (x - mu) / std, in_axes=(0, 0, 0))
        denormalize = vmap(lambda x, mu, std: x * std + mu, in_axes=(0, 0, 0))

        x = jnp.tile(x[jnp.newaxis, :, :], (self.ensemble_size, 1, 1))
        t = jnp.tile(t[jnp.newaxis, :, :], (self.ensemble_size, 1, 1))
        inputs = normalize(x, self.mu_X, self.sigma_X)

        samples, samples_Rb, samples_Q10 = vmap(self.net_forward_test, (0, 0, 0, 0))(
            self.params_best, self.Q10_best, inputs, t
        )
        samples = denormalize(samples, self.mu_y, self.sigma_y)
        samples_Rb = denormalize(samples_Rb, self.mu_y, self.sigma_y)

        return samples, samples_Rb, samples_Q10

    # @partial(jit, static_argnums=(0,))
    def predict(self, x, t):
        # accepts and returns un-normalized data
        samples, samples_Rb, samples_Q10 = self.posterior(x, t)
        return samples.mean(0), samples.std(0)

    def score(self, x, t, y):
        y_pred, *_ = self.predict(x, t)
        return mean_squared_error(y, y_pred, squared=False)
