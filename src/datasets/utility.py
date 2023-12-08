class BootstrapLoader(data.Dataset):
    def __init__(
        self,
        X,
        T,
        y,
        batch_size=128,
        ensemble_size=32,
        fraction=0.8,
        n_samples=None,
        rng_key=random.PRNGKey(1234),
        replace=True,
    ):
        "Initialization"
        self.N = X.shape[0]
        self.batch_size = batch_size
        self.ensemble_size = ensemble_size
        self.replace = replace
        if n_samples:
            self.bootstrap_size = n_samples
        else:
            self.bootstrap_size = int(self.N * fraction)
        self.key = rng_key
        # Create the bootstrapped partitions
        keys = random.split(rng_key, ensemble_size)
        self.X, self.T, self.y, self.indices = vmap(
            self.__bootstrap, (None, None, None, 0)
        )(X, T, y, keys)
        # Each bootstrapped data-set has its own normalization constants
        self.norm_const = vmap(self.normalization_constants, in_axes=(0, 0))(
            self.X, self.y
        )

    @partial(jit, static_argnums=(0,))
    def normalization_constants(self, X, y):
        mu_X, sigma_X = X.mean(0), X.std(0)
        mu_y, sigma_y = jnp.zeros(
            y.shape[1],
        ), jnp.abs(
            y
        ).max(0) * jnp.ones(
            y.shape[1],
        )  # np.abs(y).max(0)

        return (mu_X, sigma_X), (mu_y, sigma_y)

    @partial(jit, static_argnums=(0,))
    def __bootstrap(self, X, T, y, key):
        idx = random.choice(key, self.N, (self.bootstrap_size,), replace=self.replace)
        inputs = X[idx, :]
        T = T[idx, :]
        targets = y[idx, :]
        return inputs, T, targets, idx

    @partial(jit, static_argnums=(0,))
    def __data_generation(self, key, X, T, y, norm_const):
        "Generates data containing batch_size samples"
        (mu_X, sigma_X), (mu_y, sigma_y) = norm_const
        idx = random.choice(key, self.N, (self.batch_size,), replace=False)
        T = T[idx, :]
        X = X[idx, :]
        y = y[idx, :]
        X = (X - mu_X) / sigma_X
        y = (y - mu_y) / sigma_y
        return X, T, y

    def __getitem__(self, index):
        "Generate one batch of data"
        self.key, subkey = random.split(self.key)
        keys = random.split(self.key, self.ensemble_size)
        inputs, T, targets = vmap(self.__data_generation, (0, 0, 0, 0, 0))(
            keys, self.X, self.T, self.y, self.norm_const
        )
        return inputs, T, targets


class CustomBootstrapLoader(data.Dataset):
    def __init__(
        self,
        X,
        y,
        batch_size=128,
        ensemble_size=32,
        split=0.8,
        rng_key=random.PRNGKey(1234),
    ):
        #'Initialization'
        self.N = X.shape[0]
        self.batch_size = batch_size
        self.ensemble_size = ensemble_size
        self.split = split
        self.key = rng_key

        if self.N < self.batch_size:
            self.batch_size = self.N

        # Create the bootstrapped partitions
        keys = random.split(rng_key, ensemble_size)
        if split < 1:
            self.data_train, self.data_val = vmap(self.__bootstrap, (None, None, 0))(
                X, y, keys
            )
            (self.X_train, self.y_train) = self.data_train
        else:
            self.data_train, self.data_val = vmap(
                self.__bootstrap_train_only, (None, None, 0)
            )(X, y, keys)
            (self.X_train, self.y_train) = self.data_train

        # Each bootstrapped data-set has its own normalization constants
        self.norm_const = vmap(self.normalization_constants, in_axes=(0, 0))(
            self.X_train, self.y_train
        )

        # For analysis reasons
        self.norm_const_val = vmap(self.normalization_constants, in_axes=(0, 0))(
            *self.data_val
        )

    def normalization_constants(self, X, y):
        mu_X, sigma_X = X.mean(0), X.std(0)
        mu_y, sigma_y = jnp.zeros(
            y.shape[1],
        ), jnp.abs(
            y
        ).max(0) * jnp.ones(
            y.shape[1],
        )

        return (mu_X, sigma_X), (mu_y, sigma_y)

    def __bootstrap(self, X, y, key):
        # TODO Proper Bootstrap is happening outside. In here we take the whole dataset and split it
        idx = random.choice(key, self.N, (self.N,), replace=False)
        idx_train = idx[: jnp.floor(self.N * self.split).astype(int)]
        idx_test = idx[jnp.floor(self.N * self.split).astype(int) :]

        inputs_train = X[idx_train, :]
        targets_train = y[idx_train, :]

        inputs_test = X[idx_test, :]
        targets_test = y[idx_test, :]

        return (inputs_train, targets_train), (inputs_test, targets_test)

    def __bootstrap_train_only(self, X, y, key):
        idx = random.choice(key, self.N, (self.N,), replace=False).sort()

        inputs_train = X[idx]
        targets_train = y[idx]

        inputs_test = X[idx]
        targets_test = y[idx]

        return (inputs_train, targets_train), (inputs_test, targets_test)

    @partial(jit, static_argnums=(0,))
    def __data_generation(self, key, X, y, norm_const):
        "Generates data containing batch_size samples"
        (mu_X, sigma_X), (mu_y, sigma_y) = norm_const
        idx = random.choice(key, self.N, (self.batch_size,), replace=False)
        X = X[idx, :]
        y = y[idx, :]
        X = (X - mu_X) / sigma_X
        y = (y - mu_y) / sigma_y
        return X, y

    def __getitem__(self, index):
        "Generate one batch of data"
        self.key, subkey = random.split(self.key)
        keys = random.split(self.key, self.ensemble_size)
        inputs, targets = vmap(self.__data_generation, (0, 0, 0, 0))(
            keys, self.X_train, self.y_train, self.norm_const
        )
        return inputs, targets


class GeneratedDataset:
    def __init__(self, inputs, targets, model):
        self.inputs = inputs
        self.targets = targets
        self.model = model

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]

    def split(
        self,
        split,
        random=True,
        train_indices=None,
        valid_indices=None,
        test_indices=None,
    ):
        split = onp.ceil(onp.array(split) * len(self)).astype(int)

        partial_datasets = []
        if random:
            perm = onp.random.permutation(len(self))

            input_split = onp.split(self.inputs[perm, :], split)
            output_split = onp.split(self.targets[perm], split)
        else:
            if not train_indices:
                input_split = onp.split(self.inputs, split)
                output_split = onp.split(self.targets, split)
            else:
                input_split = list()
                output_split = list()
                input_split.append(self.inputs[train_indices, :])
                input_split.append(self.inputs[valid_indices, :])
                input_split.append(self.inputs[test_indices, :])

                output_split.append(self.targets[train_indices, :])
                output_split.append(self.targets[valid_indices, :])
                output_split.append(self.targets[test_indices, :])

        for i in range(len((split))):
            partial_datasets.append(
                GeneratedDataset(input_split[i], output_split[i], self.model)
            )

        return partial_datasets


def build_dataloaders(dataloader_config, samples=None):
    if not dataloader_config["split_time"]:
        traindata, valdata, testdata = dataloader_config["data"].split(
            dataloader_config["split"]
        )

        trainLoader = torch.utils.data.DataLoader(
            traindata,
            batch_size=dataloader_config["batch_size"],
            shuffle=True,
            drop_last=dataloader_config["drop_last"],
        )
        valiLoader = torch.utils.data.DataLoader(
            valdata,
            batch_size=dataloader_config["batch_size"],
            shuffle=False,
            drop_last=dataloader_config["drop_last"],
        )
        testLoader = torch.utils.data.DataLoader(
            testdata,
            batch_size=dataloader_config["batch_size"],
            shuffle=False,
            drop_last=dataloader_config["drop_last"],
        )

        print(
            "Training & validation & test batches: {} , {}, {}".format(
                len(trainLoader), len(valiLoader), len(testLoader)
            )
        )

        return {"train": trainLoader, "validation": valiLoader, "test": testLoader}
    else:
        if samples:
            subsamples = onp.ceil(
                onp.array(dataloader_config["split"]) * samples
            ).astype(int)
            train_indices = orandom.choices(
                boolean_to_list_of_ints(dataloader_config["train_indices"]),
                k=subsamples[0],
            )
            valid_indices = orandom.choices(
                boolean_to_list_of_ints(dataloader_config["valid_indices"]),
                k=subsamples[1],
            )
            test_indices = orandom.choices(
                boolean_to_list_of_ints(dataloader_config["test_indices"]),
                k=subsamples[2],
            )

        else:
            train_indices = dataloader_config["train_indices"]
            valid_indices = dataloader_config["valid_indices"]
            test_indices = dataloader_config["test_indices"]

        traindata, valdata, testdata = dataloader_config["data"].split(
            dataloader_config["split"],
            random=False,
            train_indices=train_indices,
            valid_indices=valid_indices,
            test_indices=test_indices,
        )

        trainLoader = torch.utils.data.DataLoader(
            traindata,
            batch_size=dataloader_config["batch_size"],
            shuffle=True,
            drop_last=dataloader_config["drop_last"],
        )

        if dataloader_config["norm"]:
            dataloader_config["norm_stats"] = get_mean_and_std(trainLoader)
            input_stats, output_stats = dataloader_config["norm_stats"]

            traindata = normalize(
                traindata, input_stats, output_stats, dataloader_config["include_TA"]
            )
            valdata = normalize(
                valdata, input_stats, output_stats, dataloader_config["include_TA"]
            )
            testdata = normalize(
                testdata, input_stats, output_stats, dataloader_config["include_TA"]
            )

        trainLoader = torch.utils.data.DataLoader(
            traindata,
            batch_size=dataloader_config["batch_size"],
            shuffle=True,
            drop_last=dataloader_config["drop_last"],
        )
        valiLoader = torch.utils.data.DataLoader(
            valdata,
            batch_size=dataloader_config["batch_size"],
            shuffle=False,
            drop_last=dataloader_config["drop_last"],
        )
        testLoader = torch.utils.data.DataLoader(
            testdata,
            batch_size=dataloader_config["batch_size"],
            shuffle=False,
            drop_last=dataloader_config["drop_last"],
        )

        print(
            "Training & validation & test batches: {} , {}, {}".format(
                len(trainLoader), len(valiLoader), len(testLoader)
            )
        )

        return {"train": trainLoader, "validation": valiLoader, "test": testLoader}


class Dataset:
    def __init__(self, inputs, targets):
        self.inputs = inputs.astype(onp.float32)
        self.targets = targets.astype(onp.float32)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]

    def split(
        self,
        split,
        shuffle=True,
        train_indices=None,
        valid_indices=None,
        test_indices=None,
    ):
        split = onp.ceil(onp.array(split) * len(self)).astype(int)

        partial_datasets = []
        if shuffle:
            perm = onp.random.permutation(len(self))

            input_split = onp.split(self.inputs[perm, :], split)
            output_split = onp.split(self.targets[perm], split)
        else:
            if train_indices is None:
                input_split = onp.split(self.inputs, split)
                output_split = onp.split(self.targets, split)
            else:
                input_split = list()
                output_split = list()
                input_split.append(self.inputs[train_indices])
                input_split.append(self.inputs[valid_indices])
                input_split.append(self.inputs[test_indices])

                output_split.append(self.targets[train_indices])
                output_split.append(self.targets[valid_indices])
                output_split.append(self.targets[test_indices])
        for i in range(len((split))):
            partial_datasets.append(Dataset(input_split[i], output_split[i]))

        return partial_datasets


def get_mean_and_std(dataloader):
    sum_input = torch.zeros(dataloader.dataset[0][0].size)
    sum_sq_input = torch.zeros(dataloader.dataset[0][0].size)

    sum_output = torch.zeros(dataloader.dataset[0][1].size)
    sum_sq_output = torch.zeros(dataloader.dataset[0][1].size)

    for inputs, targets in dataloader:
        # Mean over batch, height and width, but not over the channels
        sum_input += inputs.sum(0)
        sum_sq_input += (inputs**2).sum(0)

        sum_output += targets.sum(0)
        sum_sq_output += (targets**2).sum(0)

    mean_input = sum_input / len(dataloader.dataset)
    mean_sq_input = sum_sq_input / len(dataloader.dataset)
    std_input = (mean_sq_input - mean_input**2) ** 0.5

    mean_output = sum_output / len(dataloader.dataset)
    mean_sq_output = sum_sq_output / len(dataloader.dataset)
    std_output = (mean_sq_output - mean_output**2) ** 0.5

    return [mean_input, std_input], [mean_output, std_output]


def normalize(dataset, input_stats, output_stats, include_TA):
    inputs_norm = list()
    outputs_norm = list()

    if include_TA:
        for inputs, outputs in dataset:
            inputs_norm.append(
                onp.append(
                    (inputs - input_stats[0].numpy()) / input_stats[1].numpy(),
                    inputs[-1],
                )
            )
            outputs_norm.append(
                ((outputs - output_stats[0].numpy()) / output_stats[1].numpy())[0]
            )
    else:
        input_stats[0][-1] = 0
        input_stats[1][-1] = 1

        for inputs, outputs in dataset:
            inputs_norm.append(
                (inputs - input_stats[0].numpy()) / input_stats[1].numpy()
            )
            outputs_norm.append(
                ((outputs - output_stats[0].numpy()) / output_stats[1].numpy())[0]
            )

    dataset_norm = Dataset(onp.array(inputs_norm), onp.array(outputs_norm))
    return dataset_norm


def boolean_to_list_of_ints(boolean):
    list_of_ints = list()
    for i in range(len(boolean)):
        if boolean[i]:
            list_of_ints.append(i)
    return list_of_ints
