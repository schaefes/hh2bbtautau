"""
First implementation of DNN for HH analysis, generalized (TODO)
"""

from __future__ import annotations

from typing import Any
import gc
import time
import os

import law
import order as od

from columnflow.ml import MLModel
from columnflow.util import maybe_import, dev_sandbox
from columnflow.columnar_util import Route, set_ak_column, remove_ak_column
from columnflow.tasks.selection import MergeSelectionStatsWrapper
from columnflow.tasks.production import ProduceColumns
from hbt.config.categories import add_categories_ml
from columnflow.columnar_util import EMPTY_FLOAT
from hbt.ml.plotting import (
    plot_loss, plot_accuracy, plot_confusion, plot_roc_ovr, plot_output_nodes, plot_significance,
    plot_shap_values_simple_nn, write_info_file
)

np = maybe_import("numpy")
ak = maybe_import("awkward")
tf = maybe_import("tensorflow")
pickle = maybe_import("pickle")
keras = maybe_import("tensorflow.keras")
sklearn = maybe_import("sklearn")

logger = law.logger.get_logger(__name__)


# Define functions to normalize and shape inputs1 and 2
def reshape_raw_inputs1(events, n_features, input_features):
    column_counter = 0
    num_events, max_jets = ak.to_numpy(events[input_features[0]]).shape
    zeros = np.zeros((num_events, max_jets * n_features))
    for i in range(max_jets):
        for jet_features in input_features:
            zeros[:, column_counter] = events[jet_features][:, i]
            column_counter += 1
    return zeros


def reshape_raw_inputs2(events):
    events = ak.to_numpy(events)
    events = events.astype(
        [(name, np.float32) for name in events.dtype.names], copy=False,
    ).view(np.float32).reshape((-1, len(events.dtype)))

    if np.any(~np.isfinite(events)) or np.any(~np.isfinite(events)):
        raise Exception(f"Infinite values found in inputs from dataset.")

    return events


# returns a dict containg the normed and correctly shaped inputs1 and 2
def reshape_norm_inputs(events_dict, n_features, norm_features, input_features):
    # reshape train['inputs'] for DeepSets: [#events, #jets, -1] and apply standardization (z-score)
    # calculate mean and std for normalization
    events_shaped = events_dict["inputs"].reshape((-1, n_features))
    mean_feature = np.zeros(n_features)
    std_feature = np.zeros(n_features)
    delete_standardization = []

    for i, feature in enumerate(input_features[0]):
        if feature not in norm_features:
            delete_standardization.append(i)

    for i in range(n_features):
        mask_empty_floats = events_shaped[:, i] != EMPTY_FLOAT
        mean_feature[i] = np.mean(events_shaped[:, i][mask_empty_floats])
        std_feature[i] = np.std(events_shaped[:, i][mask_empty_floats])

    mean_feature[delete_standardization] = 0
    std_feature[delete_standardization] = 1

    jets_collection = []
    for i in range(events_dict['inputs'].shape[0]):
        arr_events = events_dict['inputs'][i]
        indices = np.where(arr_events != EMPTY_FLOAT)
        jets_flat = arr_events[indices]
        try:
            jets_shaped = jets_flat.reshape((-1, n_features))
        except:
            print('---?---')
            from IPython import embed; embed()
        jets_normalized = (jets_shaped - mean_feature) / std_feature
        jets_normalized = jets_normalized.flatten()
        jets_shaped = jets_normalized.reshape((1, -1, n_features))
        jets_collection.append(jets_shaped)
    stacked_events = tf.ragged.stack(jets_collection)
    events_dict['inputs'] = tf.squeeze(stacked_events, axis=1)

    # normalization of inputs2
    events_dict['inputs2'] = np.where(events_dict['inputs2'] == EMPTY_FLOAT, -1, events_dict['inputs2'])
    mass_mask_train = events_dict['inputs2'] != -1
    for i in range(events_dict['inputs2'].shape[1]):
        mean = np.mean(events_dict['inputs2'][:, i][mass_mask_train[:, i]])
        std = np.std(events_dict['inputs2'][:, i][mass_mask_train[:, i]])
        events_dict['inputs2'][:, i] = (events_dict['inputs2'][:, i] - mean) / std
    events_dict['inputs2'] = tf.reshape(events_dict['inputs2'], [events_dict['inputs2'].shape[0], 1, events_dict['inputs2'].shape[1]])

    # reshape of target
    events_dict['target'] = tf.reshape(events_dict['target'], [events_dict['target'].shape[0], 1, -1])

    return events_dict


class SimpleDNN(MLModel):

    def __init__(
            self,
            *args,
            folds: int | None = None,
            n_features: int | None = None,
            ml_process_weights: dict | None = None,
            model_name: str | None = None,
            n_output_nodes: int | None = None,
            activation_func_deepSets: str | None = None,
            activation_func_ff: str | None = None,
            batch_norm_deepSets: bool | None = None,
            batch_norm_ff: bool | None = None,
            nodes_deepSets: list | None = None,
            nodes_ff: list | None = None,
            custom_layers: list | None = None,
            L2: bool | None = None,
            norm_features: list | None = None,
            **kwargs,
    ):
        """
        Parameters that need to be set by derived model:
        folds, layers, learningrate, batchsize, epochs, eqweight, dropout,
        processes, ml_process_weights, dataset_names, input_features, store_name,
        """

        single_config = True  # noqa

        super().__init__(*args, **kwargs)

        # class- to instance-level attributes
        # (before being set, self.folds refers to a class-level attribute)
        self.folds = folds or self.folds
        self.n_features = n_features or self.n_features
        self.ml_process_weights = ml_process_weights or self.ml_process_weights
        self.model_name = model_name or self.model_name
        self.n_output_nodes = n_output_nodes or self.n_output_nodes
        self.custom_layers = custom_layers or self.custom_layers
        self.batch_norm_ff = batch_norm_ff or self.batch_norm_ff
        self.batch_norm_deepSets = batch_norm_deepSets or self.batch_norm_deepSets
        self.activation_func_deepSets = activation_func_deepSets or self.activation_func_deepSets
        self.activation_func_ff = activation_func_ff or self.activation_func_ff
        self.nodes_deepSets = nodes_deepSets or self.nodes_deepSets
        self.nodes_ff = nodes_ff or self.nodes_ff
        self.L2 = L2 or self.L2
        self.norm_features = norm_features or self.norm_features
        # DNN model parameters
        """
        self.layers = [512, 512, 512]
        self.learningrate = 0.00050
        self.batchsize = 2048
        self.epochs = 6  # 200
        self.eqweight = 0.50
        # Dropout: either False (disable) or a value between 0 and 1 (dropout_rate)
        self.dropout = False
        """

    def setup(self):
        # dynamically add variables for the quantities produced by this model
        for proc in self.processes:
            if f"{self.cls_name}.score_{proc}" not in self.config_inst.variables:
                self.config_inst.add_variable(
                    name=f"{self.cls_name}.score_{proc}",
                    null_value=-1,
                    binning=(40, 0., 1.),
                    # x_title=f"DNN output score {self.config_inst.get_process(proc).x.ml_label}",
                )
                hh_bins = [0.0, .4, .45, .5, .55, .6, .65, .7, .75, .8, .85, .92, 1.0]
                bkg_bins = [0.0, 0.4, 0.7, 1.0]
                self.config_inst.add_variable(
                    name=f"{self.cls_name}.score_{proc}_rebin1",
                    expression=f"{self.cls_name}.score_{proc}",
                    null_value=-1,
                    binning=hh_bins if "HH" in proc else bkg_bins,
                    # x_title=f"DNN output score {self.config_inst.get_process(proc).x.ml_label}",
                )

        # one variable to bookkeep truth labels
        # TODO: still needs implementation
        if f"{self.cls_name}.ml_label" not in self.config_inst.variables:
            self.config_inst.add_variable(
                name=f"{self.cls_name}.ml_label",
                null_value=-1,
                binning=(len(self.processes) + 1, -1.5, len(self.processes) -0.5),
                x_title=f"DNN truth score",
            )

        # dynamically add ml categories (but only if production categories have been added)
        if (
                self.config_inst.x("add_categories_ml", True) and
                not self.config_inst.x("add_categories_production", True)
        ):
            add_categories_ml(self.config_inst, ml_model_inst=self)
            self.config_inst.x.add_categories_ml = False

    def requires(self, task: law.Task) -> str:
        # add selection stats to requires; NOTE: not really used at the moment
        all_reqs = MergeSelectionStatsWrapper.req(
            task,
            shifts="nominal",
            configs=self.config_inst.name,
            datasets=self.dataset_names,
        )

        return all_reqs

    def sandbox(self, task: law.Task) -> str:
        return dev_sandbox("bash::$CF_BASE/sandboxes/venv_ml_tf_dev.sh")

    def datasets(self, config_inst: od.Config) -> set[od.Dataset]:
        return {config_inst.get_dataset(dataset_name) for dataset_name in self.dataset_names}

    def uses(self, config_inst: od.Config) -> set[Route | str]:
        return {"normalization_weight", "category_ids"} | set(self.input_features[0]) | set(self.input_features[1])

    def produces(self, config_inst: od.Config) -> set[Route | str]:
        produced = set()
        for proc in self.processes:
            produced.add(f"{self.cls_name}.score_{proc}")

        produced.add("category_ids")

        return produced

    def output(self, task: law.Task) -> law.FileSystemDirectoryTarget:
        return task.target(f"mlmodel_f{task.branch}of{self.folds}_{self.model_name}", dir=True)

    def open_model(self, target: law.LocalDirectoryTarget) -> tf.keras.models.Model:
        # return target.load(formatter="keras_model")

        with open(f"{target.path}/model_history.pkl", "rb") as f:
            history = pickle.load(f)
        model = tf.keras.models.load_model(target.path)
        return model, history

    def training_configs(self, requested_configs: Sequence[str]) -> list[str]:
        # default config
        # print(requested_configs)
        if len(requested_configs) == 1:
            return list(requested_configs)
        else:
            # TODO: change to "config_2017" when finished with testing phase
            return ["config_2017_limited"]

    def training_calibrators(self, config_inst: od.Config, requested_calibrators: Sequence[str]) -> list[str]:
        # fix MLTraining Phase Space
        return ["skip_jecunc"]

    def evaluation_calibrators(self, config_inst: od.Config, requested_calibrators: Sequence[str]) -> list[str]:
        # fix MLTraining Phase Space
        return ["skip_jecunc"]

    def training_selector(self, config_inst: od.Config, requested_selector: str) -> str:
        # fix MLTraining Phase Space
        return "default"

    def training_producers(self, config_inst: od.Config, requested_producers: Sequence[str]) -> list[str]:
        # fix MLTraining Phase Space
        return ["default"]

    def prepare_inputs(
        self,
        task,
        input,
    ) -> dict[str, np.array]:

        # max_events_per_fold = int(self.max_events / (self.folds - 1))
        self.process_insts = []
        for i, proc in enumerate(self.processes):
            proc_inst = self.config_insts[0].get_process(proc)
            proc_inst.x.ml_id = i
            proc_inst.x.ml_process_weight = self.ml_process_weights.get(proc, 1)

            self.process_insts.append(proc_inst)

        process_insts = [self.config_inst.get_process(proc) for proc in self.processes]
        N_events_processes = np.array(len(self.processes) * [0])
        ml_process_weights = np.array(len(self.processes) * [0])
        sum_eventweights_processes = np.array(len(self.processes) * [0])
        dataset_proc_idx = {}  # bookkeeping which process each dataset belongs to

        #
        # determine process of each dataset and count number of events & sum of eventweights for this process
        #

        for dataset, files in input["events"][self.config_inst.name].items():
            t0 = time.time()

            dataset_inst = self.config_inst.get_dataset(dataset)
            if len(dataset_inst.processes) != 1:
                raise Exception("only 1 process inst is expected for each dataset")

            # TODO: use stats here instead
            N_events = sum([len(ak.from_parquet(inp["mlevents"].fn)) for inp in files])
            # NOTE: this only works as long as each dataset only contains one process
            sum_eventweights = sum([
                ak.sum(ak.from_parquet(inp["mlevents"].fn).normalization_weight)
                for inp in files],
            )
            for i, proc in enumerate(process_insts):
                ml_process_weights[i] = self.ml_process_weights[proc.name]
                leaf_procs = [p for p, _, _ in self.config_inst.get_process(proc).walk_processes(include_self=True)]
                if dataset_inst.processes.get_first() in leaf_procs:
                    logger.info(f"the dataset *{dataset}* is used for training the *{proc.name}* output node")
                    dataset_proc_idx[dataset] = i
                    N_events_processes[i] += N_events
                    sum_eventweights_processes[i] += sum_eventweights
                    continue

            if dataset_proc_idx.get(dataset, -1) == -1:
                raise Exception(f"dataset {dataset} is not matched to any of the given processes")

            logger.info(f"Weights done for {dataset} in {(time.time() - t0):.3f}s")

        # Number to scale weights such that the largest weights are at the order of 1
        # (only implemented for eqweight = True)
        weights_scaler = min(N_events_processes / ml_process_weights)

        #
        # set inputs, weights and targets for each datset and fold
        #

        DNN_inputs = {
            "weights": None,
            "inputs": None,
            "inputs2": None,
            "target": None,
        }

        sum_nnweights_processes = {}

        self.target_dict = {}

        for dataset, files in input["events"][self.config_inst.name].items():
            t0 = time.time()
            this_proc_idx = dataset_proc_idx[dataset]
            proc_name = self.processes[this_proc_idx]
            N_events_proc = N_events_processes[this_proc_idx]
            sum_eventweights_proc = sum_eventweights_processes[this_proc_idx]

            logger.info(
                f"dataset: {dataset}, \n  #Events: {N_events_proc}, "
                f"\n  Sum Eventweights: {sum_eventweights_proc}",
            )
            sum_nnweights = 0

            self.target_dict[f'{proc_name}'] = this_proc_idx

            for inp in files:
                events = ak.from_parquet(inp["mlevents"].path)
                weights = events.normalization_weight
                if self.eqweight:
                    weights = weights * weights_scaler / sum_eventweights_proc
                    custom_procweight = self.ml_process_weights[proc_name]
                    weights = weights * custom_procweight

                weights = ak.to_numpy(weights)

                if np.any(~np.isfinite(weights)):
                    raise Exception(f"Infinite values found in weights from dataset {dataset}")

                sum_nnweights += sum(weights)
                sum_nnweights_processes.setdefault(proc_name, 0)
                sum_nnweights_processes[proc_name] += sum(weights)

                # remove columns not used in training
                for var in events.fields:
                    if var not in self.input_features[0] and var not in self.input_features[1]:
                        print(f"removing column {var}")
                        events = remove_ak_column(events, var)

                events2 = events[self.input_features[1]]
                events = events[self.input_features[0]]

                # reshape raw inputs
                events = reshape_raw_inputs1(events, self.n_features, self.input_features[0])
                events2 = reshape_raw_inputs2(events2)

                # create the truth values for the output layer
                target = np.zeros((len(events), len(self.processes)))
                target[:, this_proc_idx] = 1
                if np.any(~np.isfinite(target)):
                    raise Exception(f"Infinite values found in target from dataset {dataset}")
                if DNN_inputs["weights"] is None:
                    DNN_inputs["weights"] = weights
                    DNN_inputs["inputs"] = events
                    DNN_inputs["inputs2"] = events2
                    DNN_inputs["target"] = target
                else:
                    # check max number of jets of datasets and append EMPTY_FLOAT columns if necessary
                    if DNN_inputs["inputs"].shape[1] != events.shape[1]:
                        if DNN_inputs["inputs"].shape[1] > events.shape[1]:
                            n_extra_columns = DNN_inputs["inputs"].shape[1] - events.shape[1]
                            extra_columns = np.full((events.shape[0], n_extra_columns), 0)
                            events = np.concatenate((events, extra_columns), axis=1)
                        else:
                            n_extra_columns = events.shape[1] - DNN_inputs["inputs"].shape[1]
                            extra_columns = np.full((DNN_inputs["inputs"].shape[0], n_extra_columns), 0)
                            DNN_inputs["inputs"] = np.concatenate((DNN_inputs["inputs"], extra_columns), axis=1)
                    DNN_inputs["weights"] = np.concatenate([DNN_inputs["weights"], weights])
                    DNN_inputs["inputs"] = np.concatenate([DNN_inputs["inputs"], events])
                    DNN_inputs["inputs2"] = np.concatenate([DNN_inputs["inputs2"], events2])
                    DNN_inputs["target"] = np.concatenate([DNN_inputs["target"], target])
            logger.debug(f"   weights: {weights[:5]}")
            logger.debug(f"   Sum NN weights: {sum_nnweights}")

            logger.info(f"Inputs done for {dataset} in {(time.time() - t0):.3f}s")

        logger.info(f"Sum of weights per process: {sum_nnweights_processes}")

        #
        # shuffle events and split into train and validation fold
        #
        inputs_size = sum([arr.size * arr.itemsize for arr in DNN_inputs.values()])
        logger.info(f"inputs size is {inputs_size / 1024**3} GB")

        # from IPython import embed; embed()

        shuffle_indices = np.array(range(len(DNN_inputs["weights"])))
        np.random.shuffle(shuffle_indices)

        validation_fraction = 0.25
        N_validation_events = int(validation_fraction * len(DNN_inputs["weights"]))

        train, validation = {}, {}
        for k in DNN_inputs.keys():
            DNN_inputs[k] = DNN_inputs[k][shuffle_indices]

            validation[k] = DNN_inputs[k][:N_validation_events]
            train[k] = DNN_inputs[k][N_validation_events:]

        # reshape and normalize inputs
        train = reshape_norm_inputs(train, self.n_features, self.norm_features, self.input_features)
        validation = reshape_norm_inputs(validation, self.n_features, self.norm_features, self.input_features)
        from IPython import embed; embed()

        return train, validation

    def instant_evaluate(
        self,
        task: law.Task,
        model,
        feature_names,
        class_names,
        train: tf.data.Dataset,
        validation: tf.data.Dataset,
        output: law.LocalDirectoryTarget,
    ) -> None:
        # store the model history
        output.child("model_history.pkl", type="f").dump(model.history.history)

        def call_func_safe(func, *args, **kwargs) -> Any:
            """
            Small helper to make sure that our training does not fail due to plotting
            """
            t0 = time.perf_counter()

            try:
                outp = func(*args, **kwargs)
                # logger.info(f"Function '{func.__name__}' done; took {(time.perf_counter() - t0):.2f} seconds")
            except Exception as e:
                # logger.warning(f"Function '{func.__name__}' failed due to {type(e)}: {e}")
                print('Failed')
                from IPython import embed; embed()
                outp = None

            return outp

        # make some plots of the history
        call_func_safe(plot_accuracy, model.history.history, output)
        call_func_safe(plot_loss, model.history.history, output)

        # evaluate training and validation sets
        train['prediction'] = call_func_safe(model, [train['inputs'], train['inputs2']])
        validation['prediction'] = call_func_safe(model, [validation['inputs'], validation['inputs2']])
        train['prediction'] = np.reshape(train['prediction'], [len(train['prediction']), len(train['prediction'][0][0])])
        validation['prediction'] = np.reshape(validation['prediction'], [len(validation['prediction']), len(validation['prediction'][0][0])])
        train['target'] = np.reshape(train['target'], [len(train['target']), len(train['target'][0][0])])
        validation['target'] = np.reshape(validation['target'], [len(validation['target']), len(validation['target'][0][0])])

        # create some confusion matrices
        call_func_safe(plot_confusion, model, train, output, "train", self.process_insts)
        call_func_safe(plot_confusion, model, validation, output, "validation", self.process_insts)

        # create some ROC curves
        call_func_safe(plot_roc_ovr, train, output, "train", self.process_insts)
        call_func_safe(plot_roc_ovr, validation, output, "validation", self.process_insts)

        # create plots for all output nodes + Significance for each node
        call_func_safe(plot_output_nodes, model, train, validation, output, self.process_insts)
        call_func_safe(plot_significance, model, train, validation, output, self.process_insts)

        # create shap feature ranking
        # call_func_safe(plot_shap_values, model, train, output, self.process_insts, self.target_dict, feature_names)

    def train(
        self,
        task: law.Task,
        input: Any,
        output: law.LocalDirectoryTarget,
    ) -> ak.Array:
        # np.random.seed(1337)  # for reproducibility

        # Load Custom Model
        from hbt.ml.DNN_simple import CustomModel

        physical_devices = tf.config.list_physical_devices("GPU")
        try:
            tf.config.experimental.set_memory_growth(physical_devices[0], True)
        except:
            # Invalid device or cannot modify virtual devices once initialized.
            pass

        #
        # input preparation
        #

        train, validation = self.prepare_inputs(task, input)

        # check for infinite values
        # for key in train.keys():
        #     if np.any(~np.isfinite(train[key])):
        #         raise Exception(f"Infinite values found in training {key}")
        #     if np.any(~np.isfinite(validation[key])):
        #         raise Exception(f"Infinite values found in validation {key}")

        gc.collect()
        logger.info("garbage collected")

        #
        # model preparation
        #

        # from keras.layers import Dense, BatchNormalization

        # define the DNN model
        # TODO: do this Funcional instead of Sequential
        # string to call custom layers: Sum, Max, Min, Mean, Var, Std
        # give list of strs to choose layers
        model = CustomModel(layer_strs=self.custom_layers, n_output_nodes=self.n_output_nodes,
                            batch_norm_deepSets=self.batch_norm_deepSets, batch_norm_ff=self.batch_norm_ff,
                            nodes_deepSets=self.nodes_deepSets, nodes_ff=self.nodes_ff,
                            activation_func_deepSets=self.activation_func_deepSets,
                            activation_func_ff=self.activation_func_ff)

        activation_settings = {
            "elu": ("ELU", "he_uniform", "Dropout"),
            "relu": ("ReLU", "he_uniform", "Dropout"),
            "prelu": ("PReLU", "he_normal", "Dropout"),
            "selu": ("selu", "lecun_normal", "AlphaDropout"),
            "tanh": ("tanh", "glorot_normal", "Dropout"),
            "softmax": ("softmax", "glorot_normal", "Dropout"),
        }
        keras_act_name, init_name, dropout_layer = activation_settings[self.activation]

        optimizer = keras.optimizers.Adam(
            learning_rate=self.learningrate, beta_1=0.9, beta_2=0.999,
            epsilon=1e-6, amsgrad=False,
        )
        model.compile(
            loss="categorical_crossentropy",
            optimizer=optimizer,
            weighted_metrics=["categorical_accuracy"],
        )

        #
        # training
        #

        # early stopping to determine the 'best' model
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            min_delta=0,
            patience=int(self.epochs / 4),
            verbose=1,
            mode="auto",
            baseline=None,
            restore_best_weights=True,
            start_from_epoch=0,
        )

        reduceLR = tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.1,
            patience=int(self.epochs / 6),
            verbose=1,
            mode="auto",
            min_lr=0.01 * self.learningrate,
        )

        logger.info("input to tf Dataset")
        # .shuffle(buffer_size=len(train["inputs"], reshuffle_each_iteration=True).repeat(self.epochs).batch(self.batchsize)
        # with tf.device("CPU"):
        #     tf_train = tf.data.Dataset.from_tensor_slices(
        #         (train["inputs"], train["target"]),
        #     ).batch(self.batchsize)
        #     tf_validate = tf.data.Dataset.from_tensor_slices(
        #         (validation["inputs"], validation["target"]),
        #     ).batch(self.batchsize)

        # tf_train = tf.data.Dataset.from_tensor_slices((train["inputs"], train["target"]))
        # tf_validate = tf.data.Dataset.from_tensor_slices((validation["inputs"], validation["target"]))
        tf_train = [[train['inputs'], train['inputs2']], train['target']]
        tf_validation = [[validation['inputs'], validation['inputs2']], validation['target']]

        # for no deep sets
        # tf_train = [tf.squeeze(train['inputs2'], axis=1), tf.squeeze(train['target'], axis=1)]
        # tf_validation = [tf.squeeze(validation['inputs2'], axis=1), tf.squeeze(validation['target'], axis=1)]

        fit_kwargs = {
            "epochs": self.epochs,
            "callbacks": [early_stopping, reduceLR],
            "verbose": 2,
        }

        # train the model
        logger.info("Start training...")
        model.fit(
            tf_train[0], tf_train[1],
            validation_data=tf_validation,
            batch_size=self.batchsize,
            **fit_kwargs,
        )

        # save the model and history; TODO: use formatter
        # output.dump(model, formatter="tf_keras_model")
        output.parent.touch()
        model.save(output.path)

        # plotting of loss, acc, roc, nodes, confusion for each fold
        self.instant_evaluate(task, model, self.input_features, self.processes, train, validation, output)

        write_info_file(output, self.custom_layers, self.nodes_deepSets, self.nodes_ff,
            self.n_output_nodes, self.batch_norm_deepSets, self.batch_norm_ff, self.input_features,
            self.process_insts, self.activation_func_deepSets, self.activation_func_ff)

    def evaluate(
        self,
        task: law.Task,
        events: ak.Array,
        models: list(Any),
        fold_indices: ak.Array,
        events_used_in_training: bool = True,
    ) -> None:

        # output = task.target(f"mlmodel_f{task.branch}of{self.folds}_{self.model_name}", dir=True)
        # output_all_folds = task.target(f"mlmodel_all_folds_{self.model_name}", dir=True)
        logger.info(f"Evaluation of dataset {task.dataset}")
        models, history = zip(*models)

        # create a copy of the inputs to use for evaluation
        inputs = ak.copy(events)
        events2 = events[self.input_features[1]]
        events1 = events[self.input_features[0]]

        events1 = reshape_raw_inputs1(events1, self.n_features, self.input_features[0])
        events2 = reshape_raw_inputs2(events2)

        target_dict = {'graviton_hh_ggf_bbtautau_m400_madgraph': 0,
            'graviton_hh_vbf_bbtautau_m400_madgraph': 1}

        # create target and add to test dict
        target = np.zeros((events1.shape[0], 2))
        target[:, target_dict[task.dataset]] = 1

        test = {'inputs': events1,
                'inputs2': events2,
                'target': target,
                }

        test = reshape_norm_inputs(test, self.n_features, )

        # inputs to feed to the model
        inputs = [test["inputs"], test["inputs2"]]

        # do prediction for all models and all inputs
        predictions = []
        for i, model in enumerate(models):
            pred = model.predict(inputs)
            pred = np.reshape(pred, [len(pred), len(pred[0][0])])
            pred = ak.from_numpy(pred)
            if len(pred[0]) != len(self.processes):
                raise Exception("Number of output nodes should be equal to number of processes")
            predictions.append(pred)

        '''In pred, each model sees the complete set of data, this includes data used for training
        and validation. For each model, keep only the predictions on inputs that were not yet seen
        by the model during training/validation. Keep only prediction on the subset k that was not
        yet seen by the model. Combine all of the predictions on the k subsets by the k different
        models into one prediction array'''
        '''outputs: generate array with shape of the final pred array with only etries -1
        -> later overriden. Since all k substes combined are the complete set of data, all
        entries in outputs will later be overriden.with prediction by the model associated with
        a given subset.'''
        # combine all models into 1 output score, using the model that has not seen test set yet
        outputs = ak.where(ak.ones_like(predictions[0]), -1, -1)
        for i in range(self.folds):
            logger.info(f"Evaluation fold {i}")
            # output = task.target(f"mlmodel_f{task.branch}of{self.folds}_{self.model_name}", dir=True)
            # output = task.target(f"mlmodel_f{i}of{self.folds}_{self.model_name}", dir=True)
            # reshape mask from N*bool to N*k*bool (TODO: simpler way?)
            '''get indices of the events that belong to k subset not yet seen by the model and
            override the entries at these indices in outputs with the prediction of the model.'''
            idx = ak.to_regular(ak.concatenate([ak.singletons(fold_indices == i)] * len(self.processes), axis=1))
            outputs = ak.where(idx, predictions[i], outputs)
            events = set_ak_column(events, f"pred_model_{i}", np.squeeze(predictions[i]))

        test['prediction'] = np.squeeze(outputs)

        if len(outputs[0]) != len(self.processes):
            raise Exception("Number of output nodes should be equal to number of processes")
        '''Create on column for each proc containing the NN output score of output node associated
        with that process.'''
        for i, proc in enumerate(self.processes):
            events = set_ak_column(
                events, f"{self.cls_name}.score_{proc}", outputs[:, i],
            )

        events = set_ak_column(events, "predictions", test["prediction"])
        events = set_ak_column(events, f"{self.cls_name}.ml_truth_label", np.squeeze(test['target']))

        # ML categorization on top of existing categories
        # ml_categories = [cat for cat in self.config_inst.categories if "ml_" in cat.name]
        # ml_proc_to_id = {cat.name.replace("ml_", ""): cat.id for cat in ml_categories}

        # scores = ak.Array({
        #     f.replace("score_", ""): events[self.cls_name, f]
        #     for f in events[self.cls_name].fields if f.startswith("score_")
        # })

        # ml_category_ids = max_score = ak.Array(np.zeros(len(events)))
        # for proc in scores.fields:
        #     ml_category_ids = ak.where(scores[proc] > max_score, ml_proc_to_id[proc], ml_category_ids)
        #     max_score = ak.where(scores[proc] > max_score, scores[proc], max_score)

        # category_ids = ak.where(
        #     events.category_ids != 1,  # Do not split Inclusive category into DNN sub-categories
        #     events.category_ids + ak.values_astype(ml_category_ids, np.int32),
        #     events.category_ids,
        # )
        # events = set_ak_column(events, "category_ids", category_ids)

        return events
