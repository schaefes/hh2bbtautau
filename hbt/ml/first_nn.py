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
    write_info_file, plot_shap_baseline, plot_shap_values_deep_sets_mean,
    plot_feature_ranking_deep_sets, plot_shap_values_deep_sets_sum,
    plot_shap_values_deep_sets, plot_confusion2, check_distribution,
    plot_roc_ovr2,
)

np = maybe_import("numpy")
ak = maybe_import("awkward")
tf = maybe_import("tensorflow")
pickle = maybe_import("pickle")
keras = maybe_import("tensorflow.keras")
sklearn = maybe_import("sklearn")
# dotted_dict = maybe_import("dotted_dict")

logger = law.logger.get_logger(__name__)


def resort_jets(jet_input_shaped, jet_input_features, sorting_feature, masking_val):
    # takes as input the shaped jet input returned by reshape_norm_inputs
    jets = np.where(jet_input_shaped == masking_val, EMPTY_FLOAT, jet_input_shaped)
    sorting_idx = jet_input_features.index(sorting_feature)
    # apply argsort on the negative array to get a sorting for a descending order
    sorting_indices = np.argsort(-jets[:, :, sorting_idx])
    jets_new = np.full_like(jets, 0.0)
    for i in np.arange(jets.shape[0]):
        sorted_event = jets[np.arange(jets.shape[0])[i], sorting_indices[i]]
        jets_new[i] = sorted_event
    jets_new = np.where(jets == EMPTY_FLOAT, masking_val, jets_new)

    return jets_new


# dedicated function to get the phi array of the leading jet
def get_leading_phi_array(events, projection_phi):
    phi_str = 'phi'
    for fields_str in events.fields:
        if 'phi' in fields_str:
            phi_str = fields_str
            break
    leading_phi_array = getattr(events, phi_str)[:, 0]
    return leading_phi_array


# function that projects all phi values relative to the phi value of the leading jet
# leading jet is set to phi 0 and all other jet phi values are replaced by the distance delta phi to leading jet
def phi_projection(projection_phi_array, phi_array):
    projected_array = phi_array - projection_phi_array
    projected_array = ak.where(projected_array < -np.pi, projected_array + 2 * np.pi, projected_array)
    projected_array = ak.where(projected_array > np.pi, projected_array - 2 * np.pi, projected_array)
    # check that the values of phi are not empty float/padded jets, such that empty floats are not modified
    projected_array = ak.where(phi_array == EMPTY_FLOAT, phi_array, projected_array)
    return projected_array


# Define functions to normalize and shape inputs1 and 2
def reshape_raw_inputs1(events, n_features, input_features, projection_phi):
    # leading_phi = get_leading_phi_array(events)
    column_counter = 0
    num_events, max_jets = ak.to_numpy(events[input_features[0]]).shape
    zeros = np.zeros((num_events, max_jets * n_features))
    for i in range(max_jets):
        for jet_features in input_features:
            if 'phi' in jet_features:
                zeros[:, column_counter] = phi_projection(projection_phi, events[jet_features][:, i])
            else:
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
def reshape_norm_inputs(events_dict, n_features, norm_features, input_features, n_output_nodes, dummy, baseline_jets, masking_val, sorting_feature, train_sorting):
    # reshape train['inputs'] for DeepSets: [#events, #jets, -1] and apply standardization (z-score)
    # calculate mean and std for normalization
    # get the name of the pt jets column
    events_shaped = events_dict["inputs"].reshape((-1, n_features))
    mean_feature1 = np.zeros(n_features)
    std_feature1 = np.zeros(n_features)
    delete_standardization1 = []

    for i, feature in enumerate(input_features[0]):
        if feature not in norm_features[0]:
            delete_standardization1.append(i)

    for i in range(n_features):
        mask_empty_floats = events_shaped[:, i] != EMPTY_FLOAT
        mean_feature1[i] = np.mean(events_shaped[:, i][mask_empty_floats])
        std_feature1[i] = np.std(events_shaped[:, i][mask_empty_floats])

    std_feature1 = std_feature1 / 3
    mean_feature1[delete_standardization1] = 0
    std_feature1[delete_standardization1] = 1

    jets_collection = []
    jets_baseline_collection = []
    for i in range(events_dict['inputs'].shape[0]):
        arr_events = events_dict['inputs'][i]
        arr_shaped = np.reshape(arr_events, (-1, n_features))
        arr_mask = np.ma.masked_where(arr_shaped != EMPTY_FLOAT, arr_shaped).mask[:, 0]
        arr_normalized = (arr_shaped[arr_mask] - mean_feature1) / std_feature1
        arr_shaped[arr_mask] = arr_normalized
        arr_shaped = np.where(arr_shaped == EMPTY_FLOAT, masking_val, arr_shaped)
        baseline_model_inp = arr_shaped[:baseline_jets, :].flatten()
        jets_baseline_collection.append(baseline_model_inp)
        jets_shaped = np.reshape(arr_shaped.flatten(), (1, -1, n_features))
        jets_collection.append(jets_shaped)
    stacked_events = tf.convert_to_tensor(jets_collection)
    jets_for_resort = np.array(jets_collection)
    jets_for_resort = np.squeeze(jets_for_resort, axis=1)
    # jets resorted for comarison
    resorted_jets = resort_jets(jets_for_resort, input_features[0], sorting_feature, masking_val)
    base_jets_resorted = resorted_jets[:, :baseline_jets].reshape(resorted_jets.shape[0], -1)
    base_jets_resorted = tf.convert_to_tensor(base_jets_resorted)
    events_dict[f'inputs_{sorting_feature}'] = tf.convert_to_tensor(resorted_jets)
    # sorted jets for training
    sorted_jets = resort_jets(jets_for_resort, input_features[0], train_sorting, masking_val)
    base_jets_sorted = sorted_jets[:, :baseline_jets].reshape(sorted_jets.shape[0], -1)
    base_jets_sorted = tf.convert_to_tensor(base_jets_sorted)
    events_dict[f'inputs_{train_sorting}'] = tf.squeeze(stacked_events, axis=1)

    # normalization of inputs2
    mean_feature2 = np.zeros(len(input_features[1]))
    std_feature2 = np.zeros(len(input_features[1]))
    delete_standardization2 = []

    # get mean and std for inv masses seperately with a mass mask excluding -1/EMPTY FLOAT vals in
    # calculation of mean and std
    filled_mask = events_dict['inputs2'] != EMPTY_FLOAT
    for i in range(len(input_features[1])):
        mean = np.mean(events_dict['inputs2'][:, i][filled_mask[:, i]])
        mean_feature2[i] = mean
        std = np.std(events_dict['inputs2'][:, i][filled_mask[:, i]])
        std_feature2[i] = std
        sig_pad = mean - (3 * std)
        if dummy == "1":
            events_dict['inputs2'][:, i] = np.where(events_dict['inputs2'][:, i] == EMPTY_FLOAT, -1, events_dict['inputs2'][:, i])
        if dummy == "3sig":
            events_dict['inputs2'][:, i] = np.where(events_dict['inputs2'][:, i] == EMPTY_FLOAT, sig_pad, events_dict['inputs2'][:, i])

    for i, feature in enumerate(input_features[1]):
        if feature not in norm_features[1]:
            delete_standardization2.append(i)

    mean_feature2[delete_standardization2] = 0
    std_feature2[delete_standardization2] = 1

    # apply standardization and reshape
    for i in range(len(input_features[1])):
        events_dict['inputs2'][:, i] = (events_dict['inputs2'][:, i] - mean_feature2[i]) / std_feature2[i]

    inputs_2 = tf.reshape(events_dict['inputs2'], [-1, events_dict['inputs2'].shape[1]])
    events_dict[f'input_jets_baseline_{train_sorting}'] = np.concatenate((base_jets_sorted, inputs_2), axis=1)
    events_dict[f'input_jets_baseline_{sorting_feature}'] = np.concatenate((base_jets_resorted, inputs_2), axis=1)
    events_dict['inputs2'] = inputs_2

    # reshape of target
    events_dict['target'] = tf.reshape(events_dict['target'], [-1, n_output_nodes])

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
            aggregations: list | None = None,
            L2: bool | None = None,
            norm_features: list | None = None,
            empty_overwrite: str | None = None,
            quantity_weighting: bool | None = None,
            jet_num_cut: int | None = None,
            baseline_jets: int | None = None,
            model_type: str | None = None,
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
        self.aggregations = aggregations or self.aggregations
        self.batch_norm_ff = batch_norm_ff or self.batch_norm_ff
        self.batch_norm_deepSets = batch_norm_deepSets or self.batch_norm_deepSets
        self.activation_func_deepSets = activation_func_deepSets or self.activation_func_deepSets
        self.activation_func_ff = activation_func_ff or self.activation_func_ff
        self.nodes_deepSets = nodes_deepSets or self.nodes_deepSets
        self.nodes_ff = nodes_ff or self.nodes_ff
        self.L2 = L2 or self.L2
        self.norm_features = norm_features or self.norm_features
        self.empty_overwrite = empty_overwrite or self.empty_overwrite
        self.quantity_weighting = quantity_weighting or self.quantity_weighting
        self.jet_num_cut = jet_num_cut or self.jet_num_cut
        self.baseline_jets = baseline_jets or self.baseline_jets
        self.model_type = model_type or self.model_type
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
        return {"normalization_weight", "category_ids", "predictions_graviton_hh_vbf_bbtautau_m400_madgraph_DeepSets"} | set(self.input_features[0]) | set(self.input_features[1]) | set(self.projection_phi)

    def produces(self, config_inst: od.Config) -> set[Route | str]:
        produced = set()
        for proc in self.processes:
            produced.add(f"{self.cls_name}.score_{proc}__{self.model_name}")
            produced.add(f"{self.cls_name}.predictions_{proc}__{self.model_name}")
            produced.add(f"{self.cls_name}.ml_truth_label_{proc}__{self.model_name}")
            produced.add(f"{self.cls_name}.pred_target_{proc}__{self.model_name}")
            produced.add(f"{self.cls_name}.target_label_{proc}__{self.model_name}")
            produced.add(f"{self.cls_name}.events_weights_{proc}__{self.model_name}")
            for i in range(self.folds):
                produced.add(f"{self.cls_name}.pred_model_{proc}_fold{i}__{self.model_name}")
                produced.add(f"{self.cls_name}.DeepSetsInpPt_{proc}_fold{i}__{self.model_name}")
                produced.add(f"{self.cls_name}.DeepSetsInpEta_{proc}_fold{i}__{self.model_name}")

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

        # set seed for shuffeling for reproducebility
        np.random.seed(1337)
        tf.random.set_seed(1337)

        if self.quantity_weighting:
            # properly define the ml_process_weights through the number of events per dataset
            for dataset0, files0 in input["events"][self.config_inst.name].items():
                t0 = time.time()

                dataset_inst = self.config_inst.get_dataset(dataset0)
                if len(dataset_inst.processes) != 1:
                    raise Exception("only 1 process inst is expected for each dataset")

                N_events = sum([len(ak.from_parquet(inp["mlevents"].fn)) for inp in files0])
                proc_name_of_dataset = dataset0.split("_madgraph")[0]
                self.ml_process_weights[proc_name_of_dataset] = N_events

            max_events_of_proc = np.max(list(self.ml_process_weights.values()))
            for key_proc, value_proc in self.ml_process_weights.items():
                self.ml_process_weights[key_proc] = max_events_of_proc / value_proc

        # max_events_per_fold = int(self.max_events / (self.folds - 1))
        self.process_insts = []
        # self.input_features[0].remove(self.projection_phi[0])
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
                projection_phi = events[self.projection_phi[0]]
                for var in events.fields:
                    if var not in self.input_features[0] and var not in self.input_features[1]:
                        print(f"removing column {var}")
                        events = remove_ak_column(events, var)

                # make a cut on events based on a min number of jets required per event
                # get the string name of njets for the given jet collection
                njets_field = [i for i in self.input_features[1] if 'njets' in i][0]
                events_n_jets = getattr(events, njets_field)
                mask = events_n_jets > self.jet_num_cut
                weights = weights[mask]
                events_new = {}
                for feature in events.fields:
                    events_new[feature] = events[feature][mask]
                events_new = ak.Array(events_new)

                events2 = events_new[self.input_features[1]]
                events = events_new[self.input_features[0]]

                # reshape raw inputs
                events = reshape_raw_inputs1(events, self.n_features, self.input_features[0], projection_phi)
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
                            extra_columns = np.full((events.shape[0], n_extra_columns), EMPTY_FLOAT)
                            events = np.concatenate((events, extra_columns), axis=1)
                        else:
                            n_extra_columns = events.shape[1] - DNN_inputs["inputs"].shape[1]
                            extra_columns = np.full((DNN_inputs["inputs"].shape[0], n_extra_columns), EMPTY_FLOAT)
                            DNN_inputs["inputs"] = np.concatenate((DNN_inputs["inputs"], extra_columns), axis=1)
                    DNN_inputs["weights"] = np.concatenate([DNN_inputs["weights"], weights])
                    DNN_inputs["inputs"] = np.concatenate([DNN_inputs["inputs"], events])
                    DNN_inputs["inputs2"] = np.concatenate([DNN_inputs["inputs2"], events2])
                    DNN_inputs["target"] = np.concatenate([DNN_inputs["target"], target])
            logger.debug(f"   weights: {weights[:5]}")
            logger.debug(f"   Sum NN weights: {sum_nnweights}")

            logger.info(f"Inputs done for {dataset} in {(time.time() - t0):.3f}s")

        logger.info(f"Sum of weights per process: {sum_nnweights_processes}")

        # shuffle events and split into train and validation fold
        inputs_size = sum([arr.size * arr.itemsize for arr in DNN_inputs.values()])
        logger.info(f"inputs size is {inputs_size / 1024**3} GB")

        shuffle_indices = np.array(range(len(DNN_inputs["weights"])))
        np.random.shuffle(shuffle_indices)
        print('SHUFFLE:', shuffle_indices)

        validation_fraction = 0.25
        N_validation_events = int(validation_fraction * len(DNN_inputs["weights"]))

        train, validation = {}, {}
        for k in DNN_inputs.keys():
            DNN_inputs[k] = DNN_inputs[k][shuffle_indices]

            validation[k] = DNN_inputs[k][:N_validation_events]
            train[k] = DNN_inputs[k][N_validation_events:]
        # reshape and normalize inputs
        train = reshape_norm_inputs(train, self.n_features, self.norm_features, self.input_features, self.n_output_nodes, self.empty_overwrite, self.baseline_jets, self.masking_val, self.resorting_feature, self.train_sorting)
        validation = reshape_norm_inputs(validation, self.n_features, self.norm_features, self.input_features, self.n_output_nodes, self.empty_overwrite, self.baseline_jets, self.masking_val, self.resorting_feature, self.train_sorting)

        return train, validation

    def instant_evaluate(
        self,
        task: law.Task,
        model,
        masking_model,
        feature_names,
        class_names,
        sorting,
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
                #from IPython import embed; embed()
                outp = None

            return outp

        # evaluate training and validation sets
        if sorting == self.train_sorting:
            sorting = sorting + "_train_sorting"
            if self.model_type == "baseline":
                train['prediction'] = call_func_safe(model, train[f'input_jets_baseline_{self.train_sorting}'])
                validation['prediction'] = call_func_safe(model, validation[f'input_jets_baseline_{self.train_sorting}'])
                # plot shap for baseline model
                # call_func_safe(plot_shap_baseline, model, train, output, self.process_insts, self.target_dict, feature_names, self.baseline_jets, self.train_sorting)

            elif self.model_type == "DeepSets":
                train['prediction'] = call_func_safe(model, [train[f'inputs_{self.train_sorting}'], train['inputs2']])
                validation['prediction'] = call_func_safe(model, [validation[f'inputs_{self.train_sorting}'], validation['inputs2']])
                # plot shap for Deep Sets
                # call_func_safe(plot_feature_ranking_deep_sets, masking_model, train, output, self.process_insts, self.target_dict, feature_names, self.train_sorting)
                # call_func_safe(plot_shap_values_deep_sets_mean, model, train, output, self.process_insts, self.target_dict, feature_names, self.train_sorting)
                # call_func_safe(plot_shap_values_deep_sets, model, train, output, self.process_insts, self.target_dict, feature_names, self.train_sorting)
                # call_func_safe(plot_shap_values_deep_sets_sum, model, train, output, self.process_insts, self.target_dict, feature_names, self.train_sorting)

            # make some plots of the history
            call_func_safe(plot_accuracy, model.history.history, output)
            call_func_safe(plot_loss, model.history.history, output)

        else:
            if self.model_type == "baseline":
                train['prediction'] = call_func_safe(model, train[f'input_jets_baseline_{self.resorting_feature}'])
                validation['prediction'] = call_func_safe(model, validation[f'input_jets_baseline_{self.resorting_feature}'])
            elif self.model_type == "DeepSets":
                train['prediction'] = call_func_safe(model, [train[f'inputs_{self.resorting_feature}'], train['inputs2']])
                validation['prediction'] = call_func_safe(model, [validation[f'inputs_{self.resorting_feature}'], validation['inputs2']])

        # create some confusion matrices
        call_func_safe(plot_confusion, sorting, model, train, output, "train", self.process_insts)
        call_func_safe(plot_confusion, sorting, model, validation, output, "validation", self.process_insts)
        call_func_safe(plot_confusion2, sorting, model, validation, output, "validation", self.process_insts)

        # create some ROC curves
        call_func_safe(plot_roc_ovr, sorting, train, output, "train", self.process_insts)
        call_func_safe(plot_roc_ovr, sorting, validation, output, "validation", self.process_insts)
        call_func_safe(plot_roc_ovr2, sorting, validation, output, "validation", self.process_insts)

        # create plots for all output nodes + Significance for each node
        call_func_safe(plot_output_nodes, sorting, model, train, validation, output, self.process_insts)
        call_func_safe(plot_significance, sorting, model, train, validation, output, self.process_insts)


    def train(
        self,
        task: law.Task,
        input: Any,
        output: law.LocalDirectoryTarget,
    ) -> ak.Array:
        # set seed for shuffeling for reproducebility
        np.random.seed(1337)
        tf.random.set_seed(1337)
        # Load Custom Model
        from hbt.ml.DNN_automated import CombinedDeepSetNetwork, BaseLineFF, ShapMasking
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
        eta_feature_train = train["inputs_CustomVBFMaskJets2_pt"][:,:,3].numpy().flatten()
        pt_feature_train = train["inputs_CustomVBFMaskJets2_pt"][:,:,2].numpy().flatten()
        QG_feature_train = train["inputs_CustomVBFMaskJets2_pt"][:,:,-1].numpy().flatten()
        check_distribution(output, pt_feature_train, "pt_train_set", self.masking_val)
        # check_distribution(output, QG_feature_train, "QG_train_set", self.masking_val)
        check_distribution(output, eta_feature_train, "eta_train_set", self.masking_val)

        eta_feature_validation = validation["inputs_CustomVBFMaskJets2_pt"][:,:,3].numpy().flatten()
        pt_feature_validation = validation["inputs_CustomVBFMaskJets2_pt"][:,:,2].numpy().flatten()
        QG_feature_validation = validation["inputs_CustomVBFMaskJets2_pt"][:,:,-1].numpy().flatten()
        check_distribution(output, pt_feature_validation, "pt_val_set", self.masking_val)
        # check_distribution(output, QG_feature_validation, "QG_val_set", self.masking_val)
        check_distribution(output, eta_feature_validation, "eta_val_set", self.masking_val)
        print('DONE')

        # save process info for test set evaluation
        evalu_dict = {}
        for i, proc in enumerate(self.target_dict.keys()):
            evalu_dict[f'{proc}'] = [self.target_dict[f'{proc}'], self.process_insts[i].label]
        output.child(f"evalu_dict.json", type="f").dump(self.target_dict, formatter="json")

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

        # configureations for Deep Sets and FF Network
        # min_deepSets_nodes = train[f'inputs_{self.train_sorting}'].shape[1]
        # deepSets_nodes = self.nodes_deepSets
        # deepSets_nodes[-1] = min_deepSets_nodes + 2 * len(self.input_features[0])
        deepset_config = {'nodes': self.nodes_deepSets, 'activations': self.activation_func_deepSets,
            'aggregations': self.aggregations, 'masking_val': self.masking_val}
        feedforward_config = {'nodes': self.nodes_ff, 'activations': self.activation_func_ff,
            'n_classes': self.n_output_nodes}

        if self.model_type == "baseline":
            model = BaseLineFF(feedforward_config)
            tf_train = [train[f'input_jets_baseline_{self.train_sorting}'], train['target']]
            tf_validation = [validation[f'input_jets_baseline_{self.train_sorting}'], validation['target']]
        else:
            model = CombinedDeepSetNetwork(deepset_config, feedforward_config)
            tf_train = [[train[f'inputs_{self.train_sorting}'], train['inputs2']], train['target']]
            tf_validation = [[validation[f'inputs_{self.train_sorting}'], validation['inputs2']], validation['target']]

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
            run_eagerly=False,
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

        # transform the the keys of self.ml_process_weights to their respective idx
        # so that the dict is understood by the model
        class_weights_dict = {}
        for i, proc_key in enumerate(self.processes):
            class_weights_dict[i] = self.ml_process_weights[proc_key]

        fit_kwargs = {
            "epochs": self.epochs,
            "callbacks": [early_stopping, reduceLR],
            "verbose": 2,
            "class_weight": class_weights_dict,
        }

        # train the model
        logger.info(f"Loss training weights: {self.ml_process_weights.items()}")
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

        # plotting of loss, acc, roc, nodes, confusion, shap plots for each k-fold
        if self.model_type == 'baseline':
            # pass the same model twice, the second is used as a filler since two different models need
            # to be passed for Deep Sets
            self.instant_evaluate(task, model, model, self.input_features, self.processes, self.train_sorting, train, validation, output)
            self.instant_evaluate(task, model, model, self.input_features, self.processes, self.resorting_feature, train, validation, output)
        elif self.model_type == "DeepSets":
            slice_idx = tf_train[0][0].shape[2]
            masking_model = ShapMasking(slice_idx, model)
            self.instant_evaluate(task, model, masking_model, self.input_features, self.processes, self.train_sorting, train, validation, output)
            self.instant_evaluate(task, model, masking_model, self.input_features, self.processes, self.resorting_feature, train, validation, output)
        write_info_file(output, self.aggregations, self.nodes_deepSets, self.nodes_ff,
            self.n_output_nodes, self.batch_norm_deepSets, self.batch_norm_ff, self.input_features,
            self.process_insts, self.activation_func_deepSets, self.activation_func_ff, self.learningrate,
            self.empty_overwrite, self.ml_process_weights, self.jet_num_cut, self.ml_process_weights,
            self.model_type, self.jet_collection, self.resorting_feature, self.projection_phi, self.train_sorting)

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
        proc_name = task.dataset.split("_madgraph")[0]
        models, history = zip(*models)
        model0 = models[0]
        max_jets = model0.signatures["serving_default"].structured_input_signature[1]['input_1'].shape[1]
        # create a copy of the inputs to use for evaluation
        inputs = ak.copy(events)
        events2 = events[self.input_features[1]]
        events1 = events[self.input_features[0]]
        projection_phi = events[self.projection_phi[0]]

        # prepare the input features
        # this target dict is generated in the same way as the ml truth labels in prepare inputs
        # therefore the truth labels should match
        target_dict = {}
        for i, proc in enumerate(self.processes):
            target_dict[f'{proc}_madgraph'] = i

        events1 = reshape_raw_inputs1(events1, self.n_features, self.input_features[0], projection_phi)
        events2 = reshape_raw_inputs2(events2)

        # add extra colums to the deep sets input to fir the required input shape of the model
        # padding of extra columns must use the EMPTY_FLOAT, will be replaced with masking value
        # in reshape_norm_inputs
        n_columns = max_jets * len(self.input_features[0])
        n_extra_columns = n_columns - events1.shape[1]
        if n_extra_columns != 0:
            extra_columns = np.full((events1.shape[0], n_extra_columns), EMPTY_FLOAT)
            events1 = np.concatenate((events1, extra_columns), axis=1)

        # create target and add to test dict
        target = np.zeros((events1.shape[0], len(target_dict.keys())))
        target[:, target_dict[task.dataset]] = 1

        test = {'inputs': events1,
                'inputs2': events2,
                'target': target,
                }
        test = reshape_norm_inputs(test, self.n_features, self.norm_features, self.input_features, self.n_output_nodes, self.empty_overwrite, self.baseline_jets, self.masking_val, self.resorting_feature, self.train_sorting)
        # inputs to feed to the model
        inputs = [test[f"inputs_{self.train_sorting}"], test["inputs2"]]
        if self.model_type == "baseline":
            inputs = test[f'input_jets_baseline_{self.train_sorting}']

        # do prediction for all models and all inputs
        predictions = []
        for i, model in enumerate(models):
            pred = model.predict(inputs)
            pred = ak.from_numpy(pred)
            if len(pred[0]) != len(self.processes):
                raise Exception("Number of output nodes should be equal to number of processes")
            predictions.append(pred)

        '''In pred, each model sees the complete set of data, this includes data used for training
        and validation. For each model, keep only the predictions on inputs that were not yet seen
        by the model during training/validation. Keep only prediction on the subset k that was not
        by the model. Combine all of the predictions on the k subsets by the k different
        models into one prediction array'''
        '''outputs: generate array with shape of the final pred array with only entries -1
        -> later overriden. Since all k substes combined are the complete set of data, all
        entries in outputs will later be overriden with prediction by the model associated with
        a given subset.'''
        # combine all models into 1 output score, using the model that has not seen test set yet
        outputs = ak.where(ak.ones_like(predictions[0]), -1, -1)
        weights = np.full(len(outputs), 0)
        for i in range(self.folds):
            logger.info(f"Evaluation fold {i}")
            # output = task.target(f"mlmodel_f{task.branch}of{self.folds}_{self.model_name}", dir=True)
            # output = task.target(f"mlmodel_f{i}of{self.folds}_{self.model_name}", dir=True)
            # reshape mask from N*bool to N*k*bool (TODO: simpler way?)
            '''get indices of the events that belong to k subset not yet seen by the model and
            override the entries at these indices in outputs with the prediction of the model.'''
            idx = ak.to_regular(ak.concatenate([ak.singletons(fold_indices == i)] * len(self.processes), axis=1))
            idx = ~idx
            outputs = ak.where(idx, predictions[i], outputs)
            mask_weights = (np.sum(idx, axis=1) != 0)
            weights = ak.where(mask_weights, events.normalization_weight, weights)
            events = set_ak_column(events, f"{self.cls_name}.pred_model_{proc_name}_fold{i}__{self.model_name}", np.squeeze(predictions[i]))
            # get the inp of the Deep Sets for each test fold
            # filler = np.full_like(inputs[0], self.masking_val)
            # padded_fold_inp = np.concatenate((inputs[0].numpy()[mask_weights], filler[~mask_weights]), axis=0)
            # padded_fold_inp_pt = np.reshape(padded_fold_inp[:, :, 2], (padded_fold_inp.shape[0], -1))[:, :9]
            # events = set_ak_column(events, f"{self.cls_name}.DeepSetsInpPt_{proc_name}_fold{i}__{self.model_name}", ak.Array(padded_fold_inp_pt))
            # padded_fold_inp_eta = np.reshape(padded_fold_inp[:, :, 3], (padded_fold_inp.shape[0], -1))[:, :9]
            # events = set_ak_column(events, f"{self.cls_name}.DeepSetsInpEta_{proc_name}_fold{i}__{self.model_name}", ak.Array(padded_fold_inp_eta))

        test['prediction'] = np.squeeze(outputs)

        if len(outputs[0]) != len(self.processes):
            raise Exception("Number of output nodes should be equal to number of processes")
        '''Create on column for each proc containing the NN output score of output node associated
        with that process.'''
        for i, proc in enumerate(self.processes):
            events = set_ak_column(
                events, f"{self.cls_name}.score_{proc}", outputs[:, i],
            )
        events = set_ak_column(events, f"{self.cls_name}.predictions_{proc_name}__{self.model_name}", test["prediction"])
        target_val = np.full(len(test["prediction"]), target_dict[task.dataset])
        events = set_ak_column(events, f"{self.cls_name}.ml_truth_label_{proc_name}__{self.model_name}", target_val)
        pred_target = np.concatenate((test["prediction"], target), axis=1)
        events = set_ak_column(events, f"{self.cls_name}.pred_target_{proc_name}__{self.model_name}", pred_target)
        events = set_ak_column(events, f"{self.cls_name}.target_label_{proc_name}__{self.model_name}", np.full(target.shape[0], target_dict[f'{task.dataset}']))
        events = set_ak_column(events, f"{self.cls_name}.events_weights_{proc_name}__{self.model_name}", weights)

        return events
