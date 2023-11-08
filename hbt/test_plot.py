# encoding: utf-8

from columnflow.tasks.ml import MLEvaluation, MLEvaluationWrapper
import numpy as np
from plotting_funcs import (plot_confusion, plot_roc_ovr, plot_output_nodes,
                            plot_significance, plot_confusion2, plot_roc_ovr2,
                            check_distribution)
import os
import awkward as ak

# t = MLEvaluationWrapper(
#     version="BtagsCustomVBFMaskJets2",
#     ml_model="test",
#     datasets=("graviton_hh_vbf_bbtautau_m400_madgraph",
#              "graviton_hh_ggf_bbtautau_m400_madgraph",
#              "hh_ggf_bbtautau_madgraph"),
#     calibrators=("skip_jecunc",),
#     print_status=(3,),
# )

# t.law_run()
processes = ["graviton_hh_vbf_bbtautau_m400",
            "graviton_hh_ggf_bbtautau_m400",
            "hh_ggf_bbtautau"]

label_dict = {"graviton_hh_ggf_bbtautau_m400": 'Graviton $\\rightarrow HH_{ggf,m400}$ $\\rightarrow bb\\tau\\tau$',
              "graviton_hh_vbf_bbtautau_m400": 'Graviton $\\rightarrow HH_{vbf,m400}$ $\\rightarrow bb\\tau\\tau$',
              "hh_ggf_bbtautau": '$HH_{ggf} \\rightarrow bb\\tau\\tau$',
    }

collection_dict = {}
target_dict = {}
weights = []
DeepSetsInpPt = {}
DeepSetsInpEta = {}

# get the target and predictions from MLEvaluate
for num, proc in enumerate(processes):
    t = MLEvaluation(
        version="BtagsCustomVBFMaskJets2",
        ml_model="test",
        dataset=f"{proc}_madgraph",
        calibrators=("skip_jecunc",),
        # print_status=(3,),
    )
    files = t.output()
    data = files.collection[0]["mlcolumns"].load(formatter="awkward")
    # get part of the model name from the column name
    scores = data.test
    model_check = scores.fields[0].split("__")[-1]
    if num != 0:
        if prev_model != model_check:
            raise Exception(f"Different models mixed up for process {proc}")
    for column in scores.fields:
        if model_check != column.split("__")[-1]:
            raise Exception(f"Different models mixed up for process {proc}")
        if 'target_label' in column:
            target_dict[f'{proc}'] = scores[f'{column}'][0]
        if 'pred_target' in column:
            collection_dict[f'{column}'] = scores[f'{column}']
        if 'weights' in column:
            weights.append(scores[f'{column}'])
        if 'DeepSetsInpPt' in column:
            DeepSetsInpPt[f'{column}'] = scores[f'{column}']
        if 'DeepSetsInpEta' in column:
            DeepSetsInpEta[f'{column}'] = scores[f'{column}']
    prev_model = model_check

# create the savepath for the plots
model_name = model_check
path = files.collection[0]["mlcolumns"].sibling("dummy.json", type="f").path.split("dummy")[0]
path = os.path.join(path, model_name)
if not os.path.exists(path):
    os.makedirs(path)

# for i in range(10):
#     input_folds = []
#     for key_fold, value_fold in DeepSetsInpPt.items():
#         if f"fold{i}" in key_fold:
#             input_folds.append(value_fold)
#     inp_pt = np.concatenate(input_folds, axis=0)
#     inp_pt = np.array(inp_pt).flatten()
#     check_distribution(path, inp_pt, "pT", -2, i)

# for i in range(10):
#     input_folds = []
#     for key_fold, value_fold in DeepSetsInpEta.items():
#         if f"fold{i}" in key_fold:
#             input_folds.append(value_fold)
#     inp_eta = np.concatenate(input_folds, axis=0)
#     inp_eta = np.array(inp_eta).flatten()
#     check_distribution(path, inp_eta, "Eta", -2, i)

# calculate the proper event weights from the normalization weights
N_events_processes = np.array([len(i) for i in weights])
ml_proc_weights = np.max(N_events_processes) / N_events_processes
weight_scalar = np.min(N_events_processes / ml_proc_weights)
sum_eventweights_proc = np.array([np.sum(i) for i in weights])
sample_weights = ak.Array(weights)
sample_weights = sample_weights * weight_scalar / sum_eventweights_proc
sample_weights = sample_weights * ml_proc_weights
sample_weights = ak.to_numpy(ak.flatten(sample_weights))

collection_pred_target = np.concatenate(list(collection_dict.values()), axis=0)
predictions = collection_pred_target[:, :len(processes)]
targets = collection_pred_target[:, len(processes):]

label_sorting = np.argsort(list(target_dict.values()))
sorted_label_keys = np.array(list(target_dict.keys()))[label_sorting]
sorted_labels = [label_dict[label_key] for label_key in sorted_label_keys]
inputs = {
    'prediction': predictions,
    'target': targets,
    'weights': sample_weights,
    }

# start plotting
plot_confusion(inputs, sorted_labels, path, "test set")
plot_confusion2(inputs, sorted_labels, path, "test set")
plot_roc_ovr(inputs, sorted_labels, path, "test set")
plot_roc_ovr2(inputs, sorted_labels, path, "test set")
plot_output_nodes(inputs, sorted_label_keys, sorted_labels, path, "test set")
plot_significance(inputs, sorted_label_keys, sorted_labels, path, "test set")

