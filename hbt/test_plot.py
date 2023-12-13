# encoding: utf-8

from columnflow.tasks.ml import MLEvaluation, MLEvaluationWrapper
import numpy as np
from plotting_funcs import (plot_confusion, plot_roc_ovr, plot_output_nodes,
                            plot_significance, check_distribution, event_weights,
                            calculate_confusion, plot_confusion_std, calculate_auc)
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
fold_pred = {}

# get the target and predictions from MLEvaluate
for num, proc in enumerate(processes):
    t = MLEvaluation(
        version="PairsCustomVBFMaskJets2",
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
        if 'pred_fold' in column:
            key_name = f'pred_fold_{column.split("fold_")[1][0]}'
            mask = np.where(scores[column][:, :len(processes)] == -1., False, True)
            mask = np.transpose(np.tile(mask[:, 0], (np.array(scores[column]).shape[1], 1)))
            inp = np.array(scores[column][mask]).reshape(-1, np.array(scores[column]).shape[1])
            if key_name in fold_pred.keys():
                content = fold_pred[key_name]
                concat = np.concatenate((content, inp), axis=0)
                fold_pred[key_name] = concat
            else:
                fold_pred[key_name] = inp

    prev_model = model_check

# create the savepath for the plots
model_name = model_check
path = files.collection[0]["mlcolumns"].sibling("dummy.json", type="f").path.split("dummy")[0]
path = os.path.join(path, model_name)
if not os.path.exists(path):
    os.makedirs(path)

for arr in fold_pred.values():
    weights = arr[:, -1]
    targets = arr[:, (-len(processes) - 1):-1]
    sample_weights = event_weights(targets, weights)
    predictions = arr[:, :len(processes)]


# collection_pred_target = np.concatenate(list(collection_dict.values()), axis=0)
# predictions = collection_pred_target[:, :len(processes)]
# targets = collection_pred_target[:, len(processes):]

label_sorting = np.argsort(list(target_dict.values()))
sorted_label_keys = np.array(list(target_dict.keys()))[label_sorting]
sorted_labels = [label_dict[label_key] for label_key in sorted_label_keys]
# inputs = {
#     'prediction': predictions,
#     'target': targets,
#     'weights': sample_weights,
#     }

fold_confusion = []
fold_auc = []
for arr in fold_pred.values():
    inputs_fold = {}
    inputs_fold['prediction'] = arr[:, :len(processes)]
    inputs_fold['target'] = arr[:, len(processes): -1]
    inputs_fold['weights'] = arr[:, -1]
    fold_matrix = calculate_confusion(inputs_fold)
    fold_scores = calculate_auc(inputs_fold)
    fold_confusion.append(fold_matrix)
    fold_auc.append(fold_scores)
fold_confusion = np.array(fold_confusion)
fold_confusion = np.expand_dims(fold_confusion, axis=1)
std_confusion = np.std(fold_confusion, axis=0)
std_auc = np.std(np.array(fold_auc), axis=0)

# aggregate all folds
stacked_inp = np.vstack(list(fold_pred.values()))
inputs = {
    'prediction': stacked_inp[:, :len(processes)],
    'target': stacked_inp[:, len(processes): -1],
    'weights': stacked_inp[:, -1],
}

confusion = calculate_confusion(inputs)
# start plotting
plot_confusion_std(confusion, std_confusion, sorted_labels, path)
plot_confusion(inputs, sorted_labels, path, "test set")
plot_roc_ovr(inputs, sorted_labels, path, "test set", std_auc)
plot_output_nodes(inputs, sorted_label_keys, sorted_labels, path, "test set")
plot_significance(inputs, sorted_label_keys, sorted_labels, path, "test set")
