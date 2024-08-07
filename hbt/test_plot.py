# encoding: utf-8

from columnflow.tasks.ml import MLEvaluation, MLEvaluationWrapper
from hbt.tasks.ml import MergeMLEvaluationPerFold, MLEvaluationPerFold
from columnflow.columnar_util import EMPTY_FLOAT
import numpy as np
from plotting_funcs import (plot_confusion, plot_roc_ovr, plot_output_nodes,
                            plot_significance, check_distribution, event_weights,
                            calculate_confusion, plot_confusion_std, calculate_auc,
                            norm_weights, plot_shap, correlation_matrix_events)
import os
import awkward as ak

from vars_dict import correlations_dict, latex_dict

model_parse = "4classes_DeepSets_setup2"
jet_threshold = 4

processes_dict = {"graviton_hh_vbf_bbtautau_m400": "graviton_hh_vbf_bbtautau_m400_madgraph",
            "graviton_hh_ggf_bbtautau_m400": "graviton_hh_ggf_bbtautau_m400_madgraph",
            "tt_sl": "tt_sl_powheg",
            "tt_dl": "tt_dl_powheg",
            "dy_lep_pt50To100": "dy_lep_pt50To100_amcatnlo",
            "dy_lep_pt100To250": "dy_lep_pt100To250_amcatnlo",
            "dy_lep_pt250To400": "dy_lep_pt250To400_amcatnlo",
            "dy_lep_pt400To650": "dy_lep_pt400To650_amcatnlo",
            "dy_lep_pt650": "dy_lep_pt650_amcatnlo"}

dataset_to_proc = {"graviton_hh_vbf_bbtautau_m400_madgraph": "graviton_hh_vbf_bbtautau_m400",
            "graviton_hh_ggf_bbtautau_m400_madgraph": "graviton_hh_ggf_bbtautau_m400",
            "tt_sl_powheg": "tt",
            "tt_dl_powheg": "tt",
            "dy_lep_pt50To100_amcatnlo": "dy",
            "dy_lep_pt100To250_amcatnlo": "dy",
            "dy_lep_pt250To400_amcatnlo": "dy",
            "dy_lep_pt400To650_amcatnlo": "dy",
            "dy_lep_pt650_amcatnlo": "dy"}

processes = ["graviton_hh_vbf_bbtautau_m400", "graviton_hh_ggf_bbtautau_m400", "tt", "dy"]

label_dict = {"graviton_hh_vbf_bbtautau_m400": '$HH_{vbf,m400}$ $\\rightarrow bb\\tau\\tau$',
              "graviton_hh_ggf_bbtautau_m400": '$HH_{ggf,m400}$ $\\rightarrow bb\\tau\\tau$',
              "tt": '$t\\bar{t}$ + Jets',
              "tt_sl": '$t\\bar{t}$ + SL, Jets',
              "tt_dl": '$t\\bar{t}$ + DL, Jets',
              "dy": 'Drell-Yan'}

npz_path = f"/nfs/dust/cms/user/schaefes/models_shap_corrcoef/{model_parse}"

collection_dict = {}
target_dict = {}
weights = []
DeepSetsInpPt = {}
DeepSetsInpEta = {}
fold_pred = {}
n_jets = {}

# get the target and predictions from MLEvaluate
for num, proc in enumerate(processes_dict.keys()):
    t = MergeMLEvaluationPerFold(
        version="limits_xsecs",
        ml_model=model_parse,
        dataset=processes_dict[proc],
        calibrators=("default",),
        # print_status=(3,),
    )
    files = t.output()
    data = files.collection[0]["mlcolumns"].load(formatter="awkward")
    # get part of the model name from the column name
    scores = getattr(data, model_parse)
    score = {}
    model_check = scores.fields[0].split("__")[-1]
    c = 0
    if num != 0:
        if prev_model != model_check:
            raise Exception(f"Different models mixed up for process {proc}")
    for column in scores.fields:
        mask_empty_float = ak.to_numpy((getattr(scores, column) != EMPTY_FLOAT))
        mask_empty_float = mask_empty_float[:, 0] if len(mask_empty_float.shape) > 1 else mask_empty_float
        proc = "tt" if "tt" in proc and "tt" in processes else proc
        proc = "dy" if "dy" in proc and "dy" in processes else proc
        n_jets = scores[f'events_n_jets_{proc}__{model_parse}']
        mask_n_jets = (n_jets >= jet_threshold)
        mask_full = np.logical_and(mask_empty_float, mask_n_jets)
        score[f'{column}'] = getattr(scores, column)[mask_full]
        if model_check != column.split("__")[-1]:
            raise Exception(f"Different models mixed up for process {proc}")
        if 'target_label' in column:
            target_dict[f'{proc}'] = score[f'{column}'][0]
        if 'pred_target' in column:
            collection_dict[f'{column}'] = score[f'{column}']
        if 'weights' in column:
            weights.append(score[f'{column}'])
        if 'DeepSetsInpPt' in column:
            DeepSetsInpPt[f'{column}'] = score[f'{column}']
        if 'DeepSetsInpEta' in column:
            DeepSetsInpEta[f'{column}'] = score[f'{column}']
        if 'pred_fold' in column:
            c += 1
            key_name = f'pred_fold_{column.split("fold_")[1][0]}'
            mask = np.where(score[column][:, :len(processes)] == -1., False, True)
            mask = np.transpose(np.tile(mask[:, 0], (np.array(score[column]).shape[1], 1)))
            inp = np.array(score[column][mask]).reshape(-1, np.array(score[column]).shape[1])
            if key_name in fold_pred.keys():
                content = fold_pred[key_name]
                concat = np.concatenate((content, inp), axis=0)
                fold_pred[key_name] = concat
            else:
                fold_pred[key_name] = inp
    if c != 10:
        raise Exception(f'Not all folds collected for dataset {proc}')

    prev_model = model_check

# create the savepath for the plots
model_name = model_check
path = files.collection[0]["mlcolumns"].sibling("dummy.json", type="f").path.split("dummy")[0]
path = os.path.join(path, model_name)
print(path)
if not os.path.exists(path):
    os.makedirs(path)

glob_folds = np.concatenate(list(fold_pred.values()), axis=0)
glob_weights = glob_folds[:, -2]
glob_preds = glob_folds[:, :len(processes)]
glob_targets = glob_folds[:, len(processes): 2 * len(processes)]
normalized_glob_weights = norm_weights(glob_targets, glob_weights)

label_sorting = np.argsort(list(target_dict.values()))
sorted_label_keys = np.array(list(target_dict.keys()))[label_sorting]
sorted_labels = [label_dict[label_key] for label_key in sorted_label_keys]

fold_confusion = []
fold_auc = []
for i, arr in enumerate(fold_pred.values()):
    targets = arr[:, len(processes): 2 * len(processes)]
    weights = arr[:, -2]
    inputs_fold = {}
    inputs_fold['prediction'] = arr[:, :len(processes)]
    inputs_fold['target'] = targets
    inputs_fold['weights'] = norm_weights(targets, weights)
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
    'prediction': glob_preds,
    'target': glob_targets,
    'weights': normalized_glob_weights,
}

confusion = calculate_confusion(inputs)

plot_confusion_std(confusion, std_confusion, sorted_labels, path, jet_threshold)
plot_confusion(inputs, sorted_labels, path, "test set", jet_threshold)
print('Jets:', jet_threshold)
plot_roc_ovr(inputs, sorted_labels, path, "test set", std_auc, jet_threshold)
plot_output_nodes(inputs, sorted_label_keys, sorted_labels, path, "test set", jet_threshold)
plot_significance(inputs, sorted_label_keys, sorted_labels, path, "test set", jet_threshold)
