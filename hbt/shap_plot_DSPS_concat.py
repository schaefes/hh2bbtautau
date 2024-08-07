# encoding: utf-8

from columnflow.tasks.ml import MLEvaluation, MLEvaluationWrapper
from hbt.tasks.ml import MergeMLEvaluationPerFold
from columnflow.columnar_util import EMPTY_FLOAT
import numpy as np
from plotting_funcs import (plot_confusion, plot_roc_ovr, plot_output_nodes,
                            plot_significance, check_distribution, event_weights,
                            calculate_confusion, plot_confusion_std, calculate_auc,
                            norm_weights, plot_shap, correlation_matrix_events)
import os
import awkward as ak

from vars_dict import correlations_dict, latex_dict

model_parse = "4classes_DeepSetsPS_concat_setup2"
jet_threshold = 0

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


# start plotting
labels_directory_list = correlations_dict[model_parse]
mean_idx = correlations_dict[model_parse][0][3]
# if 'baseline' not in model_parse:
#     for l in labels_directory_list:
#         correlation_matrix_events(npz_path, processes_dict.values(), l, dataset_to_proc)

plot_shap(npz_path, processes_dict.values(), mean_idx, latex_dict, model_parse)
