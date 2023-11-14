# coding: utf-8

"""
ML models derived from the *SimpleDNN* class
"""


from hbt.ml.first_nn import SimpleDNN


processes = [
    "graviton_hh_ggf_bbtautau_m400",
    "hh_ggf_bbtautau",
    "graviton_hh_vbf_bbtautau_m400",
    # "graviton_hh_ggf_bbtautau_m1250",
]

ml_process_weights = {
    "graviton_hh_ggf_bbtautau_m400": 1.5,
    "hh_ggf_bbtautau": 1,
    "graviton_hh_vbf_bbtautau_m400": 1,
    # "graviton_hh_ggf_bbtautau_m1250": 1,
}

dataset_names = {
    "graviton_hh_ggf_bbtautau_m400_madgraph",
    "hh_ggf_bbtautau_madgraph",
    "graviton_hh_vbf_bbtautau_m400_madgraph",
    # "graviton_hh_ggf_bbtautau_m1250_madgraph",
}

# feature_list = ["pt", "eta", "phi", "mass", "e", "btag", "hadronFlavour"]

# input_features = [
#     [f"{obj}_{var}"
#     for obj in [f"jet{i}" for i in range(1, 7, 1)]
#     for var in feature_list],
#     ["mjj", "mbjetbjet", "mtautau", "mHH"]]
jet_collection = "CustomVBFMaskJets2"
kinematic_inp = ["e", "mass", "pt", "eta", "phi", "btag", "btagCvL", "btagCvB",
                 "bFlavtag", "bFlavtagCvL", "bFlavtagCvB", "btagQG"]
deepSets_inp = [f"{jet_collection}_{kin_var}" for kin_var in kinematic_inp]
event_features = ["mbjetbjet", "mtautau", "mHH", f"{jet_collection}_ht", f"{jet_collection}_mjj",
                  f"{jet_collection}_mjj_dEta", f"{jet_collection}_max_dEta", f"{jet_collection}_njets"]
projection_phi = [f"{jet_collection}_METphi"]
four_vector = [f"{jet_collection}_{k}" for k in ["e", "px", "py", "pz", "phi", "eta"]]

input_features = [deepSets_inp, event_features]
no_norm_features = [f"{jet_collection}_ones", f"{jet_collection}_btag", f"{jet_collection}_btagCvL",
                    f"{jet_collection}_btagQG", f"{jet_collection}_bFlavtag",
                    f"{jet_collection}_bFlavtagCvL", f"{jet_collection}_bFlavtagCvB",
                    f"{jet_collection}_btagCvB"]

# mask a copy of deepSets and events_features inp to make sure that input features remain unchanged
# when removing list elements that are not supposed to be normalized
deepSets_inp_copy = deepSets_inp.copy()
event_features_copy = event_features.copy()
for no_norm in no_norm_features:
    try:
        deepSets_inp_copy.remove(no_norm)
        event_features_copy.remove(no_norm)
    except:
        continue

norm_features = [deepSets_inp_copy, event_features_copy]

# Decide on dummy or proper btag of jets: If proper chosen coment out 4 lines below
# for i, name in enumerate(input_features[0]):
#     if name == 'jet1_btag' or name == 'jet2_btag':
#         name += "_dummy"
#         input_features[0][i] = name


# empty_overwrite options: "1": -1, "3sig": 3 sigma padding as replacement of EMPTY FLOAT values
# baseline_jets: max number of jets considered
# model_type: baseline-> baselie model used else deepsets
# quantity_weighting: weighting of the processes accroding to the amount of samples for each samples
default_cls_dict = {
    "folds": 10,
    # "max_events": 10**6,  # TODO
    "layers": [512, 512, 512],
    "activation": "relu",  # Options: elu, relu, prelu, selu, tanh, softmax
    "learningrate": 0.01,
    "batchsize": 256,
    "epochs": 1,
    "eqweight": True,
    "dropout": 0.50,
    "processes": processes,
    "ml_process_weights": ml_process_weights,
    "dataset_names": dataset_names,
    "input_features": input_features,
    "store_name": "inputs1",
    "n_features": len(input_features[0]),
    "n_output_nodes": len(processes),
    "batch_norm_deepSets": True,
    "batch_norm_ff": True,
    "aggregations": ["Sum", "Max", "Mean"],
    "L2": False,
    "norm_features": norm_features,
    "empty_overwrite": "1",
    "quantity_weighting": True,
    "jet_num_cut": 1,
    "baseline_jets": 4,
    "jet_collection": jet_collection,
    "masking_val": -2222,
    "projection_phi": projection_phi,
    "resorting_feature": f"{jet_collection}_bFlavtag",
    "train_sorting": f"{jet_collection}_pt",
    "pair_vectors": four_vector,
}

nodes_deepSets_op = default_cls_dict["baseline_jets"] * default_cls_dict["n_features"]
nodes_deepSets = [80, 60, 60, nodes_deepSets_op]
nodes_ff = [256, 256, 256, 256, 256, 256]

# derived model, usable on command line
default_dnn = SimpleDNN.derive("default", cls_dict=default_cls_dict)

# test model settings
model_type = "DeepSets" # choose str "baseline" for baseline nn or "DeepSets"
cls_dict = default_cls_dict
cls_dict["model_type"] = model_type
cls_dict["model_name"] = f"{len(processes)}classes_{model_type}_masking_test"
cls_dict["nodes_deepSets"] = nodes_deepSets
cls_dict["nodes_ff"] = nodes_ff
cls_dict["activation_func_deepSets"] = ["relu" for i in range(len(nodes_deepSets))]
cls_dict["activation_func_ff"] = ["relu" for i in range(len(nodes_ff))]

test_dnn = SimpleDNN.derive("test", cls_dict=cls_dict)

# 2classes_vbfjets_dr_inv_mass
# 2classes_vbfjets
# 3classes_vbfjets
# 3classes_vbfjets_dr_inv_mass
# 3classes_no_vbf
