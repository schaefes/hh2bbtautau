# coding: utf-8

"""
ML models derived from the *SimpleDNN* class
"""


from hbt.ml.first_nn_norm_layer_ensamble import SimpleDNN
from columnflow.util import maybe_import

math = maybe_import("math")
it = maybe_import("itertools")
np = maybe_import("numpy")


processes = [
    "graviton_hh_ggf_bbtautau_m400",
    # "hh_ggf_bbtautau",
    "graviton_hh_vbf_bbtautau_m400",
    "tt",
    # "tt_dl",s
    # "tt_sl",
    "dy",
    # "dy_lep_pt50To100",
    # "dy_lep_pt100To250",
    # "dy_lep_pt250To400",
    # "dy_lep_pt400To650",
    # "dy_lep_pt650",
]

ml_process_weights = {
    "graviton_hh_ggf_bbtautau_m400": 1,
    # "hh_ggf_bbtautau": 1,
    "graviton_hh_vbf_bbtautau_m400": 1,
    "tt": 1,
    # "tt_dl": 0,
    # "tt_sl": 0,
    "dy": 1,
}

dataset_names = {
    "graviton_hh_ggf_bbtautau_m400_madgraph",
    # "hh_ggf_bbtautau_madgraph",
    "graviton_hh_vbf_bbtautau_m400_madgraph",
    "tt_dl_powheg",
    "tt_sl_powheg",
    # "tt_fh_powheg",
    "dy_lep_pt50To100_amcatnlo",
    "dy_lep_pt100To250_amcatnlo",
    "dy_lep_pt250To400_amcatnlo",
    "dy_lep_pt400To650_amcatnlo",
    "dy_lep_pt650_amcatnlo",
}

proc_label_dict = {"graviton_hh_vbf_bbtautau_m400": '$HH_{vbf,m400}$ $\\rightarrow bb\\tau\\tau$',
              "graviton_hh_ggf_bbtautau_m400": '$HH_{ggf,m400}$ $\\rightarrow bb\\tau\\tau$',
              "tt": '$t\\bar{t}$ + Jets',
              "tt_sl": '$t\\bar{t}$ + SL, Jets',
              "tt_dl": '$t\\bar{t}$ + DL, Jets',
              "dy": 'Drell-Yan'}

jet_collection = "CustomVBFMaskJets2"
kinematic_inp = ["e", "mass", "pt", "eta", "phi", "jet_pt_frac", "bFlavtag", "bFlavtagCvL",
                 "bFlavtagCvB", "btagQG"]
deepSets_inp = [f"{jet_collection}_{kin_var}" for kin_var in kinematic_inp]

event_level_inp = ["mbb", "mHH", "mtautau", "mjj", "ht", "mjj_dEta", "max_dEta",
                   "thrust", "pt_thrust", "energy_corr_sqr", "sphericity",
                   "njets"]
event_features = [f"{jet_collection}_{feature}" for feature in event_level_inp]
projection_phi = [f"{jet_collection}_METphi"]

baseline_padding = [0., 0., 0., -5., -4., 0., -1., -1., -1., -1.]

# DO NOT  SWITCH THE ORDER IN four_vector, ONLY APPEND NEW AT THE END. THE COULUMS ARE EXPLICETELY
# ADRESSED BY INDEX IN THE create_pairs FUNCTION
four_vector = [f"{jet_collection}_{k}" for k in ["e", "px", "py", "pz", "phi", "eta",
               "bFlavtag", "bFlavtagCvL", "bFlavtagCvB", "btagQG"]]
four_vector_padding = [0., 0., 0., 0., -4., -5., -1., -1., -1., -1.]

input_features = [deepSets_inp, event_features]
no_norm_features = [f"{jet_collection}_ones", f"{jet_collection}_btag", f"{jet_collection}_btagCvL",
                    f"{jet_collection}_btagQG", f"{jet_collection}_bFlavtag",
                    f"{jet_collection}_bFlavtagCvL", f"{jet_collection}_bFlavtagCvB",
                    f"{jet_collection}_btagCvB"]

latex_dict = {
    f"{jet_collection}_e": "Energy",
    f"{jet_collection}_mass": "Mass",
    f"{jet_collection}_pt": r"$p_{T}$",
    f"{jet_collection}_pt_frac": r"$p_{T}$ Fraction",
    f"{jet_collection}_eta": r"$\eta$",
    f"{jet_collection}_phi": r"$\phi$",
    f"{jet_collection}_btag": "Deep B",
    f"{jet_collection}_bFlavtag": "Deep Flav B",
    f"{jet_collection}_btagCvL": "Deep CvL",
    f"{jet_collection}_bFlavtagCvL": "Deep Flav CvL",
    f"{jet_collection}_btagCvB": "Deep CvB",
    f"{jet_collection}_bFlavtagCvB": "Deep Flav CvB",
    f"{jet_collection}_btagQG": "Deep Flav QG",
    f"{jet_collection}_mbb": r"$m_{\bar{b}b}$",
    f"{jet_collection}_mHH": r"$m_{HH}$",
    f"{jet_collection}_mtautau": r"$m_{\tau^{+} \tau^{-}}$",
    f"{jet_collection}_mjj": r"$m_{JJ}$",
    f"{jet_collection}_ht": "HT",
    f"{jet_collection}_max_dEta": r"max $\Delta \eta$",
    f"{jet_collection}_mjj_dEta": r"$m_{JJ}$ to max $\Delta \eta$",
    f"{jet_collection}_njets": "N Jets",
    f"{jet_collection}_sphericity": r"$S_{T}$",
    f"{jet_collection}_thrust": "Thrust",
    f"{jet_collection}_pt_thrust": "Transversal Thrust",
    f"{jet_collection}_energy_corr_sqr": r"ECF ($\beta$=2)",
    f"{jet_collection}_energy_corr": r"ECF ($\beta$=1)",
    f"{jet_collection}_energy_corr_root": r"ECF ($\beta$=0.5)",
}

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

# Create dict for the pair idx
pairs_dict_pad = {}
for i in range(2, 11, 1):
    padded_idx = np.full([45, 2], -1)
    idx = list(it.combinations(np.arange(0, i, 1), r=2))
    idx = np.array(idx)
    padded_idx[:len(idx), :] = idx
    pairs_dict_pad[i] = padded_idx

# Decide on dummy or proper btag of jets: If proper chosen coment out 4 lines below
# for i, name in enumerate(input_features[0]):
#     if name == 'jet1_btag' or name == 'jet2_btag':
#         name += "_dummy"
#         input_features[0][i] = name


# empty_overwrite options: "1": -1, "3sig": 3 sigma padding as replacement of EMPTY FLOAT values
# baseline_jets: max number of jets considered
# model_type: baseline-> baselie model used else deepsets
# quantity_weighting: weighting of the processes accroding to the amount of samples for each samples
baseline_jets = 4
baseline_pairs = int(math.factorial(baseline_jets) / (2 * math.factorial(baseline_jets - 2)))
default_cls_dict = {
    "folds": 10,
    # "max_events": 10**6,  # TODO
    "layers": [512, 512, 512],
    "activation": "relu",  # Options: elu, relu, prelu, selu, tanh, softmax
    "learningrate": 0.01,
    "batchsize": 256,
    "epochs": 150,
    "eqweight": True,
    "dropout": 0.50,
    "l2": 1e-6,
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
    "aggregations_pairs": ["Sum", "Max", "Mean"],
    "norm_features": norm_features,
    "quantity_weighting": True,
    "jet_num_lower": 1, #incl
    "jet_num_upper": 20, #excl
    "baseline_jets": baseline_jets,
    "baseline_pairs": baseline_pairs,
    "jet_collection": jet_collection,
    "masking_val": -5,
    "projection_phi": projection_phi,
    "resorting_feature": f"{jet_collection}_bFlavtag",
    "train_sorting": f"{jet_collection}_pt",
    "pair_vectors": four_vector,
    "pairs_dict_pad": pairs_dict_pad,
    "latex_dict": latex_dict,
    "event_to_jet": False,
    "baseline_padding": baseline_padding,
    "pairs_padding": four_vector_padding,
    "random_seeds": [1201, 1337, 1598, 1730, 1922],
    "proc_label_dict": proc_label_dict,
}


# test model settings
# choose str "baseline", "baseline_pairs", "DeepSets", "DeepSetsPP", "DeepSetsPS"
# chose what kind of sequential DS mode is used: "sum", "concat", "two_inp"
model_type = "DeepSets"
sequential_mode = "sum"

# NN architecture
nodes_deepSets_op = 15
nodes_deepSets = [80, 80, 80, 80, 60, nodes_deepSets_op]
nodes_ff_ds = [256, 256, 256, 256, 256, 256]
nodes_baseline = [128, 256, 256, 256, 256, 256, 256]
model_name = f"{len(processes)}classes_{model_type}"
# used to add short description in the model name
model_name_desc = "setup1"

cls_dict = default_cls_dict
cls_dict["model_type"] = model_type
cls_dict["sequential_mode"] = sequential_mode
cls_dict["nodes_deepSets"] = nodes_deepSets
cls_dict["nodes_deepSets_pairs"] = nodes_deepSets
cls_dict["nodes_ff_ds"] = nodes_ff_ds
cls_dict["nodes_baseline"] = nodes_baseline
cls_dict["activation_func_deepSets"] = ["selu" for i in range(len(nodes_deepSets))]
cls_dict["activation_func_ff_ds"] = ["selu" for i in range(len(nodes_ff_ds))]
cls_dict["activation_func_baseline"] = ["selu" for i in range(len(nodes_baseline))]

if model_type == "baseline":
    model_name = "_".join([model_name, model_name_desc]) if model_name_desc else model_name
    cls_dict["model_name"] = model_name
    baseline_4classes = SimpleDNN.derive(model_name, cls_dict=cls_dict)
elif model_type == "baseline_pairs":
    model_name = "_".join([model_name, model_name_desc]) if model_name_desc else model_name
    cls_dict["model_name"] = model_name
    baseline_pairs_4classes = SimpleDNN.derive(model_name, cls_dict=cls_dict)
elif model_type == "DeepSets":
    model_name = "_".join([model_name, model_name_desc]) if model_name_desc else model_name
    cls_dict["model_name"] = model_name
    deepsets_4classes = SimpleDNN.derive(model_name, cls_dict=cls_dict)
elif model_type == "DeepSetsPP":
    model_name = "_".join([model_name, model_name_desc]) if model_name_desc else model_name
    cls_dict["model_name"] = model_name
    DeepSetsPP_4classes = SimpleDNN.derive(model_name, cls_dict=cls_dict)
elif model_type == "DeepSetsPS" and sequential_mode == "sum":
    model_ps = "_".join([model_name, sequential_mode])
    model_name = "_".join([model_ps, model_name_desc]) if model_name_desc else model_ps
    cls_dict["model_name"] = model_name
    deepsetsPS_sum_4classes = SimpleDNN.derive(model_name, cls_dict=cls_dict)
elif model_type == "DeepSetsPS" and sequential_mode == "concat":
    model_ps = "_".join([model_name, sequential_mode])
    model_name = "_".join([model_ps, model_name_desc]) if model_name_desc else model_ps
    cls_dict["model_name"] = model_name
    deepsetsPS_concat_4classes = SimpleDNN.derive(model_name, cls_dict=cls_dict)
elif model_type == "DeepSetsPS" and sequential_mode == "two_inp":
    model_ps = "_".join([model_name, sequential_mode])
    model_name = "_".join([model_ps, model_name_desc]) if model_name_desc else model_ps
    cls_dict["model_name"] = model_name
    deepsetsPS_two_inp_4classes = SimpleDNN.derive(model_name, cls_dict=cls_dict)

# setup 3
model_name_desc = "setup3"
cls_dict["aggregations"] = ["Sum"]
cls_dict["aggregations_pairs"] = ["Sum"]
model_name = f"{len(processes)}classes_{model_type}"
model_name = "_".join([model_name, model_name_desc]) if model_name_desc else model_name
cls_dict["model_name"] = model_name
deepsets_4classes_3 = SimpleDNN.derive(model_name, cls_dict=cls_dict)


# setup 2
model_type = "DeepSets"

# NN architecture
nodes_deepSets_op = 10
nodes_deepSets = [80, 60, 60, nodes_deepSets_op]
nodes_ff_ds = [256, 256, 256, 256]
nodes_baseline = [256, 256, 256, 256]
model_name = f"{len(processes)}classes_{model_type}"
model_name_desc = "setup2"

cls_dict = default_cls_dict
cls_dict["model_type"] = model_type
cls_dict["sequential_mode"] = sequential_mode
cls_dict["nodes_deepSets"] = nodes_deepSets
cls_dict["nodes_deepSets_pairs"] = nodes_deepSets
cls_dict["nodes_ff_ds"] = nodes_ff_ds
cls_dict["nodes_baseline"] = nodes_baseline
cls_dict["activation_func_deepSets"] = ["selu" for i in range(len(nodes_deepSets))]
cls_dict["activation_func_ff_ds"] = ["selu" for i in range(len(nodes_ff_ds))]
cls_dict["activation_func_baseline"] = ["selu" for i in range(len(nodes_baseline))]

model_name = f"{len(processes)}classes_{model_type}"
model_name = "_".join([model_name, model_name_desc]) if model_name_desc else model_name
cls_dict["model_name"] = model_name
deepsets_4classes_2 = SimpleDNN.derive(model_name, cls_dict=cls_dict)

# setup2 relu
cls_dict["activation_func_deepSets"] = ["relu" for i in range(len(nodes_deepSets))]
cls_dict["activation_func_ff_ds"] = ["relu" for i in range(len(nodes_ff_ds))]
cls_dict["activation_func_baseline"] = ["relu" for i in range(len(nodes_baseline))]

model_name_desc = "setup2_relu"
model_name = f"{len(processes)}classes_{model_type}"
model_name = "_".join([model_name, model_name_desc]) if model_name_desc else model_name
cls_dict["model_name"] = model_name
deepsets_4classes_2_relu = SimpleDNN.derive(model_name, cls_dict=cls_dict)

# setup 4
model_name_desc = "setup4"
cls_dict["activation_func_deepSets"] = ["selu" for i in range(len(nodes_deepSets))]
cls_dict["activation_func_ff_ds"] = ["selu" for i in range(len(nodes_ff_ds))]
cls_dict["activation_func_baseline"] = ["selu" for i in range(len(nodes_baseline))]
cls_dict["aggregations"] = ["Sum"]
cls_dict["aggregations_pairs"] = ["Sum"]
model_name = f"{len(processes)}classes_{model_type}"
model_name = "_".join([model_name, model_name_desc]) if model_name_desc else model_name
cls_dict["model_name"] = model_name
deepsets_4classes_4 = SimpleDNN.derive(model_name, cls_dict=cls_dict)

# setup 4 relu
cls_dict["activation_func_deepSets"] = ["relu" for i in range(len(nodes_deepSets))]
cls_dict["activation_func_ff_ds"] = ["relu" for i in range(len(nodes_ff_ds))]
cls_dict["activation_func_baseline"] = ["relu" for i in range(len(nodes_baseline))]

model_name_desc = "setup4_relu"
model_name = f"{len(processes)}classes_{model_type}"
model_name = "_".join([model_name, model_name_desc]) if model_name_desc else model_name
cls_dict["model_name"] = model_name
deepsets_4classes_4_relu = SimpleDNN.derive(model_name, cls_dict=cls_dict)

###############################################################################
# setup 1 baseline
model_type = "baseline"

# NN architecture
nodes_deepSets_op = 10
nodes_deepSets = [80, 60, 60, nodes_deepSets_op]
nodes_ff_ds = [256, 256, 256, 256, 256, 256]
nodes_baseline = [256, 256, 256, 256, 256, 256]
model_name = f"{len(processes)}classes_{model_type}"
model_name_desc = "setup1"

cls_dict = default_cls_dict
cls_dict["model_type"] = model_type
cls_dict["sequential_mode"] = sequential_mode
cls_dict["nodes_deepSets"] = nodes_deepSets
cls_dict["nodes_deepSets_pairs"] = nodes_deepSets
cls_dict["nodes_ff_ds"] = nodes_ff_ds
cls_dict["nodes_baseline"] = nodes_baseline
cls_dict["activation_func_deepSets"] = ["selu" for i in range(len(nodes_deepSets))]
cls_dict["activation_func_ff_ds"] = ["selu" for i in range(len(nodes_ff_ds))]
cls_dict["activation_func_baseline"] = ["selu" for i in range(len(nodes_baseline))]

model_name = f"{len(processes)}classes_{model_type}"
model_name = "_".join([model_name, model_name_desc]) if model_name_desc else model_name
cls_dict["model_name"] = model_name
baseline_4classes_1 = SimpleDNN.derive(model_name, cls_dict=cls_dict)

# setup 2 baseline
model_type = "baseline"

# NN architecture
nodes_deepSets_op = 10
nodes_deepSets = [80, 60, 60, nodes_deepSets_op]
nodes_ff_ds = [256, 256, 256, 256]
nodes_baseline = [256, 256, 256, 256]
model_name = f"{len(processes)}classes_{model_type}"
model_name_desc = "setup2"

cls_dict = default_cls_dict
cls_dict["model_type"] = model_type
cls_dict["sequential_mode"] = sequential_mode
cls_dict["nodes_deepSets"] = nodes_deepSets
cls_dict["nodes_deepSets_pairs"] = nodes_deepSets
cls_dict["nodes_ff_ds"] = nodes_ff_ds
cls_dict["nodes_baseline"] = nodes_baseline
cls_dict["activation_func_deepSets"] = ["selu" for i in range(len(nodes_deepSets))]
cls_dict["activation_func_ff_ds"] = ["selu" for i in range(len(nodes_ff_ds))]
cls_dict["activation_func_baseline"] = ["selu" for i in range(len(nodes_baseline))]

model_name = f"{len(processes)}classes_{model_type}"
model_name = "_".join([model_name, model_name_desc]) if model_name_desc else model_name
cls_dict["model_name"] = model_name
baseline_4classes_2 = SimpleDNN.derive(model_name, cls_dict=cls_dict)

# setup2 6 jets
cls_dict["baseline_jets"] = 6
model_name_desc = "setup2_6jets"
model_name = f"{len(processes)}classes_{model_type}"
model_name = "_".join([model_name, model_name_desc]) if model_name_desc else model_name
cls_dict["model_name"] = model_name
baseline_4classes_2_6jets = SimpleDNN.derive(model_name, cls_dict=cls_dict)

# setup2 relu
cls_dict["baseline_jets"] = 4
cls_dict["activation_func_deepSets"] = ["relu" for i in range(len(nodes_deepSets))]
cls_dict["activation_func_ff_ds"] = ["relu" for i in range(len(nodes_ff_ds))]
cls_dict["activation_func_baseline"] = ["relu" for i in range(len(nodes_baseline))]

model_name_desc = "setup2_relu"
model_name = f"{len(processes)}classes_{model_type}"
model_name = "_".join([model_name, model_name_desc]) if model_name_desc else model_name
cls_dict["model_name"] = model_name
baseline_4classes_2_relu = SimpleDNN.derive(model_name, cls_dict=cls_dict)


############################################################################
# setup 1 baseline pairs
model_type = "baseline_pairs"

# NN architecture
nodes_deepSets_op = 10
nodes_deepSets = [80, 60, 60, nodes_deepSets_op]
nodes_ff_ds = [256, 256, 256, 256, 256, 256]
nodes_baseline = [256, 256, 256, 256, 256, 256]
model_name = f"{len(processes)}classes_{model_type}"
model_name_desc = "setup1"

cls_dict = default_cls_dict
cls_dict["model_type"] = model_type
cls_dict["sequential_mode"] = sequential_mode
cls_dict["nodes_deepSets"] = nodes_deepSets
cls_dict["nodes_deepSets_pairs"] = nodes_deepSets
cls_dict["nodes_ff_ds"] = nodes_ff_ds
cls_dict["nodes_baseline"] = nodes_baseline
cls_dict["activation_func_deepSets"] = ["selu" for i in range(len(nodes_deepSets))]
cls_dict["activation_func_ff_ds"] = ["selu" for i in range(len(nodes_ff_ds))]
cls_dict["activation_func_baseline"] = ["selu" for i in range(len(nodes_baseline))]

model_name = f"{len(processes)}classes_{model_type}"
model_name = "_".join([model_name, model_name_desc]) if model_name_desc else model_name
cls_dict["model_name"] = model_name
baseline_pairs_4classes_1 = SimpleDNN.derive(model_name, cls_dict=cls_dict)

# setup 2
model_type = "baseline_pairs"

# NN architecture
nodes_deepSets_op = 10
nodes_deepSets = [80, 60, 60, nodes_deepSets_op]
nodes_ff_ds = [256, 256, 256, 256]
nodes_baseline = [256, 256, 256, 256]
model_name = f"{len(processes)}classes_{model_type}"
model_name_desc = "setup2"

cls_dict = default_cls_dict
cls_dict["model_type"] = model_type
cls_dict["sequential_mode"] = sequential_mode
cls_dict["nodes_deepSets"] = nodes_deepSets
cls_dict["nodes_deepSets_pairs"] = nodes_deepSets
cls_dict["nodes_ff_ds"] = nodes_ff_ds
cls_dict["nodes_baseline"] = nodes_baseline
cls_dict["activation_func_deepSets"] = ["selu" for i in range(len(nodes_deepSets))]
cls_dict["activation_func_ff_ds"] = ["selu" for i in range(len(nodes_ff_ds))]
cls_dict["activation_func_baseline"] = ["selu" for i in range(len(nodes_baseline))]

model_name = f"{len(processes)}classes_{model_type}"
model_name = "_".join([model_name, model_name_desc]) if model_name_desc else model_name
cls_dict["model_name"] = model_name
baseline_pairs_4classes_2 = SimpleDNN.derive(model_name, cls_dict=cls_dict)

# setup2 3jets
cls_dict["baseline_jets"] = 3
cls_dict["baseline_pairs"] = int(math.factorial(3) / (2 * math.factorial(3 - 2)))
model_name_desc = "setup2_3jets"
model_name = f"{len(processes)}classes_{model_type}"
model_name = "_".join([model_name, model_name_desc]) if model_name_desc else model_name
cls_dict["model_name"] = model_name
baseline_pairs_4classes_2_3jets = SimpleDNN.derive(model_name, cls_dict=cls_dict)

# setup2 relu
cls_dict["baseline_jets"] = 4
cls_dict["baseline_pairs"] = int(math.factorial(4) / (2 * math.factorial(4 - 2)))
cls_dict["activation_func_deepSets"] = ["relu" for i in range(len(nodes_deepSets))]
cls_dict["activation_func_ff_ds"] = ["relu" for i in range(len(nodes_ff_ds))]
cls_dict["activation_func_baseline"] = ["relu" for i in range(len(nodes_baseline))]

model_name_desc = "setup2_relu"
model_name = f"{len(processes)}classes_{model_type}"
model_name = "_".join([model_name, model_name_desc]) if model_name_desc else model_name
cls_dict["model_name"] = model_name
baseline_pairs_4classes_2_relu = SimpleDNN.derive(model_name, cls_dict=cls_dict)

model_name_desc = "test"
model_name = f"{len(processes)}classes_{model_type}"
model_name = "_".join([model_name, model_name_desc]) if model_name_desc else model_name
cls_dict["model_name"] = model_name
baseline_pairs_4classes_test = SimpleDNN.derive(model_name, cls_dict=cls_dict)


###############################################################################
# DSPP setup 1
model_type = "DeepSetsPP"

# NN architecture
nodes_deepSets_op = 10
nodes_deepSets = [80, 60, 60, nodes_deepSets_op]
nodes_ff_ds = [256, 256, 256, 256]
nodes_baseline = [256, 256, 256, 256]
model_name = f"{len(processes)}classes_{model_type}"
model_name_desc = "setup1"

cls_dict = default_cls_dict
cls_dict["model_type"] = model_type
cls_dict["sequential_mode"] = sequential_mode
cls_dict["nodes_deepSets"] = nodes_deepSets
cls_dict["nodes_deepSets_pairs"] = nodes_deepSets
cls_dict["nodes_ff_ds"] = nodes_ff_ds
cls_dict["nodes_baseline"] = nodes_baseline
cls_dict["activation_func_deepSets"] = ["selu" for i in range(len(nodes_deepSets))]
cls_dict["activation_func_ff_ds"] = ["selu" for i in range(len(nodes_ff_ds))]
cls_dict["activation_func_baseline"] = ["selu" for i in range(len(nodes_baseline))]

model_name = f"{len(processes)}classes_{model_type}"
model_name = "_".join([model_name, model_name_desc]) if model_name_desc else model_name
cls_dict["model_name"] = model_name
deepsetsPP_4classes_1 = SimpleDNN.derive(model_name, cls_dict=cls_dict)

# setup 3
model_name_desc = "setup3"
cls_dict["aggregations"] = ["Sum"]
cls_dict["aggregations_pairs"] = ["Sum"]
model_name = f"{len(processes)}classes_{model_type}"
model_name = "_".join([model_name, model_name_desc]) if model_name_desc else model_name
cls_dict["model_name"] = model_name
deepsetsPP_4classes_3 = SimpleDNN.derive(model_name, cls_dict=cls_dict)


# DSPP setup 2
model_type = "DeepSetsPP"

# NN architecture
nodes_deepSets_op = 15
nodes_deepSets = [80, 80, 80, 80, 60, nodes_deepSets_op]
nodes_ff_ds = [256, 256, 256, 256, 256, 256]
nodes_baseline = [256, 256, 256, 256, 256, 256]
model_name = f"{len(processes)}classes_{model_type}"
model_name_desc = "setup2"

cls_dict = default_cls_dict
cls_dict["model_type"] = model_type
cls_dict["sequential_mode"] = sequential_mode
cls_dict["nodes_deepSets"] = nodes_deepSets
cls_dict["nodes_deepSets_pairs"] = nodes_deepSets
cls_dict["nodes_ff_ds"] = nodes_ff_ds
cls_dict["nodes_baseline"] = nodes_baseline
cls_dict["activation_func_deepSets"] = ["selu" for i in range(len(nodes_deepSets))]
cls_dict["activation_func_ff_ds"] = ["selu" for i in range(len(nodes_ff_ds))]
cls_dict["activation_func_baseline"] = ["selu" for i in range(len(nodes_baseline))]

model_name = f"{len(processes)}classes_{model_type}"
model_name = "_".join([model_name, model_name_desc]) if model_name_desc else model_name
cls_dict["model_name"] = model_name
deepsetsPP_4classes_2 = SimpleDNN.derive(model_name, cls_dict=cls_dict)

# setup2 relu
cls_dict["activation_func_deepSets"] = ["relu" for i in range(len(nodes_deepSets))]
cls_dict["activation_func_ff_ds"] = ["relu" for i in range(len(nodes_ff_ds))]
cls_dict["activation_func_baseline"] = ["relu" for i in range(len(nodes_baseline))]

model_name_desc = "setup2_relu"
model_name = f"{len(processes)}classes_{model_type}"
model_name = "_".join([model_name, model_name_desc]) if model_name_desc else model_name
cls_dict["model_name"] = model_name
deepsetsPP_4classes_2_relu = SimpleDNN.derive(model_name, cls_dict=cls_dict)

# setup 4
model_name_desc = "setup4"
cls_dict["activation_func_deepSets"] = ["selu" for i in range(len(nodes_deepSets))]
cls_dict["activation_func_ff_ds"] = ["selu" for i in range(len(nodes_ff_ds))]
cls_dict["activation_func_baseline"] = ["selu" for i in range(len(nodes_baseline))]
cls_dict["aggregations"] = ["Sum"]
cls_dict["aggregations_pairs"] = ["Sum"]
model_name = f"{len(processes)}classes_{model_type}"
model_name = "_".join([model_name, model_name_desc]) if model_name_desc else model_name
cls_dict["model_name"] = model_name
deepsetsPP_4classes_4 = SimpleDNN.derive(model_name, cls_dict=cls_dict)

# setup 4 relu
cls_dict["activation_func_deepSets"] = ["relu" for i in range(len(nodes_deepSets))]
cls_dict["activation_func_ff_ds"] = ["relu" for i in range(len(nodes_ff_ds))]
cls_dict["activation_func_baseline"] = ["relu" for i in range(len(nodes_baseline))]

model_name_desc = "setup4_relu"
model_name = f"{len(processes)}classes_{model_type}"
model_name = "_".join([model_name, model_name_desc]) if model_name_desc else model_name
cls_dict["model_name"] = model_name
deepsetsPP_4classes_4_relu = SimpleDNN.derive(model_name, cls_dict=cls_dict)


###############################################################################
# DSPS sum setup 1
model_type = "DeepSetsPS"
sequential_mode = "sum"

# NN architecture
nodes_deepSets_op = 10
nodes_deepSets = [80, 60, 60, nodes_deepSets_op]
nodes_ff_ds = [256, 256, 256, 256]
nodes_baseline = [256, 256, 256, 256]
model_name = f"{len(processes)}classes_{model_type}"
model_name_desc = "setup1"

cls_dict = default_cls_dict
cls_dict["model_type"] = model_type
cls_dict["sequential_mode"] = sequential_mode
cls_dict["nodes_deepSets"] = nodes_deepSets
cls_dict["nodes_deepSets_pairs"] = nodes_deepSets
cls_dict["nodes_ff_ds"] = nodes_ff_ds
cls_dict["nodes_baseline"] = nodes_baseline
cls_dict["activation_func_deepSets"] = ["selu" for i in range(len(nodes_deepSets))]
cls_dict["activation_func_ff_ds"] = ["selu" for i in range(len(nodes_ff_ds))]
cls_dict["activation_func_baseline"] = ["selu" for i in range(len(nodes_baseline))]

model_name = f"{len(processes)}classes_{model_type}"
model_ps = "_".join([model_name, sequential_mode])
model_name = "_".join([model_ps, model_name_desc]) if model_name_desc else model_ps
cls_dict["model_name"] = model_name
deepsetsPS_sum_4classes_2 = SimpleDNN.derive(model_name, cls_dict=cls_dict)

# setup 3
model_name_desc = "setup3"
cls_dict["aggregations"] = ["Sum"]
cls_dict["aggregations_pairs"] = ["Sum"]
model_name = f"{len(processes)}classes_{model_type}"
model_ps = "_".join([model_name, sequential_mode])
model_name = "_".join([model_ps, model_name_desc]) if model_name_desc else model_ps
cls_dict["model_name"] = model_name
deepsetsPS_sum_4classes_1 = SimpleDNN.derive(model_name, cls_dict=cls_dict)

# DSPS sum setup 2
model_type = "DeepSetsPS"
sequential_mode = "sum"

# NN architecture
nodes_deepSets_op = 15
nodes_deepSets = [80, 80, 80, 80, 60, nodes_deepSets_op]
nodes_ff_ds = [256, 256, 256, 256, 256, 256]
nodes_baseline = [256, 256, 256, 256, 256, 256]
model_name = f"{len(processes)}classes_{model_type}"
model_name_desc = "setup2"

cls_dict = default_cls_dict
cls_dict["model_type"] = model_type
cls_dict["sequential_mode"] = sequential_mode
cls_dict["nodes_deepSets"] = nodes_deepSets
cls_dict["nodes_deepSets_pairs"] = nodes_deepSets
cls_dict["nodes_ff_ds"] = nodes_ff_ds
cls_dict["nodes_baseline"] = nodes_baseline
cls_dict["activation_func_deepSets"] = ["selu" for i in range(len(nodes_deepSets))]
cls_dict["activation_func_ff_ds"] = ["selu" for i in range(len(nodes_ff_ds))]
cls_dict["activation_func_baseline"] = ["selu" for i in range(len(nodes_baseline))]

model_name = f"{len(processes)}classes_{model_type}"
model_ps = "_".join([model_name, sequential_mode])
model_name = "_".join([model_ps, model_name_desc]) if model_name_desc else model_ps
cls_dict["model_name"] = model_name
deepsetsPS_sum_4classes_2 = SimpleDNN.derive(model_name, cls_dict=cls_dict)

# setup2 relu
cls_dict["activation_func_deepSets"] = ["relu" for i in range(len(nodes_deepSets))]
cls_dict["activation_func_ff_ds"] = ["relu" for i in range(len(nodes_ff_ds))]
cls_dict["activation_func_baseline"] = ["relu" for i in range(len(nodes_baseline))]

model_name_desc = "setup2_relu"
model_name = f"{len(processes)}classes_{model_type}"
model_ps = "_".join([model_name, sequential_mode])
model_name = "_".join([model_ps, model_name_desc]) if model_name_desc else model_ps
cls_dict["model_name"] = model_name
deepsetsPS_sum_4classes_2_relu = SimpleDNN.derive(model_name, cls_dict=cls_dict)

# setup 4
model_name_desc = "setup4"
cls_dict["activation_func_deepSets"] = ["selu" for i in range(len(nodes_deepSets))]
cls_dict["activation_func_ff_ds"] = ["selu" for i in range(len(nodes_ff_ds))]
cls_dict["activation_func_baseline"] = ["selu" for i in range(len(nodes_baseline))]
cls_dict["aggregations"] = ["Sum"]
cls_dict["aggregations_pairs"] = ["Sum"]
model_name = f"{len(processes)}classes_{model_type}"
model_ps = "_".join([model_name, sequential_mode])
model_name = "_".join([model_ps, model_name_desc]) if model_name_desc else model_ps
cls_dict["model_name"] = model_name
deepsetsPS_sum_4classes_4 = SimpleDNN.derive(model_name, cls_dict=cls_dict)

# setup 4 relu
cls_dict["activation_func_deepSets"] = ["relu" for i in range(len(nodes_deepSets))]
cls_dict["activation_func_ff_ds"] = ["relu" for i in range(len(nodes_ff_ds))]
cls_dict["activation_func_baseline"] = ["relu" for i in range(len(nodes_baseline))]

model_name_desc = "setup4_relu"
model_name = f"{len(processes)}classes_{model_type}"
model_ps = "_".join([model_name, sequential_mode])
model_name = "_".join([model_ps, model_name_desc]) if model_name_desc else model_ps
cls_dict["model_name"] = model_name
deepsetsPS_sum_4classes_4_relu = SimpleDNN.derive(model_name, cls_dict=cls_dict)

###############################################################################
# DSPS concat setup 1
model_type = "DeepSetsPS"
sequential_mode = "concat"

# NN architecture
nodes_deepSets_op = 10
nodes_deepSets = [80, 60, 60, nodes_deepSets_op]
nodes_ff_ds = [256, 256, 256, 256]
nodes_baseline = [256, 256, 256, 256]
model_name = f"{len(processes)}classes_{model_type}"
model_name_desc = "setup1"

cls_dict = default_cls_dict
cls_dict["model_type"] = model_type
cls_dict["sequential_mode"] = sequential_mode
cls_dict["nodes_deepSets"] = nodes_deepSets
cls_dict["nodes_deepSets_pairs"] = nodes_deepSets
cls_dict["nodes_ff_ds"] = nodes_ff_ds
cls_dict["nodes_baseline"] = nodes_baseline
cls_dict["activation_func_deepSets"] = ["selu" for i in range(len(nodes_deepSets))]
cls_dict["activation_func_ff_ds"] = ["selu" for i in range(len(nodes_ff_ds))]
cls_dict["activation_func_baseline"] = ["selu" for i in range(len(nodes_baseline))]

model_name = f"{len(processes)}classes_{model_type}"
model_ps = "_".join([model_name, sequential_mode])
model_name = "_".join([model_ps, model_name_desc]) if model_name_desc else model_ps
cls_dict["model_name"] = model_name
deepsetsPS_concat_4classes_1 = SimpleDNN.derive(model_name, cls_dict=cls_dict)

# setup 3
model_name_desc = "setup3"
cls_dict["aggregations"] = ["Sum"]
cls_dict["aggregations_pairs"] = ["Sum"]
model_name = f"{len(processes)}classes_{model_type}"
model_ps = "_".join([model_name, sequential_mode])
model_name = "_".join([model_ps, model_name_desc]) if model_name_desc else model_ps
cls_dict["model_name"] = model_name
deepsetsPS_concat_4classes_3 = SimpleDNN.derive(model_name, cls_dict=cls_dict)

# DSPS concat setup 2
model_type = "DeepSetsPS"
sequential_mode = "concat"

# NN architecture
nodes_deepSets_op = 15
nodes_deepSets = [80, 80, 80, 80, 60, nodes_deepSets_op]
nodes_ff_ds = [256, 256, 256, 256, 256, 256]
nodes_baseline = [256, 256, 256, 256, 256, 256]
model_name = f"{len(processes)}classes_{model_type}"
model_name_desc = "setup2"

cls_dict = default_cls_dict
cls_dict["model_type"] = model_type
cls_dict["sequential_mode"] = sequential_mode
cls_dict["nodes_deepSets"] = nodes_deepSets
cls_dict["nodes_deepSets_pairs"] = nodes_deepSets
cls_dict["nodes_ff_ds"] = nodes_ff_ds
cls_dict["nodes_baseline"] = nodes_baseline
cls_dict["activation_func_deepSets"] = ["selu" for i in range(len(nodes_deepSets))]
cls_dict["activation_func_ff_ds"] = ["selu" for i in range(len(nodes_ff_ds))]
cls_dict["activation_func_baseline"] = ["selu" for i in range(len(nodes_baseline))]

model_name = f"{len(processes)}classes_{model_type}"
model_ps = "_".join([model_name, sequential_mode])
model_name = "_".join([model_ps, model_name_desc]) if model_name_desc else model_ps
cls_dict["model_name"] = model_name
deepsetsPS_conat_4classes_2 = SimpleDNN.derive(model_name, cls_dict=cls_dict)

# setup2 relu
cls_dict["activation_func_deepSets"] = ["relu" for i in range(len(nodes_deepSets))]
cls_dict["activation_func_ff_ds"] = ["relu" for i in range(len(nodes_ff_ds))]
cls_dict["activation_func_baseline"] = ["relu" for i in range(len(nodes_baseline))]

model_name_desc = "setup2_relu"
model_name = f"{len(processes)}classes_{model_type}"
model_ps = "_".join([model_name, sequential_mode])
model_name = "_".join([model_ps, model_name_desc]) if model_name_desc else model_ps
cls_dict["model_name"] = model_name
deepsetsPS_sum_4classes_2_relu = SimpleDNN.derive(model_name, cls_dict=cls_dict)

# setup 4
model_name_desc = "setup4"
cls_dict["activation_func_deepSets"] = ["selu" for i in range(len(nodes_deepSets))]
cls_dict["activation_func_ff_ds"] = ["selu" for i in range(len(nodes_ff_ds))]
cls_dict["activation_func_baseline"] = ["selu" for i in range(len(nodes_baseline))]
cls_dict["aggregations"] = ["Sum"]
cls_dict["aggregations_pairs"] = ["Sum"]
model_name = f"{len(processes)}classes_{model_type}"
model_ps = "_".join([model_name, sequential_mode])
model_name = "_".join([model_ps, model_name_desc]) if model_name_desc else model_ps
cls_dict["model_name"] = model_name
deepsetsPS_concat_4classes_4 = SimpleDNN.derive(model_name, cls_dict=cls_dict)

# setup 4 relu
cls_dict["activation_func_deepSets"] = ["relu" for i in range(len(nodes_deepSets))]
cls_dict["activation_func_ff_ds"] = ["relu" for i in range(len(nodes_ff_ds))]
cls_dict["activation_func_baseline"] = ["relu" for i in range(len(nodes_baseline))]

model_name_desc = "setup4_relu"
model_name = f"{len(processes)}classes_{model_type}"
model_ps = "_".join([model_name, sequential_mode])
model_name = "_".join([model_ps, model_name_desc]) if model_name_desc else model_ps
cls_dict["model_name"] = model_name
deepsetsPS_concat_4classes_4_relu = SimpleDNN.derive(model_name, cls_dict=cls_dict)


###############################################################################
# DSPS two_inp setup 1
model_type = "DeepSetsPS"
sequential_mode = "two_inp"

# NN architecture
nodes_deepSets_op = 10
nodes_deepSets = [80, 60, 60, nodes_deepSets_op]
nodes_ff_ds = [256, 256, 256, 256]
nodes_baseline = [256, 256, 256, 256]
model_name = f"{len(processes)}classes_{model_type}"
model_name_desc = "setup1"

cls_dict = default_cls_dict
cls_dict["model_type"] = model_type
cls_dict["sequential_mode"] = sequential_mode
cls_dict["nodes_deepSets"] = nodes_deepSets
cls_dict["nodes_deepSets_pairs"] = nodes_deepSets
cls_dict["nodes_ff_ds"] = nodes_ff_ds
cls_dict["nodes_baseline"] = nodes_baseline
cls_dict["activation_func_deepSets"] = ["selu" for i in range(len(nodes_deepSets))]
cls_dict["activation_func_ff_ds"] = ["selu" for i in range(len(nodes_ff_ds))]
cls_dict["activation_func_baseline"] = ["selu" for i in range(len(nodes_baseline))]

model_name = f"{len(processes)}classes_{model_type}"
model_ps = "_".join([model_name, sequential_mode])
model_name = "_".join([model_ps, model_name_desc]) if model_name_desc else model_ps
cls_dict["model_name"] = model_name
deepsetsPS_two_inp_4classes_1 = SimpleDNN.derive(model_name, cls_dict=cls_dict)

# setup 3
model_name_desc = "setup3"
cls_dict["aggregations"] = ["Sum"]
cls_dict["aggregations_pairs"] = ["Sum"]
model_name = f"{len(processes)}classes_{model_type}"
model_ps = "_".join([model_name, sequential_mode])
model_name = "_".join([model_ps, model_name_desc]) if model_name_desc else model_ps
cls_dict["model_name"] = model_name
deepsetsPS_two_inp_4classes_3 = SimpleDNN.derive(model_name, cls_dict=cls_dict)

# DSPS two_inp setup 2
model_type = "DeepSetsPS"
sequential_mode = "two_inp"

# NN architecture
nodes_deepSets_op = 15
nodes_deepSets = [80, 80, 80, 80, 60, nodes_deepSets_op]
nodes_ff_ds = [256, 256, 256, 256, 256, 256]
nodes_baseline = [256, 256, 256, 256, 256, 256]
model_name = f"{len(processes)}classes_{model_type}"
model_name_desc = "setup2"

cls_dict = default_cls_dict
cls_dict["model_type"] = model_type
cls_dict["sequential_mode"] = sequential_mode
cls_dict["nodes_deepSets"] = nodes_deepSets
cls_dict["nodes_deepSets_pairs"] = nodes_deepSets
cls_dict["nodes_ff_ds"] = nodes_ff_ds
cls_dict["nodes_baseline"] = nodes_baseline
cls_dict["activation_func_deepSets"] = ["selu" for i in range(len(nodes_deepSets))]
cls_dict["activation_func_ff_ds"] = ["selu" for i in range(len(nodes_ff_ds))]
cls_dict["activation_func_baseline"] = ["selu" for i in range(len(nodes_baseline))]

model_name = f"{len(processes)}classes_{model_type}"
model_ps = "_".join([model_name, sequential_mode])
model_name = "_".join([model_ps, model_name_desc]) if model_name_desc else model_ps
cls_dict["model_name"] = model_name
deepsetsPS_two_inp_4classes_2 = SimpleDNN.derive(model_name, cls_dict=cls_dict)

# setup2 relu
cls_dict["activation_func_deepSets"] = ["relu" for i in range(len(nodes_deepSets))]
cls_dict["activation_func_ff_ds"] = ["relu" for i in range(len(nodes_ff_ds))]
cls_dict["activation_func_baseline"] = ["relu" for i in range(len(nodes_baseline))]

model_name_desc = "setup2_relu"
model_name = f"{len(processes)}classes_{model_type}"
model_ps = "_".join([model_name, sequential_mode])
model_name = "_".join([model_ps, model_name_desc]) if model_name_desc else model_ps
cls_dict["model_name"] = model_name
deepsetsPS_two_inp_4classes_2_relu = SimpleDNN.derive(model_name, cls_dict=cls_dict)

# setup 4
model_name_desc = "setup4"
cls_dict["activation_func_deepSets"] = ["selu" for i in range(len(nodes_deepSets))]
cls_dict["activation_func_ff_ds"] = ["selu" for i in range(len(nodes_ff_ds))]
cls_dict["activation_func_baseline"] = ["selu" for i in range(len(nodes_baseline))]
cls_dict["aggregations"] = ["Sum"]
cls_dict["aggregations_pairs"] = ["Sum"]
model_name = f"{len(processes)}classes_{model_type}"
model_ps = "_".join([model_name, sequential_mode])
model_name = "_".join([model_ps, model_name_desc]) if model_name_desc else model_ps
cls_dict["model_name"] = model_name
deepsetsPS_two_inp_4classes_4 = SimpleDNN.derive(model_name, cls_dict=cls_dict)

# setup 4 relu
cls_dict["activation_func_deepSets"] = ["relu" for i in range(len(nodes_deepSets))]
cls_dict["activation_func_ff_ds"] = ["relu" for i in range(len(nodes_ff_ds))]
cls_dict["activation_func_baseline"] = ["relu" for i in range(len(nodes_baseline))]

model_name_desc = "setup4_relu"
model_name = f"{len(processes)}classes_{model_type}"
model_ps = "_".join([model_name, sequential_mode])
model_name = "_".join([model_ps, model_name_desc]) if model_name_desc else model_ps
cls_dict["model_name"] = model_name
deepsetsPS_two_inp_4classes_4_relu = SimpleDNN.derive(model_name, cls_dict=cls_dict)

# law run cf.PlotMLResults --plot-function roc --general-settings "evaluation_type=ovr" --ml-model test --datasets graviton_hh_ggf_bbtautau_m400_madgraph,graviton_hh_vbf_bbtautau_m400_madgraph,tt_dl_powheg,tt_sl_powheg,dy_lep_pt50To100_amcatnlo,dy_lep_pt100To250_amcatnlo,dy_lep_pt250To400_amcatnlo,dy_lep_pt400To650_amcatnlo,dy_lep_pt650_amcatnlo --version PairsML
# law run cf.MLTraining --version PairsML --ml-model test --config run2_2017_nano_uhh_v11_limited --cf.MLTraining-workflow htcondor --cf.MLTraining-htcondor-memory 20GB

"""
law run cf.MergeMLEvaluationPerFoldWrapper --version limits_xsecs --cf.MergeMLEvaluationPerFold-ml-model 4classes_DeepSetsPS_sum_setup2 --configs run2_2017_nano_uhh_v11_limited --workers 5 --datasets tt_sl_powheg,graviton_hh_ggf_bbtautau_m400_madgraph,tt_dl_powheg --cf.MLEvaluationPerFold-workflow htcondor --cf.MLEvaluationPerFold-additional-metrics --cf.MLEvaluationPerFold-max-runtime 12h
"""
