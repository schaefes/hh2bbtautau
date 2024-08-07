# encoding: utf-8
from columnflow.columnar_util import EMPTY_FLOAT

vars_dictionary = {
    "CustomVBFMaskJets2_e": {'null_value': EMPTY_FLOAT, 'binning': [70, 0.0, 1200.0], 'unit': " / GeV", 'label': r"Jet Energy", 'steps': 200.},
    "CustomVBFMaskJets2_mass": {'null_value': EMPTY_FLOAT, 'binning': [25, 0.0, 100.0], 'unit': " / GeV", 'label': r"Jet Mass", 'steps': 20.},
    "CustomVBFMaskJets2_pt": {'null_value': EMPTY_FLOAT, 'binning': [32, 0.0, 700.0], 'unit': " / GeV", 'label': r"Jet p$_{T}$", 'steps': 100.},
    "CustomVBFMaskJets2_eta": {'null_value': EMPTY_FLOAT, 'binning': [20, -5., 5.], 'unit': "", 'label': r"Jet $\eta$", 'steps': 2.},
    "CustomVBFMaskJets2_phi": {'null_value': EMPTY_FLOAT, 'binning': [20, -3.15, 3.15], 'unit': "", 'label': r"Jet $\phi$", 'steps': 1.},
    "CustomVBFMaskJets2_jet_pt_frac": {'null_value': EMPTY_FLOAT, 'binning': [20, 0, 1], 'unit': "", 'label': r"Jet p$_{T}$ Fraction", 'steps': .2},
    "CustomVBFMaskJets2_bFlavtag": {'null_value': EMPTY_FLOAT, 'binning': [20, 0.0, 1.0], 'unit': "", 'label': r"DeepJet B Score", 'steps': .2},
    "CustomVBFMaskJets2_bFlavtagCvL": {'null_value': EMPTY_FLOAT, 'binning': [20, 0.0, 1.0], 'unit': "", 'label': r"DeepJet CvL Score", 'steps': .2},
    "CustomVBFMaskJets2_bFlavtagCvB": {'null_value': EMPTY_FLOAT, 'binning': [20, 0.0, 1.0], 'unit': "", 'label': r"DeepJet CvB Score", 'steps': .2},
    "CustomVBFMaskJets2_btagQG": {'null_value': EMPTY_FLOAT, 'binning': [20, 0.0, 1.0], 'unit': "", 'label': r"DeepJet QG Score", 'steps': .2},
    "CustomVBFMaskJets2_mbb": {'null_value': EMPTY_FLOAT, 'binning': [35, 0.0, 1000.0], 'unit': " / GeV", 'label': r"m$_{b \bar{b}}$", 'steps': 200.},
    "CustomVBFMaskJets2_mHH": {'null_value': EMPTY_FLOAT, 'binning': [40, 0.0, 2000], 'unit': " / GeV", 'label': r"m$_{HH}$", 'steps': 250.},
    "CustomVBFMaskJets2_mtautau": {'null_value': EMPTY_FLOAT, 'binning': [40, 0.0, 400.0], 'unit': " / GeV", 'label': r"m$_{\tau^{-} \tau^{+}}$", 'steps': 50.},
    "CustomVBFMaskJets2_mjj": {'null_value': EMPTY_FLOAT, 'binning': [50, 0.0, 2500.0], 'unit': " / GeV", 'label': r"Max. m$_{jj}$", 'steps': 500.},
    "CustomVBFMaskJets2_ht": {'null_value': EMPTY_FLOAT, 'binning': [45, 0.0, 2000.0], 'unit': " / GeV", 'label': r"H$_{T}$", 'steps': 400.},
    "CustomVBFMaskJets2_mjj_dEta": {'null_value': EMPTY_FLOAT, 'binning': [37, 0.0, 2500.0], 'unit': " / GeV", 'label': r"m$_{JJ, \Delta \eta}$", 'steps': 500.},
    "CustomVBFMaskJets2_max_dEta": {'null_value': EMPTY_FLOAT, 'binning': [20, 0.0, 9.0], 'unit': "", 'label': r"Max. $\Delta \eta$ of Jet Pair", 'steps': 1.},
    "CustomVBFMaskJets2_thrust": {'null_value': EMPTY_FLOAT, 'binning': [20, 0., 1.], 'unit': "", 'label': r"T", 'steps': .2},
    "CustomVBFMaskJets2_pt_thrust": {'null_value': EMPTY_FLOAT, 'binning': [20, 0., 1.], 'unit': "", 'label': r"T$_{T}$", 'steps': .2},
    "CustomVBFMaskJets2_sphericity": {'null_value': EMPTY_FLOAT, 'binning': [20, 0., 1.], 'unit': "", 'label': r"S$_{T}$", 'steps': .2},
    "CustomVBFMaskJets2_energy_corr_sqr_tev2": {'null_value': EMPTY_FLOAT, 'binning': [27, 0., 2.5], 'unit': r" / TeV²", 'label': r"ECF(N=2, $\beta$=2)", 'steps': .5},
    "CustomVBFMaskJets2_njets": {'null_value': EMPTY_FLOAT, 'binning': [11, -0.5, 10.5], 'unit': "", 'label': r"N Jets", 'steps': 1.},
}

# correlations_dict = {
#     '4classes_DeepSets_setup2': [
#         ['topDS_labels.npy', 'event_labels.npy', 'corrcoef_events', 32],
#         ['topDS_labels.npy', 'jet_labels_12.npy', 'corrcoef_jets', 32],
#         ['topDS_labels.npy', 'multiplicity3labels.npy', 'corrcoef_jets_multiplicity_3', 32],
#     ],
#     '4classes_DeepSetsPP_setup2': [
#         ['topDS_jets_labels.npy', 'event_labels.npy', 'corrcoef_events_DSJ', 25],
#         ['topDS_pairs_labels.npy', 'event_labels.npy', 'corrcoef_events_DSP', 25],
#         ['topDS_jets_labels.npy', 'jet_labels_12.npy', 'corrcoef_jets', 25],
#         ['topDS_pairs_labels.npy', 'pair_1_labels.npy', 'corrcoef_pairs', 25],
#         ['topDS_jets_labels.npy', 'multiplicity3labels_jets.npy', 'corrcoef_jets_multiplicity3', 25],
#         ['topDS_pairs_labels.npy', 'multiplicity3labels_pairs.npy', 'corrcoef_pairs_multiplicity3', 25],
#     ],
#     '4classes_DeepSetsPS_sum_setup2': [
#         ['topDS_jets_labels.npy', 'event_labels.npy', 'corrcoef_events_DSJ', 44],
#         ['topDS_pairs_labels.npy', 'event_labels.npy', 'corrcoef_events_DSP', 44],
#         ['topDS_jets_labels.npy', 'jet_labels_12.npy', 'corrcoef_jets', 44],
#         ['topDS_pairs_labels.npy', 'pair_1_labels.npy', 'corrcoef_pairs', 44],
#         ['topDS_jets_labels.npy', 'multiplicity3labels_jets.npy', 'corrcoef_jets_multiplicity3', 44],
#         ['topDS_pairs_labels.npy', 'multiplicity3labels_pairs.npy', 'corrcoef_pairs_multiplicity3', 44],
#     ],
#     '4classes_DeepSetsPS_concat_setup2': [
#         ['topDS_jets_labels.npy', 'event_labels.npy', 'corrcoef_events_DSJ', 44],
#         ['topDS_pairs_labels.npy', 'event_labels.npy', 'corrcoef_events_DSP', 44],
#         ['topDS_jets_labels.npy', 'jet_labels_12.npy', 'corrcoef_jets', 44],
#         ['topDS_pairs_labels.npy', 'pair_1_labels.npy', 'corrcoef_pairs', 44],
#         ['topDS_jets_labels.npy', 'multiplicity3labels_jets.npy', 'corrcoef_jets_multiplicity3', 44],
#         ['topDS_pairs_labels.npy', 'multiplicity3labels_pairs.npy', 'corrcoef_pairs_multiplicity3', 44],
#     ],
#     '4classes_DeepSetsPS_two_inp_setup2': [
#         ['topDS_jets_labels.npy', 'event_labels.npy', 'corrcoef_events_DSJ', 14],
#         ['topDS_pairs_labels.npy', 'event_labels.npy', 'corrcoef_events_DSP', 14],
#         ['topDS_jets_labels.npy', 'jet_labels_12.npy', 'corrcoef_jets', 14],
#         ['topDS_pairs_labels.npy', 'pair_1_labels.npy', 'corrcoef_pairs', 14],
#         ['topDS_jets_labels.npy', 'multiplicity3labels_jets.npy', 'corrcoef_jets_multiplicity3', 14],
#         ['topDS_pairs_labels.npy', 'multiplicity3labels_pairs.npy', 'corrcoef_pairs_multiplicity3', 14],
#     ],
#     '4classes_baseline_setup2': [
#         ['topDS_labels.npy', 'event_labels.npy', 'corrcoef_events', 38],
#     ],
#     '4classes_baseline_pairs_setup2': [
#         ['topDS_labels.npy', 'event_labels.npy', 'corrcoef_events', 3],
#     ]

# }

jet_collection = "CustomVBFMaskJets2"
latex_dict = {
    f"{jet_collection}_e": "Energy",
    f"{jet_collection}_mass": "Mass",
    f"{jet_collection}_pt": r"$p_{T}$",
    f"{jet_collection}_jet_pt_frac": r"$p_{T}$ Fraction",
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

correlations_dict = {
    '4classes_DeepSets_setup2': [
        ['ds_labels_jets.npy', 'event_labels.npy', 'corrcoef_events', 32],
        ['ds_labels_jets.npy', 'jet_labels_2.npy', 'corrcoef_jets', 32],
        ['ds_labels_jets.npy', 'jet_labels_3.npy', 'corrcoef_jets_multiplicity_3', 32],
    ],
    '4classes_DeepSetsPP_setup2': [
        ['ds_labels_jets_large.npy', 'event_labels.npy', 'corrcoef_events_DSJ', 25],
        ['ds_pair_labels.npy', 'event_labels.npy', 'corrcoef_events_DSP', 25],
        ['ds_labels_jets_large.npy', 'jet_labels_2.npy', 'corrcoef_jets', 25],
        ['ds_pair_labels.npy', 'pair_labels_1.npy', 'corrcoef_pairs', 25],
        ['ds_labels_jets_large.npy', 'jet_labels_3.npy', 'corrcoef_jets_multiplicity3', 25],
        ['ds_pair_labels.npy', 'pair_labels_3.npy', 'corrcoef_pairs_multiplicity3', 25],
    ],
    '4classes_DeepSetsPS_sum_setup2': [
        ['ds_labels_jets_large.npy', 'event_labels.npy', 'corrcoef_events_DSJ', 44],
        ['ds_pair_labels.npy', 'event_labels.npy', 'corrcoef_events_DSP', 44],
        ['ds_labels_jets_large.npy', 'jet_labels_2.npy', 'corrcoef_jets', 44],
        ['ds_pair_labels.npy', 'pair_labels_1.npy', 'corrcoef_pairs', 44],
        ['ds_labels_jets_large.npy', 'jet_labels_3.npy', 'corrcoef_jets_multiplicity3', 44],
        ['ds_pair_labels.npy', 'pair_labels_3.npy', 'corrcoef_pairs_multiplicity3', 44],
    ],
    '4classes_DeepSetsPS_concat_setup2': [
        ['ds_labels_jets_large.npy', 'event_labels.npy', 'corrcoef_events_DSJ', 44],
        ['ds_pair_labels.npy', 'event_labels.npy', 'corrcoef_events_DSP', 44],
        ['ds_labels_jets_large.npy', 'jet_labels_2.npy', 'corrcoef_jets', 44],
        ['ds_pair_labels.npy', 'pair_labels_1.npy', 'corrcoef_pairs', 44],
        ['ds_labels_jets_large.npy', 'jet_labels_3.npy', 'corrcoef_jets_multiplicity3', 44],
        ['ds_pair_labels.npy', 'pair_labels_3.npy', 'corrcoef_pairs_multiplicity3', 44],
    ],
    '4classes_DeepSetsPS_two_inp_setup2': [
        ['ds_labels_jets_large.npy', 'event_labels.npy', 'corrcoef_events_DSJ', 14],
        ['ds_pair_labels.npy', 'event_labels.npy', 'corrcoef_events_DSP', 14],
        ['ds_labels_jets_large.npy', 'jet_labels_2.npy', 'corrcoef_jets', 14],
        ['ds_pair_labels.npy', 'pair_labels_1.npy', 'corrcoef_pairs', 14],
        ['ds_labels_jets_large.npy', 'jet_labels_3.npy', 'corrcoef_jets_multiplicity3', 14],
        ['ds_pair_labels.npy', 'pair_labels_3.npy', 'corrcoef_pairs_multiplicity3', 14],
    ],
    '4classes_baseline_setup2': [
        ['topDS_labels.npy', 'event_labels.npy', 'corrcoef_events', 38],
    ],
    '4classes_baseline_pairs_setup2': [
        ['topDS_labels.npy', 'event_labels.npy', 'corrcoef_events', 3],
    ]
}

