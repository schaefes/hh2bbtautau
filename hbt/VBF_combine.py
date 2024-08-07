# encoding: utf-8

from columnflow.tasks.production import ProduceColumnsWrapper
from columnflow.tasks.reduction import MergeReducedEvents
# from columnflow.plotting.plot_all import draw_error_bands
import numpy as np
import os
import awkward as ak
import matplotlib.pyplot as plt
import mplhep
import hist
import matplotlib.ticker as ticker
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
import matplotlib.colors as mcolors
from hist import Hist
import hist
from columnflow.columnar_util import EMPTY_FLOAT
from columnflow.columnar_util import attach_behavior
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection


def draw_error_bands(
    ax: plt.Axes,
    h: hist.Hist,
    norm,
    **kwargs,
) -> None:
    # compute relative errors
    rel_error = h.variances()**0.5 / h.values()
    rel_error[np.isnan(rel_error)] = 0.0

    # compute the baseline
    # fill 1 in places where both numerator and denominator are 0, and 0 for remaining nan's
    baseline = h.values() / norm
    baseline[(h.values() == 0) & (norm == 0)] = 1.0
    baseline[np.isnan(baseline)] = 0.0

    defaults = {
        "x": h.axes[0].centers,
        "y": baseline,
        "width": h.axes[0].edges[1:] - h.axes[0].edges[:-1],
        "height": baseline * 2 * rel_error,
        "bottom": baseline * (1 - rel_error),
        "hatch": "///",
        "facecolor": "none",
        "linewidth": 0,
        "color": "black",
        "alpha": 1.0,
    }
    defaults.update(kwargs)

    return defaults


def make_error_boxes(ax, xdata, ydata, xerror, yerror, facecolor='grey',
                     edgecolor='none', alpha=0.5):

    # Loop over data points; create box from errors at each point
    errorboxes = [Rectangle((x - xe[0], y - ye[0]), xe.sum(), ye.sum(), hatch="//")
                  for x, y, xe, ye in zip(xdata, ydata, xerror.T, yerror.T)]

    # Create patch collection with specified colour/alpha
    pc = PatchCollection(errorboxes, facecolor=facecolor, alpha=alpha,
                         edgecolor=edgecolor, hatch="//")

    # Add collection to axes
    ax.add_collection(pc)


# t.law_run()
processes = ["graviton_hh_vbf_bbtautau_m400"]

proc_labels = {"graviton_hh_vbf_bbtautau_m400": r'$HH_{vbf,m400}$ $\rightarrow bb\tau\tau$'}

dataset_dict = {"graviton_hh_vbf_bbtautau_m400": ["graviton_hh_vbf_bbtautau_m400_madgraph"]}

colors = {'graviton_hh_vbf_bbtautau_m400': 'black'}
collection_dict = {}
target_dict = {}
weights = []
DeepSetsInpPt = {}
DeepSetsInpEta = {}
fold_pred = {}

# get the target and predictions from ProduceColumnsWrapper
vbf1 = MergeReducedEvents(
    version="NoKinSel",
    dataset="graviton_hh_vbf_bbtautau_m400_madgraph"
)
files_vbf = vbf1.output()

vbf2 = MergeReducedEvents(
    version="limits_xsecs",
    dataset="graviton_hh_vbf_bbtautau_m400_madgraph"
)
files_vbf2 = vbf2.output()

ggf = MergeReducedEvents(
    version="limits_xsecs",
    dataset="graviton_hh_ggf_bbtautau_m400_madgraph"
)
files_ggf = ggf.output()
ggf_jets = files_ggf[0]['events'].load(formatter='awkward').Jet
ggf_custom = files_ggf[0]['events'].load(formatter='awkward').CustomVBFMaskJets2

tt_jets = []
for p in ['tt_sl_powheg', 'tt_dl_powheg']:
    tt = MergeReducedEvents(
        version="limits_xsecs",
        dataset=p)
    files_tt = tt.output()
    jets_tt = files_tt[0]['events'].load(formatter='awkward')
    tt_jets.append(jets_tt)
tt_jets = ak.concatenate(tt_jets)
tt_custom = tt_jets.CustomVBFMaskJets2
tt_jets = tt_jets.Jet

dy_jets = []
for p in ["dy_lep_pt50To100_amcatnlo", "dy_lep_pt100To250_amcatnlo", "dy_lep_pt250To400_amcatnlo", "dy_lep_pt400To650_amcatnlo", "dy_lep_pt650_amcatnlo",]:
    dy = MergeReducedEvents(
        version="limits_xsecs",
        dataset=p)
    files_dy = dy.output()
    jets_dy = files_dy[0]['events'].load(formatter='awkward')
    dy_jets.append(jets_dy)
dy_jets = ak.concatenate(dy_jets)
dy_custom = dy_jets.CustomVBFMaskJets2
dy_jets = dy_jets.Jet

# vbf_data = files[('run2_2017_nano_uhh_v11_limited', 'nominal',
# 'graviton_hh_vbf_bbtautau_m400_madgraph')].collection[0]['columns'].load(formatter="awkward")
# ggf_400_data = files[('run2_2017_nano_uhh_v11_limited', 'nominal',
# 'graviton_hh_ggf_bbtautau_m400_madgraph')].collection[0]['columns'].load(formatter="awkward")

# savepath for the plots
path = "/nfs/dust/cms/user/schaefes/VBF_plots"

jet_collections = files_vbf[0]['events'].load(formatter='awkward')
jet_collections2 = files_vbf2[0]['events'].load(formatter='awkward')
auto_matched = jet_collections.GenMatchedVBFJets
auto_matched = attach_behavior(auto_matched, 'Jet')
jets = jet_collections2.Jet
jets = attach_behavior(jets, 'Jet')
custom = jet_collections2.CustomVBFMaskJets2
custom = attach_behavior(custom, 'Jet')
pt = ak.flatten(auto_matched.pt)
mask_none_flat = ~ak.is_none(pt)
pt = pt[mask_none_flat]
eta = ak.flatten(auto_matched.eta)[mask_none_flat]
inv_mass = (auto_matched[:, 0] + auto_matched[:, 1]).mass
mask_none = ~ak.is_none(inv_mass)
inv_mass = inv_mass[mask_none]
delta_eta = abs(auto_matched[:, 0].eta - auto_matched[:, 1].eta)[mask_none]

# check how many vbf jets are in each collection
"""to regular and reshape for boradcasting, different fill none value for all necessary,
else this is considered a match between jets"""
np_auto_matched = ak.to_numpy(ak.fill_none(auto_matched.pt, EMPTY_FLOAT)).reshape(len(auto_matched), 2, 1)
np_custom = ak.to_numpy(ak.fill_none(ak.pad_none(custom.pt, ak.max(ak.num(custom, axis=1))), EMPTY_FLOAT + 1)).reshape(len(auto_matched), 1, -1)
np_jets = ak.to_numpy(ak.fill_none(ak.pad_none(jets.pt, ak.max(ak.num(custom, axis=1))), EMPTY_FLOAT + 2)).reshape(len(auto_matched), 1, -1)
n_custom_matched = np.sum((np_auto_matched == np_custom).sum(axis=-1), axis=1)
n_jets_matched = np.sum((np_auto_matched == np_jets).sum(axis=-1), axis=1)
n_jets = ak.num(jets, axis=1)
n_jets_custom = ak.num(custom, axis=1)
# other processes
n_jets_ggf = ak.num(ggf_jets, axis=1)
n_jets_custom_ggf = ak.num(ggf_custom, axis=1)
n_jets_tt = ak.num(tt_jets, axis=1)
n_jets_custom_tt = ak.num(tt_custom, axis=1)
n_jets_dy = ak.num(dy_jets, axis=1)
n_jets_custom_dy = ak.num(dy_custom, axis=1)

# check how many of the added jets were VBF jets

# vars dict for all histograms
vars_dict = {'pt': {'binning': [40, 0, 700], 'data': pt, 'xlabel': r"VBF Jets $p_{T}$", 'unit': " / GeV",
        'steps': 100., 'shade_intervals': [[0., 30.]], 'txt': "selection\napplied",
        'label': r'$HH_{vbf,m400}$ $\rightarrow bb\tau\tau$', 'c': 'tab:blue', 'fill': True},
    'eta': {'binning': [20, -5, 5], 'data': eta, 'xlabel': r"VBF Jets $\eta$", 'unit': "",
        'steps': 1., 'shade_intervals': [[-5, 0.3], [4.7, 0.3]], 'txt': "selection\napplied",
        'label': r'$HH_{vbf,m400}$ $\rightarrow bb\tau\tau$', 'c': 'tab:blue', 'fill': True},
    'delta_eta': {'binning': [16, 0, 8], 'data': delta_eta, 'xlabel': r"VBF Jets $\Delta \eta$", 'unit': "",
        'steps': 2., 'shade_intervals': [[0., 3.]], 'txt': "selection\napplied",
        'label': r'$HH_{vbf,m400}$ $\rightarrow bb\tau\tau$', 'c': 'tab:blue', 'fill': True},
    'inv_mass': {'binning': [40, 0, 3000], 'data': inv_mass, 'xlabel': "VBF Jets Invariant Mass", 'unit': " / GeV",
        'steps': 500., 'shade_intervals': [[0., 500.]], 'txt': "selection\napplied",
        'label': r'$HH_{vbf,m400}$ $\rightarrow bb\tau\tau$', 'c': 'tab:blue', 'fill': True},
    'n_matched_jets': {'binning': [3, -0.5, 2.5], 'data': n_jets_matched, 'xlabel': "Included VBF Jets", 'unit': "",
        'steps': 1., 'shade_intervals': [[0., 0.]], 'txt': "selection\napplied",
        'label': r'$HH_{vbf,m400}$ $\rightarrow bb\tau\tau$', 'c': 'tab:blue', 'fill': True},
    'n_custom_match': {'binning': [3, -0.5, 2.5], 'data': n_custom_matched, 'xlabel': "Included VBF Jets", 'unit': "",
        'steps': 1., 'shade_intervals': [[0., 0.]], 'txt': "selection\napplied",
        'label': r'$HH_{vbf,m400}$ $\rightarrow bb\tau\tau$', 'c': 'tab:blue', 'fill': True},

    'n_jets': {'binning': [8, 1.5, 9.5], 'data': n_jets, 'xlabel': "Number of Jets", 'unit': "",
        'steps': 1., 'shade_intervals': [[0., 0.]], 'txt': r"$HH_{vbf,m400}$ $\rightarrow bb\tau\tau$," + "\nselection applied",
        'label': r'Standart Jet Selection', 'c': 'tab:blue', 'fill': False},
    'n_jets_custom': {'binning': [8, 1.5, 9.5], 'data': n_jets_custom, 'xlabel': "Number of Jets", 'unit': "",
        'steps': 1., 'shade_intervals': [[0., 0.]], 'txt': r"$HH_{vbf,m400}$ $\rightarrow bb\tau\tau$," + "\nselection applied",
        'label': r'Modified Jet Selection', 'c': 'black', 'fill': False},

    'n_jets_ggf': {'binning': [8, 1.5, 9.5], 'data': n_jets_ggf, 'xlabel': "Number of Jets", 'unit': "",
        'steps': 1., 'shade_intervals': [[0., 0.]], 'txt': r"$HH_{ggf,m400}$ $\rightarrow bb\tau\tau$," + "\nselection applied",
        'label': r'Standart Jet Selection', 'c': 'tab:blue', 'fill': False},
    'n_jets_custom_ggf': {'binning': [8, 1.5, 9.5], 'data': n_jets_custom_ggf, 'xlabel': "Number of Jets", 'unit': "",
        'steps': 1., 'shade_intervals': [[0., 0.]], 'txt': r"$HH_{ggf,m400}$ $\rightarrow bb\tau\tau$," + "\nselection applied",
        'label': r'Modified Jet Selection', 'c': 'black', 'fill': False},

    'n_jets_tt': {'binning': [8, 1.5, 9.5], 'data': n_jets_tt, 'xlabel': "Number of Jets", 'unit': "",
        'steps': 1., 'shade_intervals': [[0., 0.]], 'txt': r"$t \bar{t}$," + "\nselection applied",
        'label': r'Standart Jet Selection', 'c': 'tab:blue', 'fill': False},
    'n_jets_custom_tt': {'binning': [8, 1.5, 9.5], 'data': n_jets_custom_tt, 'xlabel': "Number of Jets", 'unit': "",
        'steps': 1., 'shade_intervals': [[0., 0.]], 'txt': r"$t \bar{t}$," + "\nselection applied",
        'label': r'Modified Jet Selection', 'c': 'black', 'fill': False},

    'n_jets_dy': {'binning': [8, 1.5, 9.5], 'data': n_jets_dy, 'xlabel': "Number of Jets", 'unit': "",
        'steps': 1., 'shade_intervals': [[0., 0.]], 'txt': "Drell-Yan," + "\nselection applied",
        'label': r'Standart Jet Selection', 'c': 'tab:blue', 'fill': False},
    'n_jets_custom_dy': {'binning': [8, 1.5, 9.5], 'data': n_jets_custom_dy, 'xlabel': "Number of Jets", 'unit': "",
        'steps': 1., 'shade_intervals': [[0., 0.]], 'txt': "Drell-Yan," + "\nselection applied",
        'label': 'Modified Jet Selection', 'c': 'black', 'fill': False}}
# plot the histograms of the auto matched jets
max_bin = 0
fig2, ax2 = plt.subplots()
for var in vars_dict.keys():
    mplhep.cms.label(ax=ax2, llabel="Private work", data=False, loc=0, lumi=True, lumi_format="41.48", fontsize=15)
    # get data for plotting
    bins, lower, upper = vars_dict[var]['binning']
    proc_data = vars_dict[var]['data']
    steps = vars_dict[var]['steps']
    txt = vars_dict[var]['txt']
    label = vars_dict[var]['label']
    proc_data_clip = np.clip(proc_data, lower, upper)
    c = vars_dict[var]['c']
    fill = vars_dict[var]['fill']

    # hist object to determine error bands
    fig_error, ax_error = plt.subplots()
    h = Hist(hist.axis.Regular(bins, lower, upper, name="S", label="s [units]"))
    h.fill(proc_data_clip)
    h.project("S").plot(ax=ax_error)
    norm_val = len(proc_data_clip)
    defaults = draw_error_bands(ax_error, h, norm_val)
    plt.close(fig_error)

    # create plot
    ax2.bar(defaults["x"], defaults["y"], width=defaults["width"], yerr=defaults["height"] / 2,
        color=c, edgecolor=c, error_kw=dict(ecolor=c, capthick=2),
        label=label, fill=fill)
    if 'n_jets' in var and 'custom' not in var:
        continue
    # make_error_boxes(ax2, defaults["x"], defaults["y"], np.tile / 2, [2, 1], np.tile(defaults["height"] / 2, [2, 1])
    max_bin = np.max(defaults["y"]) if np.max(defaults["y"]) > max_bin else max_bin
    # set ticks on the axis
    ax2.text(0.05, 0.95, txt, transform=ax2.transAxes, fontsize=12, verticalalignment='top')
    ax2.xaxis.set_major_locator(ticker.MultipleLocator(steps))
    ax2.xaxis.set_minor_locator(AutoMinorLocator(4))
    ax2.yaxis.set_minor_locator(AutoMinorLocator(4))
    ax2.tick_params(axis="x", which="minor", direction="in", top=True, labeltop=False, bottom=True, labelbottom=False)
    ax2.tick_params(axis="y", which="minor", direction="in", left=True, labelleft=False, right=True, labelright=False)
    ax2.tick_params(axis="x", which="major", direction="in", top=True, labeltop=False, bottom=True, labelbottom=True)
    ax2.tick_params(axis="y", which="major", direction="in", left=True, labelleft=True, right=True, labelright=False)
    if (ax2.get_ylim()[1] - max_bin) < ax2.get_ylim()[1] / 6:
        ax2.set_ylim(ymax=ax2.get_ylim()[1] * 1.2)
    for interval in vars_dict[var]['shade_intervals']:
        left, width = interval
        rect = plt.Rectangle((left, 0), width, .85 * ax2.get_ylim()[1], facecolor="black", alpha=0.3)
        ax2.add_patch(rect)
    ax2.set_ylim(ymin=0.)
    ax2.set_xlim(xmin=lower, xmax=upper)
    ax2.set_xlabel(vars_dict[var]['xlabel'] + vars_dict[var]['unit'], loc='right', fontsize=14)
    ax2.set_ylabel(r'$\Delta$N/N', loc='top', fontsize=14)
    ax2.legend(frameon=False)
    plt.setp(ax2.get_yticklabels()[0], visible=False)
    file_path = f"{path}/genmatched_VBF_jets_{var}.pdf"
    os.remove(file_path) if os.path.exists(file_path) else None
    fig2.savefig(file_path)
    fig2, ax2 = plt.subplots()
print('ggf: ', ak.sum(n_jets_custom_ggf - n_jets_ggf), 'Events: ', len(n_jets_ggf))
print('tt: ', ak.sum(n_jets_custom_tt - n_jets_tt), 'Events: ', len(n_jets_tt))
print('dy: ', ak.sum(n_jets_custom_dy - n_jets_dy), 'Events: ', len(n_jets_dy))
# for VBF: of a total of 4963 extra jets in custom collection, 3373 are VBF Jets (total events: 9825)
