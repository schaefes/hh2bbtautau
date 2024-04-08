# encoding: utf-8

from columnflow.tasks.production import ProduceColumnsWrapper
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
from vars_dict import vars_dictionary


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


# t.law_run()
processes = ["graviton_hh_vbf_bbtautau_m400",
            "graviton_hh_ggf_bbtautau_m400",
            "tt",
            "dy"]

proc_labels = {"graviton_hh_ggf_bbtautau_m400": r'$HH_{ggf,m400}$ $\rightarrow bb\tau\tau$',
              "graviton_hh_vbf_bbtautau_m400": r'$HH_{vbf,m400}$ $\rightarrow bb\tau\tau$',
              "hh_ggf_bbtautau": r'$HH_{ggf} \rightarrow bb\tau\tau$',
              "tt": r'$t \bar{t}$',
              "dy": 'Drell-Yan'}

dataset_dict = {"graviton_hh_vbf_bbtautau_m400": ["graviton_hh_ggf_bbtautau_m400_madgraph"],
            "graviton_hh_ggf_bbtautau_m400": ["graviton_hh_vbf_bbtautau_m400_madgraph"],
            "tt": ["tt_sl_powheg", "tt_dl_powheg"],
            "dy": ["dy_lep_pt50To100_amcatnlo", "dy_lep_pt100To250_amcatnlo", "dy_lep_pt250To400_amcatnlo", "dy_lep_pt400To650_amcatnlo", "dy_lep_pt650_amcatnlo"]
            }

colors = {'graviton_hh_ggf_bbtautau_m400': 'tab:blue',
          'graviton_hh_vbf_bbtautau_m400': 'black',
          'tt': 'tab:red',
          'dy': 'gold'}

collection_dict = {}
target_dict = {}
weights = []
DeepSetsInpPt = {}
DeepSetsInpEta = {}
fold_pred = {}

# get the target and predictions from ProduceColumnsWrapper
t = ProduceColumnsWrapper(
    version="limits_xsecs",
)
files = t.output()
# vbf_data = files[('run2_2017_nano_uhh_v11_limited', 'nominal',
# 'graviton_hh_vbf_bbtautau_m400_madgraph')].collection[0]['columns'].load(formatter="awkward")
# ggf_400_data = files[('run2_2017_nano_uhh_v11_limited', 'nominal',
# 'graviton_hh_ggf_bbtautau_m400_madgraph')].collection[0]['columns'].load(formatter="awkward")

# savepath for the plots
path = "/afs/desy.de/user/s/schaefes/Plots/input_features/external_plotting"

# ax2.hist(jets_pt[jets_pt <= upper], bins=bins, weights=weights_norm[jets_pt <= upper], color='tab:blue', edgecolor='tab:blue')
# ax2.errorbar(defaults["x"], defaults["y"], yerr=defaults["height"] / 2)
max_bin = 0.
for var in vars_dictionary.keys():
    bins, lower, upper = vars_dictionary[var]['binning']
    steps = vars_dictionary[var]['steps']
    s_steps = steps / 4
    fig2, ax2 = plt.subplots()
    mplhep.cms.label(ax=ax2, llabel="Private work", data=False, loc=0, lumi=True, lumi_format="41.48", fontsize=15)

    for proc in processes:
        # prepare proc data
        proc_data = [files[('run2_2017_nano_uhh_v11_limited', 'nominal', dataset_proc)].collection[0]['columns'].load(formatter="awkward") for dataset_proc in dataset_dict[proc]]
        proc_data = ak.concatenate(proc_data)[var]
        empty_float_mask = (proc_data != EMPTY_FLOAT)
        proc_data = proc_data[empty_float_mask]
        if len(proc_data) != ak.count(proc_data):
            proc_data = ak.flatten(proc_data)
        proc_data_clip = np.clip(proc_data, lower, upper)

        # hist object to determine error bands
        fig_error, ax_error = plt.subplots()
        h = Hist(hist.axis.Regular(bins, lower, upper, name="S", label="s [units]"))
        h.fill(proc_data_clip)
        h.project("S").plot(ax=ax_error)
        norm_val = len(proc_data_clip)
        defaults = draw_error_bands(ax_error, h, norm_val)
        plt.close(fig_error)

        # create plot
        c = colors[proc]
        ax2.bar(defaults["x"], defaults["y"], width=defaults["width"], yerr=defaults["height"] / 2, fill=False, edgecolor=c, error_kw=dict(ecolor=c, capthick=2), label=proc_labels[proc])
        max_bin = np.max(defaults["y"]) if np.max(defaults["y"]) > max_bin else max_bin
    # set ticks on the axis
    ax2.text(0.05, 0.95, 'inclusive', transform=ax2.transAxes, fontsize=14, verticalalignment='top')
    ax2.xaxis.set_major_locator(ticker.MultipleLocator(steps))
    ax2.xaxis.set_minor_locator(AutoMinorLocator(4))
    ax2.yaxis.set_minor_locator(AutoMinorLocator(4))
    ax2.tick_params(axis="x", which="minor", direction="in", top=True, labeltop=False, bottom=True, labelbottom=False)
    ax2.tick_params(axis="y", which="minor", direction="in", left=True, labelleft=False, right=True, labelright=False)
    ax2.tick_params(axis="x", which="major", direction="in", top=True, labeltop=False, bottom=True, labelbottom=True)
    ax2.tick_params(axis="y", which="major", direction="in", left=True, labelleft=True, right=True, labelright=False)
    if (ax2.get_ylim()[1] - max_bin) < ax2.get_ylim()[1] / 6:
        ax2.set_ylim(ymax=ax2.get_ylim()[1] * 1.15)
    ax2.set_ylim(ymin=0.)
    ax2.set_xlim(xmin=lower, xmax=upper)
    ax2.set_xlabel(vars_dictionary[var]['label'] + vars_dictionary[var]['unit'], loc='right', fontsize=14)
    ax2.set_ylabel(r'$\Delta$N/N', loc='top', fontsize=14)
    ax2.legend(frameon=False)
    plt.setp(ax2.get_yticklabels()[0], visible=False)
    file_path = f"{path}/{var}.pdf"
    os.remove(file_path) if os.path.exists(file_path) else None
    fig2.savefig(file_path)
