# encoding: utf-8

from columnflow.tasks.production import ProduceColumnsWrapper
from columnflow.plotting.plot_all import draw_error_bands
import numpy as np
import os
import awkward as ak
import matplotlib.pyplot as plt
import mplhep
import hist
from hist import Hist

# t.law_run()
processes = ["'graviton_hh_vbf_bbtautau_m400'",
            "graviton_hh_ggf_bbtautau_m400",
            "tt",
            "dy"]

label_dict = {"graviton_hh_ggf_bbtautau_m400": 'Graviton $\\rightarrow HH_{ggf,m400}$ $\\rightarrow bb\\tau\\tau$',
              "graviton_hh_vbf_bbtautau_m400": 'Graviton $\\rightarrow HH_{vbf,m400}$ $\\rightarrow bb\\tau\\tau$',
              "hh_ggf_bbtautau": '$HH_{ggf} \\rightarrow bb\\tau\\tau$'}

collection_dict = {}
target_dict = {}
weights = []
DeepSetsInpPt = {}
DeepSetsInpEta = {}
fold_pred = {}

# get the target and predictions from ProduceColumnsWrapper
t = ProduceColumnsWrapper(
    version="NoKinSel",
)
from IPython import embed; embed()
files = t.output()
vbf_data = data = files[('run2_2017_nano_uhh_v11_limited', 'nominal',
'graviton_hh_vbf_bbtautau_m400_madgraph')].collection[0]['columns'].load(formatter="awkward")
ggf_sm_data = data = files[('run2_2017_nano_uhh_v11_limited', 'nominal',
'hh_ggf_bbtautau_madgraph')].collection[0]['columns'].load(formatter="awkward")
ggf_400_data = data = files[('run2_2017_nano_uhh_v11_limited', 'nominal',
'graviton_hh_ggf_bbtautau_m400_madgraph')].collection[0]['columns'].load(formatter="awkward")

# create the savepath for the plots
path = files[('run2_2017_nano_uhh_v11_limited', 'nominal', 'graviton_hh_vbf_bbtautau_m400_madgraph')
].collection[0]['columns'].sibling("dummy.json", type="f").path.split("dummy")[0]
path = os.path.join(path, "parton_plots")
if not os.path.exists(path):
    os.makedirs(path)

# Get the correct columns
parton_pt = np.concatenate((vbf_data['GenPartVBFparton1Pt'], vbf_data['GenPartVBFparton1Pt']))
parton_pt = ak.to_numpy(parton_pt)
parton_eta = np.stack((vbf_data['GenPartVBFparton1Eta'], vbf_data['GenPartVBFparton1Eta']), axis=1)
parton_eta = ak.to_numpy(parton_eta)

# start plotting (Pt)
plt.style.use(mplhep.style.CMS)
weights = np.ones_like(parton_pt) / len(parton_pt)


fig1, ax1 = plt.subplots()
cms_label = "wip"
label_options = {
    "wip": "Work in progress",
    "pre": "Preliminary",
    "pw": "Private work",
    "sim": "Simulation",
    "simwip": "Simulation work in progress",
    "simpre": "Simulation preliminary",
    "simpw": "Simulation private work",
    "od": "OpenData",
    "odwip": "OpenData work in progress",
    "odpw": "OpenData private work",
    "public": "",
}
cms_label_kwargs = {
    "ax": ax1,
    "llabel": label_options.get(cms_label, cms_label),
    "fontsize": 22,
    "data": False,
}
mplhep.cms.label(**cms_label_kwargs)
h = Hist(hist.axis.Regular(50, 0, 1000, name="S", label="s [units]"))
h.fill(parton_pt)
h.project("S").plot(ax=ax1)
draw_error_bands(ax1, h)
file_path = f"{path}/Parton_pt.pdf"
os.remove(file_path) if os.path.exists(file_path) else None
plt.savefig(file_path)
