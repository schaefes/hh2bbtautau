# coding: utf-8

"""
Wrappers for some default sets of producers.
"""

from columnflow.production import Producer, producer
from columnflow.production.normalization import normalization_weights
from columnflow.production.categories import category_ids
from columnflow.production.cms.electron import electron_weights
from columnflow.production.cms.muon import muon_weights
from columnflow.util import maybe_import

from hbt.production.features import features
from hbt.production.weights import normalized_pu_weight, normalized_pdf_weight, normalized_murmuf_weight
from hbt.production.btag import normalized_btag_weights
from hbt.production.tau import tau_weights, trigger_weights
from columnflow.production.util import attach_coffea_behavior


from columnflow.columnar_util import EMPTY_FLOAT, Route, set_ak_column
import functools


np = maybe_import("numpy")
ak = maybe_import("awkward")


set_ak_column_f32 = functools.partial(set_ak_column, value_type=np.float32)


# Helper func to compute invariant mass, delta eta of two objects, max dEta and the corresponing inv mass
def inv_mass_helper(obj1, obj2):
    obj_mass = (obj1 + obj2).mass

    return obj_mass


def d_eta_helper(obj1, obj2):
    d_eta = abs(obj1.eta - obj2.eta)

    return d_eta


def dEta_and_inv_mass(events_jetcollection, n_jets):
    deta_table = events_jetcollection.metric_table(events_jetcollection, axis=1, metric=delta_eta)
    inv_mass_table = events_jetcollection.metric_table(events_jetcollection, axis=1, metric=inv_mass)
    n_events = len(deta_table)
    max_dEta_vals = np.zeros(n_events)
    inv_mass_vals = np.zeros(n_events)
    max_inv_mass_vals = np.zeros(n_events)
    for i, deta_matrix in enumerate(deta_table):
        if n_jets[i] < 2:
            max_dEta_vals[i] = EMPTY_FLOAT
            inv_mass_vals[i] = EMPTY_FLOAT
        else:
            max_ax0 = ak.max(deta_matrix, axis=0)
            max_mjj = ak.max(inv_mass_table[i])
            argmax_ax0 = ak.argmax(deta_matrix, axis=0)
            max_idx1 = ak.argmax(max_ax0)
            max_idx0 = argmax_ax0[max_idx1]
            max_dEta = deta_matrix[max_idx0, max_idx1]
            inv_mass_val = inv_mass_table[i][max_idx0, max_idx1]
            max_dEta_vals[i] = max_dEta
            inv_mass_vals[i] = inv_mass_val
            max_inv_mass_vals[i] = max_mjj
    max_dEta_vals = ak.from_numpy(max_dEta_vals)
    inv_mass_vals = ak.from_numpy(inv_mass_vals)
    max_inv_mass_vals = ak.from_numpy(max_inv_mass_vals)

    return max_dEta_vals, inv_mass_vals, max_inv_mass_vals


# Producers
@producer(
    uses={
        "Jet.pt", "Jet.eta", "Jet.phi", "Jet.mass",
        attach_coffea_behavior,
    },
    produces={
        "mjj",
    },
)
def invariant_mass_jets(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    # category ids
    # invariant mass of two hardest jets
    events = self[attach_coffea_behavior](events, collections=["Jet"], **kwargs)
    events = set_ak_column(events, "Jet", ak.pad_none(events.Jet, 2))
    events = set_ak_column(events, "mjj", (events.Jet[:, 0] + events.Jet[:, 1]).mass)
    events = set_ak_column(events, "mjj", ak.fill_none(events.mjj, EMPTY_FLOAT))

    return events


@producer(
    uses={
        "Tau.pt", "Tau.eta", "Tau.phi", "Tau.mass",
        "Tau.charge",
        attach_coffea_behavior,
    },
    produces={
        "mtautau",
    },
)
def invariant_mass_tau(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    # invarinat mass of tau 1 and 2
    events = self[attach_coffea_behavior](events, collections=["Tau"], **kwargs)
    ditau = events.Tau[:, :2].sum(axis=1)
    ditau_mask = ak.num(events.Tau, axis=1) >= 2
    events = set_ak_column_f32(
        events,
        "mtautau",
        ak.where(ditau_mask, ditau.mass, EMPTY_FLOAT),
    )
    return events


@producer(
    uses={
        "BJet.pt", "BJet.eta", "BJet.phi", "BJet.mass",
        attach_coffea_behavior,
    },
    produces={
        "mbjetbjet",
    },
)
def invariant_mass_bjets(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    # invarinat mass of bjet 1 and 2, sums b jets with highest pt
    events = self[attach_coffea_behavior](events, collections={"BJet": {"type_name": "Jet"}}, **kwargs)
    diBJet = events.BJet[:, :2].sum(axis=1)
    diBJet_mask = ak.num(events.BJet, axis=1) >= 2
    events = set_ak_column_f32(
        events,
        "mbjetbjet",
        ak.where(diBJet_mask, diBJet.mass, EMPTY_FLOAT),
    )
    return events


@producer(
    uses={
        "BJet.pt", "BJet.eta", "BJet.phi", "BJet.mass",
        "Tau.pt", "Tau.eta", "Tau.phi", "Tau.mass", "Tau.charge",
        attach_coffea_behavior,
    },
    produces={
        "mHH",
    },
)
def invariant_mass_HH(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    # invarinat mass of bjet 1 and 2, sums b jets with highest pt
    events = self[attach_coffea_behavior](events, collections={"BJet": {"type_name": "Jet"}, "Tau": {"type_name": "Tau"}}, **kwargs)
    diHH = events.BJet[:, :2].sum(axis=1) + events.Tau[:, :2].sum(axis=1)
    dibjet_mask = ak.num(events.BJet, axis=1) >= 2
    ditau_mask = ak.num(events.Tau, axis=1) >= 2
    diHH_mask = np.logical_and(dibjet_mask, ditau_mask)
    events = set_ak_column_f32(
        events,
        "mHH",
        ak.where(diHH_mask, diHH.mass, EMPTY_FLOAT),
    )
    return events


# functions for usage in .metric_table
def delta_eta(obj1, obj2):
    obj_eta = abs(obj1.eta - obj2.eta)

    return obj_eta


def inv_mass(obj1, obj2):
    obj_mass = (obj1 + obj2).mass

    return obj_mass


def pt_product(obj1, obj2):
    obj_product = abs(obj1.pt * obj2.pt)

    return obj_product


@producer(
    uses={
        "CollJet.pt", "CollJet.eta", "CollJet.phi", "CollJet.mass",
        attach_coffea_behavior,
    },
    produces={
        "jets_max_dr", "jets_dr_inv_mass",
    },
)
def dr_inv_mass_jets(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    events = self[attach_coffea_behavior](events, collections={"CollJet": {"type_name": "Jet"}}, **kwargs)
    n_jets = events.n_jet
    dr_table = events.CollJet.metric_table(events.CollJet, axis=1)
    inv_mass_table = events.CollJet.metric_table(events.CollJet, axis=1, metric=inv_mass)
    n_events = len(dr_table)
    max_dr_vals = np.zeros(n_events)
    inv_mass_vals = np.zeros(n_events)
    for i, dr_matrix in enumerate(dr_table):
        if n_jets[i] < 2:
            max_dr_vals[i] = EMPTY_FLOAT
            inv_mass_vals[i] = EMPTY_FLOAT
        else:
            max_ax0 = ak.max(dr_matrix, axis=0)
            argmax_ax0 = ak.argmax(dr_matrix, axis=0)
            max_idx1 = ak.argmax(max_ax0)
            max_idx0 = argmax_ax0[max_idx1]
            max_dr = dr_matrix[max_idx0, max_idx1]
            inv_mass_val = inv_mass_table[i][max_idx0, max_idx1]
            max_dr_vals[i] = max_dr
            inv_mass_vals[i] = inv_mass_val
    max_delta_r_vals = ak.from_numpy(max_dr_vals)
    inv_mass_vals = ak.from_numpy(inv_mass_vals)
    events = set_ak_column_f32(events, "jets_max_dr", max_delta_r_vals)
    events = set_ak_column_f32(events, "jets_dr_inv_mass", inv_mass_vals)

    return events


@producer(
    uses={
        "CollJet.pt", "CollJet.eta", "CollJet.phi", "CollJet.mass",
        attach_coffea_behavior,
    },
    produces={
        "jets_max_d_eta", "jets_d_eta_inv_mass",
    },
)
# jets_dr_inv_mass: invariant mass calculated for the jet pair with largest dr
# jets_deta_inv_mass: invariant mass calculated for the jet pair with largest d eta
def d_eta_inv_mass_jets(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    events = self[attach_coffea_behavior](events, collections={"CollJet": {"type_name": "Jet"}}, **kwargs)
    n_jets = ak.count(events.CollJet.pt, axis=1)
    deta_table = events.CollJet.metric_table(events.CollJet, axis=1, metric=delta_eta)
    inv_mass_table = events.CollJet.metric_table(events.CollJet, axis=1, metric=inv_mass)
    n_events = len(deta_table)
    max_deta_vals = np.zeros(n_events)
    inv_mass_vals = np.zeros(n_events)
    for i, deta_matrix in enumerate(deta_table):
        if n_jets[i] < 2:
            max_deta_vals[i] = EMPTY_FLOAT
            inv_mass_vals[i] = EMPTY_FLOAT
        else:
            max_ax0 = ak.max(deta_matrix, axis=0)
            argmax_ax0 = ak.argmax(deta_matrix, axis=0)
            max_idx1 = ak.argmax(max_ax0)
            max_idx0 = argmax_ax0[max_idx1]
            max_dr = deta_matrix[max_idx0, max_idx1]
            inv_mass_val = inv_mass_table[i][max_idx0, max_idx1]
            max_deta_vals[i] = max_dr
            inv_mass_vals[i] = inv_mass_val
    max_delta_r_vals = ak.from_numpy(max_deta_vals)
    inv_mass_vals = ak.from_numpy(inv_mass_vals)
    events = set_ak_column_f32(events, "jets_max_d_eta", max_delta_r_vals)
    events = set_ak_column_f32(events, "jets_d_eta_inv_mass", inv_mass_vals)
    return events


@producer(
    uses={
        "CollJet.pt", "CollJet.eta", "CollJet.phi", "CollJet.mass",
        attach_coffea_behavior,
    },
    produces={
        "energy_corr",
    },
)
def energy_correlation(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    events = self[attach_coffea_behavior](events, collections={"CollJet": {"type_name": "Jet"}}, **kwargs)
    dr_table = events.CollJet.metric_table(events.CollJet, axis=1)
    pt_table = events.CollJet.metric_table(events.CollJet, axis=1, metric=pt_product)
    n_events = len(pt_table)
    energy_corr = np.zeros(n_events)
    for i, (pt, dr) in enumerate(zip(pt_table, dr_table)):
        prod = pt * dr
        corr = ak.sum(prod) / 2
        energy_corr[i] = corr
    energy_corr = ak.from_numpy(energy_corr)
    events = set_ak_column_f32(events, "energy_corr", energy_corr)

    return events


# kinematic vars for collection of jets and vbf jets
@producer(
    uses={
        "CollJet.pt", "CollJet.eta", "CollJet.phi", "CollJet.mass",
        "CollJet.btagDeepFlavB", "CollJet.hadronFlavour",
        attach_coffea_behavior,
    },
    produces={
        # *[f"{obj}_{var}"
        # for obj in [f"jet{n}" for n in range(1, 7, 1)]
        # for var in ["pt", "eta", "phi", "mass", "e", "btag", "hadronFlavour"]], "nJets", "nConstituents", "jets_pt",
        "nCollJets", "Colljets_pt", "Colljets_eta", "Colljets_phi", "Colljets_mass",
        "Colljets_btag", "Colljets_hadFlav", "n_jets", "ones_count_ds", "Colljets_e",
    },
)
def kinematic_vars_colljets(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    events = self[attach_coffea_behavior](events, collections={"CollJet": {"type_name": "Jet"}}, **kwargs)
    events = set_ak_column_f32(events, "nCollJets", ak.fill_none(events.n_jet, EMPTY_FLOAT))

    n_jets = ak.count(events.CollJet.pt, axis=1)
    events = set_ak_column_f32(events, "n_jets", n_jets)

    ht = ak.sum(abs(events.CollJet.pt), axis=1)
    events = set_ak_column_f32(events, "ht", ht)

    jets_pt = ak.pad_none(events.CollJet.pt, max(n_jets))
    jets_pt = ak.to_regular(jets_pt, axis=1)
    jets_pt_filled = ak.fill_none(jets_pt, EMPTY_FLOAT)
    events = set_ak_column_f32(events, "Colljets_pt", jets_pt_filled)

    jets_eta = ak.pad_none(events.CollJet.eta, max(n_jets))
    jets_eta = ak.to_regular(jets_eta, axis=1)
    jets_eta_filled = ak.fill_none(jets_eta, EMPTY_FLOAT)
    events = set_ak_column_f32(events, "Colljets_eta", jets_eta_filled)

    jets_phi = ak.pad_none(events.CollJet.phi, max(n_jets))
    jets_phi = ak.to_regular(jets_phi, axis=1)
    jets_phi_filled = ak.fill_none(jets_phi, EMPTY_FLOAT)
    events = set_ak_column_f32(events, "Colljets_phi", jets_phi_filled)

    jets_mass = ak.pad_none(events.CollJet.mass, max(n_jets))
    jets_mass = ak.to_regular(jets_mass, axis=1)
    jets_mass_filled = ak.fill_none(jets_mass, EMPTY_FLOAT)
    events = set_ak_column_f32(events, "Colljets_mass", jets_mass_filled)

    jets_btag = ak.pad_none(events.CollJet.btagDeepFlavB, max(n_jets))
    jets_btag = ak.to_regular(jets_btag, axis=1)
    jets_btag = ak.fill_none(jets_btag, EMPTY_FLOAT)
    events = set_ak_column_f32(events, "Colljets_btag", jets_btag)

    jets_hadFlav = ak.pad_none(events.CollJet.hadronFlavour, max(n_jets))
    jets_hadFlav = ak.to_regular(jets_hadFlav, axis=1)
    jets_hadFlav = ak.fill_none(jets_hadFlav, EMPTY_FLOAT)
    events = set_ak_column_f32(events, "Colljets_hadFlav", jets_hadFlav)

    ones_count_ds = ak.ones_like(events.CollJet.pt)
    ones_count_ds = ak.pad_none(ones_count_ds, max(n_jets))
    ones_count_ds = ak.to_regular(ones_count_ds, axis=1)
    ones_count_ds = ak.fill_none(ones_count_ds, EMPTY_FLOAT)
    events = set_ak_column_f32(events, "ones_count_ds", ones_count_ds)

    # Calculate energy
    p_x = jets_pt * np.cos(jets_phi)
    p_y = jets_pt * np.sin(jets_phi)
    p_z = jets_pt * np.sinh(jets_eta)
    p = np.sqrt(p_x**2 + p_y**2 + p_z**2)
    jets_e = np.sqrt(jets_mass**2 + p**2)
    jets_e = ak.fill_none(jets_e, EMPTY_FLOAT)
    events = set_ak_column_f32(events, "Colljets_e", jets_e)

    return events


# kinematic vars for collection of all Jets that passed the default VBF mask and the default Jet Selection
@producer(
    uses={
        "VBFMaskJets.pt", "VBFMaskJets.eta", "VBFMaskJets.phi", "VBFMaskJets.mass",
        "VBFMaskJets.btagDeepFlavB", "VBFMaskJets.hadronFlavour", "VBFMaskJets.btagDeepFlavCvL",
        "VBFMaskJets.btagDeepFlavQG", "VBFMaskJets.nConstituents",
        attach_coffea_behavior,
    },
    produces={
        "VBFMaskJets_njets", "VBFMaskJets_pt", "VBFMaskJets_eta", "VBFMaskJets_phi", "VBFMaskJets_mass",
        "VBFMaskJets_btag", "VBFMaskJets_hadFlav", "VBFMaskJets_ones", "VBFMaskJets_e",
        "VBFMaskJets_ht", "VBFMaskJets_mjj", "VBFMaskJets_max_dEta", "VBFMaskJets_mjj_dEta",
        "VBFMaskJets_btagCvL", "VBFMaskJets_btagQG", "VBFMaskJets_nConstituents",
    },
)
def kinematic_vars_vbfmaskjets(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    events = self[attach_coffea_behavior](events, collections={"VBFMaskJets": {"type_name": "Jet"}}, **kwargs)
    n_jets = ak.count(events.VBFMaskJets.pt, axis=1)
    events = set_ak_column_f32(events, "VBFMaskJets_njets", n_jets)
    max_njets = np.max(n_jets)

    # Get max dEta and the corresponing invariant mass
    max_dEta, inv_mass_dEta, max_inv_mass_vals = dEta_and_inv_mass(events.VBFMaskJets, n_jets)
    events = set_ak_column(events, "VBFMaskJets_max_dEta", ak.fill_none(max_dEta, EMPTY_FLOAT))
    events = set_ak_column(events, "VBFMaskJets_mjj_dEta", ak.fill_none(inv_mass_dEta, EMPTY_FLOAT))
    events = set_ak_column(events, "VBFMaskJets_mjj", ak.fill_none(max_inv_mass_vals, EMPTY_FLOAT))

    # use padded jets only for all following operations
    events = set_ak_column(events, "VBFMaskJets", ak.pad_none(events.VBFMaskJets, max_njets))

    ht = ak.sum(abs(events.VBFMaskJets.pt), axis=1)
    events = set_ak_column_f32(events, "VBFMaskJets_ht", ht)

    jets_pt = ak.to_regular(events.VBFMaskJets.pt, axis=1)
    jets_pt_filled = ak.fill_none(jets_pt, EMPTY_FLOAT)
    events = set_ak_column_f32(events, "VBFMaskJets_pt", jets_pt_filled)

    jets_eta = ak.to_regular(events.VBFMaskJets.eta, axis=1)
    jets_eta_filled = ak.fill_none(jets_eta, EMPTY_FLOAT)
    events = set_ak_column_f32(events, "VBFMaskJets_eta", jets_eta_filled)

    jets_phi = ak.to_regular(events.VBFMaskJets.phi, axis=1)
    jets_phi_filled = ak.fill_none(jets_phi, EMPTY_FLOAT)
    events = set_ak_column_f32(events, "VBFMaskJets_phi", jets_phi_filled)

    jets_mass = ak.to_regular(events.VBFMaskJets.mass, axis=1)
    jets_mass_filled = ak.fill_none(jets_mass, EMPTY_FLOAT)
    events = set_ak_column_f32(events, "VBFMaskJets_mass", jets_mass_filled)

    jets_btag = ak.to_regular(events.VBFMaskJets.btagDeepFlavB, axis=1)
    jets_btag = ak.fill_none(jets_btag, EMPTY_FLOAT)
    events = set_ak_column_f32(events, "VBFMaskJets_btag", jets_btag)

    jets_hadFlav = ak.to_regular(events.VBFMaskJets.hadronFlavour, axis=1)
    jets_hadFlav = ak.fill_none(jets_hadFlav, EMPTY_FLOAT)
    events = set_ak_column_f32(events, "VBFMaskJets_hadFlav", jets_hadFlav)

    jets_btagCvL = ak.to_regular(events.VBFMaskJets.btagDeepFlavCvL, axis=1)
    jets_btagCvL = ak.fill_none(jets_btagCvL, EMPTY_FLOAT)
    events = set_ak_column_f32(events, "VBFMaskJets_btagCvL", jets_btagCvL)

    jets_btagQG = ak.to_regular(events.VBFMaskJets.btagDeepFlavQG, axis=1)
    jets_btagQG = ak.fill_none(jets_btagQG, EMPTY_FLOAT)
    events = set_ak_column_f32(events, "VBFMaskJets_btagQG", jets_btagQG)

    jets_nConstituents = ak.to_regular(events.VBFMaskJets.nConstituents, axis=1)
    jets_nConstituents = ak.fill_none(jets_nConstituents, EMPTY_FLOAT)
    events = set_ak_column_f32(events, "VBFMaskJets_nConstituents", jets_nConstituents)

    ones_count_ds = ak.ones_like(events.VBFMaskJets.pt)
    ones_count_ds = ak.pad_none(ones_count_ds, max(n_jets))
    ones_count_ds = ak.to_regular(ones_count_ds, axis=1)
    ones_count_ds = ak.fill_none(ones_count_ds, EMPTY_FLOAT)
    events = set_ak_column_f32(events, "VBFMaskJets_ones", ones_count_ds)

    # Calculate energy
    p_x = jets_pt * np.cos(jets_phi)
    p_y = jets_pt * np.sin(jets_phi)
    p_z = jets_pt * np.sinh(jets_eta)
    p = np.sqrt(p_x**2 + p_y**2 + p_z**2)
    jets_e = np.sqrt(jets_mass**2 + p**2)
    jets_e = ak.fill_none(jets_e, EMPTY_FLOAT)
    events = set_ak_column_f32(events, "VBFMaskJets_e", jets_e)

    return events


# kinematic vars for collection of all Jets that passed the custom VBF mask and the default Jet Selection
@producer(
    uses={
        "CustomVBFMaskJets.pt", "CustomVBFMaskJets.eta", "CustomVBFMaskJets.phi", "CustomVBFMaskJets.mass",
        "CustomVBFMaskJets.btagDeepFlavB", "CustomVBFMaskJets.hadronFlavour",
        "CustomVBFMaskJets.btagDeepFlavCvL", "CustomVBFMaskJets.btagDeepFlavQG",
        "CustomVBFMaskJets.nConstituents", "MET.*", "CustomVBFMaskJets.px",
        "CustomVBFMaskJets.py", "CustomVBFMaskJets.pz",
        attach_coffea_behavior,
    },
    produces={
        "CustomVBFMaskJets_njets", "CustomVBFMaskJets_pt", "CustomVBFMaskJets_eta", "CustomVBFMaskJets_phi",
        "CustomVBFMaskJets_mass", "CustomVBFMaskJets_btag", "CustomVBFMaskJets_hadFlav",
        "CustomVBFMaskJets_ones", "CustomVBFMaskJets_e", "CustomVBFMaskJets_ht",
        "CustomVBFMaskJets_mjj", "CustomVBFMaskJets_max_dEta", "CustomVBFMaskJets_mjj_dEta",
        "CustomVBFMaskJets_btagCvL", "CustomVBFMaskJets_btagQG", "CustomVBFMaskJets_nConstituents",
        "CustomVBFMaskJets_METphi", "CustomVBFMaskJets_px", "CustomVBFMaskJets_py",
        "CustomVBFMaskJets_pz",
    },
)
def kinematic_vars_customvbfmaskjets(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    events = self[attach_coffea_behavior](events, collections={"CustomVBFMaskJets": {"type_name": "Jet"}}, **kwargs)
    n_jets = ak.count(events.CustomVBFMaskJets.pt, axis=1)
    events = set_ak_column_f32(events, "CustomVBFMaskJets_njets", n_jets)
    max_njets = np.max(n_jets)

    # Get max dEta and the corresponing invariant mass
    max_dEta, inv_mass_dEta, max_inv_mass_vals = dEta_and_inv_mass(events.CustomVBFMaskJets, n_jets)
    events = set_ak_column(events, "CustomVBFMaskJets_max_dEta", ak.fill_none(max_dEta, EMPTY_FLOAT))
    events = set_ak_column(events, "CustomVBFMaskJets_mjj_dEta", ak.fill_none(inv_mass_dEta, EMPTY_FLOAT))
    events = set_ak_column(events, "CustomVBFMaskJets_mjj", ak.fill_none(max_inv_mass_vals, EMPTY_FLOAT))

    # use padded jets only for all following operations
    events = set_ak_column(events, "CustomVBFMaskJets", ak.pad_none(events.CustomVBFMaskJets, max_njets))

    ht = ak.sum(abs(events.CustomVBFMaskJets.pt), axis=1)
    events = set_ak_column_f32(events, "CustomVBFMaskJets_ht", ht)

    jets_pt = ak.to_regular(events.CustomVBFMaskJets.pt, axis=1)
    jets_pt_filled = ak.fill_none(jets_pt, EMPTY_FLOAT)
    events = set_ak_column_f32(events, "CustomVBFMaskJets_pt", jets_pt_filled)

    jets_px = ak.to_regular(events.CustomVBFMaskJets.px, axis=1)
    jets_px_filled = ak.fill_none(jets_px, EMPTY_FLOAT)
    events = set_ak_column_f32(events, "CustomVBFMaskJets_px", jets_px_filled)

    jets_py = ak.to_regular(events.CustomVBFMaskJets.py, axis=1)
    jets_py_filled = ak.fill_none(jets_py, EMPTY_FLOAT)
    events = set_ak_column_f32(events, "CustomVBFMaskJets_py", jets_py_filled)

    jets_pz = ak.to_regular(events.CustomVBFMaskJets.pz, axis=1)
    jets_pz_filled = ak.fill_none(jets_pz, EMPTY_FLOAT)
    events = set_ak_column_f32(events, "CustomVBFMaskJets_pz", jets_pz_filled)

    jets_eta = ak.to_regular(events.CustomVBFMaskJets.eta, axis=1)
    jets_eta_filled = ak.fill_none(jets_eta, EMPTY_FLOAT)
    events = set_ak_column_f32(events, "CustomVBFMaskJets_eta", jets_eta_filled)

    jets_phi = ak.to_regular(events.CustomVBFMaskJets.phi, axis=1)
    jets_phi_filled = ak.fill_none(jets_phi, EMPTY_FLOAT)
    events = set_ak_column_f32(events, "CustomVBFMaskJets_phi", jets_phi_filled)

    jets_mass = ak.to_regular(events.CustomVBFMaskJets.mass, axis=1)
    jets_mass_filled = ak.fill_none(jets_mass, EMPTY_FLOAT)
    events = set_ak_column_f32(events, "CustomVBFMaskJets_mass", jets_mass_filled)

    jets_btag = ak.to_regular(events.CustomVBFMaskJets.btagDeepFlavB, axis=1)
    jets_btag = ak.fill_none(jets_btag, EMPTY_FLOAT)
    events = set_ak_column_f32(events, "CustomVBFMaskJets_btag", jets_btag)

    jets_hadFlav = ak.to_regular(events.CustomVBFMaskJets.hadronFlavour, axis=1)
    jets_hadFlav = ak.fill_none(jets_hadFlav, EMPTY_FLOAT)
    events = set_ak_column_f32(events, "CustomVBFMaskJets_hadFlav", jets_hadFlav)

    jets_btagCvL = ak.to_regular(events.CustomVBFMaskJets.btagDeepFlavCvL, axis=1)
    jets_btagCvL = ak.fill_none(jets_btagCvL, EMPTY_FLOAT)
    events = set_ak_column_f32(events, "CustomVBFMaskJets_btagCvL", jets_btagCvL)

    jets_btagQG = ak.to_regular(events.CustomVBFMaskJets.btagDeepFlavQG, axis=1)
    jets_btagQG = ak.fill_none(jets_btagQG, EMPTY_FLOAT)
    events = set_ak_column_f32(events, "CustomVBFMaskJets_btagQG", jets_btagQG)

    jets_nConstituents = ak.to_regular(events.CustomVBFMaskJets.nConstituents, axis=1)
    jets_nConstituents = ak.fill_none(jets_nConstituents, EMPTY_FLOAT)
    events = set_ak_column_f32(events, "CustomVBFMaskJets_nConstituents", jets_nConstituents)

    ones_count_ds = ak.ones_like(events.CustomVBFMaskJets.pt)
    ones_count_ds = ak.pad_none(ones_count_ds, max(n_jets))
    ones_count_ds = ak.to_regular(ones_count_ds, axis=1)
    ones_count_ds = ak.fill_none(ones_count_ds, EMPTY_FLOAT)
    events = set_ak_column_f32(events, "CustomVBFMaskJets_ones", ones_count_ds)

    # Calculate energy
    p_x = jets_pt * np.cos(jets_phi)
    p_y = jets_pt * np.sin(jets_phi)
    p_z = jets_pt * np.sinh(jets_eta)
    p = np.sqrt(p_x**2 + p_y**2 + p_z**2)
    jets_e = np.sqrt(jets_mass**2 + p**2)
    jets_e = ak.fill_none(jets_e, EMPTY_FLOAT)
    events = set_ak_column_f32(events, "CustomVBFMaskJets_e", jets_e)

    # save the met phi to use for projection of the jet phis
    events = set_ak_column_f32(events, "CustomVBFMaskJets_METphi", ak.fill_none(events.MET.phi, EMPTY_FLOAT))

    return events


# kinematic vars for collection of all Jets that passed the custom VBF mask and the default Jet Selection
@producer(
    uses={
        "CustomVBFMaskJets2.pt", "CustomVBFMaskJets2.eta", "CustomVBFMaskJets2.phi", "CustomVBFMaskJets2.mass",
        "CustomVBFMaskJets2.btagDeepFlavB", "CustomVBFMaskJets2.hadronFlavour",
        "CustomVBFMaskJets2.btagDeepFlavCvL", "CustomVBFMaskJets2.btagDeepFlavQG",
        "CustomVBFMaskJets2.nConstituents", "MET.*","CustomVBFMaskJets2.btagDeepCvB",
        "CustomVBFMaskJets2.btagDeepFlavCvB", "CustomVBFMaskJets2.btagDeepCvL",
        "CustomVBFMaskJets2.btagDeepB",
        attach_coffea_behavior,
    },
    produces={
        "CustomVBFMaskJets2_njets", "CustomVBFMaskJets2_pt", "CustomVBFMaskJets2_eta", "CustomVBFMaskJets2_phi",
        "CustomVBFMaskJets2_mass", "CustomVBFMaskJets2_btag", "CustomVBFMaskJets2_hadFlav",
        "CustomVBFMaskJets2_ones", "CustomVBFMaskJets2_e", "CustomVBFMaskJets2_ht",
        "CustomVBFMaskJets2_mjj", "CustomVBFMaskJets2_max_dEta", "CustomVBFMaskJets2_mjj_dEta",
        "CustomVBFMaskJets2_btagCvL", "CustomVBFMaskJets2_btagCvB",
        "CustomVBFMaskJets2_btagQG", "CustomVBFMaskJets2_nConstituents",
        "CustomVBFMaskJets2_METphi", "CustomVBFMaskJets2_bFlavtagCvL",
        "CustomVBFMaskJets2_bFlavtagCvB", "CustomVBFMaskJets2_bFlavtag", "CustomVBFMaskJets2_px",
        "CustomVBFMaskJets2_py", "CustomVBFMaskJets2_pz",
    },
)
def kinematic_vars_customvbfmaskjets2(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    events = self[attach_coffea_behavior](events, collections={"CustomVBFMaskJets2": {"type_name": "Jet"}}, **kwargs)
    n_jets = ak.count(events.CustomVBFMaskJets2.pt, axis=1)
    events = set_ak_column_f32(events, "CustomVBFMaskJets2_njets", n_jets)
    max_njets = np.max(n_jets)

    # Get max dEta and the corresponing invariant mass
    max_dEta, inv_mass_dEta, max_inv_mass_vals = dEta_and_inv_mass(events.CustomVBFMaskJets2, n_jets)
    events = set_ak_column(events, "CustomVBFMaskJets2_max_dEta", ak.fill_none(max_dEta, EMPTY_FLOAT))
    events = set_ak_column(events, "CustomVBFMaskJets2_mjj_dEta", ak.fill_none(inv_mass_dEta, EMPTY_FLOAT))
    events = set_ak_column(events, "CustomVBFMaskJets2_mjj", ak.fill_none(max_inv_mass_vals, EMPTY_FLOAT))

    # use padded jets only for all following operations
    events = set_ak_column(events, "CustomVBFMaskJets2", ak.pad_none(events.CustomVBFMaskJets2, max_njets))

    ht = ak.sum(abs(events.CustomVBFMaskJets2.pt), axis=1)
    events = set_ak_column_f32(events, "CustomVBFMaskJets2_ht", ht)

    jets_pt = ak.to_regular(events.CustomVBFMaskJets2.pt, axis=1)
    jets_pt_filled = ak.fill_none(jets_pt, EMPTY_FLOAT)
    events = set_ak_column_f32(events, "CustomVBFMaskJets2_pt", jets_pt_filled)

    jets_eta = ak.to_regular(events.CustomVBFMaskJets2.eta, axis=1)
    jets_eta_filled = ak.fill_none(jets_eta, EMPTY_FLOAT)
    events = set_ak_column_f32(events, "CustomVBFMaskJets2_eta", jets_eta_filled)

    jets_phi = ak.to_regular(events.CustomVBFMaskJets2.phi, axis=1)
    jets_phi_filled = ak.fill_none(jets_phi, EMPTY_FLOAT)
    events = set_ak_column_f32(events, "CustomVBFMaskJets2_phi", jets_phi_filled)

    jets_mass = ak.to_regular(events.CustomVBFMaskJets2.mass, axis=1)
    jets_mass_filled = ak.fill_none(jets_mass, EMPTY_FLOAT)
    events = set_ak_column_f32(events, "CustomVBFMaskJets2_mass", jets_mass_filled)

    jets_hadFlav = ak.to_regular(events.CustomVBFMaskJets2.hadronFlavour, axis=1)
    jets_hadFlav = ak.fill_none(jets_hadFlav, EMPTY_FLOAT)
    events = set_ak_column_f32(events, "CustomVBFMaskJets2_hadFlav", jets_hadFlav)

    jets_bFlavtag = ak.to_regular(events.CustomVBFMaskJets2.btagDeepFlavB, axis=1)
    jets_bFlavtag = ak.fill_none(jets_bFlavtag, EMPTY_FLOAT)
    events = set_ak_column_f32(events, "CustomVBFMaskJets2_bFlavtag", jets_bFlavtag)

    jets_bFlavtagCvL = ak.to_regular(events.CustomVBFMaskJets2.btagDeepFlavCvL, axis=1)
    jets_bFlavtagCvL = ak.fill_none(jets_bFlavtagCvL, EMPTY_FLOAT)
    events = set_ak_column_f32(events, "CustomVBFMaskJets2_bFlavtagCvL", jets_bFlavtagCvL)

    jets_bFlavtagCvB = ak.to_regular(events.CustomVBFMaskJets2.btagDeepFlavCvB, axis=1)
    jets_bFlavtagCvB = ak.fill_none(jets_bFlavtagCvB, EMPTY_FLOAT)
    events = set_ak_column_f32(events, "CustomVBFMaskJets2_bFlavtagCvB", jets_bFlavtagCvB)

    jets_btagQG = ak.to_regular(events.CustomVBFMaskJets2.btagDeepFlavQG, axis=1)
    jets_btagQG = ak.fill_none(jets_btagQG, EMPTY_FLOAT)
    events = set_ak_column_f32(events, "CustomVBFMaskJets2_btagQG", jets_btagQG)

    jets_btag = ak.to_regular(events.CustomVBFMaskJets2.btagDeepB, axis=1)
    jets_btag = ak.fill_none(jets_btag, EMPTY_FLOAT)
    events = set_ak_column_f32(events, "CustomVBFMaskJets2_btag", jets_btag)

    jets_btagCvL = ak.to_regular(events.CustomVBFMaskJets2.btagDeepCvL, axis=1)
    jets_btagCvL = ak.fill_none(jets_btagCvL, EMPTY_FLOAT)
    events = set_ak_column_f32(events, "CustomVBFMaskJets2_btagCvL", jets_btagCvL)

    jets_btagCvB = ak.to_regular(events.CustomVBFMaskJets2.btagDeepCvB, axis=1)
    jets_btagCvB = ak.fill_none(jets_btagCvB, EMPTY_FLOAT)
    events = set_ak_column_f32(events, "CustomVBFMaskJets2_btagCvB", jets_btagCvB)

    jets_nConstituents = ak.to_regular(events.CustomVBFMaskJets2.nConstituents, axis=1)
    jets_nConstituents = ak.fill_none(jets_nConstituents, EMPTY_FLOAT)
    events = set_ak_column_f32(events, "CustomVBFMaskJets2_nConstituents", jets_nConstituents)

    ones_count_ds = ak.ones_like(events.CustomVBFMaskJets2.pt)
    ones_count_ds = ak.pad_none(ones_count_ds, max(n_jets))
    ones_count_ds = ak.to_regular(ones_count_ds, axis=1)
    ones_count_ds = ak.fill_none(ones_count_ds, EMPTY_FLOAT)
    events = set_ak_column_f32(events, "CustomVBFMaskJets2_ones", ones_count_ds)

    # Calculate energy
    p_x = jets_pt * np.cos(jets_phi)
    p_y = jets_pt * np.sin(jets_phi)
    p_z = jets_pt * np.sinh(jets_eta)
    p = np.sqrt(p_x**2 + p_y**2 + p_z**2)
    jets_e = np.sqrt(jets_mass**2 + p**2)
    jets_e = ak.fill_none(jets_e, EMPTY_FLOAT)
    events = set_ak_column_f32(events, "CustomVBFMaskJets2_e", jets_e)
    jets_px = ak.fill_none(p_x, EMPTY_FLOAT)
    events = set_ak_column_f32(events, "CustomVBFMaskJets2_px", jets_px)
    jets_py = ak.fill_none(p_y, EMPTY_FLOAT)
    events = set_ak_column_f32(events, "CustomVBFMaskJets2_py", jets_py)
    jets_pz = ak.fill_none(p_z, EMPTY_FLOAT)
    events = set_ak_column_f32(events, "CustomVBFMaskJets2_pz", jets_pz)

    # save the met phi to use for projection of the jet phis
    events = set_ak_column_f32(events, "CustomVBFMaskJets2_METphi", ak.fill_none(events.MET.phi, EMPTY_FLOAT))

    return events


@producer(
    uses={
        "BJet.pt", "BJet.eta", "BJet.phi", "BJet.mass", "BJet.E",
        attach_coffea_behavior,
    },
    produces={
        f"{obj}_{var}"
        for obj in ["bjet1", "bjet2"]
        for var in ["pt", "eta", "phi", "mass", "e"]
    },
)
def kinematic_vars_bjets(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    # BJet 1 and 2 kinematic variables
    # BJet 1
    events = self[attach_coffea_behavior](events, collections={"BJet": {"type_name": "Jet"}}, **kwargs)
    events = set_ak_column_f32(events, "bjet1_mass", Route("BJet.mass[:,0]").apply(events, EMPTY_FLOAT))
    events = set_ak_column_f32(events, "bjet1_pt", Route("BJet.pt[:,0]").apply(events, EMPTY_FLOAT))
    events = set_ak_column_f32(events, "bjet1_eta", Route("BJet.eta[:,0]").apply(events, EMPTY_FLOAT))
    events = set_ak_column_f32(events, "bjet1_phi", Route("BJet.phi[:,0]").apply(events, EMPTY_FLOAT))

    # BJet 2
    events = set_ak_column_f32(events, "bjet2_mass", Route("BJet.mass[:,1]").apply(events, EMPTY_FLOAT))
    events = set_ak_column_f32(events, "bjet2_pt", Route("BJet.pt[:,1]").apply(events, EMPTY_FLOAT))
    events = set_ak_column_f32(events, "bjet2_eta", Route("BJet.eta[:,1]").apply(events, EMPTY_FLOAT))
    events = set_ak_column_f32(events, "bjet2_phi", Route("BJet.phi[:,1]").apply(events, EMPTY_FLOAT))
    # get energy seperately
    bjets_e = ak.pad_none(events.BJet.E, 2, axis=1)
    events = set_ak_column_f32(events, "bjet1_e", ak.fill_none(bjets_e[:, 0], EMPTY_FLOAT))
    events = set_ak_column_f32(events, "bjet2_e", ak.fill_none(bjets_e[:, 1], EMPTY_FLOAT))
    return events


@producer(
    uses={
        "BJet.pt", "BJet.eta", "BJet.phi", "BJet.mass",
        attach_coffea_behavior,
    },
    produces={
        "BJetsdR", "BJetsdEta",
    },
)
def dR_bjets(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    events = self[attach_coffea_behavior](events, collections={"BJet": {"type_name": "Jet"}}, **kwargs)
    events = set_ak_column(events, "BJet", ak.pad_none(events.BJet, 2))
    jets_eta = ak.pad_none(events.BJet.eta, 2)
    jets_eta = ak.to_regular(jets_eta[:, :2], axis=1)
    jets_phi = ak.pad_none(events.BJet.phi, 2)
    jets_phi = ak.to_regular(jets_phi[:, :2], axis=1)
    d_eta = abs(jets_eta[:, 0] - jets_eta[:, 1])
    events = set_ak_column_f32(events, "BJetsdEta", ak.fill_none(d_eta, EMPTY_FLOAT))
    dR = np.sqrt(d_eta**2 + abs(jets_phi[:, 0] - jets_phi[:, 1])**2)
    events = set_ak_column_f32(events, "BJetsdR", ak.fill_none(dR, EMPTY_FLOAT))

    return events


@producer(
    uses={
        "Tau.pt", "Tau.eta", "Tau.phi", "Tau.mass",
        attach_coffea_behavior,
    },
    produces={
        "TaudR", "TaudEta",
    },
)
def dR_tau(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    events = self[attach_coffea_behavior](events, collections=["Tau"], **kwargs)
    events = set_ak_column(events, "Tau", ak.pad_none(events.Tau, 2))
    jets_eta = ak.pad_none(events.Tau.eta, 2)
    jets_eta = ak.to_regular(jets_eta[:, :2], axis=1)
    jets_phi = ak.pad_none(events.Tau.phi, 2)
    jets_phi = ak.to_regular(jets_phi[:, :2], axis=1)
    d_eta = abs(jets_eta[:, 0] - jets_eta[:, 1])
    d_eta = ak.fill_none(d_eta, EMPTY_FLOAT)
    events = set_ak_column_f32(events, "TaudEta", d_eta)
    dR = np.sqrt(d_eta**2 + abs(jets_phi[:, 0] - jets_phi[:, 1])**2)
    dR = ak.fill_none(dR, EMPTY_FLOAT)
    events = set_ak_column_f32(events, "TaudR", dR)

    return events


@producer(
    uses={
        "Tau.pt", "Tau.eta", "Tau.phi", "Tau.mass",
        "Tau.charge",
        attach_coffea_behavior,
    },
    produces={
        f"{obj}_{var}"
        for obj in ["tau1", "tau2"]
        for var in ["pt", "eta", "phi", "mass", "e"]
    },
)
def kinematic_vars_taus(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    # Tau 1 and 2 kinematic variables
    # Tau 1
    events = self[attach_coffea_behavior](events, collections=["Tau"], **kwargs)
    events = set_ak_column_f32(events, "tau1_mass", Route("Tau.mass[:,0]").apply(events, EMPTY_FLOAT))
    events = set_ak_column_f32(events, "tau1_pt", Route("Tau.pt[:,0]").apply(events, EMPTY_FLOAT))
    events = set_ak_column_f32(events, "tau1_eta", Route("Tau.eta[:,0]").apply(events, EMPTY_FLOAT))
    events = set_ak_column_f32(events, "tau1_phi", Route("Tau.phi[:,0]").apply(events, EMPTY_FLOAT))

    # Tau 2
    events = set_ak_column_f32(events, "tau2_mass", Route("Tau.mass[:,1]").apply(events, EMPTY_FLOAT))
    events = set_ak_column_f32(events, "tau2_pt", Route("Tau.pt[:,1]").apply(events, EMPTY_FLOAT))
    events = set_ak_column_f32(events, "tau2_eta", Route("Tau.eta[:,1]").apply(events, EMPTY_FLOAT))
    events = set_ak_column_f32(events, "tau2_phi", Route("Tau.phi[:,1]").apply(events, EMPTY_FLOAT))
    # get energy seperately
    taus_e = ak.pad_none(events.Tau.E, 2, axis=1)
    events = set_ak_column_f32(events, "tau1_e", ak.fill_none(taus_e[:, 0], EMPTY_FLOAT))
    events = set_ak_column_f32(events, "tau2_e", ak.fill_none(taus_e[:, 1], EMPTY_FLOAT))
    return events


# Producers for aditinal bjet exclusive features, dummy fills and one Hot encoding
@producer(
    uses={
        "BJet.btagDeepFlavB", "BJet.mass",
        attach_coffea_behavior,
    },
    produces={
        *[f"{obj}_{var}"
        for obj in ["bjet1", "bjet2"]
        for var in ["btag", "DeepTau_e", "jet_oneHot", "bjet_oneHot", "tau_oneHot", "object_type"]]
    },
)
def bjet_information(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    # Produce bjet exclusive features
    events = self[attach_coffea_behavior](events, collections={"BJet": {"type_name": "Jet"}}, **kwargs)
    events = set_ak_column_f32(events, "bjet1_btag", Route("BJet.btagDeepFlavB[:,0]").apply(events, EMPTY_FLOAT))
    events = set_ak_column_f32(events, "bjet2_btag", Route("BJet.btagDeepFlavB[:,1]").apply(events, EMPTY_FLOAT))

    # Produce dummy fills for non bjet features
    padded_mass = ak.pad_none(events.BJet.mass, 2, axis=1)
    dummy_fill_1 = ak.full_like(padded_mass[:, 0], -1)
    dummy_fill_2 = ak.full_like(padded_mass[:, 1], -1)
    events = set_ak_column_f32(events, "bjet1_DeepTau_e", ak.fill_none(dummy_fill_1, EMPTY_FLOAT))
    events = set_ak_column_f32(events, "bjet2_DeepTau_e", ak.fill_none(dummy_fill_2, EMPTY_FLOAT))

    # Create one Hot encoding
    oneHot_fill_0 = ak.full_like(padded_mass[:, 0], 0)
    oneHot_fill_1 = ak.full_like(padded_mass[:, 1], 1)
    events = set_ak_column_f32(events, "bjet1_jet_oneHot", ak.fill_none(oneHot_fill_0, EMPTY_FLOAT))
    events = set_ak_column_f32(events, "bjet2_jet_oneHot", ak.fill_none(oneHot_fill_0, EMPTY_FLOAT))
    events = set_ak_column_f32(events, "bjet1_tau_oneHot", ak.fill_none(oneHot_fill_0, EMPTY_FLOAT))
    events = set_ak_column_f32(events, "bjet2_tau_oneHot", ak.fill_none(oneHot_fill_0, EMPTY_FLOAT))
    events = set_ak_column_f32(events, "bjet1_bjet_oneHot", ak.fill_none(oneHot_fill_1, EMPTY_FLOAT))
    events = set_ak_column_f32(events, "bjet2_bjet_oneHot", ak.fill_none(oneHot_fill_1, EMPTY_FLOAT))

    # Create object type integer
    object_type_1 = ak.full_like(padded_mass[:, 0], 2)
    object_type_2 = ak.full_like(padded_mass[:, 1], 2)
    events = set_ak_column_f32(events, "bjet1_object_type", ak.fill_none(object_type_1, EMPTY_FLOAT))
    events = set_ak_column_f32(events, "bjet2_object_type", ak.fill_none(object_type_2, EMPTY_FLOAT))

    return events


# Producers for aditinal tau exclusive features, dummy fills and one Hot encoding
@producer(
    uses={
        "Tau.idDeepTau2017v2p1VSe", "Tau.idDeepTau2017v2p1VSmu", "Tau.idDeepTau2017v2p1VSjet",
        "Tau.decayMode", "Tau.mass",
        attach_coffea_behavior,
    },
    produces={
        *[f"{obj}_{var}"
        for obj in ["tau1", "tau2"]
        for var in ["DeepTau_e", "DeepTau_mu", "DeepTau_jet", "decayMode", "btag", "jet_oneHot",
        "bjet_oneHot", "tau_oneHot", "object_type"]]
    },
)
def tau_information(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    # Produce tau exclusive features
    events = self[attach_coffea_behavior](events, collections=["Tau"], **kwargs)
    events = set_ak_column_f32(events, "tau1_DeepTau_e", Route("Tau.idDeepTau2017v2p1VSe[:,0]").apply(events, EMPTY_FLOAT))
    events = set_ak_column_f32(events, "tau2_DeepTau_e", Route("Tau.idDeepTau2017v2p1VSe[:,1]").apply(events, EMPTY_FLOAT))
    events = set_ak_column_f32(events, "tau1_DeepTau_mu", Route("Tau.idDeepTau2017v2p1VSmu[:,0]").apply(events, EMPTY_FLOAT))
    events = set_ak_column_f32(events, "tau2_DeepTau_mu", Route("Tau.idDeepTau2017v2p1VSmu[:,1]").apply(events, EMPTY_FLOAT))
    events = set_ak_column_f32(events, "tau1_DeepTau_jet", Route("Tau.idDeepTau2017v2p1VSjet[:,0]").apply(events, EMPTY_FLOAT))
    events = set_ak_column_f32(events, "tau2_DeepTau_jet", Route("Tau.idDeepTau2017v2p1VSjet[:,1]").apply(events, EMPTY_FLOAT))
    events = set_ak_column_f32(events, "tau1_decayMode", Route("Tau.idDeepTau2017v2p1VSjet[:,0]").apply(events, EMPTY_FLOAT))
    events = set_ak_column_f32(events, "tau2_decayMode", Route("Tau.idDeepTau2017v2p1VSjet[:,1]").apply(events, EMPTY_FLOAT))

    # Produce dummy fills for non tau features
    padded_mass = ak.pad_none(events.Tau.mass, 2, axis=1)
    dummy_fill_1 = ak.full_like(padded_mass[:, 0], -1)
    dummy_fill_2 = ak.full_like(padded_mass[:, 1], -1)
    events = set_ak_column_f32(events, "tau1_btag", ak.fill_none(dummy_fill_1, EMPTY_FLOAT))
    events = set_ak_column_f32(events, "tau2_btag", ak.fill_none(dummy_fill_2, EMPTY_FLOAT))

    # Create one Hot encoding
    oneHot_fill_0 = ak.full_like(padded_mass[:, 0], 0)
    oneHot_fill_1 = ak.full_like(padded_mass[:, 1], 1)
    events = set_ak_column_f32(events, "tau1_jet_oneHot", ak.fill_none(oneHot_fill_0, EMPTY_FLOAT))
    events = set_ak_column_f32(events, "tau2_jet_oneHot", ak.fill_none(oneHot_fill_0, EMPTY_FLOAT))
    events = set_ak_column_f32(events, "tau1_bjet_oneHot", ak.fill_none(oneHot_fill_0, EMPTY_FLOAT))
    events = set_ak_column_f32(events, "tau2_bjet_oneHot", ak.fill_none(oneHot_fill_0, EMPTY_FLOAT))
    events = set_ak_column_f32(events, "tau1_tau_oneHot", ak.fill_none(oneHot_fill_1, EMPTY_FLOAT))
    events = set_ak_column_f32(events, "tau2_tau_oneHot", ak.fill_none(oneHot_fill_1, EMPTY_FLOAT))

    # Create object type integer
    object_type_1 = ak.full_like(padded_mass[:, 0], 3)
    object_type_2 = ak.full_like(padded_mass[:, 1], 3)
    events = set_ak_column_f32(events, "tau1_object_type", ak.fill_none(object_type_1, EMPTY_FLOAT))
    events = set_ak_column_f32(events, "tau2_object_type", ak.fill_none(object_type_2, EMPTY_FLOAT))

    return events
