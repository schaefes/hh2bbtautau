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


# Helper func to compute invariant mass, delta eta of two objects
def inv_mass_helper(obj1, obj2):
    obj_mass = (obj1 + obj2).mass

    return obj_mass


def d_eta_helper(obj1, obj2):
    d_eta = abs(obj1.eta - obj2.eta)

    return d_eta


# Invariant Mass Producers
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


# Producers for the columns of the kinetmatic variables (four vectors) of the jets, bjets and taus
@producer(
    uses={
        "Jet.pt", "Jet.nJet", "Jet.eta", "Jet.phi", "Jet.mass", "Jet.area",# "Jet.E",
        "Jet.nConstituents", "Jet.jetID", "Jet.btagDeepFlavB", "Jet.hadronFlavour",
        attach_coffea_behavior,
    },
    produces={
        # *[f"{obj}_{var}"
        # for obj in [f"jet{n}" for n in range(1, 7, 1)]
        # for var in ["pt", "eta", "phi", "mass", "e", "btag", "hadronFlavour"]], "nJets", "nConstituents", "jets_pt",
        "nJets", "jets_pt", "jets_e", "jets_eta", "jets_phi", "jets_mass", "jets_btag", "jets_hadFlav",
    },
)
def kinematic_vars_jets(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    events = self[attach_coffea_behavior](events, collections=["Jet"], **kwargs)
    events = set_ak_column_f32(events, "nJets", ak.fill_none(events.n_jet, EMPTY_FLOAT))
    # jets_mass = ak.pad_none(events.Jet.mass, 6, axis=1)
    # jets_e = ak.pad_none(events.Jet.E, 6, axis=1)
    # jets_pt = ak.pad_none(events.Jet.pt, 6, axis=1)
    # jets_eta = ak.pad_none(events.Jet.eta, 6, axis=1)
    # jets_phi = ak.pad_none(events.Jet.phi, 6, axis=1)
    # jets_btag = ak.pad_none(events.Jet.btagDeepFlavB, 6, axis=1)
    # jets_hadFlav = ak.pad_none(events.Jet.hadronFlavour, 6, axis=1)
    jets_pt = ak.pad_none(events.Jet.pt, max(events.n_jet))
    jets_pt = ak.to_regular(jets_pt, axis=1)
    jets_pt = ak.fill_none(jets_pt, EMPTY_FLOAT)
    events = set_ak_column_f32(events, "jets_pt", jets_pt)
    jets_eta = ak.pad_none(events.Jet.eta, max(events.n_jet))
    jets_eta = ak.to_regular(jets_eta, axis=1)
    jets_eta = ak.fill_none(jets_eta, EMPTY_FLOAT)
    events = set_ak_column_f32(events, "jets_eta", jets_eta)
    jets_phi = ak.pad_none(events.Jet.phi, max(events.n_jet))
    jets_phi = ak.to_regular(jets_phi, axis=1)
    jets_phi = ak.fill_none(jets_phi, EMPTY_FLOAT)
    events = set_ak_column_f32(events, "jets_phi", jets_phi)
    jets_mass = ak.pad_none(events.Jet.mass, max(events.n_jet))
    jets_mass = ak.to_regular(jets_mass, axis=1)
    jets_mass = ak.fill_none(jets_mass, EMPTY_FLOAT)
    events = set_ak_column_f32(events, "jets_mass", jets_mass)
    jets_e = ak.pad_none(events.Jet.E, max(events.n_jet))
    jets_e = ak.to_regular(jets_e, axis=1)
    jets_e = ak.fill_none(jets_e, EMPTY_FLOAT)
    events = set_ak_column_f32(events, "jets_e", jets_e)
    jets_btag = ak.pad_none(events.Jet.btagDeepFlavB, max(events.n_jet))
    jets_btag = ak.to_regular(jets_btag, axis=1)
    jets_btag = ak.fill_none(jets_btag, EMPTY_FLOAT)
    events = set_ak_column_f32(events, "jets_btag", jets_btag)
    jets_hadFlav = ak.pad_none(events.Jet.hadronFlavour, max(events.n_jet))
    jets_hadFlav = ak.to_regular(jets_hadFlav, axis=1)
    jets_hadFlav = ak.fill_none(jets_hadFlav, EMPTY_FLOAT)
    events = set_ak_column_f32(events, "jets_hadFlav", jets_hadFlav)
    # for i in range(0, 6, 1):
    #     events = set_ak_column_f32(events, f"jet{i+1}_mass", ak.fill_none(jets_mass[:, i], EMPTY_FLOAT))
    #     events = set_ak_column_f32(events, f"jet{i+1}_e", ak.fill_none(jets_e[:, i], EMPTY_FLOAT))
    #     events = set_ak_column_f32(events, f"jet{i+1}_pt", ak.fill_none(jets_pt[:, i], EMPTY_FLOAT))
    #     events = set_ak_column_f32(events, f"jet{i+1}_eta", ak.fill_none(jets_eta[:, i], EMPTY_FLOAT))
    #     events = set_ak_column_f32(events, f"jet{i+1}_phi", ak.fill_none(jets_phi[:, i], EMPTY_FLOAT))
    #     events = set_ak_column_f32(events, f"jet{i+1}_btag", ak.fill_none(jets_btag[:, i], EMPTY_FLOAT))
    #     events = set_ak_column_f32(events, f"jet{i+1}_hadronFlavour", ak.fill_none(jets_hadFlav[:, i], EMPTY_FLOAT))

    return events


# kinematic vars for coollection of jets and cbf jets
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


# kinematic vars for coollection of jets and cbf jets
@producer(
    uses={
        "VBFJet.pt", "VBFJet.eta", "VBFJet.phi", "VBFJet.mass",
        attach_coffea_behavior,
    },
    produces={
        # *[f"{obj}_{var}"
        # for obj in [f"jet{n}" for n in range(1, 7, 1)]
        # for var in ["pt", "eta", "phi", "mass", "e", "btag", "hadronFlavour"]], "nJets", "nConstituents", "jets_pt",
        "VBFjets_pt", "VBFjets_eta", "VBFjets_phi", "VBFjets_mass",
        "VBFjets_e", "VBFJetsdR", "VBFJetsdEta", "nVBFJets", "VBFjetsInvMass",
    },
)
def kinematic_vars_VBFjets(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    events = self[attach_coffea_behavior](events, collections={"VBFJet": {"type_name": "Jet"}}, **kwargs)

    ht = ak.sum(abs(events.VBFJet.pt), axis=1)
    events = set_ak_column_f32(events, "VBFht", ht)

    jets_pt = ak.pad_none(events.VBFJet.pt, 2)
    jets_pt = ak.to_regular(jets_pt, axis=1)
    jets_pt_filled = ak.fill_none(jets_pt, EMPTY_FLOAT)
    events = set_ak_column_f32(events, "VBFjets_pt", jets_pt_filled)

    jets_eta = ak.pad_none(events.VBFJet.eta, 2)
    jets_eta = ak.to_regular(jets_eta, axis=1)
    jets_eta_filled = ak.fill_none(jets_eta, EMPTY_FLOAT)
    events = set_ak_column_f32(events, "VBFjets_eta", jets_eta_filled)

    jets_phi = ak.pad_none(events.VBFJet.phi, 2)
    jets_phi = ak.to_regular(jets_phi, axis=1)
    jets_phi_filled = ak.fill_none(jets_phi, EMPTY_FLOAT)
    events = set_ak_column_f32(events, "VBFjets_phi", jets_phi_filled)

    jets_mass = ak.pad_none(events.VBFJet.mass, 2)
    jets_mass = ak.to_regular(jets_mass, axis=1)
    jets_mass_filled = ak.fill_none(jets_mass, EMPTY_FLOAT)
    events = set_ak_column_f32(events, "VBFjets_mass", jets_mass_filled)

    # Calculate energy
    p_x = jets_pt * np.cos(jets_phi)
    p_y = jets_pt * np.sin(jets_phi)
    p_z = jets_pt * np.sinh(jets_eta)
    p = np.sqrt(p_x**2 + p_y**2 + p_z**2)
    jets_e = np.sqrt(jets_mass**2 + p**2)
    jets_e = ak.fill_none(jets_e, EMPTY_FLOAT)
    events = set_ak_column_f32(events, "VBFjets_e", jets_e)

    # Calculate dR and d Eta
    # dR
    events = set_ak_column(events, "VBFJet", ak.pad_none(events.VBFJet, 2))
    dR_table = events.VBFJet.metric_table(events.VBFJet, axis=1)
    dR_values = dR_table[:, 0, 1]
    events = set_ak_column_f32(events, "VBFJetsdR", ak.fill_none(dR_values, EMPTY_FLOAT))

    # dEta
    d_eta = events.VBFJet.metric_table(events.VBFJet, axis=1, metric=d_eta_helper)
    d_eta = d_eta[:, 0, 1]
    events = set_ak_column_f32(events, "VBFJetsdEta", ak.fill_none(d_eta, EMPTY_FLOAT))

    # dR
    dR = np.sqrt(d_eta**2 + abs(jets_phi[:, 0] - jets_phi[:, 1])**2)
    events = set_ak_column_f32(events, "VBFJetsdR", ak.fill_none(dR, EMPTY_FLOAT))

    # Calculate Invariant Mass
    events = set_ak_column(events, "VBFJet", ak.pad_none(events.VBFJet, 2))
    events = set_ak_column(events, "VBFjetsInvMass", (events.VBFJet[:, 0] + events.VBFJet[:, 1]).mass)
    events = set_ak_column(events, "VBFjetsInvMass", ak.fill_none(events.VBFjetsInvMass, EMPTY_FLOAT))

    # Number of VBF Jets
    jets_filled = ak.fill_none(jets_pt, 0)
    mask = np.where(jets_filled == 0, 0, 1)
    n_jets = np.sum(mask, axis=1)
    events = set_ak_column_f32(events, "nVBFJets", n_jets)

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


# Producers for aditinal jet exclusive features, dummy fills and one Hot encoding
@producer(
    uses={
        "Jet.hadronFlavour", "Jet.btagDeepFlavB", "Jet.mass",
        attach_coffea_behavior,
    },
    produces={
        *[f"{obj}_{var}"
        for obj in ["jet1", "jet2"]
        for var in ["hadronFlavour", "btag", "DeepTau_e", "btag_dummy", "jet_oneHot", "bjet_oneHot",
        "tau_oneHot", "object_type"]]
    },
)
def jet_information(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    # Produce jet exclusive features
    events = self[attach_coffea_behavior](events, collections=["Jet"], **kwargs)
    events = set_ak_column_f32(events, "jet1_hadronFlavour", Route("Jet.hadronFlavour[:,0]").apply(events, EMPTY_FLOAT))
    events = set_ak_column_f32(events, "jet2_hadronFlavour", Route("Jet.hadronFlavour[:,1]").apply(events, EMPTY_FLOAT))
    events = set_ak_column_f32(events, "jet1_btag", Route("Jet.btagDeepFlavB[:,0]").apply(events, EMPTY_FLOAT))
    events = set_ak_column_f32(events, "jet2_btag", Route("Jet.btagDeepFlavB[:,1]").apply(events, EMPTY_FLOAT))
    # Produce dummy fills for non jet features
    padded_mass = ak.pad_none(events.Jet.mass, 2, axis=1)
    dummy_fill_1 = ak.full_like(padded_mass[:, 0], -1)
    dummy_fill_2 = ak.full_like(padded_mass[:, 1], -1)
    events = set_ak_column_f32(events, "jet1_DeepTau_e", ak.fill_none(dummy_fill_1, EMPTY_FLOAT))
    events = set_ak_column_f32(events, "jet2_DeepTau_e", ak.fill_none(dummy_fill_2, EMPTY_FLOAT))
    events = set_ak_column_f32(events, "jet1_btag_dummy", ak.fill_none(dummy_fill_1, EMPTY_FLOAT))
    events = set_ak_column_f32(events, "jet2_btag_dummy", ak.fill_none(dummy_fill_2, EMPTY_FLOAT))

    # Create one Hot encoding
    oneHot_fill_0 = ak.full_like(padded_mass[:, 0], 0)
    oneHot_fill_1 = ak.full_like(padded_mass[:, 1], 1)
    events = set_ak_column_f32(events, "jet1_tau_oneHot", ak.fill_none(oneHot_fill_0, EMPTY_FLOAT))
    events = set_ak_column_f32(events, "jet2_tau_oneHot", ak.fill_none(oneHot_fill_0, EMPTY_FLOAT))
    events = set_ak_column_f32(events, "jet1_bjet_oneHot", ak.fill_none(oneHot_fill_0, EMPTY_FLOAT))
    events = set_ak_column_f32(events, "jet2_bjet_oneHot", ak.fill_none(oneHot_fill_0, EMPTY_FLOAT))
    events = set_ak_column_f32(events, "jet1_jet_oneHot", ak.fill_none(oneHot_fill_1, EMPTY_FLOAT))
    events = set_ak_column_f32(events, "jet2_jet_oneHot", ak.fill_none(oneHot_fill_1, EMPTY_FLOAT))

    # Alternateively use integer to define object type (Jet:1, BJet:2, Tau:3) and apply embendding
    # layer in network
    events = set_ak_column_f32(events, "jet1_object_type", ak.fill_none(oneHot_fill_1, EMPTY_FLOAT))
    events = set_ak_column_f32(events, "jet2_object_type", ak.fill_none(oneHot_fill_1, EMPTY_FLOAT))

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


# Gen Parton B Producer, invariant mass, mass, pt, eta, phi
@producer(
    uses={
        "genBpartonH.pt", "genBpartonH.eta", "genBpartonH.phi", "genBpartonH.mass",
        attach_coffea_behavior,
    },
    produces={
        "GenPartBpartonInvMass", "GenPartBparton1Mass", "GenPartBparton1Pt", "GenPartBparton1Eta", "GenPartBparton1Phi",
        "GenPartBparton2Mass", "GenPartBparton2Pt", "GenPartBparton2Eta", "GenPartBparton2Phi", "GenPartBpartondR",
        "GenPartBpartondEta", "GenPartBpartondR_1"
    },
)
def genBPartonProducer(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    events = self[attach_coffea_behavior](events, collections={"genBpartonH": {"type_name": "GenParticle", "skip_fields": "*Idx*G"}} , **kwargs)

    # invariant mass of B partons
    inv_mass_Bpartons = inv_mass_helper(events.genBpartonH[:, 0], events.genBpartonH[:, 1])
    events = set_ak_column_f32(events, "GenPartBpartonInvMass", ak.fill_none(inv_mass_Bpartons, EMPTY_FLOAT))

    # kinematic information of B partons
    # B Parton 1
    Bparton1_mass = events.genBpartonH[:, 0].mass
    Bparton1_pt = events.genBpartonH[:, 0].pt
    Bparton1_eta = events.genBpartonH[:, 0].eta
    Bparton1_phi = events.genBpartonH[:, 0].phi

    events = set_ak_column_f32(events, "GenPartBparton1Mass", ak.fill_none(Bparton1_mass, EMPTY_FLOAT))
    events = set_ak_column_f32(events, "GenPartBparton1Pt", ak.fill_none(Bparton1_pt, EMPTY_FLOAT))
    events = set_ak_column_f32(events, "GenPartBparton1Eta", ak.fill_none(Bparton1_eta, EMPTY_FLOAT))
    events = set_ak_column_f32(events, "GenPartBparton1Phi", ak.fill_none(Bparton1_phi, EMPTY_FLOAT))

    # B Parton 2
    Bparton2_mass = events.genBpartonH[:, 1].mass
    Bparton2_pt = events.genBpartonH[:, 1].pt
    Bparton2_eta = events.genBpartonH[:, 1].eta
    Bparton2_phi = events.genBpartonH[:, 1].phi

    events = set_ak_column_f32(events, "GenPartBparton2Mass", ak.fill_none(Bparton2_mass, EMPTY_FLOAT))
    events = set_ak_column_f32(events, "GenPartBparton2Pt", ak.fill_none(Bparton2_pt, EMPTY_FLOAT))
    events = set_ak_column_f32(events, "GenPartBparton2Eta", ak.fill_none(Bparton2_eta, EMPTY_FLOAT))
    events = set_ak_column_f32(events, "GenPartBparton2Phi", ak.fill_none(Bparton2_phi, EMPTY_FLOAT))

    # dR
    dR_table = events.genBpartonH.metric_table(events.genBpartonH, axis=1)
    dR_values = dR_table[:, 0, 1]
    events = set_ak_column_f32(events, "GenPartBpartondR", ak.fill_none(dR_values, EMPTY_FLOAT))
    dR_values_1 = np.sqrt((events.genBpartonH.eta[:,0] - events.genBpartonH.eta[:,1])**2 + (events.genBpartonH.phi[:,0] - events.genBpartonH.phi[:,1])**2)
    events = set_ak_column_f32(events, "GenPartBpartondR_1", ak.fill_none(dR_values_1, EMPTY_FLOAT))

    # d eta
    d_eta_table = events.genBpartonH.metric_table(events.genBpartonH, axis=1, metric=d_eta_helper)
    d_eta_values = d_eta_table[:, 0, 1]
    events = set_ak_column_f32(events, "GenPartBpartondEta", ak.fill_none(d_eta_values, EMPTY_FLOAT))

    return events


# Gen Parton Tau Producer, invariant mass, mass, pt, eta, phi
@producer(
    uses={
        "genTaupartonH.*",
        attach_coffea_behavior,
    },
    produces={
        "GenPartTaupartonInvMass", "GenPartTauparton1Mass", "GenPartTauparton1Pt", "GenPartTauparton1Eta", "GenPartTauparton1Phi",
        "GenPartTauparton2Mass", "GenPartTauparton2Pt", "GenPartTauparton2Eta", "GenPartTauparton2Phi", "GenPartTaupartondR",
        "GenPartTaupartondEta",
    },
)
def genTauPartonProducer(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    events = self[attach_coffea_behavior](events, collections={"genTaupartonH": {"type_name": "GenParticle", "skip_fields": "*Idx*G"}} , **kwargs)

    # invariant mass of Tau partons
    inv_mass_Taupartons = inv_mass_helper(events.genTaupartonH[:, 0], events.genTaupartonH[:, 1])
    events = set_ak_column_f32(events, "GenPartTaupartonInvMass", ak.fill_none(inv_mass_Taupartons, EMPTY_FLOAT))

    # kinematic information of Tau partons
    # Tau Parton 1
    Tauparton1_mass = events.genTaupartonH[:, 0].mass
    Tauparton1_pt = events.genTaupartonH[:, 0].pt
    Tauparton1_eta = events.genTaupartonH[:, 0].eta
    Tauparton1_phi = events.genTaupartonH[:, 0].phi

    events = set_ak_column_f32(events, "GenPartTauparton1Mass", ak.fill_none(Tauparton1_mass, EMPTY_FLOAT))
    events = set_ak_column_f32(events, "GenPartTauparton1Pt", ak.fill_none(Tauparton1_pt, EMPTY_FLOAT))
    events = set_ak_column_f32(events, "GenPartTauparton1Eta", ak.fill_none(Tauparton1_eta, EMPTY_FLOAT))
    events = set_ak_column_f32(events, "GenPartTauparton1Phi", ak.fill_none(Tauparton1_phi, EMPTY_FLOAT))

    # Tau Parton 2
    Tauparton2_mass = events.genTaupartonH[:, 1].mass
    Tauparton2_pt = events.genTaupartonH[:, 1].pt
    Tauparton2_eta = events.genTaupartonH[:, 1].eta
    Tauparton2_phi = events.genTaupartonH[:, 1].phi

    events = set_ak_column_f32(events, "GenPartTauparton2Mass", ak.fill_none(Tauparton2_mass, EMPTY_FLOAT))
    events = set_ak_column_f32(events, "GenPartTauparton2Pt", ak.fill_none(Tauparton2_pt, EMPTY_FLOAT))
    events = set_ak_column_f32(events, "GenPartTauparton2Eta", ak.fill_none(Tauparton2_eta, EMPTY_FLOAT))
    events = set_ak_column_f32(events, "GenPartTauparton2Phi", ak.fill_none(Tauparton2_phi, EMPTY_FLOAT))

    # dR
    dR_table = events.genTaupartonH.metric_table(events.genTaupartonH, axis=1)
    dR_values = dR_table[:, 0, 1]
    events = set_ak_column_f32(events, "GenPartTaupartondR", ak.fill_none(dR_values, EMPTY_FLOAT))

    # d eta
    d_eta_table = events.genTaupartonH.metric_table(events.genTaupartonH, axis=1, metric=d_eta_helper)
    d_eta_values = d_eta_table[:, 0, 1]
    events = set_ak_column_f32(events, "GenPartTaupartondEta", ak.fill_none(d_eta_values, EMPTY_FLOAT))

    return events


# Gen Parton H Producer, invariant mass, mass, pt, eta, phi
@producer(
    uses={
        "genHpartonH.*",
        attach_coffea_behavior,
    },
    produces={
        "GenPartHpartonInvMass", "GenPartHparton1Mass", "GenPartHparton1Pt", "GenPartHparton1Eta", "GenPartHparton1Phi",
        "GenPartHparton2Mass", "GenPartHparton2Pt", "GenPartHparton2Eta", "GenPartHparton2Phi", "GenPartHpartondR",
        "GenPartHpartondEta", "GenHparton1E", "GenHparton1Gamma", "GenHparton2E", "GenHparton2Gamma", "GenHHGamma", "GenHHE"
    },
)
def genHPartonProducer(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    events = self[attach_coffea_behavior](events, collections={"genHpartonH": {"type_name": "GenParticle", "skip_fields": "*Idx*G"}} , **kwargs)
    # invariant mass of H partons
    inv_mass_Hpartons = inv_mass_helper(events.genHpartonH[:, 0], events.genHpartonH[:, 1])
    events = set_ak_column_f32(events, "GenPartHpartonInvMass", ak.fill_none(inv_mass_Hpartons, EMPTY_FLOAT))

    # kinematic information of H partons
    # H Parton 1
    Hparton1_mass = events.genHpartonH[:, 0].mass
    Hparton1_pt = events.genHpartonH[:, 0].pt
    Hparton1_eta = events.genHpartonH[:, 0].eta
    Hparton1_phi = events.genHpartonH[:, 0].phi

    events = set_ak_column_f32(events, "GenPartHparton1Mass", ak.fill_none(Hparton1_mass, EMPTY_FLOAT))
    events = set_ak_column_f32(events, "GenPartHparton1Pt", ak.fill_none(Hparton1_pt, EMPTY_FLOAT))
    events = set_ak_column_f32(events, "GenPartHparton1Eta", ak.fill_none(Hparton1_eta, EMPTY_FLOAT))
    events = set_ak_column_f32(events, "GenPartHparton1Phi", ak.fill_none(Hparton1_phi, EMPTY_FLOAT))

    # H Parton 2
    Hparton2_mass = events.genHpartonH[:, 1].mass
    Hparton2_pt = events.genHpartonH[:, 1].pt
    Hparton2_eta = events.genHpartonH[:, 1].eta
    Hparton2_phi = events.genHpartonH[:, 1].phi

    events = set_ak_column_f32(events, "GenPartHparton2Mass", ak.fill_none(Hparton2_mass, EMPTY_FLOAT))
    events = set_ak_column_f32(events, "GenPartHparton2Pt", ak.fill_none(Hparton2_pt, EMPTY_FLOAT))
    events = set_ak_column_f32(events, "GenPartHparton2Eta", ak.fill_none(Hparton2_eta, EMPTY_FLOAT))
    events = set_ak_column_f32(events, "GenPartHparton2Phi", ak.fill_none(Hparton2_phi, EMPTY_FLOAT))

    # dR
    dR_table = events.genHpartonH.metric_table(events.genHpartonH, axis=1)
    dR_values = dR_table[:, 0, 1]
    events = set_ak_column_f32(events, "GenPartHpartondR", ak.fill_none(dR_values, EMPTY_FLOAT))

    # d eta
    d_eta_table = events.genHpartonH.metric_table(events.genHpartonH, axis=1, metric=d_eta_helper)
    d_eta_values = d_eta_table[:, 0, 1]
    events = set_ak_column_f32(events, "GenPartHpartondEta", ak.fill_none(d_eta_values, EMPTY_FLOAT))

    # calculate gamma lorentz factor
    # for daughter Higgs
    p1_x = Hparton1_pt * np.cos(Hparton1_phi)
    p1_y = Hparton1_pt * np.sin(Hparton1_phi)
    p1_z = Hparton1_pt * np.sinh(Hparton1_eta)
    p1 = np.sqrt(p1_x**2 + p1_y**2 + p1_z**2)
    Hparton1_e = np.sqrt(Hparton1_mass**2 + p1**2)
    Hparton1_e = ak.fill_none(Hparton1_e, EMPTY_FLOAT)
    Hparton1_gamma = Hparton1_e / Hparton1_mass
    events = set_ak_column_f32(events, "GenHparton1E", Hparton1_e)
    events = set_ak_column_f32(events, "GenHparton1Gamma", Hparton1_gamma)

    p2_x = Hparton2_pt * np.cos(Hparton2_phi)
    p2_y = Hparton2_pt * np.sin(Hparton2_phi)
    p2_z = Hparton2_pt * np.sinh(Hparton2_eta)
    p2 = np.sqrt(p2_x**2 + p2_y**2 + p2_z**2)
    Hparton2_e = np.sqrt(Hparton2_mass**2 + p2**2)
    Hparton2_e = ak.fill_none(Hparton2_e, EMPTY_FLOAT)
    Hparton2_gamma = Hparton2_e / Hparton2_mass
    events = set_ak_column_f32(events, "GenHparton2E", Hparton2_e)
    events = set_ak_column_f32(events, "GenHparton2Gamma", Hparton2_gamma)

    # for mother Higgs
    pHH = p1 + p2
    HH_e = np.sqrt(inv_mass_Hpartons**2 + pHH**2)
    HH_gamma = HH_e / inv_mass_Hpartons
    events = set_ak_column_f32(events, "GenHHGamma", HH_gamma)
    events = set_ak_column_f32(events, "GenHHE", HH_e)

    return events


# Gen Parton VBF Producer, invariant mass, mass, pt, eta, phi
@producer(
    uses={
        "genVBFparton.*",
        attach_coffea_behavior,
    },
    produces={
        "GenPartVBFpartonInvMass", "GenPartVBFparton1Mass", "GenPartVBFparton1Pt", "GenPartVBFparton1Eta", "GenPartVBFparton1Phi",
        "GenPartVBFparton2Mass", "GenPartVBFparton2Pt", "GenPartVBFparton2Eta", "GenPartVBFparton2Phi", "GenPartVBFpartondR",
        "GenPartVBFpartondEta",# "GenVBFparton1E", "GenVBFparton1Gamma", "GenVBFparton2E", "GenVBFparton2Gamma"
    },
)
def genVBFPartonProducer(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    events = self[attach_coffea_behavior](events, collections={"genVBFparton": {"type_name": "GenParticle", "skip_fields": "*Idx*G"}} , **kwargs)
    # invariant mass of H partons
    inv_mass_VBFpartons = inv_mass_helper(events.genVBFparton[:, 0], events.genVBFparton[:, 1])
    events = set_ak_column_f32(events, "GenPartVBFpartonInvMass", ak.fill_none(inv_mass_VBFpartons, EMPTY_FLOAT))

    # kinematic information of H partons
    # H Parton 1
    VBFparton1_mass = events.genVBFparton[:, 0].mass
    VBFparton1_pt = events.genVBFparton[:, 0].pt
    VBFparton1_eta = events.genVBFparton[:, 0].eta
    VBFparton1_phi = events.genVBFparton[:, 0].phi

    events = set_ak_column_f32(events, "GenPartVBFparton1Mass", ak.fill_none(VBFparton1_mass, EMPTY_FLOAT))
    events = set_ak_column_f32(events, "GenPartVBFparton1Pt", ak.fill_none(VBFparton1_pt, EMPTY_FLOAT))
    events = set_ak_column_f32(events, "GenPartVBFparton1Eta", ak.fill_none(VBFparton1_eta, EMPTY_FLOAT))
    events = set_ak_column_f32(events, "GenPartVBFparton1Phi", ak.fill_none(VBFparton1_phi, EMPTY_FLOAT))

    # VBF Parton 2
    VBFparton2_mass = events.genVBFparton[:, 1].mass
    VBFparton2_pt = events.genVBFparton[:, 1].pt
    VBFparton2_eta = events.genVBFparton[:, 1].eta
    VBFparton2_phi = events.genVBFparton[:, 1].phi

    events = set_ak_column_f32(events, "GenPartVBFparton2Mass", ak.fill_none(VBFparton2_mass, EMPTY_FLOAT))
    events = set_ak_column_f32(events, "GenPartVBFparton2Pt", ak.fill_none(VBFparton2_pt, EMPTY_FLOAT))
    events = set_ak_column_f32(events, "GenPartVBFparton2Eta", ak.fill_none(VBFparton2_eta, EMPTY_FLOAT))
    events = set_ak_column_f32(events, "GenPartVBFparton2Phi", ak.fill_none(VBFparton2_phi, EMPTY_FLOAT))

    # dR
    dR_table = events.genVBFparton.metric_table(events.genVBFparton, axis=1)
    dR_values = dR_table[:, 0, 1]
    events = set_ak_column_f32(events, "GenPartVBFpartondR", ak.fill_none(dR_values, EMPTY_FLOAT))

    # d eta
    d_eta_table = events.genVBFparton.metric_table(events.genVBFparton, axis=1, metric=d_eta_helper)
    d_eta_values = d_eta_table[:, 0, 1]
    events = set_ak_column_f32(events, "GenPartVBFpartondEta", ak.fill_none(d_eta_values, EMPTY_FLOAT))

    return events


@producer(
    uses={
        "BJet.*", "GenMatchedBJets.*", "HHBJet.*",
        attach_coffea_behavior,
    },
    produces={
        "Btagging_results", "HHBtagging_results",
    },
)
def Btagging_efficiency_Bpartons(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    events = self[attach_coffea_behavior](events, collections={"BJet": {"type_name": "Jet"}}, **kwargs)
    events = self[attach_coffea_behavior](events, collections={"HHBJet": {"type_name": "Jet"}}, **kwargs)
    events = self[attach_coffea_behavior](events, collections={"GenMatchedBJets": {"type_name": "Jet"}}, **kwargs)
    Btagging_results = np.zeros(len(events.GenMatchedBJets.pt[:, 0]))
    HHBtagging_results = np.zeros(len(events.GenMatchedBJets.pt[:, 0]))
    BJets_filled_pt = ak.fill_none(events.BJet.pt, 0)
    HHBJets_filled_pt = ak.fill_none(events.HHBJet.pt, 0)
    for i, _ in enumerate(BJets_filled_pt):
        Btagging_counter = 0
        HHBtagging_counter = 0
        if np.sum(BJets_filled_pt, axis=1)[i] == 0:
            Btagging_counter = -1
        if np.sum(HHBJets_filled_pt, axis=1)[i] == 0:
            HHBtagging_counter = -1
        for matched_pt_b, matched_pt_hh in zip(BJets_filled_pt[i], HHBJets_filled_pt[i]):
            if matched_pt_b in events.GenMatchedBJets.pt[i]:
                Btagging_counter += 1
            if matched_pt_hh in events.GenMatchedBJets.pt[i]:
                HHBtagging_counter += 1
        Btagging_results[i] = Btagging_counter
        HHBtagging_results[i] = HHBtagging_counter
    events = set_ak_column_f32(events, "Btagging_results", ak.fill_none(Btagging_results, 0))
    events = set_ak_column_f32(events, "HHBtagging_results", ak.fill_none(HHBtagging_results, 0))

    return events


@producer(
    uses={
        "VBFJet.*", "AutoGenMatchedVBFJets.*", "GenMatchedVBFJets.*",
        attach_coffea_behavior,
    },
    produces={
        "VBFtagging_results_auto", "VBFtagging_results_dr"
    },
)
def VBFtagging_efficiency(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    events = self[attach_coffea_behavior](events, collections={"VBFJet": {"type_name": "Jet"}}, **kwargs)
    events = self[attach_coffea_behavior](events, collections={"AutoGenMatchedVBFJets": {"type_name": "Jet"}}, **kwargs)
    events = self[attach_coffea_behavior](events, collections={"GenMatchedVBFJets": {"type_name": "Jet"}}, **kwargs)
    # from IPython import embed; embed()
    VBFtagging_results_auto = np.zeros(len(events.AutoGenMatchedVBFJets.pt[:, 0]))
    VBFtagging_results_dr = np.zeros(len(events.AutoGenMatchedVBFJets.pt[:, 0]))
    VBFJets_filled_pt = ak.fill_none(events.VBFJet.pt, 0)
    for i, _ in enumerate(VBFJets_filled_pt):
        VBFtagging_counter_auto = 0
        VBFtagging_counter_dr = 0
        if np.sum(VBFJets_filled_pt, axis=1)[i] == 0:
            VBFtagging_counter_auto = -1
            VBFtagging_counter_dr = -1
        for matched_pt in VBFJets_filled_pt[i]:
            if matched_pt in events.AutoGenMatchedVBFJets.pt[i]:
                VBFtagging_counter_auto += 1
            if matched_pt in events.GenMatchedVBFJets.pt[i]:
                VBFtagging_counter_dr += 1
        VBFtagging_results_auto[i] = VBFtagging_counter_auto
        VBFtagging_results_dr[i] = VBFtagging_counter_dr
    print(f'Hist Counts Auto: -1: {np.sum(np.where(VBFtagging_results_auto==-1, 1, 0), axis=0)} ',
        f'0: {np.sum(np.where(VBFtagging_results_auto==0, 1, 0), axis=0)} ',
        f'1: {np.sum(np.where(VBFtagging_results_auto==1, 1, 0), axis=0)} ',
        f'2: {np.sum(np.where(VBFtagging_results_auto==2, 1, 0), axis=0)} '
    )
    print(f'Hist Counts dR: -1: {np.sum(np.where(VBFtagging_results_dr==-1, 1, 0), axis=0)} ',
        f'0: {np.sum(np.where(VBFtagging_results_dr==0, 1, 0), axis=0)} ',
        f'1: {np.sum(np.where(VBFtagging_results_dr==1, 1, 0), axis=0)} ',
        f'2: {np.sum(np.where(VBFtagging_results_dr==2, 1, 0), axis=0)} '
    )
    events = set_ak_column_f32(events, "VBFtagging_results_auto", ak.fill_none(VBFtagging_results_auto, 0))
    events = set_ak_column_f32(events, "VBFtagging_results_dr", ak.fill_none(VBFtagging_results_dr, 0))

    return events


@producer(
    uses={
        "VBFmask_step.*", "VBFpairs_step.*", "VBFtrigger_step.*", "AutoGenMatchedVBFJets.*",
        "genVBFparton.*", "VBFak4_step.*",
        attach_coffea_behavior,
    },
    produces={
        "VBFMaskStep", "VBFPairStep", "VBFTriggerStep", "GenMatchedVBFJets_ak4_pt",
        "GenMatchedVBFJets_ak4_eta", "GenMatchedVBFJets_mask_pt", "GenMatchedVBFJets_mask_eta",
        "GenMatchedVBFJets_pairs_pt", "GenMatchedVBFJets_pairs_eta",
        "GenMatchedVBFJets_trigger_pt", "GenMatchedVBFJets_trigger_eta", "VBFpartons_ak4_pt",
        "VBFpartons_ak4_eta", "VBFpartons_mask_pt", "VBFpartons_mask_eta", "VBFpartons_pairs_pt",
        "VBFpartons_pairs_eta", "VBFpartons_trigger_pt", "VBFpartons_trigger_eta",
        "GenMatchedVBFJets_ak4_inv_mass", "GenMatchedVBFJets_ak4_dEta",
        "GenMatchedVBFJets_mask_inv_mass", "GenMatchedVBFJets_mask_dEta",
        "GenMatchedVBFJets_pairs_inv_mass", "GenMatchedVBFJets_pairs_dEta",
        "GenMatchedVBFJets_trigger_inv_mass", "GenMatchedVBFJets_trigger_dEta",
        "GenVBFPartons_ak4_inv_mass", "GenVBFPartons_ak4_dEta",
        "GenVBFPartons_mask_inv_mass", "GenVBFPartons_mask_dEta",
        "GenVBFPartons_pairs_inv_mass", "GenVBFPartons_pairs_dEta",
        "GenVBFPartons_trigger_inv_mass", "GenVBFPartons_trigger_dEta",
    },
)
def VBFsteps_analysis(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    events = self[attach_coffea_behavior](events, collections={"VBFak4_step": {"type_name": "Jet"}}, **kwargs)
    events = self[attach_coffea_behavior](events, collections={"VBFmask_step": {"type_name": "Jet"}}, **kwargs)
    events = self[attach_coffea_behavior](events, collections={"VBFpairs_step": {"type_name": "Jet"}}, **kwargs)
    events = self[attach_coffea_behavior](events, collections={"VBFtrigger_step": {"type_name": "Jet"}}, **kwargs)
    events = self[attach_coffea_behavior](events, collections={"AutoGenMatchedVBFJets": {"type_name": "Jet"}}, **kwargs)
    events = self[attach_coffea_behavior](events, collections={"genVBFparton": {"type_name": "GenParticle", "skip_fields": "*Idx*G"}} , **kwargs)

    auto_matched_filled = ak.fill_none(events.AutoGenMatchedVBFJets.pt, 0)

    padding_val_ak4 = max(ak.count(ak.fill_none(events.VBFak4_step.pt, 0), axis=1))
    padded_ak4 = ak.fill_none(ak.pad_none(events.VBFak4_step.pt, padding_val_ak4), -1)
    padding_val_mask = max(ak.count(ak.fill_none(events.VBFmask_step.pt, 0), axis=1))
    padded_vbfmask = ak.fill_none(ak.pad_none(events.VBFmask_step.pt, padding_val_mask), -1)
    padding_val_pair = max(ak.count(ak.fill_none(events.VBFpairs_step.pt, 0), axis=1))
    padded_pairs = ak.fill_none(ak.pad_none(events.VBFpairs_step.pt, padding_val_pair), -1)
    padding_val_trigger = max(ak.count(ak.fill_none(events.VBFtrigger_step.pt, 0), axis=1))
    padded_trigger = ak.fill_none(ak.pad_none(events.VBFtrigger_step.pt, padding_val_trigger), -1)

    # returns masks for the matched VBF Jets that are still in the events after aech selection step
    ak4_mask0 = np.isin(auto_matched_filled[:, 0], padded_ak4)
    ak4_mask1 = np.isin(auto_matched_filled[:, 1], padded_ak4)
    ak4_mask = np.stack((ak4_mask0, ak4_mask1), axis=1)

    vbfmask_mask0 = np.isin(auto_matched_filled[:, 0], padded_vbfmask)
    vbfmask_mask1 = np.isin(auto_matched_filled[:, 1], padded_vbfmask)
    vbfmask_mask = np.stack((vbfmask_mask0, vbfmask_mask1), axis=1)

    pairs_mask0 = np.isin(auto_matched_filled[:, 0], padded_pairs)
    pairs_mask1 = np.isin(auto_matched_filled[:, 1], padded_pairs)
    pairs_mask = np.stack((pairs_mask0, pairs_mask1), axis=1)

    trigger_mask0 = np.isin(auto_matched_filled[:, 0], padded_trigger)
    trigger_mask1 = np.isin(auto_matched_filled[:, 1], padded_trigger)
    trigger_mask = np.stack((trigger_mask0, trigger_mask1), axis=1)

    # Sum mask to get number of correct VBF Jets left per event
    vbfmask_count = np.sum(vbfmask_mask, axis=1)
    vbfpair_count = np.sum(pairs_mask, axis=1)
    vbftrigger_count = np.sum(trigger_mask, axis=1)

    # Use mask to get Gen matched Jets left after selection steps
    genMatchedVBFJets_ak4_pt = ak.fill_none(ak.mask(events.AutoGenMatchedVBFJets.pt, ak4_mask)[:, 0], EMPTY_FLOAT)
    genMatchedVBFJets_ak4_eta = ak.fill_none(ak.mask(events.AutoGenMatchedVBFJets.eta, ak4_mask)[:, 0], EMPTY_FLOAT)

    genMatchedVBFJets_mask_pt = np.full(len(auto_matched_filled), EMPTY_FLOAT)
    genMatchedVBFJets_mask_pt[0:len(events.AutoGenMatchedVBFJets.pt[vbfmask_mask])] = events.AutoGenMatchedVBFJets.pt[vbfmask_mask]
    genMatchedVBFJets_mask_eta = np.full(len(auto_matched_filled), EMPTY_FLOAT)
    genMatchedVBFJets_mask_eta[0:len(events.AutoGenMatchedVBFJets.eta[vbfmask_mask])] = events.AutoGenMatchedVBFJets.eta[vbfmask_mask]

    genMatchedVBFJets_pairs_pt = np.full(len(auto_matched_filled), EMPTY_FLOAT)
    genMatchedVBFJets_pairs_pt[0:len(events.AutoGenMatchedVBFJets.pt[pairs_mask])] = events.AutoGenMatchedVBFJets.pt[pairs_mask]
    genMatchedVBFJets_pairs_eta = np.full(len(auto_matched_filled), EMPTY_FLOAT)
    genMatchedVBFJets_pairs_eta[0:len(events.AutoGenMatchedVBFJets.eta[pairs_mask])] = events.AutoGenMatchedVBFJets.eta[pairs_mask]

    genMatchedVBFJets_trigger_pt = np.full(len(auto_matched_filled), EMPTY_FLOAT)
    genMatchedVBFJets_trigger_pt[0:len(events.AutoGenMatchedVBFJets.pt[trigger_mask])] = events.AutoGenMatchedVBFJets.pt[trigger_mask]
    genMatchedVBFJets_trigger_eta = np.full(len(auto_matched_filled), EMPTY_FLOAT)
    genMatchedVBFJets_trigger_eta[0:len(events.AutoGenMatchedVBFJets.eta[trigger_mask])] = events.AutoGenMatchedVBFJets.eta[trigger_mask]

    # Use mask to get according partons of the correct VBF Jets left after selection steps
    VBFpartons_ak4_pt = ak.fill_none(ak.mask(events.genVBFparton.pt, ak4_mask)[:, 0], EMPTY_FLOAT)
    VBFpartons_ak4_eta = ak.fill_none(ak.mask(events.genVBFparton.eta, ak4_mask)[:, 0], EMPTY_FLOAT)

    VBFpartons_mask_pt = np.full(len(auto_matched_filled), EMPTY_FLOAT)
    VBFpartons_mask_pt[0:len(events.genVBFparton.pt[vbfmask_mask])] = events.genVBFparton.pt[vbfmask_mask]
    VBFpartons_mask_eta = np.full(len(auto_matched_filled), EMPTY_FLOAT)
    VBFpartons_mask_eta[0:len(events.genVBFparton.eta[vbfmask_mask])] = events.genVBFparton.eta[vbfmask_mask]

    VBFpartons_pairs_pt = np.full(len(auto_matched_filled), EMPTY_FLOAT)
    VBFpartons_pairs_pt[0:len(events.genVBFparton.pt[pairs_mask])] = events.genVBFparton.pt[pairs_mask]
    VBFpartons_pairs_eta = np.full(len(auto_matched_filled), EMPTY_FLOAT)
    VBFpartons_pairs_eta[0:len(events.genVBFparton.eta[pairs_mask])] = events.genVBFparton.eta[pairs_mask]

    VBFpartons_trigger_pt = np.full(len(auto_matched_filled), EMPTY_FLOAT)
    VBFpartons_trigger_pt[0:len(events.genVBFparton.pt[trigger_mask])] = events.genVBFparton.pt[trigger_mask]
    VBFpartons_trigger_eta = np.full(len(auto_matched_filled), EMPTY_FLOAT)
    VBFpartons_trigger_eta[0:len(events.genVBFparton.eta[trigger_mask])] = events.genVBFparton.eta[trigger_mask]

    # do the same for the gen matched jets and the corresponding partons using the mask defined above
    # Gen matched Jets
    events = set_ak_column(events, "AutoGenMatchedVBFJets", ak.pad_none(events.AutoGenMatchedVBFJets, 2))
    events = set_ak_column(events, "genVBFparton", ak.pad_none(events.genVBFparton, 2))

    genMatchedJets_inv_mass_ak4 = (ak.mask(events.AutoGenMatchedVBFJets, ak4_mask)[:, 0] + ak.mask(events.AutoGenMatchedVBFJets, ak4_mask)[:, 1]).mass
    genMatchedJets_dEta_ak4 = abs(ak.mask(events.AutoGenMatchedVBFJets, ak4_mask)[:, 0].eta - ak.mask(events.AutoGenMatchedVBFJets, ak4_mask)[:, 1].eta)

    genMatchedJets_inv_mass_mask = (ak.mask(events.AutoGenMatchedVBFJets, vbfmask_mask)[:, 0] + ak.mask(events.AutoGenMatchedVBFJets, vbfmask_mask)[:, 1]).mass
    genMatchedJets_dEta_mask = abs(ak.mask(events.AutoGenMatchedVBFJets, vbfmask_mask)[:, 0].eta - ak.mask(events.AutoGenMatchedVBFJets, vbfmask_mask)[:, 1].eta)

    genMatchedJets_inv_mass_pairs = (ak.mask(events.AutoGenMatchedVBFJets, pairs_mask)[:, 0] + ak.mask(events.AutoGenMatchedVBFJets, pairs_mask)[:, 1]).mass
    genMatchedJets_dEta_pairs = abs(ak.mask(events.AutoGenMatchedVBFJets, pairs_mask)[:, 0].eta - ak.mask(events.AutoGenMatchedVBFJets, pairs_mask)[:, 1].eta)
    mask_pairs = ak.fill_none(((genMatchedJets_inv_mass_pairs > 500.0) & (genMatchedJets_dEta_pairs > 3.0)), False)

    genMatchedJets_inv_mass_trigger = (ak.mask(events.AutoGenMatchedVBFJets, trigger_mask)[:, 0] + ak.mask(events.AutoGenMatchedVBFJets, trigger_mask)[:, 1]).mass
    genMatchedJets_dEta_trigger = abs(ak.mask(events.AutoGenMatchedVBFJets, trigger_mask)[:, 0].eta - ak.mask(events.AutoGenMatchedVBFJets, trigger_mask)[:, 1].eta)
    mask_trigger = ak.fill_none(((genMatchedJets_inv_mass_trigger > 500.0) & (genMatchedJets_dEta_trigger > 3.0)), False)

    # Corresponing Partons
    vbfpartons_inv_mass_ak4 = (events.genVBFparton[:, 0] + events.genVBFparton[:, 1]).mass
    vbfpartons_dEta_ak4 = abs(events.genVBFparton[:, 0].eta - events.genVBFparton[:, 1].eta)

    vbfpartons_inv_mass_mask = (events.genVBFparton[:, 0] + events.genVBFparton[:, 1]).mass
    vbfpartons_dEta_mask = abs(events.genVBFparton[:, 0].eta - events.genVBFparton[:, 1].eta)

    vbfpartons_inv_mass_pairs = (ak.mask(events.genVBFparton, pairs_mask)[:, 0] + ak.mask(events.genVBFparton, pairs_mask)[:, 1]).mass
    vbfpartons_dEta_pairs = abs(ak.mask(events.genVBFparton, pairs_mask)[:, 0].eta - ak.mask(events.genVBFparton, pairs_mask)[:, 1].eta)

    vbfpartons_inv_mass_trigger = (ak.mask(events.genVBFparton, trigger_mask)[:, 0] + ak.mask(events.genVBFparton, trigger_mask)[:, 1]).mass
    vbfpartons_dEta_trigger = abs(ak.mask(events.genVBFparton, trigger_mask)[:, 0].eta - ak.mask(events.genVBFparton, trigger_mask)[:, 1].eta)

    events = set_ak_column_f32(events, "GenMatchedVBFJets_ak4_pt", ak.fill_none(genMatchedVBFJets_ak4_pt, EMPTY_FLOAT))
    events = set_ak_column_f32(events, "GenMatchedVBFJets_ak4_eta", ak.fill_none(genMatchedVBFJets_ak4_eta, EMPTY_FLOAT))
    events = set_ak_column_f32(events, "GenMatchedVBFJets_mask_pt", ak.fill_none(genMatchedVBFJets_mask_pt, EMPTY_FLOAT))
    events = set_ak_column_f32(events, "GenMatchedVBFJets_mask_eta", ak.fill_none(genMatchedVBFJets_mask_eta, EMPTY_FLOAT))
    events = set_ak_column_f32(events, "GenMatchedVBFJets_pairs_pt", ak.fill_none(genMatchedVBFJets_pairs_pt, EMPTY_FLOAT))
    events = set_ak_column_f32(events, "GenMatchedVBFJets_pairs_eta", ak.fill_none(genMatchedVBFJets_pairs_eta, EMPTY_FLOAT))
    events = set_ak_column_f32(events, "GenMatchedVBFJets_trigger_pt", ak.fill_none(genMatchedVBFJets_trigger_pt, EMPTY_FLOAT))
    events = set_ak_column_f32(events, "GenMatchedVBFJets_trigger_eta", ak.fill_none(genMatchedVBFJets_trigger_eta, EMPTY_FLOAT))

    events = set_ak_column_f32(events, "VBFpartons_ak4_pt", ak.fill_none(VBFpartons_ak4_pt, EMPTY_FLOAT))
    events = set_ak_column_f32(events, "VBFpartons_ak4_eta", ak.fill_none(VBFpartons_ak4_eta, EMPTY_FLOAT))
    events = set_ak_column_f32(events, "VBFpartons_mask_pt", ak.fill_none(VBFpartons_mask_pt, EMPTY_FLOAT))
    events = set_ak_column_f32(events, "VBFpartons_mask_eta", ak.fill_none(VBFpartons_mask_eta, EMPTY_FLOAT))
    events = set_ak_column_f32(events, "VBFpartons_pairs_pt", ak.fill_none(VBFpartons_pairs_pt, EMPTY_FLOAT))
    events = set_ak_column_f32(events, "VBFpartons_pairs_eta", ak.fill_none(VBFpartons_pairs_eta, EMPTY_FLOAT))
    events = set_ak_column_f32(events, "VBFpartons_trigger_pt", ak.fill_none(VBFpartons_trigger_pt, EMPTY_FLOAT))
    events = set_ak_column_f32(events, "VBFpartons_trigger_eta", ak.fill_none(VBFpartons_trigger_eta, EMPTY_FLOAT))

    events = set_ak_column_f32(events, "VBFMaskStep", vbfmask_count)
    events = set_ak_column_f32(events, "VBFPairStep", vbfpair_count)
    events = set_ak_column_f32(events, "VBFTriggerStep", vbftrigger_count)

    events = set_ak_column_f32(events, "GenMatchedVBFJets_ak4_inv_mass", ak.fill_none(genMatchedJets_inv_mass_ak4, EMPTY_FLOAT))
    events = set_ak_column_f32(events, "GenMatchedVBFJets_ak4_dEta", ak.fill_none(genMatchedJets_dEta_ak4, EMPTY_FLOAT))
    events = set_ak_column_f32(events, "GenMatchedVBFJets_mask_inv_mass", ak.fill_none(genMatchedJets_inv_mass_mask, EMPTY_FLOAT))
    events = set_ak_column_f32(events, "GenMatchedVBFJets_mask_dEta", ak.fill_none(genMatchedJets_dEta_mask, EMPTY_FLOAT))
    events = set_ak_column_f32(events, "GenMatchedVBFJets_pairs_inv_mass", ak.fill_none(ak.mask(genMatchedJets_inv_mass_pairs, mask_pairs), EMPTY_FLOAT))
    events = set_ak_column_f32(events, "GenMatchedVBFJets_pairs_dEta", ak.fill_none(ak.mask(genMatchedJets_dEta_pairs, mask_pairs), EMPTY_FLOAT))
    events = set_ak_column_f32(events, "GenMatchedVBFJets_trigger_inv_mass", ak.fill_none(ak.mask(genMatchedJets_inv_mass_trigger, mask_trigger), EMPTY_FLOAT))
    events = set_ak_column_f32(events, "GenMatchedVBFJets_trigger_dEta", ak.fill_none(ak.mask(genMatchedJets_dEta_trigger, mask_trigger), EMPTY_FLOAT))

    events = set_ak_column_f32(events, "GenVBFPartons_ak4_inv_mass", ak.fill_none(vbfpartons_inv_mass_ak4, EMPTY_FLOAT))
    events = set_ak_column_f32(events, "GenVBFPartons_ak4_dEta", ak.fill_none(vbfpartons_dEta_ak4, EMPTY_FLOAT))
    events = set_ak_column_f32(events, "GenVBFPartons_mask_inv_mass", ak.fill_none(vbfpartons_inv_mass_mask, EMPTY_FLOAT))
    events = set_ak_column_f32(events, "GenVBFPartons_mask_dEta", ak.fill_none(vbfpartons_dEta_mask, EMPTY_FLOAT))
    events = set_ak_column_f32(events, "GenVBFPartons_pairs_inv_mass", ak.fill_none(ak.mask(vbfpartons_inv_mass_pairs, mask_pairs), EMPTY_FLOAT))
    events = set_ak_column_f32(events, "GenVBFPartons_pairs_dEta", ak.fill_none(ak.mask(vbfpartons_dEta_pairs, mask_pairs), EMPTY_FLOAT))
    events = set_ak_column_f32(events, "GenVBFPartons_trigger_inv_mass", ak.fill_none(ak.mask(vbfpartons_inv_mass_trigger, mask_trigger), EMPTY_FLOAT))
    events = set_ak_column_f32(events, "GenVBFPartons_trigger_dEta", ak.fill_none(ak.mask(vbfpartons_dEta_trigger, mask_trigger), EMPTY_FLOAT))

    return events


@producer(
    uses={
        "AutoGenMatchedVBFJets.*", "matchedGenVBFparton.*",
        attach_coffea_behavior,
    },
    produces={
        "AutoGenMatchedVBFJetsInvMass", "AutoGenMatchedVBFJetsdEta", "matchedGenPartVBFpartondEta",
        "matchedGenPartVBFpartonInvMass",
    },
)
def GenInvMassdEta(self: Producer, events: ak.Array, **kwargs) -> ak.Array:

    events = self[attach_coffea_behavior](events, collections={"AutoGenMatchedVBFJets": {"type_name": "Jet"}}, **kwargs)
    events = self[attach_coffea_behavior](events, collections={"matchedGenVBFparton": {"type_name": "GenParticle", "skip_fields": "*Idx*G"}} , **kwargs)

    events = set_ak_column(events, "AutoGenMatchedVBFJets", ak.pad_none(events.AutoGenMatchedVBFJets, 2))

    events = set_ak_column(events, "AutoGenMatchedVBFJetsInvMass", (events.AutoGenMatchedVBFJets[:, 0] + events.AutoGenMatchedVBFJets[:, 1]).mass)
    events = set_ak_column(events, "AutoGenMatchedVBFJetsInvMass", ak.fill_none(events.AutoGenMatchedVBFJetsInvMass, EMPTY_FLOAT))

    events = set_ak_column(events, "AutoGenMatchedVBFJetsdEta", abs(events.AutoGenMatchedVBFJets[:, 0].eta - events.AutoGenMatchedVBFJets[:, 1].eta))
    events = set_ak_column(events, "AutoGenMatchedVBFJetsdEta", ak.fill_none(events.AutoGenMatchedVBFJetsdEta, EMPTY_FLOAT))

    # same for the gen partons that could be matched to a reco jet
    events = set_ak_column(events, "matchedGenVBFparton", ak.pad_none(events.matchedGenVBFparton, 2))

    events = set_ak_column(events, "matchedGenPartVBFpartonInvMass", (events.matchedGenVBFparton[:, 0] + events.matchedGenVBFparton[:, 1]).mass)
    events = set_ak_column(events, "matchedGenPartVBFpartonInvMass", ak.fill_none(events.matchedGenPartVBFpartonInvMass, EMPTY_FLOAT))

    events = set_ak_column(events, "matchedGenPartVBFpartondEta", abs(events.matchedGenVBFparton[:, 0].eta - events.matchedGenVBFparton[:, 1].eta))
    events = set_ak_column(events, "matchedGenPartVBFpartondEta", ak.fill_none(events.matchedGenPartVBFpartondEta, EMPTY_FLOAT))

    return events


@producer(
    uses={
        "AutoGenMatchedVBFJets.*",
        attach_coffea_behavior,
    },
    produces={
        "MatchedPartonsCounts",
    },
)
def NumberOfMatchedPartons(self: Producer, events: ak.Array, **kwargs) -> ak.Array:

    events = self[attach_coffea_behavior](events, collections={"AutoGenMatchedVBFJets": {"type_name": "Jet"}}, **kwargs)

    # ak count only counts entries different from None, eg. [2, None] is countes as len 1
    matched_parton_count = ak.count(events.AutoGenMatchedVBFJets.pt, axis=1)
    events = set_ak_column(events, "MatchedPartonsCounts", ak.fill_none(matched_parton_count, EMPTY_FLOAT))

    return events


@producer(
    uses={
        "VBFmask_step.*", "VBFpairs_step.*", "VBFtrigger_step.*", "VBFak4_step.*",
        attach_coffea_behavior,
    },
    produces={
        "Jets_inv_mass_ak4_step", "Jets_dEta_ak4_step", "Jets_inv_mass_mask_step", "Jets_dEta_mask_step",
        "Jets_inv_mass_pairs_step", "Jets_dEta_pairs_step", "Jets_inv_mass_trigger_step", "Jets_dEta_trigger_step",
        "Jets_ak4_pt", "Jets_ak4_eta", "Jets_mask_pt", "Jets_mask_eta", "Jets_pairs_pt", "Jets_pairs_eta",
        "Jets_trigger_pt", "Jets_trigger_eta",
    },
)
def VBFCandidatesInvMassdEta(self: Producer, events: ak.Array, **kwargs) -> ak.Array:

    events = self[attach_coffea_behavior](events, collections={"VBFmask_step": {"type_name": "Jet"}}, **kwargs)
    events = self[attach_coffea_behavior](events, collections={"VBFak4_step": {"type_name": "Jet"}}, **kwargs)
    events = self[attach_coffea_behavior](events, collections={"VBFpairs_step": {"type_name": "Jet"}}, **kwargs)
    events = self[attach_coffea_behavior](events, collections={"VBFtrigger_step": {"type_name": "Jet"}}, **kwargs)

    events = set_ak_column(events, "VBFmask_step", ak.pad_none(events.VBFmask_step, 2))
    events = set_ak_column(events, "VBFak4_step", ak.pad_none(events.VBFak4_step, 2))
    events = set_ak_column(events, "VBFpairs_step", ak.pad_none(events.VBFpairs_step, 2))
    events = set_ak_column(events, "VBFtrigger_step", ak.pad_none(events.VBFtrigger_step, 2))

    # Get pt and eta of highest pt Jet left after each sel step
    ak4_sort = ak.argsort(events.VBFak4_step.pt, axis=1, ascending=False)
    events = set_ak_column_f32(events, "Jets_ak4_pt", ak.fill_none(events.VBFak4_step.pt[ak4_sort][:, 0], EMPTY_FLOAT))
    events = set_ak_column_f32(events, "Jets_ak4_eta", ak.fill_none(events.VBFak4_step.eta[ak4_sort][:, 0], EMPTY_FLOAT))

    vbfmask_sort = ak.argsort(events.VBFmask_step.pt, axis=1, ascending=False)
    events = set_ak_column_f32(events, "Jets_mask_pt", ak.fill_none(events.VBFmask_step.pt[vbfmask_sort][:, 0], EMPTY_FLOAT))
    events = set_ak_column_f32(events, "Jets_mask_eta", ak.fill_none(events.VBFmask_step.eta[vbfmask_sort][:, 0], EMPTY_FLOAT))

    pairs_sort = ak.argsort(events.VBFpairs_step.pt, axis=1, ascending=False)
    events = set_ak_column_f32(events, "Jets_pairs_pt", ak.fill_none(events.VBFpairs_step.pt[pairs_sort][:, 0], EMPTY_FLOAT))
    events = set_ak_column_f32(events, "Jets_pairs_eta", ak.fill_none(events.VBFpairs_step.eta[pairs_sort][:, 0], EMPTY_FLOAT))

    trigger_sort = ak.argsort(events.VBFtrigger_step.pt, axis=1, ascending=False)
    events = set_ak_column_f32(events, "Jets_trigger_pt", ak.fill_none(events.VBFtrigger_step.pt[trigger_sort][:, 0], EMPTY_FLOAT))
    events = set_ak_column_f32(events, "Jets_trigger_eta", ak.fill_none(events.VBFtrigger_step.eta[trigger_sort][:, 0], EMPTY_FLOAT))

    # Calculate the (max) invariant mass and corresponding delta eta at each step
    jets_inv_mass_ak4 = (events.VBFak4_step[:, 0] + events.VBFak4_step[:, 1]).mass
    jets_dEta_ak4 = abs(events.VBFak4_step[:, 0].eta - events.VBFak4_step[:, 1].eta)

    jets_inv_mass_mask = (events.VBFmask_step[:, 0] + events.VBFmask_step[:, 1]).mass
    jets_dEta_mask = abs(events.VBFmask_step[:, 0].eta - events.VBFmask_step[:, 1].eta)

    jets_inv_mass_pairs = (events.VBFpairs_step[:, 0] + events.VBFpairs_step[:, 1]).mass
    jets_dEta_pairs = abs(events.VBFpairs_step[:, 0].eta - events.VBFpairs_step[:, 1].eta)

    jets_inv_mass_trigger = (events.VBFtrigger_step[:, 0] + events.VBFtrigger_step[:, 1]).mass
    jets_dEta_trigger = abs(events.VBFtrigger_step[:, 0].eta - events.VBFtrigger_step[:, 1].eta)

    events = set_ak_column_f32(events, "Jets_inv_mass_ak4_step", ak.fill_none(jets_inv_mass_ak4, EMPTY_FLOAT))
    events = set_ak_column_f32(events, "Jets_dEta_ak4_step", ak.fill_none(jets_dEta_ak4, EMPTY_FLOAT))
    events = set_ak_column_f32(events, "Jets_inv_mass_mask_step", ak.fill_none(jets_inv_mass_mask, EMPTY_FLOAT))
    events = set_ak_column_f32(events, "Jets_dEta_mask_step", ak.fill_none(jets_dEta_mask, EMPTY_FLOAT))
    events = set_ak_column_f32(events, "Jets_inv_mass_pairs_step", ak.fill_none(jets_inv_mass_pairs, EMPTY_FLOAT))
    events = set_ak_column_f32(events, "Jets_dEta_pairs_step", ak.fill_none(jets_dEta_pairs, EMPTY_FLOAT))
    events = set_ak_column_f32(events, "Jets_inv_mass_trigger_step", ak.fill_none(jets_inv_mass_trigger, EMPTY_FLOAT))
    events = set_ak_column_f32(events, "Jets_dEta_trigger_step", ak.fill_none(jets_dEta_trigger, EMPTY_FLOAT))

    return events
