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
