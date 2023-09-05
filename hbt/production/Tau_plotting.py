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


# Helper function for matching between Gen Partons and Reco Objects
# Indices are returned with respect to array2
def find_genjet_indices(array1: ak.Array, array2: ak.Array, deltaR):
    """calculates indices of jets of a specific genmatching step
    in which array1 is matched to array2.

    :param array1: First simulation step.
    :param array2: Second simullation step.
    :return: indices of genmatched jets, where array1 was matched to array2.
    """
    # calculate delta R between jets:
    metrics_genjets = array1.metric_table(array2, axis=1)
    # get indices of minimum delta R value:
    minimum_deltar_indices = ak.argmin(metrics_genjets, axis=2, keepdims=True)
    # filter only indices of minimum delta R value:
    metric = ak.firsts(metrics_genjets[minimum_deltar_indices], axis=2)
    # get indices:
    genjet_indices = ak.firsts(minimum_deltar_indices.mask[metric <= deltaR], axis=2)

    return genjet_indices


@producer(
    uses={
        "genTaupartonH.*",
        attach_coffea_behavior,
    },
    produces={
        "GenPartTauParton1Pt", "GenPartTauParton2Pt", "GenPartTauParton1Eta", "GenPartTauParton2Eta",
        "GenPartTauPartonInvMass", "GenPartTauPartondEta",
    },
)
def kinematics_tau_partons(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    events = self[attach_coffea_behavior](events, collections={"genTaupartonH": {"type_name": "GenParticle", "skip_fields": "*Idx*G"}} , **kwargs)
    events = set_ak_column(events, "genTaupartonH", ak.pad_none(events.genTaupartonH, 2))

    # Tau Parton pt
    TauParton1_pt = events.genTaupartonH.pt[:, 0]
    TauParton2_pt = events.genTaupartonH.pt[:, 1]
    events = set_ak_column_f32(events, "GenPartTauParton1Pt", ak.fill_none(TauParton1_pt, EMPTY_FLOAT))
    events = set_ak_column_f32(events, "GenPartTauParton2Pt", ak.fill_none(TauParton2_pt, EMPTY_FLOAT))

    # Tau Parton Eta
    TauParton1_eta = events.genTaupartonH.eta[:, 0]
    TauParton2_eta = events.genTaupartonH.eta[:, 1]
    events = set_ak_column_f32(events, "GenPartTauParton1Eta", ak.fill_none(TauParton1_eta, EMPTY_FLOAT))
    events = set_ak_column_f32(events, "GenPartTauParton2Eta", ak.fill_none(TauParton2_eta, EMPTY_FLOAT))

    # Invariant Mass of the Tau Pair
    inv_mass_tau_partons = inv_mass_helper(events.genTaupartonH[:, 0], events.genTaupartonH[:, 1])
    events = set_ak_column_f32(events, "GenPartTauPartonInvMass", ak.fill_none(inv_mass_tau_partons, EMPTY_FLOAT))

    # Delta Eta of Tau parton Pair
    dEta_tau_partons = d_eta_helper(events.genTaupartonH[:, 0], events.genTaupartonH[:, 1])
    events = set_ak_column_f32(events, "GenPartTauPartondEta", ak.fill_none(dEta_tau_partons, EMPTY_FLOAT))

    return events


@producer(
    uses={
        "GenPart.*", "Tau.*", "genTaupartonH.*",
        attach_coffea_behavior,
    },
    produces={
        "MatchedTauParton1Pt", "MatchedTauParton2Pt", "MatchedTauParton1Eta", "MatchedTauParton2Eta",
        "UnmatchedTauParton1Pt", "UnmatchedTauParton2Pt", "UnmatchedTauPartonPt",
        "UnmatchedTauParton1Eta", "UnmatchedTauParton2Eta", "UnmatchedTauPartonEta",
        "UnmatchedTau1Pt", "UnmatchedTau2Pt", "UnmatchedTauPt",
        "UnmatchedTau1Eta", "UnmatchedTau2Eta", "UnmatchedTauEta",
    },
)
def PartonsFromTauIdx(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    events = self[attach_coffea_behavior](events, collections={"genTaupartonH": {"type_name": "GenParticle", "skip_fields": "*Idx*G"}} , **kwargs)
    events = set_ak_column(events, "genTaupartonH", ak.pad_none(events.genTaupartonH, 2))
    events = self[attach_coffea_behavior](events, collections=["Tau"], **kwargs)

    # TauPartonIdx = events.Tau.genPartIdx
    # PartonbyIdx = events.GenPart[TauPartonIdx]
    # Get Partons by delta R matching using a smaller radius of 0.3 instead of 0.4
    TauPartonIdx = find_genjet_indices(events.Tau, events.genTaupartonH, 0.3)
    TauPartonIdx = ak.pad_none(TauPartonIdx, 2, clip=True)
    mask_partons = ak.where(TauPartonIdx != None, True, False)
    mask_partons = ak.fill_none(mask_partons, False)
    # sum(mask_partons)=10352, tot numer of partons 7671*2->10352 of a total of 15242 could e matched
    # through delta R matching
    # Total num of Taus if a 3rd tau per event in incl. 10476, if 3rd tau not incl in count 10469
    # numbers are for ggf 1250
    unmatchedPartonPt1 = ak.fill_none(ak.mask(events.genTaupartonH, ~mask_partons).pt[:,0], EMPTY_FLOAT)
    unmatchedPartonPt2 = ak.fill_none(ak.mask(events.genTaupartonH, ~mask_partons).pt[:,1], EMPTY_FLOAT)
    unmatchedPartonEta1 = ak.fill_none(ak.mask(events.genTaupartonH, ~mask_partons).eta[:,0], EMPTY_FLOAT)
    unmatchedPartonEta2 = ak.fill_none(ak.mask(events.genTaupartonH, ~mask_partons).eta[:,1], EMPTY_FLOAT)

    events = set_ak_column(events, "Tau", ak.pad_none(events.Tau, 2))

    unmatchedTauPt1 = ak.fill_none(ak.mask(events.Tau[:, :2], ~mask_partons).pt[:,0], EMPTY_FLOAT)
    unmatchedTauPt2 = ak.fill_none(ak.mask(events.Tau[:, :2], ~mask_partons).pt[:,1], EMPTY_FLOAT)
    unmatchedTauEta1 = ak.fill_none(ak.mask(events.Tau[:, :2], ~mask_partons).eta[:,0], EMPTY_FLOAT)
    unmatchedTauEta2 = ak.fill_none(ak.mask(events.Tau[:, :2], ~mask_partons).eta[:,1], EMPTY_FLOAT)

    # put all unmatched taus in one column
    cat_pt = np.concatenate((unmatchedPartonPt1[unmatchedPartonPt1 != EMPTY_FLOAT], unmatchedPartonPt2[unmatchedPartonPt2 != EMPTY_FLOAT]))
    unmatchedPartonPt = np.full(len(events.genTaupartonH), EMPTY_FLOAT)
    unmatchedPartonPt[:len(cat_pt)] = cat_pt

    cat_eta = np.concatenate((unmatchedPartonEta1[unmatchedPartonEta1 != EMPTY_FLOAT], unmatchedPartonEta2[unmatchedPartonEta2 != EMPTY_FLOAT]))
    unmatchedPartonEta = np.full(len(events.genTaupartonH), EMPTY_FLOAT)
    unmatchedPartonEta[:len(cat_eta)] = cat_eta

    # same for Tau that could not be matched to a Parton
    cat_pt_tau = np.concatenate((unmatchedTauPt1[unmatchedTauPt1 != EMPTY_FLOAT], unmatchedTauPt2[unmatchedTauPt2 != EMPTY_FLOAT]))
    unmatchedTauPt = np.full(len(events.genTaupartonH), EMPTY_FLOAT)
    unmatchedTauPt[:len(cat_pt_tau)] = cat_pt_tau

    cat_eta_tau = np.concatenate((unmatchedTauEta1[unmatchedTauEta1 != EMPTY_FLOAT], unmatchedTauEta2[unmatchedTauEta2 != EMPTY_FLOAT]))
    unmatchedTauEta = np.full(len(events.genTaupartonH), EMPTY_FLOAT)
    unmatchedTauEta[:len(cat_eta_tau)] = cat_eta_tau

    events = set_ak_column_f32(events, "MatchedTauParton1Pt", ak.fill_none(events.genTaupartonH[TauPartonIdx].pt[:,0], EMPTY_FLOAT))
    events = set_ak_column_f32(events, "MatchedTauParton2Pt", ak.fill_none(events.genTaupartonH[TauPartonIdx].pt[:,1], EMPTY_FLOAT))
    events = set_ak_column_f32(events, "MatchedTauParton1Eta", ak.fill_none(events.genTaupartonH[TauPartonIdx].eta[:,0], EMPTY_FLOAT))
    events = set_ak_column_f32(events, "MatchedTauParton2Eta", ak.fill_none(events.genTaupartonH[TauPartonIdx].eta[:,1], EMPTY_FLOAT))
    events = set_ak_column_f32(events, "UnmatchedTauParton1Pt", ak.fill_none(unmatchedPartonPt1, EMPTY_FLOAT))
    events = set_ak_column_f32(events, "UnmatchedTauParton2Pt", ak.fill_none(unmatchedPartonPt2, EMPTY_FLOAT))
    events = set_ak_column_f32(events, "UnmatchedTauParton1Eta", ak.fill_none(unmatchedPartonEta1, EMPTY_FLOAT))
    events = set_ak_column_f32(events, "UnmatchedTauParton2Eta", ak.fill_none(unmatchedPartonEta2, EMPTY_FLOAT))
    events = set_ak_column_f32(events, "UnmatchedTauPartonPt", ak.fill_none(unmatchedPartonPt, EMPTY_FLOAT))
    events = set_ak_column_f32(events, "UnmatchedTauPartonEta", ak.fill_none(unmatchedPartonEta, EMPTY_FLOAT))
    events = set_ak_column_f32(events, "UnmatchedTau1Pt", ak.fill_none(unmatchedTauPt1, EMPTY_FLOAT))
    events = set_ak_column_f32(events, "UnmatchedTau2Pt", ak.fill_none(unmatchedTauPt2, EMPTY_FLOAT))
    events = set_ak_column_f32(events, "UnmatchedTau1Eta", ak.fill_none(unmatchedTauEta1, EMPTY_FLOAT))
    events = set_ak_column_f32(events, "UnmatchedTau2Eta", ak.fill_none(unmatchedTauEta2, EMPTY_FLOAT))
    events = set_ak_column_f32(events, "UnmatchedTauPt", ak.fill_none(unmatchedTauPt, EMPTY_FLOAT))
    events = set_ak_column_f32(events, "UnmatchedTauEta", ak.fill_none(unmatchedTauEta, EMPTY_FLOAT))

    return events
