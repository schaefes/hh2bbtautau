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
        "VBFJet.*", "AutoGenMatchedVBFJets.*", "GenMatchedVBFJets.*",
        attach_coffea_behavior,
    },
    produces={
        "VBFtagging_results_auto_1", "VBFtagging_results_dr_1",
        "VBFtagging_results_auto_0", "VBFtagging_results_dr_0",
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

    VBFtagging_results_auto_1 = ak.fill_none(VBFtagging_results_auto, 0)
    VBFtagging_results_dr_1 = ak.fill_none(VBFtagging_results_dr, 0)
    VBFtagging_results_dr_0 = ak.where(VBFtagging_results_dr_1 == -1, 0, VBFtagging_results_auto_1)
    VBFtagging_results_auto_0 = ak.where(VBFtagging_results_auto_1 == -1, 0, VBFtagging_results_auto_1)
    events = set_ak_column_f32(events, "VBFtagging_results_auto_1", ak.fill_none(VBFtagging_results_auto, 0))
    events = set_ak_column_f32(events, "VBFtagging_results_dr_1", VBFtagging_results_dr_1)
    events = set_ak_column_f32(events, "VBFtagging_results_auto_0", VBFtagging_results_auto_0)
    events = set_ak_column_f32(events, "VBFtagging_results_dr_0", VBFtagging_results_dr_0)

    return events


@producer(
    uses={
        "VBFJet.*",
        attach_coffea_behavior,
    },
    produces={
        "VBFPairsInEvent",

    },
)
def VBFPairsInEvent(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    events = self[attach_coffea_behavior](events, collections={"VBFJet": {"type_name": "Jet"}}, **kwargs)

    # Get events that have a VBF pair
    num_VBFJets = ak.num(events.VBFJet.pt, axis=1) / 2
    events = set_ak_column_f32(events, "VBFPairsInEvent", ak.fill_none(num_VBFJets, EMPTY_FLOAT))

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
    mask_ak4 = ak.fill_none(((genMatchedJets_inv_mass_ak4 > 500.0) & (genMatchedJets_dEta_ak4 > 3.0)), False)

    genMatchedJets_inv_mass_mask = (ak.mask(events.AutoGenMatchedVBFJets, vbfmask_mask)[:, 0] + ak.mask(events.AutoGenMatchedVBFJets, vbfmask_mask)[:, 1]).mass
    genMatchedJets_dEta_mask = abs(ak.mask(events.AutoGenMatchedVBFJets, vbfmask_mask)[:, 0].eta - ak.mask(events.AutoGenMatchedVBFJets, vbfmask_mask)[:, 1].eta)
    mask_mask = ak.fill_none(((genMatchedJets_inv_mass_mask > 500.0) & (genMatchedJets_dEta_mask > 3.0)), False)

    genMatchedJets_inv_mass_pairs = (ak.mask(events.AutoGenMatchedVBFJets, pairs_mask)[:, 0] + ak.mask(events.AutoGenMatchedVBFJets, pairs_mask)[:, 1]).mass
    genMatchedJets_dEta_pairs = abs(ak.mask(events.AutoGenMatchedVBFJets, pairs_mask)[:, 0].eta - ak.mask(events.AutoGenMatchedVBFJets, pairs_mask)[:, 1].eta)
    mask_pairs = ak.fill_none(((genMatchedJets_inv_mass_pairs > 500.0) & (genMatchedJets_dEta_pairs > 3.0)), False)

    genMatchedJets_inv_mass_trigger = (ak.mask(events.AutoGenMatchedVBFJets, trigger_mask)[:, 0] + ak.mask(events.AutoGenMatchedVBFJets, trigger_mask)[:, 1]).mass
    genMatchedJets_dEta_trigger = abs(ak.mask(events.AutoGenMatchedVBFJets, trigger_mask)[:, 0].eta - ak.mask(events.AutoGenMatchedVBFJets, trigger_mask)[:, 1].eta)
    mask_trigger = ak.fill_none(((genMatchedJets_inv_mass_trigger > 500.0) & (genMatchedJets_dEta_trigger > 3.0)), False)

    # Corresponing Partons
    vbfpartons_inv_mass_ak4 = (ak.mask(events.genVBFparton, ak4_mask)[:, 0] + ak.mask(events.genVBFparton, ak4_mask)[:, 1]).mass
    vbfpartons_dEta_ak4 = abs(ak.mask(events.genVBFparton, ak4_mask)[:, 0].eta - ak.mask(events.genVBFparton, ak4_mask)[:, 1].eta)

    vbfpartons_inv_mass_mask = (ak.mask(events.genVBFparton, vbfmask_mask)[:, 0] + ak.mask(events.genVBFparton, vbfmask_mask)[:, 1]).mass
    vbfpartons_dEta_mask = abs(ak.mask(events.genVBFparton, vbfmask_mask)[:, 0].eta - ak.mask(events.genVBFparton, vbfmask_mask)[:, 1].eta)

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

    events = set_ak_column_f32(events, "GenMatchedVBFJets_ak4_inv_mass", ak.fill_none(ak.mask(genMatchedJets_inv_mass_ak4, mask_ak4), EMPTY_FLOAT))
    events = set_ak_column_f32(events, "GenMatchedVBFJets_ak4_dEta", ak.fill_none(ak.mask(genMatchedJets_dEta_ak4, mask_ak4), EMPTY_FLOAT))
    events = set_ak_column_f32(events, "GenMatchedVBFJets_mask_inv_mass", ak.fill_none(ak.mask(genMatchedJets_inv_mass_mask, mask_mask), EMPTY_FLOAT))
    events = set_ak_column_f32(events, "GenMatchedVBFJets_mask_dEta", ak.fill_none(ak.mask(genMatchedJets_dEta_mask, mask_mask), EMPTY_FLOAT))
    events = set_ak_column_f32(events, "GenMatchedVBFJets_pairs_inv_mass", ak.fill_none(ak.mask(genMatchedJets_inv_mass_pairs, mask_pairs), EMPTY_FLOAT))
    events = set_ak_column_f32(events, "GenMatchedVBFJets_pairs_dEta", ak.fill_none(ak.mask(genMatchedJets_dEta_pairs, mask_pairs), EMPTY_FLOAT))
    events = set_ak_column_f32(events, "GenMatchedVBFJets_trigger_inv_mass", ak.fill_none(ak.mask(genMatchedJets_inv_mass_trigger, mask_trigger), EMPTY_FLOAT))
    events = set_ak_column_f32(events, "GenMatchedVBFJets_trigger_dEta", ak.fill_none(ak.mask(genMatchedJets_dEta_trigger, mask_trigger), EMPTY_FLOAT))

    events = set_ak_column_f32(events, "GenVBFPartons_ak4_inv_mass", ak.fill_none(ak.mask(vbfpartons_inv_mass_ak4, mask_ak4), EMPTY_FLOAT))
    events = set_ak_column_f32(events, "GenVBFPartons_ak4_dEta", ak.fill_none(ak.mask(vbfpartons_dEta_ak4, mask_ak4), EMPTY_FLOAT))
    events = set_ak_column_f32(events, "GenVBFPartons_mask_inv_mass", ak.fill_none(ak.mask(vbfpartons_inv_mass_mask, mask_mask), EMPTY_FLOAT))
    events = set_ak_column_f32(events, "GenVBFPartons_mask_dEta", ak.fill_none(ak.mask(vbfpartons_dEta_mask, mask_mask), EMPTY_FLOAT))
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

    # Get the invariant mass of the two VBF Partons and the Gen Matched VBF Jets and the delta eta of the pair
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

    # ak count only counts entries different from None, eg. [2, None] is counted as len 1
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

    # Get pt and eta of first Jet
    ak4_sort = ak.argsort(events.VBFak4_step.pt, axis=1)
    events = set_ak_column_f32(events, "Jets_ak4_pt", ak.fill_none(events.VBFak4_step.pt[ak4_sort][:, 0], EMPTY_FLOAT))
    events = set_ak_column_f32(events, "Jets_ak4_eta", ak.fill_none(events.VBFak4_step.eta[ak4_sort][:, 0], EMPTY_FLOAT))

    vbfmask_sort = ak.argsort(events.VBFmask_step.pt, axis=1)
    events = set_ak_column_f32(events, "Jets_mask_pt", ak.fill_none(events.VBFmask_step.pt[vbfmask_sort][:, 0], EMPTY_FLOAT))
    events = set_ak_column_f32(events, "Jets_mask_eta", ak.fill_none(events.VBFmask_step.eta[vbfmask_sort][:, 0], EMPTY_FLOAT))

    pairs_sort = ak.argsort(events.VBFpairs_step.pt, axis=1)
    events = set_ak_column_f32(events, "Jets_pairs_pt", ak.fill_none(events.VBFpairs_step.pt[pairs_sort][:, 0], EMPTY_FLOAT))
    events = set_ak_column_f32(events, "Jets_pairs_eta", ak.fill_none(events.VBFpairs_step.eta[pairs_sort][:, 0], EMPTY_FLOAT))

    trigger_sort = ak.argsort(events.VBFtrigger_step.pt, axis=1)
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
