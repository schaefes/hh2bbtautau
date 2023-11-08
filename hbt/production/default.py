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
from hbt.production.invariant_mass import (invariant_mass_jets, invariant_mass_tau,
    invariant_mass_bjets, invariant_mass_HH, kinematic_vars_taus, kinematic_vars_jets,
    kinematic_vars_bjets, jet_information, bjet_information, tau_information,
    kinematic_vars_colljets, dr_inv_mass_jets, d_eta_inv_mass_jets, energy_correlation,
    genBPartonProducer, genTauPartonProducer, genHPartonProducer, genVBFPartonProducer,
    kinematic_vars_VBFjets, dR_bjets, dR_tau, Btagging_efficiency_Bpartons,
    VBFsteps_analysis, GenInvMassdEta, NumberOfMatchedPartons, VBFCandidatesInvMassdEta)
from hbt.production.Tau_plotting import (kinematics_tau_partons, PartonsFromTauIdx, PartonsFromHadDecay)
from hbt.production.ml_producer import (kinematic_vars_vbfmaskjets, kinematic_vars_customvbfmaskjets,
                                        kinematic_vars_customvbfmaskjets2)
from hbt.production.VBF_plotting import (VBFtagging_efficiency, VBFPairsInEvent)


ak = maybe_import("awkward")


@producer(
    uses={
        category_ids, features, normalization_weights, normalized_pdf_weight,
        normalized_murmuf_weight, normalized_pu_weight, normalized_btag_weights,
        tau_weights, electron_weights, muon_weights, trigger_weights, invariant_mass_jets,
        invariant_mass_tau, invariant_mass_bjets, invariant_mass_HH, kinematic_vars_taus,
        kinematic_vars_jets, kinematic_vars_bjets, jet_information, bjet_information,
        tau_information, kinematic_vars_colljets, dr_inv_mass_jets, d_eta_inv_mass_jets,
        energy_correlation, genBPartonProducer, genTauPartonProducer, genHPartonProducer, genVBFPartonProducer,
        kinematic_vars_VBFjets, dR_bjets, dR_tau, Btagging_efficiency_Bpartons,
        VBFtagging_efficiency, VBFsteps_analysis, GenInvMassdEta, NumberOfMatchedPartons,
        VBFCandidatesInvMassdEta, kinematics_tau_partons, PartonsFromTauIdx,
        kinematic_vars_vbfmaskjets, kinematic_vars_customvbfmaskjets, PartonsFromHadDecay,
        VBFPairsInEvent, kinematic_vars_customvbfmaskjets2,
    },
    produces={
        category_ids, features, normalization_weights, normalized_pdf_weight,
        normalized_murmuf_weight, normalized_pu_weight, normalized_btag_weights,
        tau_weights, electron_weights, muon_weights, trigger_weights, invariant_mass_jets,
        invariant_mass_tau, invariant_mass_bjets, invariant_mass_HH, kinematic_vars_taus,
        kinematic_vars_jets, kinematic_vars_bjets, jet_information, bjet_information,
        tau_information, kinematic_vars_colljets, dr_inv_mass_jets, d_eta_inv_mass_jets,
        energy_correlation, genBPartonProducer, genTauPartonProducer,
        genHPartonProducer, genVBFPartonProducer, kinematic_vars_VBFjets, dR_bjets, dR_tau,
        Btagging_efficiency_Bpartons, VBFtagging_efficiency, VBFsteps_analysis,
        GenInvMassdEta, NumberOfMatchedPartons, VBFCandidatesInvMassdEta,
        kinematics_tau_partons, PartonsFromTauIdx, kinematic_vars_vbfmaskjets,
        kinematic_vars_customvbfmaskjets, PartonsFromHadDecay, VBFPairsInEvent,
        kinematic_vars_customvbfmaskjets2,
    },
)
def default(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    '''category ids'''
    events = self[category_ids](events, **kwargs)

    '''features'''
    events = self[features](events, **kwargs)

    '''invariant masses'''

    events = self[invariant_mass_jets](events, **kwargs)

    events = self[invariant_mass_bjets](events, **kwargs)

    events = self[invariant_mass_tau](events, **kwargs)

    events = self[invariant_mass_HH](events, **kwargs)

    '''kinetmatatic vars for jets, bjets, taus'''

    # events = self[kinematic_vars_colljets](events, **kwargs)

    # events = self[kinematic_vars_vbfmaskjets](events, **kwargs)

    # events = self[kinematic_vars_customvbfmaskjets](events, **kwargs)

    events = self[kinematic_vars_customvbfmaskjets2](events, **kwargs)

    # events = self[dr_inv_mass_jets](events, **kwargs)

    # events = self[d_eta_inv_mass_jets](events, **kwargs)

    # events = self[energy_correlation](events, **kwargs)

    # ----

    # events = self[kinematic_vars_bjets](events, **kwargs)

    # events = self[kinematic_vars_taus](events, **kwargs)

    # information on the jets and bjets

    # events = self[jet_information](events, **kwargs)

    # events = self[bjet_information](events, **kwargs)

    # events = self[tau_information](events, **kwargs)

    '''GenPart Information'''

    # events = self[genBPartonProducer](events, **kwargs)

    # events = self[genTauPartonProducer](events, **kwargs)

    # events = self[genHPartonProducer](events, **kwargs)

    # events = self[dR_bjets](events, **kwargs)

    # events = self[dR_tau](events, **kwargs)

    # events = self[Btagging_efficiency_Bpartons](events, **kwargs)

    '''(Gen) VBF Plotting'''
    # events = self[genVBFPartonProducer](events, **kwargs)

    # events = self[kinematic_vars_VBFjets](events, **kwargs)

    # events = self[VBFtagging_efficiency](events, **kwargs)

    # events = self[VBFPairsInEvent](events, **kwargs)

    # events = self[VBFsteps_analysis](events, **kwargs)

    # events = self[GenInvMassdEta](events, **kwargs)

    # events = self[NumberOfMatchedPartons](events, **kwargs)

    # events = self[VBFCandidatesInvMassdEta](events, **kwargs)

    '''Tau Parton Plotting'''
    # events = self[kinematics_tau_partons](events, **kwargs)

    # events = self[PartonsFromTauIdx](events, **kwargs)

    # events = self[PartonsFromHadDecay](events, **kwargs)

    # mc-only weights
    if self.dataset_inst.is_mc:
        # normalization weights
        events = self[normalization_weights](events, **kwargs)

        # normalized pdf weight
        events = self[normalized_pdf_weight](events, **kwargs)

        # normalized renorm./fact. weight
        events = self[normalized_murmuf_weight](events, **kwargs)

        # normalized pu weights
        events = self[normalized_pu_weight](events, **kwargs)

        # btag weights
        events = self[normalized_btag_weights](events, **kwargs)

        # tau weights
        events = self[tau_weights](events, **kwargs)

        # electron weights
        events = self[electron_weights](events, **kwargs)

        # muon weights
        events = self[muon_weights](events, **kwargs)

        # trigger weights
        events = self[trigger_weights](events, **kwargs)

        # invariant mass of jets, bjets, taus, HH sys
        events = self[invariant_mass_jets](events, **kwargs)

        events = self[invariant_mass_tau](events, **kwargs)

        events = self[invariant_mass_bjets](events, **kwargs)

        events = self[invariant_mass_HH](events, **kwargs)

    return events
