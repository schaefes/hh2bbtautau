# coding: utf-8

"""
Wrappers for some default sets of producers.
"""

from columnflow.production import Producer, producer
from columnflow.production.normalization import normalization_weights
from columnflow.production.categories import category_ids
from columnflow.production.cms.electron import electron_weights
from columnflow.production.cms.muon import muon_weights
from columnflow.util import maybe_import, dev_sandbox

from hbt.production.features import features
from hbt.production.weights import normalized_pu_weight, normalized_pdf_weight, normalized_murmuf_weight
from hbt.production.btag import normalized_btag_weights
from hbt.production.tau import tau_weights, trigger_weights
from hbt.production.invariant_mass import genBPartonProducer,genVBFPartonProducer
from columnflow.production.util import attach_coffea_behavior


from columnflow.columnar_util import EMPTY_FLOAT, Route, set_ak_column
import functools


np = maybe_import("numpy")
ak = maybe_import("awkward")


set_ak_column_f32 = functools.partial(set_ak_column, value_type=np.float32)


# Helper func to compute invariant mass of two objects
def inv_mass(obj1, obj2):
    obj_mass = (obj1 + obj2).mass

    return obj_mass


# B parton indices
@producer(
    uses={
        # nano columns
        "event",
        "GenPart.*",
    },
    sandbox=dev_sandbox("bash::$HBT_BASE/sandboxes/venv_columnar.sh"),
)
def gen_HH_decay_product_idx_b(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    """
    Creates a new ragged column "gen_top_decay" with one element per hard top quark. Each element is
    a GenParticleArray with five or more objects in a distinct order: top quark, bottom quark,
    W boson, down-type quark or charged lepton, up-type quark or neutrino, and any additional decay
    produces of the W boson (if any, then most likly photon radiations). Per event, the structure
    will be similar to:

    .. code-block:: python

        [[t1, b1, W1, q1/l, q2/n(, additional_w_decay_products)], [t2, ...], ...]
    """

    def find_partons(events: ak.Array, pdgId: int, mother_pdgId: int = 25):
        # get all GenPart indices
        idx = ak.local_index(events.GenPart, axis=1)

        # filter requirements for interesting partons
        abs_id = abs(events.GenPart.pdgId)
        mask = (
            (abs_id == pdgId) &
            events.GenPart.hasFlags("isHardProcess") &
            (abs(events.GenPart.distinctParent.pdgId) == mother_pdgId)
        )

        # fill None values with False
        mask = ak.fill_none(mask, False)
        idx = idx[mask]
        return idx

    return find_partons(events, 5, 25)


# Tau Producer
@producer(
    uses={
        # nano columns
        "event",
        "GenPart.*",
        "Tau.*"
    },
    sandbox=dev_sandbox("bash::$HBT_BASE/sandboxes/venv_columnar.sh"),
)
def gen_HH_decay_product_idx_tau(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    """
    Creates a new ragged column "gen_top_decay" with one element per hard top quark. Each element is
    a GenParticleArray with five or more objects in a distinct order: top quark, bottom quark,
    W boson, down-type quark or charged lepton, up-type quark or neutrino, and any additional decay
    produces of the W boson (if any, then most likly photon radiations). Per event, the structure
    will be similar to:

    .. code-block:: python

        [[t1, b1, W1, q1/l, q2/n(, additional_w_decay_products)], [t2, ...], ...]
    """

    def find_partons(events: ak.Array, pdgId: int, mother_pdgId: int = 25):
        # get all GenPart indices
        idx = ak.local_index(events.GenPart, axis=1)

        # filter requirements for interesting partons
        abs_id = abs(events.GenPart.pdgId)
        mask = (
            (abs_id == pdgId) &
            events.GenPart.hasFlags("isHardProcess") &
            (abs(events.GenPart.distinctParent.pdgId) == mother_pdgId)
        )

        # fill None values with False
        mask = ak.fill_none(mask, False)
        idx = idx[mask]

        return idx

    return find_partons(events, 15, 25)


# Higgs Producer
@producer(
    uses={
        # nano columns
        "event",
        "GenPart.*",
    },
    sandbox=dev_sandbox("bash::$HBT_BASE/sandboxes/venv_columnar.sh"),
)
def gen_HH_decay_product_idx_H(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    """
    Creates a new ragged column "gen_top_decay" with one element per hard top quark. Each element is
    a GenParticleArray with five or more objects in a distinct order: top quark, bottom quark,
    W boson, down-type quark or charged lepton, up-type quark or neutrino, and any additional decay
    produces of the W boson (if any, then most likly photon radiations). Per event, the structure
    will be similar to:

    .. code-block:: python

        [[t1, b1, W1, q1/l, q2/n(, additional_w_decay_products)], [t2, ...], ...]
    """

    def find_partons(events: ak.Array, pdgId: int, mother_pdgId: int = 25):
        # get all GenPart indices
        idx = ak.local_index(events.GenPart, axis=1)

        # filter requirements for interesting partons
        abs_id = abs(events.GenPart.pdgId)
        mask = (
            (abs_id == pdgId) &
            events.GenPart.hasFlags("isHardProcess")
        )

        # fill None values with False
        mask = ak.fill_none(mask, False)
        idx = idx[mask]

        return idx

    return find_partons(events, 25, 25)


# Producer for finding the VBF partons
@producer(
    uses={
        # nano columns
        "event",
        "GenPart.*",
    },
    sandbox=dev_sandbox("bash::$HBT_BASE/sandboxes/venv_columnar.sh"),
)
def gen_HH_decay_product_VBF_idx(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    """
    Creates a new ragged column "gen_top_decay" with one element per hard top quark. Each element is
    a GenParticleArray with five or more objects in a distinct order: top quark, bottom quark,
    W boson, down-type quark or charged lepton, up-type quark or neutrino, and any additional decay
    produces of the W boson (if any, then most likly photon radiations). Per event, the structure
    will be similar to:

    .. code-block:: python

        [[t1, b1, W1, q1/l, q2/n(, additional_w_decay_products)], [t2, ...], ...]
    """

    # find parton indices of the VBF Jets
    def find_partons(events: ak.Array):
        # get all GenPart indices
        idx = ak.local_index(events.GenPart, axis=1)

        # filter requirements for interesting partons
        abs_id = abs(events.GenPart.pdgId)
        mask_incoming = (
            (events.GenPart.status != 21) &
            events.GenPart.hasFlags("isHardProcess") &
            (abs_id < 6)
        )
        mask_incoming = ak.fill_none(mask_incoming, False)

        mask_b_from_H = (
            (abs_id == 5) &
            events.GenPart.hasFlags("isHardProcess") &
            (abs(events.GenPart.distinctParent.pdgId) == 25)
        )
        mask_b_from_H = ak.fill_none(mask_b_from_H, False)

        # fill None values with False
        mask = mask_incoming & ~mask_b_from_H
        idx = idx[mask]

        return idx

    return find_partons(events)


# Producer for the VBF Selection, filtering out VHH events and applyies the same kinemtaic selection to the VBF partons
# that is required for the VBF Jets if kinemtaic set to True (returns a mask)
@producer(
    uses={
        # nano columns
        "event",
        "GenPart.*",
    },
    sandbox=dev_sandbox("bash::$HBT_BASE/sandboxes/venv_columnar.sh"),
)
def gen_HH_decay_product_VBF_sel(self: Producer, events: ak.Array, genVBFpartonIndices, kinemtaic, **kwargs) -> ak.Array:
    """
    Creates a new ragged column "gen_top_decay" with one element per hard top quark. Each element is
    a GenParticleArray with five or more objects in a distinct order: top quark, bottom quark,
    W boson, down-type quark or charged lepton, up-type quark or neutrino, and any additional decay
    produces of the W boson (if any, then most likly photon radiations). Per event, the structure
    will be similar to:

    .. code-block:: python

        [[t1, b1, W1, q1/l, q2/n(, additional_w_decay_products)], [t2, ...], ...]
    """

    # selection mask for the VBF events
    def VBF_sel(events: ak.Array, genVBFpartonIndices, kinemtaic):

        # filter requirements for interesting partons
        abs_id = abs(events.GenPart.pdgId)
        mask = (
            (events.GenPart.status != 21) &
            (events.GenPart.hasFlags("isHardProcess")) &
            (abs_id < 7) &
            ((abs(events.GenPart.distinctParent.pdgId) == 24) | (abs(events.GenPart.distinctParent.pdgId) == 23))
        )

        mask = ak.fill_none(mask, False)
        mask_VBF_events = (np.sum(mask, axis=1) == 0)
        VBF_partons = events.GenPart[genVBFpartonIndices]

        # kinemetaic selection applied to VBF Jets: eta < 4.7 and pt > 30.0 GeV
        mask_row_0_eta = np.where(abs(VBF_partons.eta[:, 0]) < 4.7, True, False)
        mask_row_0_pt = np.where(VBF_partons.pt[:, 0] > 30.0, True, False)
        mask_row_0 = np.logical_and(mask_row_0_pt, mask_row_0_eta)
        mask_row_1_eta = np.where(abs(VBF_partons.eta[:, 1]) < 4.7, True, False)
        mask_row_1_pt = np.where(VBF_partons.pt[:, 1] > 30.0, True, False)
        mask_row_1 = np.logical_and(mask_row_1_pt, mask_row_1_eta)

        kinemtaic_mask = np.logical_and(mask_row_0, mask_row_1)

        if kinemtaic:
            mask_VBF_events = np.logical_and(mask_VBF_events, kinemtaic_mask)

        return mask_VBF_events

    return VBF_sel(events, genVBFpartonIndices, kinemtaic)


@producer(
    uses={
        "GenJet.*", "Jet.*", "GenPart.*",
        genBPartonProducer,
        genVBFPartonProducer,

    },
    sandbox=dev_sandbox("bash::$HBT_BASE/sandboxes/venv_columnar.sh"),
)
def GenMatchingBJets(self: Producer, events: ak.Array, genBpartonIndices, **kwargs) -> ak.Array:

    def find_genjet_indices(array1: ak.Array, array2: ak.Array):
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
        genjet_indices = ak.firsts(minimum_deltar_indices.mask[metric <= 0.4], axis=2)

        return genjet_indices

    genBjets = events.GenJet[abs(events.GenJet.partonFlavour) == 5]
    genBpartonH = events.GenPart[genBpartonIndices]

    genmatchedGenBjet_indices = find_genjet_indices(array1=genBpartonH, array2=genBjets)
    genmatchedBJet_indices = find_genjet_indices(array1=genBjets[genmatchedGenBjet_indices], array2=events.Jet)

    return genmatchedBJet_indices, genmatchedGenBjet_indices


@producer(
    uses={
        "GenJet.*", "Jet.*", "GenPart.*",
        genBPartonProducer,
        genVBFPartonProducer,

    },
    sandbox=dev_sandbox("bash::$HBT_BASE/sandboxes/venv_columnar.sh"),
)
def GenMatchingVBFJets(self: Producer, events: ak.Array, genVBFpartonIndices, **kwargs) -> ak.Array:

    def find_genjet_indices(array1: ak.Array, array2: ak.Array):
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
        genjet_indices = ak.firsts(minimum_deltar_indices.mask[metric <= 0.4], axis=2)

        return genjet_indices

    # Matching only through dR criterium
    genJets = events.GenJet
    genVBFparton = events.GenPart[genVBFpartonIndices]

    genmatchedGenVBFjet_indices = find_genjet_indices(array1=genVBFparton, array2=genJets)
    genmatchedVBFJet_indices = find_genjet_indices(array1=genJets[genmatchedGenVBFjet_indices], array2=events.Jet)
    # End: Matching only through dR criterium

    # Matching throgh dR from Parton to Gen Jets and usage of genJetIdx of Jet collection
    # to match Gen Jets to Reco Jets
    events = set_ak_column(events, "GenJet", ak.pad_none(events.GenJet, max(ak.count(events.Jet.genJetIdx, axis=1))))
    pad_idx = ak.pad_none(events.Jet.genJetIdx, max(ak.count(events.Jet.genJetIdx, axis=1)))
    auto_genJets = events.GenJet[pad_idx]

    auto_genmatchedGenVBFjet_indices = find_genjet_indices(array1=genVBFparton, array2=auto_genJets)
    auto_genmatchedVBFJets_indices = ak.to_regular(auto_genmatchedGenVBFjet_indices)
    # End: Matching dR and genJetIdx.

    return genmatchedVBFJet_indices, genmatchedGenVBFjet_indices, auto_genmatchedVBFJets_indices
