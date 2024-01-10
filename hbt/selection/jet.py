# coding: utf-8

"""
Jet selection methods.
"""

from operator import or_
from functools import reduce

from columnflow.selection import Selector, SelectionResult, selector
from columnflow.util import maybe_import
from columnflow.columnar_util import set_ak_column

from hbt.production.hhbtag import hhbtag

from hbt.production.gen_producer import (gen_HH_decay_product_idx_b, gen_HH_decay_product_idx_tau,
    gen_HH_decay_product_idx_H, gen_HH_decay_product_VBF_idx, gen_HH_decay_product_VBF_sel,
    GenMatchingBJets, GenMatchingVBFJets)

from hbt.selection.helper_funcs import (mask_to_indices, get_jet_indices_from_pair_mask,
    get_unique_rows)


np = maybe_import("numpy")
ak = maybe_import("awkward")


@selector(
    uses={
        hhbtag,
        # custom columns created upstream, probably by a selector
        "trigger_ids",
        # nano columns
        "Jet.pt", "Jet.eta", "Jet.phi", "Jet.mass", "Jet.jetId", "Jet.puId",
        "Jet.btagDeepFlavB",
        "FatJet.pt", "FatJet.eta", "FatJet.phi", "FatJet.mass", "FatJet.msoftdrop",
        "FatJet.jetId", "FatJet.subJetIdx1", "FatJet.subJetIdx2",
        "SubJet.pt", "SubJet.eta", "SubJet.phi", "SubJet.mass", "SubJet.btagDeepB",
        "GenPart.*", gen_HH_decay_product_idx_b, gen_HH_decay_product_idx_tau,
        gen_HH_decay_product_idx_H, gen_HH_decay_product_VBF_idx, gen_HH_decay_product_VBF_sel,
        GenMatchingBJets, GenMatchingVBFJets,
    },
    produces={
        # new columns
        "Jet.hhbtag",
    },
    # shifts are declared dynamically below in jet_selection_init
)
def jet_selection(
    self: Selector,
    events: ak.Array,
    trigger_results: SelectionResult,
    lepton_results: SelectionResult,
    **kwargs,
) -> tuple[ak.Array, SelectionResult]:
    """
    Jet selection based on ultra-legacy recommendations.

    Resources:
    https://twiki.cern.ch/twiki/bin/view/CMS/JetID?rev=107#nanoAOD_Flags
    https://twiki.cern.ch/twiki/bin/view/CMS/JetID13TeVUL?rev=15#Recommendations_for_the_13_T_AN1
    https://twiki.cern.ch/twiki/bin/view/CMS/PileupJetIDUL?rev=17
    https://twiki.cern.ch/twiki/bin/view/CMSPublic/WorkBookNanoAOD?rev=100#Jets
    """
    is_2016 = self.config_inst.campaign.x.year == 2016

    # local jet index
    li = ak.local_index(events.Jet)

    # common ak4 jet mask for normal and vbf jets
    ak4_mask = (
        (events.Jet.jetId == 6) &  # tight plus lepton veto
        ((events.Jet.pt >= 50.0) | (events.Jet.puId == (1 if is_2016 else 4))) &  # flipped in 2016
        ak.all(events.Jet.metric_table(lepton_results.x.lepton_pair) > 0.5, axis=2)
    )
    indices_ak4_step = ak.mask(li, ak4_mask)

    # default jets
    default_mask = (
        ak4_mask &
        (events.Jet.pt > 20.0) &
        (abs(events.Jet.eta) < 2.4)
    )

    # get the scores of the hhbtag and per event get the two indices corresponding to the best pick
    hhbtag_scores = self[hhbtag](events, default_mask, lepton_results.x.lepton_pair, **kwargs)
    score_indices = ak.argsort(hhbtag_scores, axis=1, ascending=False)
    # pad the indices to simplify creating the hhbjet mask
    padded_hhbjet_indices = ak.pad_none(score_indices, 2, axis=1)[..., :2][..., :2]
    hhbjet_mask = ((li == padded_hhbjet_indices[..., [0]]) | (li == padded_hhbjet_indices[..., [1]]))
    # get indices for actual book keeping only for events with both lepton candidates and where at
    # least two jets pass the default mask (bjet candidates)
    valid_score_mask = (
        default_mask &
        (ak.sum(default_mask, axis=1) >= 2) &
        (ak.num(lepton_results.x.lepton_pair, axis=1) == 2)
    )
    hhbjet_indices = score_indices[valid_score_mask[score_indices]][..., :2]

    # vbf jets
    vbf_mask = (
        ak4_mask &
        (events.Jet.pt > 30.0) & #default is 30.0
        (abs(events.Jet.eta) < 4.7) &
        (~hhbjet_mask)
    )
    custom_vbf_mask = (
        ak4_mask &
        (events.Jet.pt > 30.0) & #default is 30.0
        (abs(events.Jet.eta) < 4.7) &
        (abs(events.Jet.eta) > 1.0)
    )
    custom_vbf_mask_2 = (
        ak4_mask &
        (events.Jet.pt > 30.0) & #default is 30.0
        (abs(events.Jet.eta) < 4.7)
    )
    indices_vbfmask_step = ak.mask(li, vbf_mask)
    indices_vbfmask = li[ak.fill_none(vbf_mask, False)]
    indices_custom_vbfmask = li[ak.fill_none(custom_vbf_mask, False)]
    indices_custom_vbfmask_2 = li[ak.fill_none(custom_vbf_mask_2, False)]

    # build vectors of vbf jets representing all combinations and apply selections
    vbf1, vbf2 = ak.unzip(ak.combinations(events.Jet[vbf_mask], 2, axis=1))
    vbf_pair = ak.concatenate([vbf1[..., None], vbf2[..., None]], axis=2)
    vbfjj = vbf1 + vbf2
    vbf_pair_mask = (
        (vbfjj.mass > 500.0) & # default is 500
        (abs(vbf1.eta - vbf2.eta) > 3.0) # default is 3.0
    )
    ### Pair Conditions Step
    # get the indices to the given invariant mass column
    vbf_mass_indices = ak.argsort(vbfjj.mass, axis=1, ascending=False)
    vbf_pairs = vbf_mass_indices[vbf_pair_mask[vbf_mass_indices]]
    indices_list = []
    for i in range(max(ak.count(vbf_pairs, axis=1))):
        vbf_pair_index = vbf_pairs[..., i:i+1]
        idx_jets = get_jet_indices_from_pair_mask(events.Jet, vbf_mask, li, vbf_pair_index)
        indices_list.append(idx_jets)
    indices_vbfpair_step = ak.concatenate(indices_list, axis=1)
    indices_vbfpair_step = get_unique_rows(indices_vbfpair_step)
    ###

    # extra requirements for events for which only the tau tau vbf cross trigger fired
    cross_vbf_ids = [t.id for t in self.config_inst.x.triggers if t.has_tag("cross_tau_tau_vbf")]
    if not cross_vbf_ids:
        cross_vbf_mask = ak.full_like(1 * events.event, False, dtype=bool)
    else:
        cross_vbf_masks = [events.trigger_ids == tid for tid in cross_vbf_ids]
        cross_vbf_mask = ak.all(reduce(or_, cross_vbf_masks), axis=1)
    vbf_pair_mask = vbf_pair_mask & (
        (~cross_vbf_mask) | (
            (vbfjj.mass > 800) &
            (ak.max(vbf_pair.pt, axis=2) > 140.0) &
            (ak.min(vbf_pair.pt, axis=2) > 60.0)
        )
    )
    # get the index to the pair with the highest pass
    vbf_mass_indices = ak.argsort(vbfjj.mass, axis=1, ascending=False)
    vbf_pair_index = vbf_mass_indices[vbf_pair_mask[vbf_mass_indices]][..., :1]

    # get the two indices referring to jets passing vbf_mask
    # and change them so that they point to jets in the full set, sorted by pt
    vbf_indices_local = ak.concatenate(
        [
            ak.singletons(idx) for idx in
            ak.unzip(ak.firsts(ak.argcombinations(events.Jet[vbf_mask], 2, axis=1)[vbf_pair_index]))
        ],
        axis=1,
    )
    vbfjet_indices = li[vbf_mask][vbf_indices_local]
    vbfjet_indices = vbfjet_indices[ak.argsort(events.Jet[vbfjet_indices].pt, axis=1, ascending=False)]

    ### Trigger Step
    # get the indices to the given invariant mass column
    vbf_mass_indices = ak.argsort(vbfjj.mass, axis=1, ascending=False)
    vbf_pairs = vbf_mass_indices[vbf_pair_mask[vbf_mass_indices]]
    indices_list = []
    for i in range(max(ak.count(vbf_pairs, axis=1))):
        vbf_pair_index = vbf_pairs[..., i:i+1]
        idx_jets = get_jet_indices_from_pair_mask(events.Jet, vbf_mask, li, vbf_pair_index)
        indices_list.append(idx_jets)
    indices_vbftrigger_step = ak.concatenate(indices_list, axis=1)
    indices_vbftrigger_step = get_unique_rows(indices_vbftrigger_step)
    ###

    # check whether the two bjets were matched by fatjet subjets to mark it as boosted
    fatjet_mask = (
        (events.FatJet.jetId == 6) &  # tight plus lepton veto
        (events.FatJet.msoftdrop > 30.0) &
        (abs(events.FatJet.eta) < 2.4) &
        ak.all(events.FatJet.metric_table(lepton_results.x.lepton_pair) > 0.5, axis=2) &
        (events.FatJet.subJetIdx1 >= 0) &
        (events.FatJet.subJetIdx2 >= 0)
    )

    # unique subjet matching
    metrics = events.FatJet.subjets.metric_table(events.Jet[hhbjet_indices])
    subjets_match = (
        ak.all(ak.sum(metrics < 0.4, axis=3) == 1, axis=2) &
        (ak.num(hhbjet_indices, axis=1) == 2)
    )
    fatjet_mask = fatjet_mask & subjets_match

    # store fatjet and subjet indices
    fatjet_indices = ak.local_index(events.FatJet.pt)[fatjet_mask]
    subjet_indices = ak.concatenate(
        [
            events.FatJet[fatjet_mask].subJetIdx1[..., None],
            events.FatJet[fatjet_mask].subJetIdx2[..., None],
        ],
        axis=2,
    )

    # discard the event in case the (first) fatjet with matching subjets is found
    # but they are not b-tagged (TODO: move to deepjet when available for subjets)
    wp = self.config_inst.x.btag_working_points.deepcsv.loose
    subjets_btagged = ak.all(events.SubJet[ak.firsts(subjet_indices)].btagDeepB > wp, axis=1)

    # pt sorted indices to convert mask
    sorted_indices = ak.argsort(events.Jet.pt, axis=-1, ascending=False)
    ak4_btag_wp = self.config_inst.x.btag_working_points.deepjet.medium
    bjet_indices = sorted_indices[events.Jet[sorted_indices].btagDeepFlavB > ak4_btag_wp]
    bjet_indices = bjet_indices[default_mask[bjet_indices]]
    jet_indices = sorted_indices[default_mask[sorted_indices]]

    # keep indices of default jets that are explicitly not selected as hhbjets for easier handling
    non_hhbjet_mask = default_mask & (~hhbjet_mask)
    non_hhbjet_indices = sorted_indices[non_hhbjet_mask[sorted_indices]]

    # final event selection
    jet_sel = (
        (ak.sum(default_mask, axis=1) >= 2) &
        ak.fill_none(subjets_btagged, True)  # was none for events with no matched fatjet
    )

    # GenVBF_sel = self[gen_HH_decay_product_VBF](events, **kwargs)[0]

    colljet_indices = ak.concatenate((jet_indices, vbfjet_indices), axis=1)
    vbfmask_indices = ak.concatenate((jet_indices, indices_vbfmask), axis=1)
    custom_vbfmask_indices = ak.concatenate((jet_indices, indices_custom_vbfmask), axis=1)
    custom_vbfmask_indices_2 = ak.concatenate((jet_indices, indices_custom_vbfmask_2), axis=1)
    unique_colljet_indices = []
    unique_vbfmask_indices = []
    unique_custom_vbfmask_indices = []
    unique_custom_vbfmask_indices_2 = []
    for (coll_idx, vbfmask_idx, custom_vbfmask_idx, custom_vbfmask_idx_2) in zip(colljet_indices, vbfmask_indices, custom_vbfmask_indices, custom_vbfmask_indices_2):
        unique_colljet_indices.append(np.unique(coll_idx))
        unique_vbfmask_indices.append(np.unique(vbfmask_idx))
        unique_custom_vbfmask_indices.append(np.unique(custom_vbfmask_idx))
        unique_custom_vbfmask_indices_2.append(np.unique(custom_vbfmask_idx_2))
    colljet_indices = ak.Array(unique_colljet_indices)
    vbfmask_indices = ak.Array(unique_vbfmask_indices)
    custom_vbfmask_indices = ak.Array(unique_custom_vbfmask_indices)
    custom_vbfmask_indices_2 = ak.Array(unique_custom_vbfmask_indices_2)

    # some final type conversions
    jet_indices = ak.values_astype(ak.fill_none(jet_indices, 0), np.int32)
    subjet_indices = ak.values_astype(ak.fill_none(subjet_indices, 0), np.int32)
    hhbjet_indices = ak.values_astype(hhbjet_indices, np.int32)
    bjet_indices = ak.values_astype(bjet_indices, np.int32)
    non_hhbjet_indices = ak.values_astype(ak.fill_none(non_hhbjet_indices, 0), np.int32)
    fatjet_indices = ak.values_astype(fatjet_indices, np.int32)
    vbfjet_indices = ak.values_astype(ak.fill_none(vbfjet_indices, 0), np.int32)
    colljet_indices = ak.values_astype(ak.fill_none(colljet_indices, 0), np.int32)
    custom_vbfmask_indices = ak.values_astype(ak.fill_none(custom_vbfmask_indices, 0), np.int32)
    custom_vbfmask_indices_2 = ak.values_astype(ak.fill_none(custom_vbfmask_indices_2, 0), np.int32)

    genBpartonHidx = self[gen_HH_decay_product_idx_b](events, **kwargs)
    genTaupartonHidx, genTaupartonHidxE, genTaupartonHidxMu, genTaupartonHidxHad = self[gen_HH_decay_product_idx_tau](events, **kwargs)
    genHpartonidx = self[gen_HH_decay_product_idx_H](events, **kwargs)
    genVBFpartonIdx = self[gen_HH_decay_product_VBF_idx](events, **kwargs)

    # Gen Matchd indices
    # events = set_ak_column(events, "genBpartonH", events.GenPart[genBpartonHidx])
    # events = set_ak_column(events, "genVBFparton", events.GenPart[genVBFpartonIdx])
    # genMatchingBJets_indices, genMatchingGenBJets_indices = self[GenMatchingBJets](events, genBpartonHidx, **kwargs)
    # genMatchingVBFJets_indices, genMatchingGenVBFJets_indices, auto_genMatchingVBFJets_indices = self[GenMatchingVBFJets](events, genVBFpartonIdx, **kwargs)

    # partons that can be matched to a reco jet
    # mask_partons = ~ak.is_none(auto_genMatchingVBFJets_indices, axis=1)
    # matchedGenVBFpartonIdx = ak.mask(genVBFpartonIdx, mask_partons)
    if self.dataset_inst.has_tag("is_vbf"): # and self.get("dataset_inst", False):
        VBF_sel = self[gen_HH_decay_product_VBF_sel](events, genVBFpartonIdx, False, **kwargs)
        # If VBF is considered, fuse VBF_sel and jet_sel using logical and, excludes VHH events
        jet_sel = np.logical_and(jet_sel, VBF_sel)
    # store some columns
    events = set_ak_column(events, "Jet.hhbtag", hhbtag_scores)
    # build and return selection results plus new columns (src -> dst -> indices)
    object_dict = {
        "Jet": {
            "Jet": jet_indices,
            "BJet": bjet_indices,
            "HHBJet": hhbjet_indices,
            "NonHHBJet": non_hhbjet_indices,
            "SubJet1": subjet_indices[..., 0],
            "SubJet2": subjet_indices[..., 1],
            "VBFJet": vbfjet_indices,
            "CustomVBFMaskJets": custom_vbfmask_indices,
            "CustomVBFMaskJets2": custom_vbfmask_indices_2,
            # "GenMatchedBJets": genMatchingBJets_indices,
            # "VBFak4_step": indices_ak4_step,
            # "VBFmask_step": indices_vbfmask_step,
            # "VBFpairs_step": indices_vbfpair_step,
            # "VBFtrigger_step": indices_vbftrigger_step,
        },
        # "FatJet": {
        #     "FatJet": fatjet_indices,
        # },
        # "GenPart": {
        #     "genBpartonH": genBpartonHidx,
        #     "genTaupartonH": genTaupartonHidx,
        #     "genTaupartonHE": genTaupartonHidxE,
        #     "genTaupartonHMu": genTaupartonHidxMu,
        #     "genTaupartonHHad": genTaupartonHidxHad,
        #     "genHpartonH": genHpartonidx,

        # },
        # "GenJet": {
        #     "genMatchedGenBJets": genMatchingGenBJets_indices,

        # }
    }

    # if self.dataset_inst.has_tag("is_vbf"):
    #     object_dict["Jet"].update({
    #         "GenMatchedVBFJets": genMatchingVBFJets_indices,
    #         "AutoGenMatchedVBFJets": auto_genMatchingVBFJets_indices,
    #         }
    #     )
    #     object_dict["GenPart"].update({
    #         "genVBFparton": genVBFpartonIdx,
    #         "matchedGenVBFparton": matchedGenVBFpartonIdx,
    #     })
    #     object_dict["GenJet"].update({
    #         "genMatchedGenVBFJets": genMatchingGenVBFJets_indices,
    #     })
    return events, SelectionResult(
        steps={
            "jet": jet_sel,
            # the btag weight normalization requires a selection with everything but the bjet
            # selection, so add this step here
            # note: there is currently no b-tag discriminant cut at this point, so take jet_sel
            "bjet": jet_sel,
        },
        objects=object_dict,
        aux={
            # jet mask that lead to the jet_indices
            "jet_mask": default_mask,
            # used to determine sum of weights in increment_stats
            "n_central_jets": ak.num(jet_indices, axis=1),
        },
    )


@jet_selection.init
def jet_selection_init(self: Selector) -> None:
    # register shifts
    self.shifts |= {
        shift_inst.name
        for shift_inst in self.config_inst.shifts
        if shift_inst.has_tag(("jec", "jer"))
    }
