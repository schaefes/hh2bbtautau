# coding: utf-8

"""
Definition of categories.
"""

import order as od

from columnflow.config_util import add_category

from collections import OrderedDict


def add_categories(config: od.Config) -> None:
    """
    Adds all categories to a *config*.
    """
    add_category(
        config,
        name="incl",
        id=1,
        selection="sel_incl",
        label="inclusive",
    )
    add_category(
        config,
        name="2j",
        id=100,
        selection="sel_2j",
        label="2 jets",
    )

    # add extra categories for VBF Sel plots
    add_category(
        config,
        name="kin_selections_ak4",
        id=2,
        selection="sel_incl",
        label="Kinematic Gen Sel.\nAk4 Sel.",
    )
    add_category(
        config,
        name="no_kin_selections_ak4",
        id=3,
        selection="sel_incl",
        label="Ak4 Sel.",
    )
    add_category(
        config,
        name="kin_selections_vbfmask",
        id=4,
        selection="sel_incl",
        label="Kinematic Gen Sel.\nVBF Mask Sel.",
    )
    add_category(
        config,
        name="no_kin_selections_vbfmask",
        id=5,
        selection="sel_incl",
        label="VBF Mask Sel.",
    )
    add_category(
        config,
        name="kin_selections_pairs",
        id=6,
        selection="sel_incl",
        label="Kinematic Gen Sel.\nVBF Pair Sel.",
    )
    add_category(
        config,
        name="no_kin_selections_pairs",
        id=7,
        selection="sel_incl",
        label="VBF Pair Sel.",
    )
    add_category(
        config,
        name="kin_selections_trigger",
        id=8,
        selection="sel_incl",
        label="Kinematic Gen Sel.\nVBF Trigger Sel.",
    )
    add_category(
        config,
        name="no_kin_selections_trigger",
        id=9,
        selection="sel_incl",
        label="VBF Trigger Sel.",
    )

    # add categires for the lepton sel cutflow plots
    add_category(
        config,
        name="no_lepton_veto",
        id=10,
        selection="sel_incl",
        label="No Lepton Veto",
    )
    add_category(
        config,
        name="no_tau_iso",
        id=11,
        selection="sel_incl",
        label="No $\\tau$ Iso",
    )
    add_category(
        config,
        name="no_tau_indices1",
        id=12,
        selection="sel_incl",
        label="No min #$\\tau$ for\n$e$ and $\\mu$ ",
    )
    add_category(
        config,
        name="tau_15pt_2_5eta",
        id=13,
        selection="sel_incl",
        label="Min $\\tau$ $p_{T}$=15GeV\nMax $\\eta$=2.5",
    )
    add_category(
        config,
        name="tau_e_mu_15pt_2_5eta",
        id=14,
        selection="sel_incl",
        label="Min $\\tau$,$e$,$\\mu$\n$p_{T}$=15GeV\nMax $\\eta$=2.5",
    )
    add_category(
        config,
        name="DeepTau_keep_highest_bin",
        id=15,
        selection="sel_incl",
        label="DeepTauId max\nscore kept",
    )
    add_category(
        config,
        name="Combinedv1",
        id=16,
        selection="sel_incl",
        label="Kinematic\nDeepTauId\nNo $\\tau$ Iso",
    )
    add_category(
        config,
        name="no_tau_indices_at_all",
        id=17,
        selection="sel_incl",
        label="No min #$\\tau$ for\n$e$, $\\mu$, $\\tau$",
    )
    add_category(
        config,
        name="no_tau_iso_mask",
        id=18,
        selection="sel_incl",
        label="No $\\tau$ Iso Mask",
    )
    add_category(
        config,
        name="Combinedv2",
        id=19,
        selection="sel_incl",
        label="Combined Cuts\nV2",
    )

def add_categories_ml(config, ml_model_inst):

    # add ml categories directly to the config
    ml_categories = []
    for i, proc in enumerate(ml_model_inst.processes):
        ml_categories.append(config.add_category(
            # NOTE: name and ID is unique as long as we don't use
            #       multiple ml_models simutaneously
            name=f"ml_{proc}",
            id=(i + 1) * 10000,
            selection=f"catid_ml_{proc}",
            label=f"ml_{proc}",
        ))

    category_blocks = OrderedDict({
        "lep": [config.get_category("1e"), config.get_category("1mu")],
        "jet": [config.get_category("resolved"), config.get_category("boosted")],
        "b": [config.get_category("1b"), config.get_category("2b")],
        "dnn": ml_categories,
    })

    # create combination of categories
    n_cats = create_category_combinations(
        config,
        category_blocks,
        name_fn=name_fn,
        kwargs_fn=kwargs_fn,
        skip_existing=True,
    )
    logger.info(f"Number of produced ml category insts: {n_cats}")
