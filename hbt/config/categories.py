# coding: utf-8

"""
Definition of categories.
"""

import order as od
import law

from columnflow.config_util import add_category, create_category_combinations
from columnflow.ml import MLModel
from hbt.util import call_once_on_config

from collections import OrderedDict


logger = law.logger.get_logger(__name__)


def add_categories(config: od.Config) -> None:
    """
    Adds all categories to a *config*.
    """
    add_category(
        config,
        name="incl",
        id=1,
        selection="cat_incl",
        label="selection\napplied",
        aux={
            "root_cats": {"incl": "incl"},
        },
    )
    add_category(
        config,
        name="2j",
        id=100,
        selection="cat_2j",
        label="2 jets",
    )

    # add extra categories for VBF Sel plots
    add_category(
        config,
        name="kin_selections_ak4",
        id=2,
        selection="cat_incl",
        label="Kinematic Gen Sel.\nAk4 Sel.",
    )
    add_category(
        config,
        name="no_kin_selections_ak4",
        id=3,
        selection="cat_incl",
        label="Ak4 Sel.",
    )
    add_category(
        config,
        name="kin_selections_vbfmask",
        id=4,
        selection="cat_incl",
        label="Kinematic Gen Sel.\nVBF Mask Sel.",
    )
    add_category(
        config,
        name="no_kin_selections_vbfmask",
        id=5,
        selection="cat_incl",
        label="VBF Mask\nSel.",
    )
    add_category(
        config,
        name="kin_selections_pairs",
        id=6,
        selection="cat_incl",
        label="Kinematic Gen Sel.\nVBF Pair Sel.",
    )
    add_category(
        config,
        name="no_kin_selections_pairs",
        id=7,
        selection="cat_incl",
        label="VBF Pair Sel.",
    )
    add_category(
        config,
        name="kin_selections_trigger",
        id=8,
        selection="cat_incl",
        label="Kinematic Gen Sel.\nVBF Trigger Sel.",
    )
    add_category(
        config,
        name="no_kin_selections_trigger",
        id=9,
        selection="cat_incl",
        label="VBF Trigger Sel.",
    )

    # add categires for the lepton sel cutflow plots
    add_category(
        config,
        name="no_lepton_veto",
        id=10,
        selection="cat_incl",
        label="No Lepton Veto",
    )
    add_category(
        config,
        name="no_tau_iso",
        id=11,
        selection="cat_incl",
        label="No $\\tau$ Iso",
    )
    add_category(
        config,
        name="no_tau_indices1",
        id=12,
        selection="cat_incl",
        label="No min #$\\tau$ for\n$e$ and $\\mu$ ",
    )
    add_category(
        config,
        name="tau_15pt_2_5eta",
        id=13,
        selection="cat_incl",
        label="Min $\\tau$ $p_{T}$=15GeV\nMax $\\eta$=2.5",
    )
    add_category(
        config,
        name="tau_e_mu_15pt_2_5eta",
        id=14,
        selection="cat_incl",
        label="Min $\\tau$,$e$,$\\mu$\n$p_{T}$=15GeV\nMax $\\eta$=2.5",
    )
    add_category(
        config,
        name="DeepTau_keep_highest_bin",
        id=15,
        selection="cat_incl",
        label="DeepTauId max\nscore kept",
    )
    add_category(
        config,
        name="Combinedv1",
        id=16,
        selection="cat_incl",
        label="Kinematic\nDeepTauId\nNo $\\tau$ Iso",
    )
    add_category(
        config,
        name="no_tau_indices_at_all",
        id=17,
        selection="cat_incl",
        label="No min #$\\tau$ for\n$e$, $\\mu$, $\\tau$",
    )
    add_category(
        config,
        name="no_tau_iso_mask",
        id=18,
        selection="cat_incl",
        label="No $\\tau$ Iso Mask",
    )
    add_category(
        config,
        name="Combinedv2",
        id=19,
        selection="cat_incl",
        label="Combined Cuts\nV2",
    )
    add_category(
        config,
        name="had_tau_parton",
        id=20,
        selection="cat_incl",
        label="Hadronic Decay",
    )
    add_category(
        config,
        name="e_tau_parton",
        id=21,
        selection="cat_incl",
        label="Decay to e",
    )
    add_category(
        config,
        name="mu_tau_parton",
        id=22,
        selection="cat_incl",
        label=r"Decay to $\mu$",
    )
    add_category(
        config,
        name="all_tau_partons_channel",
        id=23,
        selection="cat_incl",
        label=r"All decay channels",
    )
    add_category(
        config,
        name="CustomVBFMask",
        id=24,
        selection="cat_incl",
        label="Custom\nVBF Mask",
    )


def name_fn(root_cats):
    cat_name = "__".join(cat.name for cat in root_cats.values())
    return cat_name


def kwargs_fn(root_cats):
    kwargs = {
        "id": sum([c.id for c in root_cats.values()]),
        "label": ",\n".join([root_cats['incl'].label, root_cats['dnn'].label]),
        "aux": {
            "root_cats": {key: value.name for key, value in root_cats.items()},
        },
    }
    return kwargs


@call_once_on_config()
def add_categories_ml(config, ml_model_inst):

    # if not already done, get the ml_model instance
    if isinstance(ml_model_inst, str):
        ml_model_inst = MLModel.get_cls(ml_model_inst)(config)

    # add ml categories directly to the config
    ml_categories = []
    for i, proc in enumerate(ml_model_inst.processes):
        l = f"{config.get_process(proc).label}"
        if '$\\rightarrow' in l:
            l = '$' + l.split('$\\rightarrow')[1]
        ml_categories.append(config.add_category(
            # NOTE: name and ID is unique as long as we don't use
            #       multiple ml_models simutaneously
            name=f"ml_{proc}",
            id=(i + 1) * 10000,
            selection=f"catid_ml_{proc}",
            label=l,
        ))

    category_blocks = OrderedDict({
        "incl": [config.get_category("incl")],
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
