"""
Gen matching selection methods.
"""

from columnflow.selection import Selector, SelectionResult, selector
from columnflow.util import maybe_import, dev_sandbox
from columnflow.columnar_util import set_ak_column

from collections import defaultdict, OrderedDict


np = maybe_import("numpy")
ak = maybe_import("awkward")

@selector(
    uses={
        "Jet.pt", "Jet.eta", "Jet.phi", "Jet.jetId", "Jet.puId",
        "nGenJet", "GenJet.*",
        "nGenVisTau", "GenVisTau.*", "Jet.genJetIdx", "Tau.genPartIdx",
        # gen_HH_decay_products.PRODUCES,
    },
    produces={
        "GenmatchedJets", "GenmatchedHHBtagJets", "GenBPartons",
    },
    sandbox=dev_sandbox("bash::$HBT_BASE/sandboxes/venv_columnar.sh"),
)
def genmatching_selector(
    self: Selector,
    events: ak.Array,
    jet_collection: ak.Array, # different collections can be matched now
    jet_results: SelectionResult,
    **kwargs,
) -> tuple[ak.Array, SelectionResult]:
    from IPython import embed; embed()
    genBjets = events.GenJet[abs(events.GenJet.partonFlavour) == 5]

