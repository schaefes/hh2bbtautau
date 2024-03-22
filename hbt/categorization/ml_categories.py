from columnflow.util import maybe_import
from columnflow.categorization import Categorizer, categorizer

np = maybe_import("numpy")
ak = maybe_import("awkward")

# TODO: not hard-coded -> use config?
ml_processes = [
    "graviton_hh_ggf_bbtautau_m400",
    # "hh_ggf_bbtautau",
    "graviton_hh_vbf_bbtautau_m400",
    "tt",
    # "tt_dl",s
    # "tt_sl",
    "dy",
    # "dy_lep_pt50To100",
    # "dy_lep_pt100To250",
    # "dy_lep_pt250To400",
    # "dy_lep_pt400To650",
    # "dy_lep_pt650",
]
for proc in ml_processes:
    @categorizer(
        uses=set(f"mlscore.{proc1}" for proc1 in ml_processes),
        cls_name=f"catid_ml_{proc}",
        proc_col_name=f"{proc}",
        # skip check because we don't know which ML processes were used from the MLModel
        check_used_columns=False,
        call_force=True,
    )
    def dnn_mask(self: Categorizer, events: ak.Array, **kwargs) -> tuple[ak.Array, ak.Array]:
        """
        dynamically built Categorizer that categorizes events based on dnn scores
        """
        # start with true mask
        outp_mask = np.ones(len(events), dtype=bool)
        for col_name in events.mlscore.fields:
            # check for each mlscore if *this* score is larger and combine all masks
            mask = events.mlscore[self.proc_col_name] >= events.mlscore[col_name]
            outp_mask = outp_mask & mask

        return events, outp_mask
