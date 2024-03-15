# coding: utf-8

"""
Set of producers to reconstruct categories at different states of the analysis
"""

import law

from columnflow.production import Producer, producer
from columnflow.util import maybe_import, InsertableDict
from columnflow.production.categories import category_ids

from hbt.config.categories import add_categories_ml
from hbt.util import get_subclasses_deep

np = maybe_import("numpy")
ak = maybe_import("awkward")

logger = law.logger.get_logger(__name__)


@producer(
    uses={category_ids},
    produces={category_ids},
    ml_model_name=None,
)
def ml_cats(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    """
    Reproduces category ids after ML Training. Calling this producer also
    automatically adds `MLEvaluation` to the requirements.
    """
    # category ids
    events = self[category_ids](events, **kwargs)

    return events


@ml_cats.requires
def ml_cats_reqs(self: Producer, reqs: dict) -> None:
    if "ml" in reqs:
        return

    from columnflow.tasks.ml import MLEvaluation
    reqs["ml"] = MLEvaluation.req(self.task, ml_model=self.ml_model_name)


@ml_cats.setup
def ml_cats_setup(self: Producer, reqs: dict, inputs: dict, reader_targets: InsertableDict) -> None:
    reader_targets["mlcolumns"] = inputs["ml"]["mlcolumns"]


@ml_cats.init
def ml_cats_init(self: Producer) -> None:
    if not self.ml_model_name:
        self.ml_model_name = "test"

    # add categories to config inst
    add_categories_ml(self.config_inst, self.ml_model_name)


# get all the derived MLModels and instantiate a corresponding producer for each one
import hbt.ml.derived
from hbt.ml.first_nn_norm_layer_ensamble import SimpleDNN

ml_model_names = get_subclasses_deep(SimpleDNN)
logger.info(f"deriving {len(ml_model_names)} ML categorizer...")
flat_names = ", ".join(ml_model_names)
logger.info(f"deriving {ml_model_names} ML categorizer...")


for ml_model_name in ml_model_names:
    prod_name = f"ml_{ml_model_name}"
    logger.info(f"deriving producer: {prod_name}")
    ml_cats.derive(prod_name, cls_dict={"ml_model_name": ml_model_name, "exposed": True})
