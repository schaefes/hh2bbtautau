# coding: utf-8

"""
Helpful utils.
"""

from __future__ import annotations
__all__ = ["IF_NANO_V9", "IF_NANO_V11"]

from functools import wraps
from columnflow.types import Any
from columnflow.columnar_util import ArrayFunction, deferred_column


@deferred_column
def IF_NANO_V9(self: ArrayFunction.DeferredColumn, func: ArrayFunction) -> Any | set[Any]:
    return self.get() if func.config_inst.campaign.x.version == 9 else None


@deferred_column
def IF_NANO_V11(self: ArrayFunction.DeferredColumn, func: ArrayFunction) -> Any | set[Any]:
    return self.get() if func.config_inst.campaign.x.version == 11 else None

# "borrowed" from hbw analysis, thanks Mathis!
def get_subclasses_deep(*classes):
    """
    Helper that gets all subclasses from input classes based on the '_subclasses' attribute.
    """
    classes = {_cls.__name__: _cls for _cls in classes}
    all_classes = {}

    while classes:
        for key, _cls in classes.copy().items():
            classes.update(getattr(_cls, "_subclasses", {}))
            all_classes[key] = classes.pop(key)

    return all_classes


def call_once_on_config(include_hash=False):
    """
    Parametrized decorator to ensure that function *func* is only called once for the config *config*
    """
    def outer(func):
        @wraps(func)
        def inner(config, *args, **kwargs):
            tag = f"{func.__name__}_called"
            if include_hash:
                tag += f"_{func.__hash__()}"

            if config.has_tag(tag):
                return

            config.add_tag(tag)
            return func(config, *args, **kwargs)
        return inner
    return outer
