__version__ = "0.5.1"
import os
import warnings
from typing import Dict

import paddle

_OPT_DEFAULTS: Dict[str, bool] = dict(specialized_code=True, optimize_einsums=True, jit_script_fx=True)


def set_optimization_defaults(**kwargs) -> None:
    """Globally set the default optimization settings.

    Parameters
    ----------
    **kwargs
        Keyword arguments to set the default optimization settings.
    """
    for k, v in kwargs.items():
        if k not in _OPT_DEFAULTS:
            raise ValueError(f"Unknown optimization option: {k}")
        _OPT_DEFAULTS[k] = v


def get_optimization_defaults() -> Dict[str, bool]:
    """Get the global default optimization settings."""
    return dict(_OPT_DEFAULTS)


def set_paddle_flags():
    # NOTE: set NVIDIA_TF32_OVERRIDE=0 to improve precision.
    if paddle.device.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0:
        os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
        warnings.warn("Set NVIDIA_TF32_OVERRIDE=0 to disables all TF32 kernels to improve precision.")


from . import io as io  # noqa
from . import nn as nn  # noqa
from . import o3 as o3  # noqa

set_paddle_flags()
