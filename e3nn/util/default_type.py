from typing import Optional
from typing import Tuple
from typing import Union

import paddle


def paddle_get_default_tensor_type():
    return str(paddle.empty(shape=[0]).dtype)


def _paddle_get_default_dtype() -> paddle.dtype:
    return paddle.empty(shape=[0]).dtype


def paddle_get_default_device() -> Union[paddle.CPUPlace, paddle.CUDAPlace]:
    return paddle.empty(shape=[0]).place


def explicit_default_types(
    dtype: Optional[paddle.dtype] = None,
    device: Optional[Union[paddle.CPUPlace, paddle.CUDAPlace]] = None,
) -> Tuple[paddle.dtype, Union[paddle.CPUPlace, paddle.CUDAPlace]]:
    if dtype is None:
        dtype = _paddle_get_default_dtype()
    if device is None:
        device = paddle_get_default_device()
    return dtype, device
