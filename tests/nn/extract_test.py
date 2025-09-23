import copy

import paddle
import pytest

from e3nn.nn import Extract
from e3nn.nn import ExtractIr
from e3nn.util.test import assert_equivariant


def test_extract() -> None:
    def build_module():
        return Extract("1e + 0e + 0e", ["0e", "0e"], [(1,), (2,)])

    c = build_module()
    out = c(paddle.to_tensor([0.0, 0.0, 0.0, 1.0, 2.0]))
    assert out == (paddle.to_tensor([1.0]), paddle.to_tensor([2.0]))
    assert_equivariant(c, irreps_out=list(c.irreps_outs))


@pytest.mark.parametrize("squeeze", [True, False])
def test_extract_single(squeeze) -> None:
    def build_module():
        return Extract("1e + 0e + 0e", ["0e"], [(1,)], squeeze_out=squeeze)

    c = build_module()
    out = c(paddle.to_tensor([0.0, 0.0, 0.0, 1.0, 2.0]))
    if squeeze:
        assert isinstance(out, paddle.Tensor)
    else:
        assert len(out) == 1
        out = out[0]
    assert out == paddle.to_tensor([1.0])
    assert_equivariant(c, irreps_out=list(c.irreps_outs))


def test_extract_ir() -> None:
    def build_module():
        return ExtractIr("1e + 0e + 0e", "0e")

    c = build_module()
    out = c(paddle.to_tensor([0.0, 0.0, 0.0, 1.0, 2.0]))
    assert paddle.all(out == paddle.to_tensor([1.0, 2.0]))
    assert_equivariant(c)


def test_copy() -> None:
    c = Extract("1e + 0e + 0e", ["0e", "0e"], [(1,), (2,)])
    _ = copy.deepcopy(c)
    c = ExtractIr("1e + 0e + 0e", "0e")
    _ = copy.deepcopy(c)
