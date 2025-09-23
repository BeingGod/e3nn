import paddle
import pytest

from e3nn import o3
from e3nn.nn import Activation
from e3nn.util.test import assert_equivariant
from e3nn.util.test import assert_normalized


@pytest.mark.parametrize(
    "irreps_in,acts",
    [("256x0o", [paddle.abs]), ("37x0e", [paddle.tanh]), ("4x0e + 3x0o", [paddle.nn.functional.silu, paddle.abs])],
)
def test_activation(irreps_in, acts) -> None:
    irreps_in = o3.Irreps(irreps_in)

    def build_module(irreps_in, acts):
        return Activation(irreps_in, acts)

    a = build_module(irreps_in, acts)
    assert_equivariant(a)

    inp = irreps_in.randn(13, -1)
    out = a(inp)
    for ir_slice, act in zip(irreps_in.slices(), acts):
        this_out = out[:, ir_slice]
        true_up_to_factor = act(inp[:, ir_slice])
        factors = this_out / true_up_to_factor
        assert paddle.allclose(factors, paddle.broadcast_to(factors[0], factors.shape))

    assert_normalized(a)
