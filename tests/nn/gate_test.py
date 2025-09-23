import paddle

from e3nn.nn import Gate
from e3nn.o3 import Irreps
from e3nn.util.test import assert_equivariant
from e3nn.util.test import assert_normalized


def test_gate() -> None:
    irreps_scalars, act_scalars, irreps_gates, act_gates, irreps_gated = (
        Irreps("16x0o"),
        [paddle.tanh],
        Irreps("32x0o"),
        [paddle.tanh],
        Irreps("16x1e+16x1o"),
    )

    def build_module(irreps_scalars, act_scalars, irreps_gates, act_gates, irreps_gated):
        return Gate(irreps_scalars, act_scalars, irreps_gates, act_gates, irreps_gated)

    g = build_module(irreps_scalars, act_scalars, irreps_gates, act_gates, irreps_gated)
    assert_equivariant(g)
    assert_normalized(g)
