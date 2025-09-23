import copy

import paddle

from e3nn.nn import Dropout
from e3nn.util.test import assert_equivariant


def test_dropout() -> None:
    def build_module():
        return Dropout(irreps="10x1e + 10x0e", p=0.75)

    c = build_module()
    x = c.irreps.randn(5, 2, -1)

    for c in [c]:
        c.eval()
        assert (c(x) == x).all()

        c.train()
        y = c(x)
        assert ((y == x / 0.25) | (y == 0)).all()

        def wrap(x):
            paddle.seed(0)
            return c(x)

        assert_equivariant(wrap, args_in=[x], irreps_in=[c.irreps], irreps_out=[c.irreps])


def test_copy() -> None:
    c = Dropout(irreps="0e + 1e", p=0.5)
    _ = copy.deepcopy(c)
