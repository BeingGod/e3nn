import paddle
import pytest

from e3nn import o3
from e3nn.util.test import assert_equivariant
from e3nn.util.test import random_irreps


@pytest.mark.parametrize("irreps_in", ["", "5x0e", "1e + 2e + 4x1e + 3x3o"] + random_irreps(n=4))
@pytest.mark.parametrize("squared", [True, False])
def test_norm(irreps_in, squared) -> None:
    def build_module(irreps_in, squared):
        return o3.Norm(irreps_in, squared=squared)

    m = build_module(irreps_in, squared=squared)
    m(paddle.randn([m.irreps_in.dim]))
    if m.irreps_in.dim == 0:
        return
    assert_equivariant(m)


@pytest.mark.parametrize("squared", [True, False])
def test_grad(squared) -> None:
    """Confirm has zero grad at zero"""
    irreps_in = o3.Irreps("2x0e + 3x0o")
    norm = o3.Norm(irreps_in, squared=squared)
    inp = paddle.zeros(norm.irreps_in.dim)
    inp.stop_gradient = False
    out = norm(inp)
    grads = paddle.autograd.grad(
        outputs=out.sum(),
        inputs=inp,
    )[0]
    assert paddle.allclose(grads, paddle.zeros_like(grads))


@pytest.mark.parametrize("squared", [True, False])
def test_vector_norm(squared) -> None:
    n = 10
    batch = 3
    irreps_in = o3.Irreps([(n, (1, -1))])
    vecs = paddle.randn([batch, n, 3])
    norm_mod = o3.Norm(irreps_in, squared=squared)
    norms = norm_mod(vecs.reshape(batch, -1))
    norms_true = vecs.norm(axis=-1)
    if squared:
        norms_true.square_()
    assert paddle.allclose(norms_true, norms.reshape(batch, n))
