import paddle
import pytest

from e3nn import o3
from e3nn.util.test import assert_equivariant


@pytest.mark.skip(reason="paddle not support save paddle.nn.Layer.")
def test_save_load() -> None:
    pass


def test_antisymmetric_matrix(float_tolerance) -> None:
    def build_module():
        return o3.ReducedTensorProducts("ij=-ji", i="5x0e + 1e")

    tp = build_module()

    assert_equivariant(tp, irreps_in=tp.irreps_in, irreps_out=tp.irreps_out)

    Q = tp.change_of_basis
    x = paddle.randn([2, 5 + 3])
    assert (tp(*x) - paddle.einsum("xij,i,j", Q, *x)).abs().max() < float_tolerance

    assert (Q + paddle.einsum("xij->xji", Q)).abs().max() < float_tolerance


def test_reduce_tensor_Levi_Civita_symbol(float_tolerance) -> None:
    tp = o3.ReducedTensorProducts("ijk=-ikj=-jik", i="1e")
    assert tp.irreps_out == o3.Irreps("0e")

    assert_equivariant(tp, irreps_in=tp.irreps_in, irreps_out=tp.irreps_out)

    Q = tp.change_of_basis
    x = paddle.randn([3, 3])
    assert (tp(*x) - paddle.einsum("xijk,i,j,k", Q, *x)).abs().max() < float_tolerance

    assert (Q + paddle.einsum("xijk->xikj", Q)).abs().max() < float_tolerance
    assert (Q + paddle.einsum("xijk->xjik", Q)).abs().max() < float_tolerance


def test_reduce_tensor_antisymmetric_L2(float_tolerance) -> None:

    tp = o3.ReducedTensorProducts("ijk=-ikj=-jik", i="2e")

    assert_equivariant(tp, irreps_in=tp.irreps_in, irreps_out=tp.irreps_out)

    Q = tp.change_of_basis
    x = paddle.randn([3, 5])
    assert (tp(*x) - paddle.einsum("xijk,i,j,k", Q, *x)).abs().max() < float_tolerance

    assert (Q + paddle.einsum("xijk->xikj", Q)).abs().max() < float_tolerance
    assert (Q + paddle.einsum("xijk->xjik", Q)).abs().max() < float_tolerance


def test_reduce_tensor_elasticity_tensor(float_tolerance) -> None:

    tp = o3.ReducedTensorProducts("ijkl=jikl=klij", i="1e")
    assert tp.irreps_out.dim == 21

    assert_equivariant(tp, irreps_in=tp.irreps_in, irreps_out=tp.irreps_out)

    Q = tp.change_of_basis
    x = paddle.randn([4, 3])
    assert (tp(*x) - paddle.einsum("xijkl,i,j,k,l", Q, *x)).abs().max() < float_tolerance

    assert (Q - paddle.einsum("xijkl->xjikl", Q)).abs().max() < float_tolerance
    assert (Q - paddle.einsum("xijkl->xijlk", Q)).abs().max() < float_tolerance
    assert (Q - paddle.einsum("xijkl->xklij", Q)).abs().max() < float_tolerance


def test_reduce_tensor_elasticity_tensor_parity(float_tolerance) -> None:

    tp = o3.ReducedTensorProducts("ijkl=jikl=klij", i="1o")
    assert tp.irreps_out.dim == 21
    assert all(ir.p == 1 for _, ir in tp.irreps_out)

    assert_equivariant(tp, irreps_in=tp.irreps_in, irreps_out=tp.irreps_out)

    Q = tp.change_of_basis
    x = paddle.randn([4, 3])
    assert (tp(*x) - paddle.einsum("xijkl,i,j,k,l", Q, *x)).abs().max() < float_tolerance

    assert (Q - paddle.einsum("xijkl->xjikl", Q)).abs().max() < float_tolerance
    assert (Q - paddle.einsum("xijkl->xijlk", Q)).abs().max() < float_tolerance
    assert (Q - paddle.einsum("xijkl->xklij", Q)).abs().max() < float_tolerance
