import paddle
import pytest

from e3nn import o3


def test_wigner_3j_symmetry() -> None:
    assert paddle.allclose(o3.wigner_3j(1, 2, 3), o3.wigner_3j(1, 3, 2).transpose([0, 2, 1]))
    assert paddle.allclose(o3.wigner_3j(1, 2, 3), o3.wigner_3j(2, 1, 3).transpose([1, 0, 2]))
    assert paddle.allclose(o3.wigner_3j(1, 2, 3), o3.wigner_3j(3, 2, 1).transpose([2, 1, 0]))
    assert paddle.allclose(o3.wigner_3j(1, 2, 3), o3.wigner_3j(3, 1, 2).transpose([1, 0, 2]).transpose([0, 2, 1]))
    assert paddle.allclose(o3.wigner_3j(1, 2, 3), o3.wigner_3j(2, 3, 1).transpose([2, 1, 0]).transpose([0, 2, 1]))


@pytest.mark.parametrize("l1,l2,l3", [(1, 2, 3), (2, 3, 4), (3, 4, 5), (1, 1, 1), (1, 1, 0), (1, 0, 1), (0, 1, 1), (2, 2, 2)])
def test_wigner_3j(l1, l2, l3, float_tolerance) -> None:

    abc = o3.rand_angles(10)

    C = o3.wigner_3j(l1, l2, l3)
    D1 = o3.Irrep(l1, 1).D_from_angles(*abc)
    D2 = o3.Irrep(l2, 1).D_from_angles(*abc)
    D3 = o3.Irrep(l3, 1).D_from_angles(*abc)

    C2 = paddle.einsum("ijk,zil,zjm,zkn->zlmn", C, D1, D2, D3)
    assert (C - C2).abs().max() < float_tolerance


def test_cartesian(float_tolerance) -> None:
    if paddle.get_default_dtype() == "float64":
        pytest.skip(reason="loosen unit test in float64 precision")

    abc = o3.rand_angles(10)
    R = o3.angles_to_matrix(*abc)
    D = o3.wigner_D(1, *abc)
    assert (R - D).abs().max() < float_tolerance


def commutator(A, B):
    return A @ B - B @ A


@pytest.mark.parametrize("j", [0, 1 / 2, 1, 3 / 2, 2, 5 / 2])
def test_su2_algebra(j, float_tolerance) -> None:
    X = o3.su2_generators(j)
    assert paddle.allclose(commutator(X[0], X[1]).real(), X[2].real(), atol=float_tolerance) and paddle.allclose(
        commutator(X[0], X[1]).imag(), X[2].imag(), atol=float_tolerance
    )
    assert paddle.allclose(commutator(X[1], X[2]).real(), X[0].real(), atol=float_tolerance) and paddle.allclose(
        commutator(X[1], X[2]).imag(), X[0].imag(), atol=float_tolerance
    )
