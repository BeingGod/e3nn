import paddle.nn.functional as F
import pytest

from e3nn import o3
from e3nn.util.test import assert_equivariant
from e3nn.util.test import assert_normalized
from e3nn.util.test import random_irreps


def test_assert_equivariant() -> None:
    def not_equivariant(x1, x2):
        return x1 * x2

    not_equivariant.irreps_in1 = o3.Irreps("2x0e + 1x1e + 3x2o + 1x4e")
    not_equivariant.irreps_in2 = o3.Irreps("2x0o + 3x0o + 3x2e + 1x4o")
    not_equivariant.irreps_out = o3.Irreps("1x1e + 2x0o + 3x2e + 1x4o")
    assert not_equivariant.irreps_in1.dim == not_equivariant.irreps_in2.dim
    assert not_equivariant.irreps_in1.dim == not_equivariant.irreps_out.dim
    with pytest.raises(AssertionError):
        assert_equivariant(not_equivariant)


def test_bad_normalize() -> None:
    def not_normal(x1) -> float:
        return 870.0 * F.relu(x1.square())

    not_normal.irreps_in = random_irreps(clean=True, allow_empty=False)
    not_normal.irreps_out = not_normal.irreps_in
    with pytest.raises(AssertionError):
        assert_normalized(not_normal)


def test_normalized_ident() -> None:
    def ident(x1):
        return x1

    ident.irreps_in = random_irreps(clean=True, allow_empty=False)
    ident.irreps_out = ident.irreps_in
    assert_normalized(ident)
