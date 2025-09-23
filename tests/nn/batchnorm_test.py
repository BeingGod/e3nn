import paddle
import pytest

from e3nn import o3
from e3nn.nn import BatchNorm
from e3nn.util.test import assert_equivariant


def test_equivariant() -> None:
    irreps = o3.Irreps("3x0e + 3x0o + 4x1e")
    m = BatchNorm(irreps)
    m(irreps.randn(16, -1))
    m(irreps.randn(16, -1))
    m.train()
    assert_equivariant(m, irreps_in=[irreps], irreps_out=[irreps])
    m.eval()
    assert_equivariant(m, irreps_in=[irreps], irreps_out=[irreps])


@pytest.mark.parametrize("affine", [True, False])
@pytest.mark.parametrize("reduce", ["mean", "max"])
@pytest.mark.parametrize("normalization", ["norm", "component"])
@pytest.mark.parametrize("instance", [True, False])
def test_modes(affine, reduce, normalization, instance) -> None:
    irreps = o3.Irreps("10x0e + 5x1e")

    m = BatchNorm(irreps, affine=affine, reduce=reduce, normalization=normalization, instance=instance)
    repr(m)

    m.train()
    m(irreps.randn(20, 20, -1))

    m.eval()
    m(irreps.randn(20, 20, -1))


@pytest.mark.parametrize("instance", [True, False])
def test_normalization(float_tolerance, instance) -> None:
    sqrt_float_tolerance = paddle.sqrt(float_tolerance)

    batch, n = 20, 20
    irreps = o3.Irreps("3x0e + 4x1e")

    m = BatchNorm(irreps, normalization="norm", instance=instance)

    x = paddle.randn([batch, n, irreps.dim]).multiply(paddle.to_tensor(5.0)).add(paddle.to_tensor(10.0))
    x = m(x)

    a = x[..., :3]  # [batch, space, mul]
    assert a.mean([0, 1]).abs().max() < float_tolerance
    assert a.pow(2).mean([0, 1]).sub(paddle.to_tensor(1)).abs().max() < sqrt_float_tolerance

    a = x[..., 3:].reshape(batch, n, 4, 3)  # [batch, space, mul, repr]
    assert a.pow(2).sum(3).mean([0, 1]).sub(paddle.to_tensor(1)).abs().max() < sqrt_float_tolerance

    m = BatchNorm(irreps, normalization="component", instance=instance)

    x = paddle.randn([batch, n, irreps.dim]).multiply(paddle.to_tensor(5.0)).add(paddle.to_tensor(10.0))
    x = m(x)

    a = x[..., :3]  # [batch, space, mul]
    assert a.mean([0, 1]).abs().max() < float_tolerance
    assert a.pow(2).mean([0, 1]).sub(paddle.to_tensor(1)).abs().max() < sqrt_float_tolerance

    a = x[..., 3:].reshape(batch, n, 4, 3)  # [batch, space, mul, repr]
    assert a.pow(2).mean(3).mean([0, 1]).sub(paddle.to_tensor(1)).abs().max() < sqrt_float_tolerance
