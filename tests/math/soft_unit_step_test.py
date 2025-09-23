import paddle
import pytest

from e3nn.math import soft_unit_step


@pytest.mark.skip(reason="gradcheck is not support on paddle")
def test_grad() -> None:
    paddle.set_default_dtype("float64")
    x = paddle.linspace(-1, 1, 1000)
    x.stop_gradient = False

    def f(x):
        return soft_unit_step(x).sum()

    # assert torch.autograd.gradcheck(f, (x,), check_undefined_grad=False)


@pytest.mark.skip(reason="lack of double grad and triple grad ops.")
def test_grads() -> None:
    x = paddle.linspace(-1, 1, 1000)
    x.stop_gradient = False

    y0 = soft_unit_step(x)
    assert paddle.isfinite(y0).all()

    (y1,) = paddle.autograd.grad(y0.sum(), x, create_graph=True)
    assert paddle.isfinite(y1).all()

    (y2,) = paddle.autograd.grad(y1.sum(), x, create_graph=True)
    assert paddle.isfinite(y2).all()

    (y3,) = paddle.autograd.grad(y2.sum(), x, create_graph=True)
    assert paddle.isfinite(y3).all()
