import paddle

from e3nn.math import normalize2mom


def test_device() -> None:
    act = paddle.nn.ReLU()
    act = normalize2mom(act)


def test_identity() -> None:
    act1 = normalize2mom(paddle.nn.functional.relu)
    act2 = normalize2mom(act1)

    x = paddle.randn((10,))
    assert (act1(x) == act2(x)).all()


def test_deterministic() -> None:
    act1 = normalize2mom(paddle.tanh)
    act2 = normalize2mom(paddle.tanh)

    x = paddle.randn((10,))
    assert (act1(x) == act2(x)).all()
