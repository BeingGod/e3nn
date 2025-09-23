import paddle
import pytest

from e3nn.nn import FullyConnectedNet


@pytest.mark.parametrize("act", [None, paddle.tanh])
@pytest.mark.parametrize("var_in, var_out, out_act", [(1, 1, False), (1, 1, True), (0.1, 10.0, False), (0.1, 0.05, True)])
def test_variance(act, var_in, var_out, out_act) -> None:
    hs = (1000, 500, 1500, 4)

    def build_module(hs, act, var_in, var_out, out_act):
        return FullyConnectedNet(hs, act, var_in, var_out, out_act)

    f = build_module(hs, act, var_in, var_out, out_act)

    x = paddle.randn([2000, hs[0]]) * var_in**0.5
    y = f(x) / var_out**0.5

    if not out_act:
        assert y.mean().abs() < 0.5
    assert y.pow(2).mean().log10().abs() < paddle.to_tensor(1.5).log10()

    f(x)


@pytest.mark.skipif(not paddle.device.cuda.device_count() > 1, reason="requires cuda")
def test_data_parallel() -> None:
    fc = paddle.DataParallel(FullyConnectedNet([10, 20, 30]))
    y = fc(paddle.randn([32, 10]))
    y.sum().backward()
