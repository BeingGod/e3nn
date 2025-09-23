from typing import Callable
from typing import Optional

import paddle

from e3nn import o3


class NormActivation(paddle.nn.Layer):
    """Norm-based activation function
    Applies a scalar nonlinearity to the norm of each irrep and ouputs a (normalized) version of that irrep multiplied by the
    scalar output of the scalar nonlinearity.
    Parameters
    ----------
    irreps_in : `e3nn.o3.Irreps`
        representation of the input
    scalar_nonlinearity : callable
        scalar nonlinearity such as ``paddle.nn.functional.sigmoid``
    normalize : bool
        whether to normalize the input features before multiplying them by the scalars from the nonlinearity
    epsilon : float, optional
        when ``normalize``ing, norms smaller than ``epsilon`` will be clamped up to ``epsilon`` to avoid division by zero and
        NaN gradients. Not allowed when ``normalize`` is False.
    bias : bool
        whether to apply a learnable additive bias to the inputs of the ``scalar_nonlinearity``
    Examples
    --------
    >>> n = NormActivation("2x1e", paddle.nn.functional.sigmoid)
    >>> feats = paddle.ones([1, 2*3])
    >>> print(feats.reshape(1, 2, 3).norm(axis=-1))
    Tensor(shape=[1, 2], dtype=float32, place=Place(gpu:0), stop_gradient=True,
           [[1.73205078, 1.73205078]])
    >>> print(paddle.nn.functional.sigmoid(feats.reshape(1, 2, 3).norm(axis=-1)))
    Tensor(shape=[1, 2], dtype=float32, place=Place(gpu:0), stop_gradient=True,
           [[0.84967452, 0.84967452]])
    >>> print(n(feats).reshape(1, 2, 3).norm(axis=-1))
    Tensor(shape=[1, 2], dtype=float32, place=Place(gpu:0), stop_gradient=True,
           [[0.84967452, 0.84967452]])
    """

    epsilon: Optional[float]
    _eps_squared: float

    def __init__(
        self,
        irreps_in,
        scalar_nonlinearity: Callable,
        normalize: bool = True,
        epsilon: Optional[float] = None,
        bias: bool = False,
    ):
        super().__init__()
        self.irreps_in = o3.Irreps(irreps_in)
        self.irreps_out = o3.Irreps(irreps_in)
        if epsilon is None and normalize:
            epsilon = 1e-08
        elif epsilon is not None and not normalize:
            raise ValueError("epsilon and normalize = False don't make sense together")
        elif not epsilon > 0:
            raise ValueError(f"epsilon {epsilon} is invalid, must be strictly positive.")
        self.epsilon = epsilon
        if self.epsilon is not None:
            self._eps_squared = epsilon * epsilon
        else:
            self._eps_squared = 0.0
        self.norm = o3.Norm(irreps_in, squared=epsilon is not None)
        self.scalar_nonlinearity = scalar_nonlinearity
        self.normalize = normalize
        self.bias = bias
        if self.bias:
            self.biases = paddle.base.framework.EagerParamBase.from_tensor(tensor=paddle.zeros(shape=irreps_in.num_irreps))
        self.scalar_multiplier = o3.ElementwiseTensorProduct(irreps_in1=self.norm.irreps_out, irreps_in2=irreps_in)

    def forward(self, features):
        """evaluate
        Parameters
        ----------
        features : `paddle.Tensor`
            tensor of shape ``(..., irreps_in.dim)``
        Returns
        -------
        `paddle.Tensor`
            tensor of shape ``(..., irreps_in.dim)``
        """
        norms = self.norm(features)
        if self._eps_squared > 0:
            norms[norms < self._eps_squared] = self._eps_squared
            norms = norms.sqrt()
        nonlin_arg = norms
        if self.bias:
            nonlin_arg = nonlin_arg + self.biases
        scalings = self.scalar_nonlinearity(nonlin_arg)
        if self.normalize:
            scalings = scalings / norms
        return self.scalar_multiplier(scalings, features)
