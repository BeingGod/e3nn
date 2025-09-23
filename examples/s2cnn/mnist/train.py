import gzip
import math
import pickle

import numpy as np
import paddle

from e3nn import o3
from e3nn.nn import SO3Activation
from e3nn.util.paddle_utils import dim2perm


def s2_near_identity_grid(max_beta: float = math.pi / 8, n_alpha: int = 8, n_beta: int = 3) -> paddle.Tensor:
    """
    :return: rings around the north pole
    size of the kernel = n_alpha * n_beta
    """
    beta = paddle.arange(start=1, end=n_beta + 1) * max_beta / n_beta
    alpha = paddle.linspace(start=0, stop=2 * math.pi, num=n_alpha + 1)[:-1]
    a, b = paddle.meshgrid(alpha, beta)
    b = b.flatten()
    a = a.flatten()
    return paddle.stack(x=(a, b))


def so3_near_identity_grid(
    max_beta: float = math.pi / 8,
    max_gamma: float = 2 * math.pi,
    n_alpha: int = 8,
    n_beta: int = 3,
    n_gamma=None,
) -> paddle.Tensor:
    """
    :return: rings of rotations around the identity, all points (rotations) in
    a ring are at the same distance from the identity
    size of the kernel = n_alpha * n_beta * n_gamma
    """
    if n_gamma is None:
        n_gamma = n_alpha
    beta = paddle.arange(start=1, end=n_beta + 1) * max_beta / n_beta
    alpha = paddle.linspace(start=0, stop=2 * math.pi, num=n_alpha)[:-1]
    pre_gamma = paddle.linspace(start=-max_gamma, stop=max_gamma, num=n_gamma)
    A, B, preC = paddle.meshgrid(alpha, beta, pre_gamma)
    C = preC - A
    A = A.flatten()
    B = B.flatten()
    C = C.flatten()
    return paddle.stack(x=(A, B, C))


def s2_irreps(lmax: int) -> o3.Irreps:
    return o3.Irreps([(1, (l, 1)) for l in range(lmax + 1)])


def so3_irreps(lmax: int) -> o3.Irreps:
    return o3.Irreps([(2 * l + 1, (l, 1)) for l in range(lmax + 1)])


def flat_wigner(lmax: int, alpha: paddle.Tensor, beta: paddle.Tensor, gamma: paddle.Tensor) -> paddle.Tensor:
    return paddle.concat(
        x=[((2 * l + 1) ** 0.5 * o3.wigner_D(l, alpha, beta, gamma).flatten(-2)) for l in range(lmax + 1)],
        axis=-1,
    )


class S2Convolution(paddle.nn.Layer):
    def __init__(self, f_in, f_out, lmax, kernel_grid) -> None:
        super().__init__()
        self.add_parameter(
            name="w",
            parameter=paddle.base.framework.EagerParamBase.from_tensor(
                tensor=paddle.randn(shape=[f_in, f_out, tuple(kernel_grid.shape)[1]])
            ),
        )
        self.register_buffer(
            name="Y",
            tensor=o3.spherical_harmonics_alpha_beta(range(lmax + 1), *kernel_grid, normalization="component"),
        )
        self.lin = o3.Linear(
            s2_irreps(lmax),
            so3_irreps(lmax),
            f_in=f_in,
            f_out=f_out,
            internal_weights=False,
        )

    def forward(self, x):
        psi = paddle.einsum("ni,xyn->xyi", self.Y, self.w) / tuple(self.Y.shape)[0] ** 0.5
        return self.lin(x, weight=psi)


class SO3Convolution(paddle.nn.Layer):
    def __init__(self, f_in, f_out, lmax, kernel_grid) -> None:
        super().__init__()
        self.add_parameter(
            name="w",
            parameter=paddle.base.framework.EagerParamBase.from_tensor(
                tensor=paddle.randn(shape=[f_in, f_out, tuple(kernel_grid.shape)[1]])
            ),
        )
        self.register_buffer(name="D", tensor=flat_wigner(lmax, *kernel_grid))
        self.lin = o3.Linear(
            so3_irreps(lmax),
            so3_irreps(lmax),
            f_in=f_in,
            f_out=f_out,
            internal_weights=False,
        )

    def forward(self, x):
        psi = paddle.einsum("ni,xyn->xyi", self.D, self.w) / tuple(self.D.shape)[0] ** 0.5
        return self.lin(x, weight=psi)


class S2ConvNet_original(paddle.nn.Layer):
    def __init__(self) -> None:
        super().__init__()
        f1 = 20
        f2 = 40
        f_output = 10
        b_in = 60
        lmax1 = 10
        b_l1 = 10
        lmax2 = 5
        b_l2 = 6
        grid_s2 = s2_near_identity_grid()
        grid_so3 = so3_near_identity_grid()
        self.from_s2 = o3.FromS2Grid((b_in, b_in), lmax1)
        self.conv1 = S2Convolution(1, f1, lmax1, kernel_grid=grid_s2)
        self.act1 = SO3Activation(lmax1, lmax2, paddle.nn.functional.relu, b_l1)
        self.conv2 = SO3Convolution(f1, f2, lmax2, kernel_grid=grid_so3)
        self.act2 = SO3Activation(lmax2, 0, paddle.nn.functional.relu, b_l2)
        self.w_out = paddle.base.framework.EagerParamBase.from_tensor(tensor=paddle.randn(shape=[f2, f_output]))

    def forward(self, x):
        x = x.transpose(perm=dim2perm(x.ndim, -1, -2))
        x = self.from_s2(x)
        x = self.conv1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.act2(x)
        x = x.flatten(start_axis=1) @ self.w_out / tuple(self.w_out.shape)[0]
        return x


MNIST_PATH = "s2_mnist.gz"
DEVICE = "gpu" if paddle.device.cuda.device_count() >= 1 else "cpu"
NUM_EPOCHS = 20
BATCH_SIZE = 32
LEARNING_RATE = 0.005


def load_data(path, batch_size):
    with gzip.open(path, "rb") as f:
        dataset = pickle.load(f)
    train_data = paddle.to_tensor(data=dataset["train"]["images"][:, None, :, :].astype(np.float32))
    train_labels = paddle.to_tensor(data=dataset["train"]["labels"].astype(np.int64))
    train_dataset = paddle.io.TensorDataset([train_data, train_labels])
    train_loader = paddle.io.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_data = paddle.to_tensor(data=dataset["test"]["images"][:, None, :, :].astype(np.float32))
    test_labels = paddle.to_tensor(data=dataset["test"]["labels"].astype(np.int64))
    test_dataset = paddle.io.TensorDataset([test_data, test_labels])
    test_loader = paddle.io.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)
    return train_loader, test_loader, train_dataset, test_dataset


def main() -> None:
    paddle.device.set_device(DEVICE)

    train_loader, test_loader, train_dataset, _ = load_data(MNIST_PATH, BATCH_SIZE)
    classifier = S2ConvNet_original()
    print("#params", sum(x.size for x in classifier.parameters()))
    optimizer = paddle.optimizer.Adam(
        parameters=classifier.parameters(),
        learning_rate=LEARNING_RATE,
        weight_decay=0.0,
    )
    for epoch in range(NUM_EPOCHS):
        for i, (images, labels) in enumerate(train_loader):
            classifier.train()
            optimizer.clear_gradients(set_to_zero=False)
            outputs = classifier(images)
            loss = paddle.nn.functional.cross_entropy(input=outputs, label=labels)
            loss.backward()
            optimizer.step()
            print(
                "\rEpoch [{0}/{1}], Iter [{2}/{3}] Loss: {4:.4f}".format(
                    epoch + 1,
                    NUM_EPOCHS,
                    i + 1,
                    len(train_dataset) // BATCH_SIZE,
                    loss.item(),
                ),
                end="",
            )
        print("")
        correct = 0
        total = 0
        for images, labels in test_loader:
            classifier.eval()
            with paddle.no_grad():
                outputs = classifier(images)
                _, predicted = paddle.max(x=outputs, axis=1), paddle.argmax(x=outputs, axis=1)
                total += labels.shape[0]
                correct += (predicted == labels).astype(dtype="int64").sum().item()
        print(f"Test Accuracy: {100 * correct / total}")


if __name__ == "__main__":
    main()
