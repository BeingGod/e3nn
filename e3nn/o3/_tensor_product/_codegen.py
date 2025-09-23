from math import sqrt
from typing import Callable
from typing import List

import paddle

from e3nn import o3
from e3nn.util import prod

from ._instruction import Instruction


def _sum_tensors(xs: List[paddle.Tensor], shape: list, like: paddle.Tensor):
    if len(xs) > 0:
        out = xs[0]
        for x in xs[1:]:
            out = out + x
        return out
    return paddle.zeros(shape=shape, dtype=like.dtype)


def codegen_tensor_product_right(
    irreps_in1: o3.Irreps,
    irreps_in2: o3.Irreps,
    irreps_out: o3.Irreps,
    instructions: List[Instruction],
    shared_weights: bool = False,
    specialized_code: bool = True,
    optimize_einsums: bool = True,
) -> Callable:
    """ """
    filtered_instructions = [ins for ins in instructions if 0 not in ins.path_shape]
    w3j_dict = {}
    for ins in filtered_instructions:
        i_in1, i_in2, i_out = ins.i_in1, ins.i_in2, ins.i_out
        mul_ir_in1 = irreps_in1[i_in1]
        mul_ir_in2 = irreps_in2[i_in2]
        mul_ir_out = irreps_out[i_out]
        if (mul_ir_in1.ir.l, mul_ir_in2.ir.l, mul_ir_out.ir.l) not in w3j_dict:
            w3j_name = f"_w3j_{mul_ir_in1.ir.l}_{mul_ir_in2.ir.l}_{mul_ir_out.ir.l}"
            w3j_dict[w3j_name] = o3.wigner_3j(mul_ir_in1.ir.l, mul_ir_in2.ir.l, mul_ir_out.ir.l)

    def tp_right(x2: paddle.Tensor, w: paddle.Tensor = None) -> paddle.Tensor:
        x2s, weights = x2, w

        if shared_weights:
            output_shape = tuple(x2s.shape[:-1])
        else:
            output_shape = paddle.broadcast_tensors(x2s[..., 0], weights[..., 0])[0].shape

        if len(filtered_instructions) == 0:
            outputs = paddle.zeros(
                output_shape
                + (
                    irreps_in1.dim,
                    irreps_out.dim,
                ),
                x2s.dtype,
            )

            return outputs

        # = Broadcast inputs =
        if not shared_weights:
            x2s, weights = x2s.broadcast_to(output_shape + (-1,)), weights.broadcast_to(output_shape + (-1,))

        output_shape = output_shape + (
            irreps_in1.dim,
            irreps_out.dim,
        )

        x2s = x2s.reshape(-1, irreps_in2.dim)

        batch_numel = x2s.shape[0]

        # = Determine number of weights and reshape weights ==
        weight_numel = sum(prod(ins.path_shape) for ins in instructions if ins.has_weight)
        if weight_numel > 0:
            weights = weights.reshape(-1, weight_numel)
        del weight_numel

        # = book-keeping for wigners =

        # = extract individual input irreps =
        # If only one input irrep, can avoid creating a view
        x2_list = []
        # If only one input irrep, can avoid creating a view
        if len(irreps_in2) == 1:
            x2_list.append(x2s.reshape(batch_numel, irreps_in2[0].mul, irreps_in2[0].ir.dim))
        else:
            for i, mul_ir in zip(irreps_in2.slices(), irreps_in2):
                x2_list.append(x2s[:, i].reshape(batch_numel, mul_ir.mul, mul_ir.ir.dim))

        # The einsum string index to prepend to the weights if the weights are not shared and have a batch dimension
        z = "" if shared_weights else "z"

        # Current index in the flat weight tensor
        flat_weight_index = 0

        outputs = []

        for ins in instructions:
            mul_ir_in1 = irreps_in1[ins.i_in1]
            mul_ir_in2 = irreps_in2[ins.i_in2]
            mul_ir_out = irreps_out[ins.i_out]

            assert mul_ir_in1.ir.p * mul_ir_in2.ir.p == mul_ir_out.ir.p
            assert abs(mul_ir_in1.ir.l - mul_ir_in2.ir.l) <= mul_ir_out.ir.l <= mul_ir_in1.ir.l + mul_ir_in2.ir.l

            if mul_ir_in1.dim == 0 or mul_ir_in2.dim == 0 or mul_ir_out.dim == 0:
                continue

            x2 = x2_list[ins.i_in2]

            e1 = paddle.eye(mul_ir_in1.mul, dtype=x2s.dtype)
            e2 = paddle.eye(mul_ir_in2.mul, dtype=x2s.dtype)
            i1 = paddle.eye(mul_ir_in1.ir.dim, dtype=x2s.dtype)

            assert ins.connection_mode in ["uvw", "uvu", "uvv", "uuw", "uuu", "uvuv", "uvu<v", "u<vw"]

            if ins.has_weight:
                # Extract the weight from the flattened weight tensor
                w = weights[:, flat_weight_index : flat_weight_index + prod(ins.path_shape)].reshape(
                    (() if shared_weights else (-1,)) + tuple(ins.path_shape)
                )
                flat_weight_index += prod(ins.path_shape)

            # Create a proxy & request for the relevant wigner w3j
            # If not used (because of specialized code), will get removed later.
            w3j_name = f"_w3j_{mul_ir_in1.ir.l}_{mul_ir_in2.ir.l}_{mul_ir_out.ir.l}"
            w3j = w3j_dict[w3j_name]

            if ins.connection_mode == "uvw":
                assert ins.has_weight
                if specialized_code and (mul_ir_in1.ir.l, mul_ir_in2.ir.l, mul_ir_out.ir.l) == (0, 0, 0):
                    result = paddle.einsum(f"{z}uvw,zv->zuw", w, x2.reshape(batch_numel, mul_ir_in2.dim))
                elif specialized_code and mul_ir_in1.ir.l == 0:
                    result = paddle.einsum(f"{z}uvw,zvi->zuwi", w, x2) / sqrt(mul_ir_out.ir.dim)
                elif specialized_code and mul_ir_in2.ir.l == 0:
                    result = paddle.einsum(f"{z}uvw,ij,zv->zuiwj", w, i1, x2.reshape(batch_numel, mul_ir_in2.dim)) / sqrt(
                        mul_ir_out.ir.dim
                    )
                elif specialized_code and mul_ir_out.ir.l == 0:
                    result = paddle.einsum(f"{z}uvw,zvi->zuiw", w, x2) / sqrt(mul_ir_in1.ir.dim)
                else:
                    result = paddle.einsum(f"{z}uvw,ijk,zvj->zuiwk", w, w3j, x2)
            if ins.connection_mode == "uvu":
                assert mul_ir_in1.mul == mul_ir_out.mul
                if ins.has_weight:
                    if specialized_code and (mul_ir_in1.ir.l, mul_ir_in2.ir.l, mul_ir_out.ir.l) == (0, 0, 0):
                        result = paddle.einsum(f"{z}uv,uw,zv->zuw", w, e1, x2.reshape(batch_numel, mul_ir_in2.dim))
                    elif specialized_code and mul_ir_in1.ir.l == 0:
                        result = paddle.einsum(f"{z}uv,uw,zvi->zuwi", w, e1, x2) / sqrt(mul_ir_out.ir.dim)
                    elif specialized_code and mul_ir_in2.ir.l == 0:
                        result = paddle.einsum(
                            f"{z}uv,ij,uw,zv->zuiwj", w, i1, e1, x2.reshape(batch_numel, mul_ir_in2.dim)
                        ) / sqrt(mul_ir_out.ir.dim)
                    elif specialized_code and mul_ir_out.ir.l == 0:
                        result = paddle.einsum(f"{z}uv,uw,zvi->zuiw", w, e1, x2) / sqrt(mul_ir_in1.ir.dim)
                    else:
                        result = paddle.einsum(f"{z}uv,ijk,uw,zvj->zuiwk", w, w3j, e1, x2)
                else:
                    # not so useful operation because v is summed
                    result = paddle.einsum("ijk,uw,zvj->zuiwk", w3j, e1, x2)
            if ins.connection_mode == "uvv":
                assert mul_ir_in2.mul == mul_ir_out.mul
                if ins.has_weight:
                    if specialized_code and (mul_ir_in1.ir.l, mul_ir_in2.ir.l, mul_ir_out.ir.l) == (0, 0, 0):
                        result = paddle.einsum(f"{z}uv,vw,zv->zuw", w, e2, x2.reshape(batch_numel, mul_ir_in2.dim))
                    elif specialized_code and mul_ir_in1.ir.l == 0:
                        result = paddle.einsum(f"{z}uv,vw,zvi->zuwi", w, e2, x2) / sqrt(mul_ir_out.ir.dim)
                    elif specialized_code and mul_ir_in2.ir.l == 0:
                        result = paddle.einsum(
                            f"{z}uv,ij,vw,zv->zuiwj", w, i1, e2, x2.reshape(batch_numel, mul_ir_in2.dim)
                        ) / sqrt(mul_ir_out.ir.dim)
                    elif specialized_code and mul_ir_out.ir.l == 0:
                        result = paddle.einsum(f"{z}uv,vw,zvi->zuiw", w, e2, x2) / sqrt(mul_ir_in1.ir.dim)
                    else:
                        result = paddle.einsum(f"{z}uv,ijk,zvj->zuivk", w, w3j, x2)
                else:
                    # not so useful operation because u is summed
                    # only specialize out for this path
                    s2ones = paddle.ones(mul_ir_in1.mul, dtype=x2s.dtype)
                    result = paddle.einsum("u,ijk,zvj->zuivk", s2ones, w3j, x2)
            if ins.connection_mode == "uuw":
                assert mul_ir_in1.mul == mul_ir_in2.mul
                if ins.has_weight:
                    # TODO: specialize right()
                    result = paddle.einsum(f"{z}uw,ijk,zuj->zuiwk", w, w3j, x2)
                else:
                    # equivalent to tp(x, y, 'uuu').sum('u')
                    assert mul_ir_out.mul == 1
                    result = paddle.einsum("ijk,zuj->zuik", w3j, x2)
            if ins.connection_mode == "uuu":
                assert mul_ir_in1.mul == mul_ir_in2.mul == mul_ir_out.mul
                if ins.has_weight:
                    if specialized_code and (mul_ir_in1.ir.l, mul_ir_in2.ir.l, mul_ir_out.ir.l) == (0, 0, 0):
                        result = paddle.einsum(f"{z}u,uw,zu->zuw", w, e2, x2.reshape(batch_numel, mul_ir_in2.dim))
                    elif specialized_code and (mul_ir_in1.ir.l, mul_ir_in2.ir.l, mul_ir_out.ir.l) == (1, 1, 1):
                        # For cross product, use the general case right()
                        result = paddle.einsum(f"{z}u,ijk,uw,zuj->zuiwk", w, w3j, e1, x2)
                    elif specialized_code and mul_ir_in1.ir.l == 0:
                        result = paddle.einsum(f"{z}u,uw,zui->zuwi", w, e2, x2) / sqrt(mul_ir_out.ir.dim)
                    elif specialized_code and mul_ir_in2.ir.l == 0:
                        result = paddle.einsum(
                            f"{z}u,ij,uw,zu->zuiwj", w, i1, e2, x2.reshape(batch_numel, mul_ir_in2.dim)
                        ) / sqrt(mul_ir_out.ir.dim)
                    elif specialized_code and mul_ir_out.ir.l == 0:
                        result = paddle.einsum(f"{z}u,uw,zui->zuiw", w, e2, x2) / sqrt(mul_ir_in1.ir.dim)
                    else:
                        result = paddle.einsum(f"{z}u,ijk,uw,zuj->zuiwk", w, w3j, e1, x2)
                else:
                    if specialized_code and (mul_ir_in1.ir.l, mul_ir_in2.ir.l, mul_ir_out.ir.l) == (0, 0, 0):
                        result = paddle.einsum("uw,zu->zuw", e2, x2.reshape(batch_numel, mul_ir_in2.dim))
                    elif specialized_code and (mul_ir_in1.ir.l, mul_ir_in2.ir.l, mul_ir_out.ir.l) == (1, 1, 1):
                        # For cross product, use the general case right()
                        result = paddle.einsum("ijk,uw,zuj->zuiwk", w3j, e1, x2)
                    elif specialized_code and mul_ir_in1.ir.l == 0:
                        result = paddle.einsum("uw,zui->zuwi", e2, x2) / sqrt(mul_ir_out.ir.dim)
                    elif specialized_code and mul_ir_in2.ir.l == 0:
                        result = paddle.einsum("ij,uw,zu->zuiwj", i1, e2, x2.reshape(batch_numel, mul_ir_in2.dim)) / sqrt(
                            mul_ir_out.ir.dim
                        )
                    elif specialized_code and mul_ir_out.ir.l == 0:
                        result = paddle.einsum("uw,zui->zuiw", e2, x2) / sqrt(mul_ir_in1.ir.dim)
                    else:
                        result = paddle.einsum("ijk,uw,zuj->zuiwk", w3j, e1, x2)
            if ins.connection_mode == "uvuv":
                assert mul_ir_in1.mul * mul_ir_in2.mul == mul_ir_out.mul
                if ins.has_weight:
                    # TODO implement specialized code
                    result = paddle.einsum(f"{z}uv,ijk,uw,zvj->zuiwvk", w, w3j, e1, x2)
                else:
                    # TODO implement specialized code
                    result = paddle.einsum("ijk,uw,zvj->zuiwvk", w3j, e1, x2)
            if ins.connection_mode == "uvu<v":
                raise NotImplementedError
            if ins.connection_mode == "u<vw":
                raise NotImplementedError

            result = ins.path_weight * result

            outputs += [result.reshape(batch_numel, mul_ir_in1.dim, mul_ir_out.dim)]

        # = Return the result =
        outputs = [
            paddle.concat(
                [
                    _sum_tensors(
                        [out for ins, out in zip(instructions, outputs) if (ins.i_in1, ins.i_out) == (i_in1, i_out)],
                        shape=(batch_numel, mul_ir_in1.dim, mul_ir_out.dim),
                        like=x2s,
                    )
                    for i_out, mul_ir_out in enumerate(irreps_out)
                    if mul_ir_out.mul > 0
                ],
                axis=2,
            )
            for i_in1, mul_ir_in1 in enumerate(irreps_in1)
            if mul_ir_in1.mul > 0
        ]
        if len(outputs) > 1:
            outputs = paddle.concat(outputs, axis=1)
        else:
            outputs = outputs[0]

        outputs = outputs.reshape(output_shape)

        return outputs

    return tp_right


def codegen_tensor_product_left_right(
    irreps_in1: o3.Irreps,
    irreps_in2: o3.Irreps,
    irreps_out: o3.Irreps,
    instructions: List,
    shared_weights: bool = False,
    specialized_code: bool = True,
    optimize_einsums: bool = True,
) -> Callable:
    w3j_dict = {}
    filtered_instructions = [ins for ins in instructions if 0 not in ins.path_shape]
    for ins in filtered_instructions:
        if 0 in ins.path_shape:
            continue
        mul_ir_in1 = irreps_in1[ins.i_in1]
        mul_ir_in2 = irreps_in2[ins.i_in2]
        mul_ir_out = irreps_out[ins.i_out]
        if (mul_ir_in1.ir.l, mul_ir_in2.ir.l, mul_ir_out.ir.l) not in w3j_dict:
            w3j_name = f"_w3j_{mul_ir_in1.ir.l}_{mul_ir_in2.ir.l}_{mul_ir_out.ir.l}"
            w3j_dict[w3j_name] = o3.wigner_3j(mul_ir_in1.ir.l, mul_ir_in2.ir.l, mul_ir_out.ir.l)

    def tp_forward(x1: paddle.Tensor, x2: paddle.Tensor, w: paddle.Tensor = None) -> paddle.Tensor:
        x1s, x2s, weights = x1, x2, w

        if shared_weights:
            # by broadcasting all but the final irrep dim, we broadcast any and all batch dimensions
            # the use of `:1` is important, rather than `0`, to ensure the case with an empty irrep dimension doesn't error out
            output_shape = paddle.broadcast_tensors([x1s[..., :1], x2s[..., :1]])[0].shape[:-1]
        else:
            output_shape = paddle.broadcast_tensors([x1s[..., :1], x2s[..., :1], weights[..., :1]])[0].shape[:-1]

        output_shape = tuple(output_shape)
        if len(instructions) == 0:
            outputs = paddle.zeros(output_shape + (irreps_out.dim,), dtype=x1s.dtype)

            return outputs

        # = Broadcast inputs =
        output_shape = tuple(output_shape)
        bc_shape = output_shape + (-1,)
        x1s, x2s = x1s.expand(bc_shape), x2s.expand(bc_shape)
        if not shared_weights:
            weights = weights.expand(bc_shape)

        output_shape = output_shape + (irreps_out.dim,)

        x1s = x1s.reshape(-1, irreps_in1.dim)
        x2s = x2s.reshape(-1, irreps_in2.dim)

        batch_numel = x1s.shape[0]

        # = Determine number of weights and reshape weights ==
        weight_numel = sum(prod(ins.path_shape) for ins in instructions if ins.has_weight)
        if weight_numel > 0:
            weights = weights.reshape(-1, weight_numel)
        del weight_numel

        # = extract individual input irreps =
        # If only one input irrep, can avoid creating a view
        if len(irreps_in1) == 1:
            x1_list = [x1s.reshape(batch_numel, irreps_in1[0].mul, irreps_in1[0].ir.dim)]
        else:
            x1_list = [
                x1s[:, i].reshape(batch_numel, mul_ir.mul, mul_ir.ir.dim) for i, mul_ir in zip(irreps_in1.slices(), irreps_in1)
            ]

        x2_list = []
        # If only one input irrep, can avoid creating a view
        if len(irreps_in2) == 1:
            x2_list.append(x2s.reshape(batch_numel, irreps_in2[0].mul, irreps_in2[0].ir.dim))
        else:
            for i, mul_ir in zip(irreps_in2.slices(), irreps_in2):
                x2_list.append(x2s[:, i].reshape(batch_numel, mul_ir.mul, mul_ir.ir.dim))

        # The einsum string index to prepend to the weights if the weights are not shared and have a batch dimension
        z = "" if shared_weights else "z"

        # Cache of input irrep pairs whose outer products (xx) have already been computed
        xx_dict = dict()

        # Current index in the flat weight tensor
        flat_weight_index = 0

        outputs = []

        for ins in instructions:
            mul_ir_in1 = irreps_in1[ins.i_in1]
            mul_ir_in2 = irreps_in2[ins.i_in2]
            mul_ir_out = irreps_out[ins.i_out]

            assert mul_ir_in1.ir.p * mul_ir_in2.ir.p == mul_ir_out.ir.p
            assert abs(mul_ir_in1.ir.l - mul_ir_in2.ir.l) <= mul_ir_out.ir.l <= mul_ir_in1.ir.l + mul_ir_in2.ir.l

            if mul_ir_in1.dim == 0 or mul_ir_in2.dim == 0 or mul_ir_out.dim == 0:
                continue

            x1 = x1_list[ins.i_in1]
            x2 = x2_list[ins.i_in2]

            assert ins.connection_mode in ["uvw", "uvu", "uvv", "uuw", "uuu", "uvuv", "uvu<v", "u<vw"]

            if ins.has_weight:
                # Extract the weight from the flattened weight tensor
                w = weights[:, flat_weight_index : flat_weight_index + prod(ins.path_shape)].reshape(
                    (() if shared_weights else (-1,)) + tuple(ins.path_shape)
                )
                flat_weight_index += prod(ins.path_shape)

            # Construct the general xx in case this instruction isn't specialized
            # If this isn't used, the dead code will get removed
            key = (ins.i_in1, ins.i_in2, ins.connection_mode[:2])
            if key not in xx_dict:
                if ins.connection_mode[:2] == "uu":
                    xx_dict[key] = paddle.einsum("zui,zuj->zuij", x1, x2)
                else:
                    xx_dict[key] = paddle.einsum("zui,zvj->zuvij", x1, x2)
            xx = xx_dict[key]
            del key

            # Create a proxy & request for the relevant wigner w3j
            # If not used (because of specialized code), will get removed later.
            w3j_name = f"_w3j_{mul_ir_in1.ir.l}_{mul_ir_in2.ir.l}_{mul_ir_out.ir.l}"
            w3j = w3j_dict[w3j_name]

            l1l2l3 = (mul_ir_in1.ir.l, mul_ir_in2.ir.l, mul_ir_out.ir.l)

            if ins.connection_mode == "uvw":
                assert ins.has_weight
                if specialized_code and l1l2l3 == (0, 0, 0):
                    result = paddle.einsum(
                        f"{z}uvw,zu,zv->zw",
                        w,
                        x1.reshape(batch_numel, mul_ir_in1.dim),
                        x2.reshape(batch_numel, mul_ir_in2.dim),
                    )
                elif specialized_code and mul_ir_in1.ir.l == 0:
                    result = paddle.einsum(f"{z}uvw,zu,zvj->zwj", w, x1.reshape(batch_numel, mul_ir_in1.dim), x2) / sqrt(
                        mul_ir_out.ir.dim
                    )
                elif specialized_code and mul_ir_in2.ir.l == 0:
                    result = paddle.einsum(f"{z}uvw,zui,zv->zwi", w, x1, x2.reshape(batch_numel, mul_ir_in2.dim)) / sqrt(
                        mul_ir_out.ir.dim
                    )
                elif specialized_code and mul_ir_out.ir.l == 0:
                    result = paddle.einsum(f"{z}uvw,zui,zvi->zw", w, x1, x2) / sqrt(mul_ir_in1.ir.dim)
                else:
                    result = paddle.einsum(f"{z}uvw,ijk,zuvij->zwk", w, w3j, xx)
            if ins.connection_mode == "uvu":
                assert mul_ir_in1.mul == mul_ir_out.mul
                if ins.has_weight:
                    if specialized_code and l1l2l3 == (0, 0, 0):
                        result = paddle.einsum(
                            f"{z}uv,zu,zv->zu",
                            w,
                            x1.reshape(batch_numel, mul_ir_in1.dim),
                            x2.reshape(batch_numel, mul_ir_in2.dim),
                        )
                    elif specialized_code and mul_ir_in1.ir.l == 0:
                        result = paddle.einsum(f"{z}uv,zu,zvj->zuj", w, x1.reshape(batch_numel, mul_ir_in1.dim), x2) / sqrt(
                            mul_ir_out.ir.dim
                        )
                    elif specialized_code and mul_ir_in2.ir.l == 0:
                        result = paddle.einsum(f"{z}uv,zui,zv->zui", w, x1, x2.reshape(batch_numel, mul_ir_in2.dim)) / sqrt(
                            mul_ir_out.ir.dim
                        )
                    elif specialized_code and mul_ir_out.ir.l == 0:
                        result = paddle.einsum(f"{z}uv,zui,zvi->zu", w, x1, x2) / sqrt(mul_ir_in1.ir.dim)
                    else:
                        result = paddle.einsum(f"{z}uv,ijk,zuvij->zuk", w, w3j, xx)
                else:
                    # not so useful operation because v is summed
                    result = paddle.einsum("ijk,zuvij->zuk", w3j, xx)
            if ins.connection_mode == "uvv":
                assert mul_ir_in2.mul == mul_ir_out.mul
                if ins.has_weight:
                    if specialized_code and l1l2l3 == (0, 0, 0):
                        result = paddle.einsum(
                            f"{z}uv,zu,zv->zv",
                            w,
                            x1.reshape(batch_numel, mul_ir_in1.dim),
                            x2.reshape(batch_numel, mul_ir_in2.dim),
                        )
                    elif specialized_code and mul_ir_in1.ir.l == 0:
                        result = paddle.einsum(f"{z}uv,zu,zvj->zvj", w, x1.reshape(batch_numel, mul_ir_in1.dim), x2) / sqrt(
                            mul_ir_out.ir.dim
                        )
                    elif specialized_code and mul_ir_in2.ir.l == 0:
                        result = paddle.einsum(f"{z}uv,zui,zv->zvi", w, x1, x2.reshape(batch_numel, mul_ir_in2.dim)) / sqrt(
                            mul_ir_out.ir.dim
                        )
                    elif specialized_code and mul_ir_out.ir.l == 0:
                        result = paddle.einsum(f"{z}uv,zui,zvi->zv", w, x1, x2) / sqrt(mul_ir_in1.ir.dim)
                    else:
                        result = paddle.einsum(f"{z}uv,ijk,zuvij->zvk", w, w3j, xx)
                else:
                    # not so useful operation because u is summed
                    # only specialize out for this path
                    if specialized_code and l1l2l3 == (0, 0, 0):
                        result = paddle.einsum(
                            "zu,zv->zv", x1.reshape(batch_numel, mul_ir_in1.dim), x2.reshape(batch_numel, mul_ir_in2.dim)
                        )
                    elif specialized_code and mul_ir_in1.ir.l == 0:
                        result = paddle.einsum("zu,zvj->zvj", x1.reshape(batch_numel, mul_ir_in1.dim), x2) / sqrt(
                            mul_ir_out.ir.dim
                        )
                    elif specialized_code and mul_ir_in2.ir.l == 0:
                        result = paddle.einsum("zui,zv->zvi", x1, x2.reshape(batch_numel, mul_ir_in2.dim)) / sqrt(
                            mul_ir_out.ir.dim
                        )
                    elif specialized_code and mul_ir_out.ir.l == 0:
                        result = paddle.einsum("zui,zvi->zv", x1, x2) / sqrt(mul_ir_in1.ir.dim)
                    else:
                        result = paddle.einsum("ijk,zuvij->zvk", w3j, xx)
            if ins.connection_mode == "uuw":
                assert mul_ir_in1.mul == mul_ir_in2.mul
                if ins.has_weight:
                    if specialized_code and l1l2l3 == (0, 0, 0):
                        result = paddle.einsum(
                            f"{z}uw,zu,zu->zw",
                            w,
                            x1.reshape(batch_numel, mul_ir_in1.dim),
                            x2.reshape(batch_numel, mul_ir_in2.dim),
                        )
                    elif specialized_code and mul_ir_in1.ir.l == 0:
                        result = paddle.einsum(f"{z}uw,zu,zuj->zwj", w, x1.reshape(batch_numel, mul_ir_in1.dim), x2) / sqrt(
                            mul_ir_out.ir.dim
                        )
                    elif specialized_code and mul_ir_in2.ir.l == 0:
                        result = paddle.einsum(f"{z}uw,zui,zu->zwi", w, x1, x2.reshape(batch_numel, mul_ir_in2.dim)) / sqrt(
                            mul_ir_out.ir.dim
                        )
                    elif specialized_code and mul_ir_out.ir.l == 0:
                        result = paddle.einsum(f"{z}uw,zui,zui->zw", w, x1, x2) / sqrt(mul_ir_in1.ir.dim)
                    else:
                        result = paddle.einsum(f"{z}uw,ijk,zuij->zwk", w, w3j, xx)
                else:
                    # equivalent to tp(x, y, 'uuu').sum('u')
                    assert mul_ir_out.mul == 1
                    result = paddle.einsum("ijk,zuij->zk", w3j, xx)
            if ins.connection_mode == "uuu":
                assert mul_ir_in1.mul == mul_ir_in2.mul == mul_ir_out.mul
                if ins.has_weight:
                    if specialized_code and l1l2l3 == (0, 0, 0):
                        result = paddle.einsum(
                            f"{z}u,zu,zu->zu",
                            w,
                            x1.reshape(batch_numel, mul_ir_in1.dim),
                            x2.reshape(batch_numel, mul_ir_in2.dim),
                        )
                    elif specialized_code and l1l2l3 == (1, 1, 1):
                        result = paddle.einsum(f"{z}u,zui->zui", w, paddle.cross(x1, x2, axis=2)) / sqrt(2 * 3)
                    elif specialized_code and mul_ir_in1.ir.l == 0:
                        result = paddle.einsum(f"{z}u,zu,zuj->zuj", w, x1.reshape(batch_numel, mul_ir_in1.dim), x2) / sqrt(
                            mul_ir_out.ir.dim
                        )
                    elif specialized_code and mul_ir_in2.ir.l == 0:
                        result = paddle.einsum(f"{z}u,zui,zu->zui", w, x1, x2.reshape(batch_numel, mul_ir_in2.dim)) / sqrt(
                            mul_ir_out.ir.dim
                        )
                    elif specialized_code and mul_ir_out.ir.l == 0:
                        result = paddle.einsum(f"{z}u,zui,zui->zu", w, x1, x2) / sqrt(mul_ir_in1.ir.dim)
                    else:
                        result = paddle.einsum(f"{z}u,ijk,zuij->zuk", w, w3j, xx)
                else:
                    if specialized_code and l1l2l3 == (0, 0, 0):
                        result = paddle.einsum(
                            "zu,zu->zu", x1.reshape(batch_numel, mul_ir_in1.dim), x2.reshape(batch_numel, mul_ir_in2.dim)
                        )
                    elif specialized_code and l1l2l3 == (1, 1, 1):
                        result = paddle.cross(x1, x2, axis=2) * (1.0 / sqrt(2 * 3))
                    elif specialized_code and mul_ir_in1.ir.l == 0:
                        result = paddle.einsum("zu,zuj->zuj", x1.reshape(batch_numel, mul_ir_in1.dim), x2) / sqrt(
                            mul_ir_out.ir.dim
                        )
                    elif specialized_code and mul_ir_in2.ir.l == 0:
                        result = paddle.einsum("zui,zu->zui", x1, x2.reshape(batch_numel, mul_ir_in2.dim)) / sqrt(
                            mul_ir_out.ir.dim
                        )
                    elif specialized_code and mul_ir_out.ir.l == 0:
                        result = paddle.einsum("zui,zui->zu", x1, x2) / sqrt(mul_ir_in1.ir.dim)
                    else:
                        result = paddle.einsum("ijk,zuij->zuk", w3j, xx)
            if ins.connection_mode == "uvuv":
                assert mul_ir_in1.mul * mul_ir_in2.mul == mul_ir_out.mul
                if ins.has_weight:
                    # TODO implement specialized code
                    result = paddle.einsum(f"{z}uv,ijk,zuvij->zuvk", w, w3j, xx)
                else:
                    # TODO implement specialized code
                    result = paddle.einsum("ijk,zuvij->zuvk", w3j, xx)
            if ins.connection_mode == "uvu<v":
                assert mul_ir_in1.mul == mul_ir_in2.mul
                assert mul_ir_in1.mul * (mul_ir_in1.mul - 1) // 2 == mul_ir_out.mul
                i = paddle.triu_indices(mul_ir_in1.mul, mul_ir_in1.mul, 1)
                xx = xx[:, i[0], i[1]]  # zuvij -> zwij
                if ins.has_weight:
                    # TODO implement specialized code
                    result = paddle.einsum(f"{z}w,ijk,zwij->zwk", w, w3j, xx)
                else:
                    # TODO implement specialized code
                    result = paddle.einsum("ijk,zwij->zwk", w3j, xx)
            if ins.connection_mode == "u<vw":
                assert mul_ir_in1.mul == mul_ir_in2.mul
                assert ins.has_weight
                i = paddle.triu_indices(mul_ir_in1.mul, mul_ir_in1.mul, 1)
                xx = xx[:, i[0], i[1]]  # zuvij -> zqij
                # TODO implement specialized code
                result = paddle.einsum(f"{z}qw,ijk,zqij->zwk", w, w3j, xx)

            result = ins.path_weight * result

            outputs += [result.reshape(batch_numel, mul_ir_out.dim)]

        # = Return the result =
        outputs = [
            _sum_tensors(
                [out for ins, out in zip(instructions, outputs) if ins.i_out == i_out],
                shape=(batch_numel, mul_ir_out.dim),
                like=x1s,
            )
            for i_out, mul_ir_out in enumerate(irreps_out)
            if mul_ir_out.mul > 0
        ]
        if len(outputs) > 1:
            outputs = paddle.concat(outputs, axis=1)
        else:
            # Avoid an unnecessary copy in a size one paddle.cat
            outputs = outputs[0]

        outputs = outputs.reshape(output_shape)

        return outputs

    return tp_forward
