import torch
import torch.nn as nn
import torch.nn.functional as F
import tvm.relax as relax
import tvm
MAIN_TRANSFORM_LIST = {}

def map_param(param: nn.Parameter):
    return relax.const(
        param.data.cpu().numpy(), relax.TensorStructInfo(param.data.shape, "float32")
    )

def register_convert_function(name, **kwargs):
    def _register_convert_function(cls_or_func):
        MAIN_TRANSFORM_LIST[name] = cls_or_func
        return cls_or_func
    return _register_convert_function


@register_convert_function(torch.nn.Linear)
def torch_nn_linear_op(bb, node_map, node, nn_mod):
    x = node_map[node.args[0]]
    w = map_param(nn_mod.weight)
    b = map_param(nn_mod.bias)
    return bb.emit(relax.op.linear(x, w, b))

@tvm.register_func("torch.nn.ReLU", override=True)
def lnumpy_relu(x: tvm.nd.NDArray,
                out: tvm.nd.NDArray):
    x_torch = torch.from_dlpack(x)
    out_torch = torch.from_dlpack(out)
    torch.maximum(x_torch, torch.Tensor([0.0]), out=out_torch)