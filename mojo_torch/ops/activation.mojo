from mojo_torch import Tensor


fn relu[Type: DType](x: Tensor[Type]) -> Tensor[Type]:
    """ReLU activation: max(0, x)."""
    var output = Tensor[Type](x.layout.shape)
    for i in range(x.size):
        var val = x.data.load(i)
        if val > 0:
            output.data.store(i, val)
        else:
            output.data.store(i, 0.0)
    return output^


fn relu_backward[
    Type: DType
](output: Tensor[Type], grad_output: Tensor[Type]) -> Tensor[Type]:
    """ReLU backward: gradient = grad_output if output > 0 else 0."""
    var grad_input = Tensor[Type](grad_output.layout.shape)
    for i in range(grad_output.size):
        var out_val = output.data.load(i)
        if out_val > 0:
            grad_input.data.store(i, grad_output.data.load(i))
        else:
            grad_input.data.store(i, 0.0)
    return grad_input^
