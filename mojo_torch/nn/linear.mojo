from random import rand, randn
from math import sqrt
from memory import memset_zero
from memory.unsafe_pointer import UnsafePointer
from mojo_torch import Tensor
from mojo_torch.ops.matmul import matmul


struct Linear[Type: DType]:
    """Linear (Fully Connected) Layer: y = xW^T + b."""

    var in_features: Int
    var out_features: Int
    var weight: Tensor[Type]
    var bias: Tensor[Type]

    fn __init__(out self, in_features: Int, out_features: Int, bias: Bool):
        self.in_features = in_features
        self.out_features = out_features

        # Initialize weights with Xavier initialization
        var xavier_bound = sqrt(6.0 / (in_features + out_features))
        self.weight = Tensor[Type](out_features, in_features)
        randn(self.weight.data, out_features * in_features)
        self.weight = self.weight * Scalar[Type](xavier_bound)

        # Initialize bias to zeros
        if bias:
            self.bias = Tensor[Type](1, out_features)
        else:
            self.bias = Tensor[Type](1, out_features)
            memset_zero(self.bias.data, 1 * out_features)

    fn forward(self, input: Tensor[Type]) -> Tensor[Type]:
        """Forward pass: output = input @ weight.T + bias."""
        var batch_size = input.layout.shape[0]

        # Matrix multiplication: [batch, in] @ [out, in].T = [batch, out]
        output: Tensor[Type] = matmul[Type](self.weight.T(), input)

        # Add bias
        output = output + self.bias

        return output^

    fn backward(
        self, input: Tensor[Type], grad_output: Tensor[Type], lr: Float32
    ) -> Tuple[Tensor[Type], Tensor[Type], Tensor[Type]]:
        """Complete backward pass for linear layer."""
        var batch_size = input.layout.shape[0]

        # 1. Gradient w.r.t. input: [batch, out] @ [out, in] = [batch, in]
        var grad_input: Tensor[Type] = matmul(grad_output, self.weight)

        # 2. Gradient w.r.t. weight: [out, batch] @ [batch, in] = [out, in]
        var grad_output_t: Tensor[Type] = grad_output.T()
        var grad_weight: Tensor[Type] = matmul(grad_output_t, input)

        # 3. Gradient w.r.t. bias: sum over batch dimension
        var grad_bias: Tensor[Type] = Tensor[Type](1, self.out_features)
        for j in range(self.out_features):
            var sum: Scalar[Type] = 0.0
            for i in range(batch_size):
                sum = sum + grad_output[i, j]
            grad_bias[0, j] = sum

        return grad_input^, grad_weight^, grad_bias^

    fn __str__(self) -> String:
        return (
            "Linear("
            + String(self.in_features)
            + ", "
            + String(self.out_features)
            + ")"
        )
