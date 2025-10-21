from random import rand, randn
from math import sqrt
from memory import memset_zero
from memory.unsafe_pointer import UnsafePointer
from tensor import Tensor


struct Linear:
    """Linear (Fully Connected) Layer: y = xW^T + b."""
    var in_features: Int
    var out_features: Int
    var weight: Tensor
    var bias: Tensor

    fn __init__(out self, in_features: Int, out_features: Int, bias: Bool):
        self.in_features = in_features
        self.out_features = out_features

        # Initialize weights with Xavier initialization
        var xavier_bound = sqrt(6.0 / (in_features + out_features))
        self.weight = Tensor(out_features, in_features)
        randn(self.weight.data, out_features * in_features)
        self.weight = self.weight * Float32(xavier_bound)
        
        # Initialize bias to zeros
        if bias:
            self.bias = Tensor(1, out_features)
        else:
            self.bias = Tensor(1, out_features)
            memset_zero(self.bias.data, 1 * out_features)

    fn forward(self, input: Tensor) -> Tensor:
        """Forward pass: output = input @ weight.T + bias."""
        var batch_size = input.shape[0]
        
        # Matrix multiplication: [batch, in] @ [out, in].T = [batch, out]
        output = input @ self.weight.T()

        # Add bias
        output = output + self.bias

        return output^

    fn backward(self, input: Tensor, grad_output: Tensor, lr: Float32) -> Tuple[Tensor, Tensor, Tensor]:
        """Complete backward pass for linear layer."""
        var batch_size = input.shape[0]

        # 1. Gradient w.r.t. input: [batch, out] @ [out, in] = [batch, in]
        var grad_input: Tensor = grad_output @ self.weight

        # 2. Gradient w.r.t. weight: [out, batch] @ [batch, in] = [out, in]
        var grad_output_t: Tensor = grad_output.T()
        var grad_weight: Tensor = grad_output_t @ input

        # 3. Gradient w.r.t. bias: sum over batch dimension
        var grad_bias: Tensor = Tensor(1, self.out_features)
        for j in range(self.out_features):
            var sum: Float32 = 0.0
            for i in range(batch_size):
                sum = sum + grad_output[i, j]
            grad_bias[0, j] = sum

        return grad_input^, grad_weight^, grad_bias^
    
    fn __str__(self) -> String:
        return "Linear(" + String(self.in_features) + ", " + String(self.out_features) + ")"