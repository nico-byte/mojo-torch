from mojo_torch import Tensor, matmul
from mojo_torch import nn
from time import perf_counter_ns


fn main():
    print("Running mojo matmul benchmarks...")

    var a = Tensor[DType.float32]([1024, 1024])
    var b = Tensor[DType.float32]([1024, 1024])

    start = perf_counter_ns()
    var _ = matmul(a, b, tiled=True)  # pytorch like
    end = perf_counter_ns()
    print(
        "Time taken for matmul (pytorch like): ",
        (end - start) / 1e6,
        " milliseconds",
    )

    #linear_layer = nn.Linear(512, 256, True)
    #var input = Tensor(12, 512)  # Batch size of 12
    #var output = linear_layer.forward(input)
    #var grads = linear_layer.backward(input, output, 0.01)
    #print(
    #    "Output shape from Linear layer: [",
    #    output.shape[0],
    #    ", ",
    #    output.shape[1],
    #    "]",
    #)
