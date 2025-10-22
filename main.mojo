from mojo_torch import Tensor, matmul
from mojo_torch import nn
from time import perf_counter_ns


fn main():
    var a = Tensor(1024, 1024)
    var b = Tensor(1024, 1024)

    var start = perf_counter_ns()
    var _ = a @ b  # Uses vectorized+parallelized
    var end = perf_counter_ns()
    print(
        "Time taken for matmul (vectorized + parallelized): ",
        (end - start) / 1e6,
        " milliseconds",
    )

    start = perf_counter_ns()
    var _ = a.__matmul_tiled__(b)  # Uses tiled accumulation
    end = perf_counter_ns()
    print(
        "Time taken for matmul (tiled): ", (end - start) / 1e6, " milliseconds"
    )

    start = perf_counter_ns()
    var _ = a.__matmul_simd__(b)  # Uses pure SIMD
    end = perf_counter_ns()
    print(
        "Time taken for matmul (SIMD): ", (end - start) / 1e6, " milliseconds"
    )

    start = perf_counter_ns()
    var _ = matmul(a, b, tiled=True)  # pytorch like
    end = perf_counter_ns()
    print(
        "Time taken for matmul (pytorch like): ",
        (end - start) / 1e6,
        " milliseconds",
    )

    linear_layer = nn.Linear(512, 256, True)
    var input = Tensor(12, 512)  # Batch size of 12
    var output = linear_layer.forward(input)
    var grads = linear_layer.backward(input, output, 0.01)
    print(
        "Output shape from Linear layer: [",
        output.shape[0],
        ", ",
        output.shape[1],
        "]",
    )
