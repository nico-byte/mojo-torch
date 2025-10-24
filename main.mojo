from mojo_torch import Tensor, matmul
from mojo_torch import nn
from time import perf_counter_ns
from python import Python


fn main() raises:
    mojo_shape1 = [1024, 1024]
    mojo_shape2 = [1024, 1024]

    python_shape1 = Python.tuple(1024, 1024)
    python_shape2 = Python.tuple(1024, 1024)

    NUM_ITER = 500

    print("Running mojo matmul benchmarks...")

    var avg_time = 0.0
    var min_time = 1e12

    for _ in range(NUM_ITER):
        var a = Tensor[DType.float32](mojo_shape1, 1.0)
        var b = Tensor[DType.float32](mojo_shape2, 1.0)

        a.rand()
        b.rand()

        start = perf_counter_ns()
        var c = matmul(a, b, tiled=True)
        end = perf_counter_ns()
        var exec_time = end - start
        avg_time += Float64(exec_time)
        min_time = min(min_time, exec_time)

        a.data.free()
        b.data.free()
        c.data.free()

    avg_time /= NUM_ITER

    print(
        "Average time taken for matmul (mojo): ",
        avg_time / 1e3,
        " microseconds",
    )
    print(
        "Minimum time taken for matmul (mojo): ",
        min_time / 1e3,
        " microseconds",
    )

    print("\nRunning python matmul benchmarks...")
    np = Python.import_module("numpy")
    torch = Python.import_module("torch")

    device = "cpu"

    avg_time = 0.0
    min_time = 1e12

    for _ in range(NUM_ITER):
        a = np.ones(python_shape1).astype(np.float32)
        b = np.ones(python_shape2).astype(np.float32)

        start = perf_counter_ns()
        c = np.matmul(a, b)
        end = perf_counter_ns()
        exec_time = end - start
        avg_time += Float64(exec_time)
        min_time = min(min_time, exec_time)

    avg_time /= NUM_ITER
    print(
        "Average time taken for matmul (numpy):",
        avg_time / 1e3,
        " microseconds",
    )
    print(
        "Minimum time taken for matmul (numpy):",
        min_time / 1e3,
        " microseconds",
    )

    avg_time = 0.0
    min_time = 1e12

    for _ in range(NUM_ITER):
        torch_a = torch.ones(python_shape1).to(device)
        torch_b = torch.ones(python_shape2).to(device)

        start = perf_counter_ns()
        c = torch.matmul(torch_a, torch_b)
        end = perf_counter_ns()
        exec_time = end - start
        avg_time += Float64(exec_time)
        min_time = min(min_time, exec_time)

    avg_time /= NUM_ITER
    print(
        "Average time taken for matmul (torch):",
        avg_time / 1e3,
        " microseconds",
    )
    print(
        "Minimum time taken for matmul (torch):",
        min_time / 1e3,
        " microseconds",
    )
    print()

    # linear_layer = nn.Linear(512, 256, True)
    # var input = Tensor(12, 512)  # Batch size of 12
    # var output = linear_layer.forward(input)
    # var grads = linear_layer.backward(input, output, 0.01)
    # print(
    #    "Output shape from Linear layer: [",
    #    output.shape[0],
    #    ", ",
    #    output.shape[1],
    #    "]",
    # )
