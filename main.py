from time import perf_counter_ns
import numpy as np
import torch


def main():
    shape1 = (1024, 1024)
    shape2 = (1024, 1024)

    NUM_ITER = 500

    device = "cpu"

    print("Running torch and numpy matmul benchmarks...")

    avg_time = 0.0
    min_time = 1e12

    for _ in range(NUM_ITER):
        a = np.ones(shape1).astype(np.float32)
        b = np.ones(shape2).astype(np.float32)

        start = perf_counter_ns()
        c = np.matmul(a, b)
        end = perf_counter_ns()
        exec_time = end - start
        avg_time += exec_time
        min_time = min(min_time, exec_time)

    avg_time /= NUM_ITER
    print(
        "Average time taken for matmul (numpy):",
        round(avg_time / 1e3, 2),
        "microseconds",
    )
    print(
        "Minimum time taken for matmul (numpy):",
        round(min_time / 1e3, 2),
        "microseconds",
    )

    avg_time = 0.0
    min_time = 1e12

    for _ in range(NUM_ITER):
        torch_a = torch.ones(shape1).to(device)
        torch_b = torch.ones(shape2).to(device)

        start = perf_counter_ns()
        torch_c = torch.matmul(torch_a, torch_b)
        end = perf_counter_ns()
        exec_time = end - start
        avg_time += exec_time
        min_time = min(min_time, exec_time)

    avg_time /= NUM_ITER
    print(
        "Average time taken for matmul (torch):",
        round(avg_time / 1e3, 2),
        "microseconds",
    )
    print(
        "Minimum time taken for matmul (torch):",
        round(min_time / 1e3, 2),
        "microseconds",
    )


if __name__ == "__main__":
    main()
