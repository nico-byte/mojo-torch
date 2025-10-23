from time import perf_counter_ns
import numpy as np
import torch


def main():
    device = "cpu"

    print("Running torch and numpy matmul benchmarks...")
    a = np.ones((1024, 1024)).astype(np.float32)
    b = np.ones((1024, 1024)).astype(np.float32)

    start = perf_counter_ns()
    c = np.matmul(a, b)
    end = perf_counter_ns()
    
    print("Time taken for matmul (numpy):", (end - start) / 1e6, "milliseconds")

    torch_a = torch.ones((1024, 1024)).to(device)
    torch_b = torch.ones((1024, 1024)).to(device)

    start = perf_counter_ns()
    torch_result = torch.matmul(torch_a, torch_b)
    end = perf_counter_ns()
    print("Time taken for matmul (torch):", (end - start) / 1e6, "milliseconds")
    print()


if __name__ == "__main__":
    main()
