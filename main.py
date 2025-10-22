from time import perf_counter_ns
import numpy as np
import torch


def main():
    device = "cpu"

    print("Hello from mojo-nn!")
    a = np.ones((1024, 1024)).astype(np.float32)
    b = np.ones((1024, 1024)).astype(np.float32)

    start = perf_counter_ns()
    c = np.matmul(a, b)
    end = perf_counter_ns()

    torch_a = torch.tensor([1.0, 2.0, 3.0]).to(device)
    torch_b = torch.tensor([4.0, 5.0, 6.0]).to(device)
    torch_result = torch.matmul(torch_a, torch_b)
    print(torch_result.shape)

    print(c)
    print("Time taken for matmul (numpy):", (end - start) / 1e6, "milliseconds")


if __name__ == "__main__":
    main()
