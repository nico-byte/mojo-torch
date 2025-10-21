from time import perf_counter_ns
import numpy as np


def main():
    print("Hello from mojo-nn!")
    a = np.ones((1024, 1024)).astype(np.float32)
    b = np.ones((1024, 1024)).astype(np.float32)

    start = perf_counter_ns()
    c = np.matmul(a, b)
    end = perf_counter_ns()

    print(c)
    print("Time taken for matmul (numpy):", (end - start) / 1e6, "milliseconds")


if __name__ == "__main__":
    main()
