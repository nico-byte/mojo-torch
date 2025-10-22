from tensor.tensor import Tensor
from ops.matmul import matmul
from python import Python, PythonObject


fn abs_f32(x: Float32) -> Float32:
    return x if x >= 0 else -x


struct TestResults:
    var passed: Int
    var failed: Int
    var total: Int

    fn __init__(out self):
        self.passed = 0
        self.failed = 0
        self.total = 0

    fn assert_equal(
        mut self,
        actual: Float32,
        expected: Float32,
        test_name: String,
        tolerance: Float32 = 1e-6,
    ):
        self.total += 1
        if abs_f32(actual - expected) < tolerance:
            self.passed += 1
            print("✓ PASS:", test_name)
        else:
            self.failed += 1
            print("✗ FAIL:", test_name, "- Expected:", expected, "Got:", actual)

    fn assert_shape_equal(
        mut self,
        actual_shape: List[Int],
        expected_shape: List[Int],
        test_name: String,
    ):
        self.total += 1
        var shapes_match = True

        if actual_shape.__len__() != expected_shape.__len__():
            shapes_match = False
        else:
            for i in range(actual_shape.__len__()):
                if actual_shape[i] != expected_shape[i]:
                    shapes_match = False
                    break

        if shapes_match:
            self.passed += 1
            print("✓ PASS:", test_name)
        else:
            self.failed += 1
            print("✗ FAIL:", test_name, "- Shape mismatch")
            print("  Expected: [", end="")
            for i in range(expected_shape.__len__()):
                print(expected_shape[i], end="")
                if i < expected_shape.__len__() - 1:
                    print(", ", end="")
            print("]")
            print("  Got: [", end="")
            for i in range(actual_shape.__len__()):
                print(actual_shape[i], end="")
                if i < actual_shape.__len__() - 1:
                    print(", ", end="")
            print("]")

    fn print_summary(self):
        print("\n" + "=" * 50)
        print("TEST SUMMARY")
        print("=" * 50)
        print("Total tests:", self.total)
        print("Passed:", self.passed)
        print("Failed:", self.failed)
        if self.failed == 0:
            print("All tests passed!")
        else:
            print("Some tests failed. Please check the details above.")
        print("=" * 50)


fn test_1d_x_1d_dot_product(mut results: TestResults) raises:
    print("\nTest 1: 1D x 1D (dot product)")

    # Mojo implementation
    var a = Tensor(3)
    var b = Tensor(3)
    a[0] = 1.0
    a[1] = 2.0
    a[2] = 3.0
    b[0] = 4.0
    b[1] = 5.0
    b[2] = 6.0

    var result = matmul(a, b)

    # PyTorch reference
    var torch = Python.import_module("torch")
    var torch_a = torch.tensor([1.0, 2.0, 3.0])
    var torch_b = torch.tensor([4.0, 5.0, 6.0])
    var torch_result = torch.matmul(torch_a, torch_b)

    # Assertions
    var expected_shape = List[Int](1)
    results.assert_shape_equal(result.shape, expected_shape, "1D x 1D shape")
    results.assert_equal(
        result[0], Float32(torch_result.item()), "1D x 1D value"
    )
    results.assert_equal(result[0], 32.0, "1D x 1D manual calculation")


fn test_2d_x_2d_matrix_multiply(mut results: TestResults) raises:
    print("\nTest 2: 2D x 2D (matrix multiplication)")

    # Mojo implementation
    var c = Tensor(2, 3)
    var d = Tensor(3, 2)

    # Fill c with values [[1, 2, 3], [4, 5, 6]]
    c[0, 0] = 1.0
    c[0, 1] = 2.0
    c[0, 2] = 3.0
    c[1, 0] = 4.0
    c[1, 1] = 5.0
    c[1, 2] = 6.0

    # Fill d with values [[7, 8], [9, 10], [11, 12]]
    d[0, 0] = 7.0
    d[0, 1] = 8.0
    d[1, 0] = 9.0
    d[1, 1] = 10.0
    d[2, 0] = 11.0
    d[2, 1] = 12.0

    var result = matmul(c, d)

    # PyTorch reference
    var torch = Python.import_module("torch")
    var init_c = Python.list(
        Python.list(1.0, 2.0, 3.0), Python.list(4.0, 5.0, 6.0)
    )
    var init_d = Python.list(
        Python.list(7.0, 8.0), Python.list(9.0, 10.0), Python.list(11.0, 12.0)
    )
    var torch_c = torch.tensor(init_c)
    var torch_d = torch.tensor(init_d)
    var torch_result = torch.matmul(torch_c, torch_d)

    # Assertions
    var expected_shape = List[Int](2, 2)
    results.assert_shape_equal(result.shape, expected_shape, "2D x 2D shape")
    results.assert_equal(
        result[0, 0], Float32(torch_result[0, 0].item()), "2D x 2D [0,0]"
    )
    results.assert_equal(
        result[0, 1], Float32(torch_result[0, 1].item()), "2D x 2D [0,1]"
    )
    results.assert_equal(
        result[1, 0], Float32(torch_result[1, 0].item()), "2D x 2D [1,0]"
    )
    results.assert_equal(
        result[1, 1], Float32(torch_result[1, 1].item()), "2D x 2D [1,1]"
    )

    # Manual calculations
    results.assert_equal(
        result[0, 0], 58.0, "2D x 2D [0,0] manual: 1*7 + 2*9 + 3*11"
    )
    results.assert_equal(
        result[0, 1], 64.0, "2D x 2D [0,1] manual: 1*8 + 2*10 + 3*12"
    )


fn test_1d_x_2d_broadcast(mut results: TestResults) raises:
    print("\nTest 3: 1D x 2D (broadcast)")

    # Mojo implementation
    var e = Tensor(3)
    e[0] = 1.0
    e[1] = 2.0
    e[2] = 3.0

    var d = Tensor(3, 2)
    d[0, 0] = 7.0
    d[0, 1] = 8.0
    d[1, 0] = 9.0
    d[1, 1] = 10.0
    d[2, 0] = 11.0
    d[2, 1] = 12.0

    var result = matmul(e, d)

    # PyTorch reference
    var torch = Python.import_module("torch")
    var torch_e = torch.tensor([1.0, 2.0, 3.0])
    var init_d = Python.list(
        Python.list(7.0, 8.0), Python.list(9.0, 10.0), Python.list(11.0, 12.0)
    )
    var torch_d = torch.tensor(init_d)
    var torch_result = torch.matmul(torch_e, torch_d)

    # Assertions
    var expected_shape = List[Int](2)
    results.assert_shape_equal(result.shape, expected_shape, "1D x 2D shape")
    results.assert_equal(
        result[0], Float32(torch_result[0].item()), "1D x 2D [0]"
    )
    results.assert_equal(
        result[1], Float32(torch_result[1].item()), "1D x 2D [1]"
    )


fn test_2d_x_1d_matvec(mut results: TestResults) raises:
    print("\nTest 4: 2D x 1D (matrix-vector)")

    # Mojo implementation
    var c = Tensor(2, 3)
    c[0, 0] = 1.0
    c[0, 1] = 2.0
    c[0, 2] = 3.0
    c[1, 0] = 4.0
    c[1, 1] = 5.0
    c[1, 2] = 6.0

    var f = Tensor(3)
    f[0] = 1.0
    f[1] = 2.0
    f[2] = 3.0

    var result = matmul(c, f)

    # PyTorch reference
    var torch = Python.import_module("torch")
    var init_c = Python.list(
        Python.list(1.0, 2.0, 3.0), Python.list(4.0, 5.0, 6.0)
    )
    var torch_c = torch.tensor(init_c)
    var torch_f = torch.tensor([1.0, 2.0, 3.0])
    var torch_result = torch.matmul(torch_c, torch_f)

    # Assertions
    var expected_shape = List[Int](2)
    results.assert_shape_equal(result.shape, expected_shape, "2D x 1D shape")
    results.assert_equal(
        result[0], Float32(torch_result[0].item()), "2D x 1D [0]"
    )
    results.assert_equal(
        result[1], Float32(torch_result[1].item()), "2D x 1D [1]"
    )

    # Manual calculations
    results.assert_equal(result[0], 14.0, "2D x 1D [0] manual: 1*1 + 2*2 + 3*3")
    results.assert_equal(result[1], 32.0, "2D x 1D [1] manual: 4*1 + 5*2 + 6*3")


fn test_3d_x_2d_batched(mut results: TestResults) raises:
    print("\nTest 5: 3D x 2D (batched)")

    # Mojo implementation
    var g = Tensor(2, 3, 4)
    var h = Tensor(4, 2)

    # Fill tensors
    for i in range(2):
        for j in range(3):
            for k in range(4):
                g[i, j, k] = Float32(i + j + k)

    for i in range(4):
        for j in range(2):
            h[i, j] = Float32(i * j)

    var result = matmul(g, h)

    # PyTorch reference
    var torch = Python.import_module("torch")
    var torch_g = torch.zeros([2, 3, 4])
    var torch_h = torch.zeros([4, 2])

    for i in range(2):
        for j in range(3):
            for k in range(4):
                torch_g[i, j, k] = Float32(i + j + k)

    for i in range(4):
        for j in range(2):
            torch_h[i, j] = Float32(i * j)

    var torch_result = torch.matmul(torch_g, torch_h)

    # Assertions
    var expected_shape = List[Int](2, 3, 2)
    results.assert_shape_equal(result.shape, expected_shape, "3D x 2D shape")
    results.assert_equal(
        result[0, 0, 0],
        Float32(torch_result[0, 0, 0].item()),
        "3D x 2D [0,0,0]",
    )
    results.assert_equal(
        result[0, 0, 1],
        Float32(torch_result[0, 0, 1].item()),
        "3D x 2D [0,0,1]",
    )
    results.assert_equal(
        result[1, 0, 0],
        Float32(torch_result[1, 0, 0].item()),
        "3D x 2D [1,0,0]",
    )
    results.assert_equal(
        result[1, 0, 1],
        Float32(torch_result[1, 0, 1].item()),
        "3D x 2D [1,0,1]",
    )
    results.assert_equal(
        result[0, 1, 1],
        Float32(torch_result[0, 1, 1].item()),
        "3D x 2D [0,1,1]",
    )
    results.assert_equal(
        result[1, 2, 1],
        Float32(torch_result[1, 2, 1].item()),
        "3D x 2D [1,2,1]",
    )

    # Manual calculations
    results.assert_equal(
        result[0, 0, 0], 0.0, "3D x 2D [0,0,0] manual: 0*0 + 1*0 + 2*0 + 3*0"
    )
    results.assert_equal(
        result[0, 0, 1], 14.0, "3D x 2D [0,0,1] manual: 0*0 + 1*1 + 2*2 + 3*3"
    )
    results.assert_equal(
        result[1, 0, 1], 20.0, "3D x 2D [1,0,1] manual: 1*0 + 2*1 + 3*2 + 4*3"
    )


fn matmul_test() raises:
    var results = TestResults()

    # Run all tests
    test_1d_x_1d_dot_product(results)
    test_2d_x_2d_matrix_multiply(results)
    test_1d_x_2d_broadcast(results)
    test_2d_x_1d_matvec(results)
    test_3d_x_2d_batched(results)

    # Print final results
    results.print_summary()
