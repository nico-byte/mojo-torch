from mojo_torch import Tensor, matmul
from test.utils.test_results import TensorTestResults
from python import Python, PythonObject


fn test_1d_x_1d_dot_product(
    mut results: TensorTestResults, tiled: Bool = False
) raises:
    print("\nTest 1: 1D x 1D (dot product)")

    # Mojo implementation
    var a = Tensor[DType.float32](3)
    var b = Tensor[DType.float32](3)
    a[0] = 1.0
    a[1] = 2.0
    a[2] = 3.0
    b[0] = 4.0
    b[1] = 5.0
    b[2] = 6.0

    var result = matmul(a, b, tiled)

    # PyTorch reference
    var torch = Python.import_module("torch")
    var torch_a = torch.tensor([1.0, 2.0, 3.0])
    var torch_b = torch.tensor([4.0, 5.0, 6.0])
    var torch_result = torch.matmul(torch_a, torch_b)
    print(torch_result)

    # Assertions - dot product results in a scalar stored in a 1D tensor with 1 element
    var expected_shape = List[Int]()
    for i in range(torch_result.shape.__len__()):
        expected_shape.append(Int(torch_result.shape[i]))
    results.assert_shape_equal(
        result.layout.shape, expected_shape, "1D x 1D shape (scalar in tensor)"
    )
    results.assert_equal(
        result[0], Float32(torch_result.item()), "1D x 1D value"
    )


fn test_2d_x_2d_matrix_multiply(
    mut results: TensorTestResults, tiled: Bool = False
) raises:
    print("\nTest 2: 2D x 2D (matrix multiplication)")

    # Mojo implementation
    var c = Tensor[DType.float32]([2, 3])
    var d = Tensor[DType.float32]([3, 2])

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

    var result = matmul(c, d, tiled)

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
    var expected_shape = List[Int]()
    for i in range(torch_result.shape.__len__()):
        expected_shape.append(Int(torch_result.shape[i]))
    results.assert_shape_equal(
        result.layout.shape, expected_shape, "2D x 2D shape"
    )
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


fn test_2d_x_2dT_matrix_multiply(
    mut results: TensorTestResults, tiled: Bool = False
) raises:
    print("\nTest 2: 2D x 2D^T (matrix multiplication)")

    # Mojo implementation
    var c = Tensor[DType.float32]([2, 3])
    var d = Tensor[DType.float32]([2, 3])

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
    d[0, 2] = 9.0
    d[1, 0] = 10.0
    d[1, 1] = 11.0
    d[1, 2] = 12.0

    var result = matmul(c, d.transpose(0, 1), tiled)

    # PyTorch reference
    var torch = Python.import_module("torch")
    var init_c = Python.list(
        Python.list(1.0, 2.0, 3.0), Python.list(4.0, 5.0, 6.0)
    )
    var init_d = Python.list(
        Python.list(7.0, 8.0, 9.0), Python.list(10.0, 11.0, 12.0)
    )
    var torch_c = torch.tensor(init_c)
    var torch_d = torch.tensor(init_d)
    var torch_result = torch.matmul(torch_c, torch_d.transpose(0, 1))

    # Assertions
    var expected_shape = List[Int]()
    for i in range(torch_result.shape.__len__()):
        expected_shape.append(Int(torch_result.shape[i]))
    results.assert_shape_equal(
        result.layout.shape, expected_shape, "2D x 2D^T shape"
    )
    results.assert_equal(
        result[0, 0], Float32(torch_result[0, 0].item()), "2D x 2D^T [0,0]"
    )
    results.assert_equal(
        result[0, 1], Float32(torch_result[0, 1].item()), "2D x 2D^T [0,1]"
    )
    results.assert_equal(
        result[1, 0], Float32(torch_result[1, 0].item()), "2D x 2D^T [1,0]"
    )
    results.assert_equal(
        result[1, 1], Float32(torch_result[1, 1].item()), "2D x 2D^T [1,1]"
    )


fn test_1d_x_2d_broadcast(
    mut results: TensorTestResults, tiled: Bool = False
) raises:
    print("\nTest 3: 1D x 2D (broadcast)")

    # Mojo implementation
    var e = Tensor[DType.float32](3)
    e[0] = 1.0
    e[1] = 2.0
    e[2] = 3.0

    var d = Tensor[DType.float32]([3, 2])
    d[0, 0] = 7.0
    d[0, 1] = 8.0
    d[1, 0] = 9.0
    d[1, 1] = 10.0
    d[2, 0] = 11.0
    d[2, 1] = 12.0

    var result = matmul(e, d, tiled)

    # PyTorch reference
    var torch = Python.import_module("torch")
    var torch_e = torch.tensor([1.0, 2.0, 3.0])
    var init_d = Python.list(
        Python.list(7.0, 8.0), Python.list(9.0, 10.0), Python.list(11.0, 12.0)
    )
    var torch_d = torch.tensor(init_d)
    var torch_result = torch.matmul(torch_e, torch_d)

    # Assertions
    var expected_shape = List[Int]()
    for i in range(torch_result.shape.__len__()):
        expected_shape.append(Int(torch_result.shape[i]))
    results.assert_shape_equal(
        result.layout.shape, expected_shape, "1D x 2D shape"
    )
    results.assert_equal(
        result[0], Float32(torch_result[0].item()), "1D x 2D [0]"
    )
    results.assert_equal(
        result[1], Float32(torch_result[1].item()), "1D x 2D [1]"
    )


fn test_2d_x_1d_matvec(
    mut results: TensorTestResults, tiled: Bool = False
) raises:
    print("\nTest 4: 2D x 1D (matrix-vector)")

    # Mojo implementation
    var c = Tensor[DType.float32]([2, 3])
    c[0, 0] = 1.0
    c[0, 1] = 2.0
    c[0, 2] = 3.0
    c[1, 0] = 4.0
    c[1, 1] = 5.0
    c[1, 2] = 6.0

    var f = Tensor[DType.float32]([3])
    f[0] = 1.0
    f[1] = 2.0
    f[2] = 3.0

    var result = matmul(c, f, tiled)

    # PyTorch reference
    var torch = Python.import_module("torch")
    var init_c = Python.list(
        Python.list(1.0, 2.0, 3.0), Python.list(4.0, 5.0, 6.0)
    )
    var torch_c = torch.tensor(init_c)
    var torch_f = torch.tensor([1.0, 2.0, 3.0])
    var torch_result = torch.matmul(torch_c, torch_f)

    # Assertions
    var expected_shape = List[Int]()
    for i in range(torch_result.shape.__len__()):
        expected_shape.append(Int(torch_result.shape[i]))
    results.assert_shape_equal(
        result.layout.shape, expected_shape, "2D x 1D shape"
    )
    results.assert_equal(
        result[0], Float32(torch_result[0].item()), "2D x 1D [0]"
    )
    results.assert_equal(
        result[1], Float32(torch_result[1].item()), "2D x 1D [1]"
    )


fn test_3d_x_2d_batched(
    mut results: TensorTestResults, tiled: Bool = False
) raises:
    print("\nTest 5: 3D x 2D (batched)")

    # Mojo implementation
    var g = Tensor[DType.float32]([2, 3, 4])
    var h = Tensor[DType.float32]([4, 2])

    # Fill tensors
    for i in range(2):
        for j in range(3):
            for k in range(4):
                g[i, j, k] = Float32(i + j + k)

    for i in range(4):
        for j in range(2):
            h[i, j] = Float32(i * j)

    var result = matmul(g, h, tiled)

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
    var expected_shape = List[Int]()
    for i in range(torch_result.shape.__len__()):
        expected_shape.append(Int(torch_result.shape[i]))
    results.assert_shape_equal(
        result.layout.shape, expected_shape, "3D x 2D shape"
    )
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


fn matmul_test() raises:
    var results = TensorTestResults()

    # Run all tests
    test_1d_x_1d_dot_product(results)
    test_2d_x_2d_matrix_multiply(results)
    test_2d_x_2dT_matrix_multiply(results)
    test_1d_x_2d_broadcast(results)
    test_2d_x_1d_matvec(results)
    test_3d_x_2d_batched(results)

    # Print final results
    results.print_summary()

    # Raise an error if any tests failed
    if results.failed > 0:
        raise Error(
            "Test suite failed: "
            + String(results.failed)
            + " out of "
            + String(results.total)
            + " tests failed"
        )
