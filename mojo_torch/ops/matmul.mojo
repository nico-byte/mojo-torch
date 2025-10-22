from mojo_torch import Tensor
from collections.list import List
from math import sqrt


fn matmul(input: Tensor, other: Tensor, tiled: Bool = False) -> Tensor:
    """
    Matrix product of two tensors following PyTorch's torch.matmul behavior.

    The behavior depends on the dimensionality of the tensors as follows:
    - If both tensors are 1-dimensional, the dot product (scalar) is returned.
    - If both arguments are 2-dimensional, the matrix-matrix product is returned.
    - If the first argument is 1-dimensional and the second argument is 2-dimensional,
      a 1 is prepended to its dimension for the purpose of the matrix multiply.
      After the matrix multiply, the prepended dimension is removed.
    - If the first argument is 2-dimensional and the second argument is 1-dimensional,
      the matrix-vector product is returned.
    - If both arguments are at least 1-dimensional and at least one argument is N-dimensional
      (where N > 2), then a batched matrix multiply is returned with broadcasting support.
    """
    var input_ndim = input.shape.__len__()
    var other_ndim = other.shape.__len__()

    # Case 1: Both tensors are 1-dimensional (dot product)
    if input_ndim == 1 and other_ndim == 1:
        return _dot_product(input, other)

    # Case 2: Both tensors are 2-dimensional (matrix-matrix product)
    elif input_ndim == 2 and other_ndim == 2:
        if tiled:
            return input.__matmul_tiled__(other)
        else:
            return input.__matmul__(other)

    # Case 3: First is 1D, second is 2D (prepend 1 to first, then remove)
    elif input_ndim == 1 and other_ndim == 2:
        return _matmul_1d_2d(input, other)

    # Case 4: First is 2D, second is 1D (matrix-vector product)
    elif input_ndim == 2 and other_ndim == 1:
        return _matmul_2d_1d(input, other)

    # Case 5: At least one tensor is N-dimensional (N > 2) - batched matrix multiply
    else:
        return _batched_matmul(input, other)


fn _dot_product(input: Tensor, other: Tensor) -> Tensor:
    """Compute dot product of two 1D tensors."""
    if input.shape[0] != other.shape[0]:
        print("Dot product requires tensors of same length")
        return Tensor.scalar(0.0)  # Return scalar tensor with 0

    var sum_val: Float32 = 0.0
    for i in range(input.shape[0]):
        sum_val += input.data.load(i) * other.data.load(i)

    return Tensor.scalar(sum_val)


fn _matmul_1d_2d(input: Tensor, other: Tensor, tiled: Bool = False) -> Tensor:
    """
    Matrix multiply where first tensor is 1D and second is 2D.
    Prepends 1 to first tensor's dimension, performs matmul, then removes prepended dimension.
    """
    var n = input.shape[0]
    var m = other.shape[0]
    var p = other.shape[1]

    if n != m:
        print(
            "MatMul not possible -> input.size: "
            + String(n)
            + " != other.rows: "
            + String(m)
        )
        return Tensor([1], 0.0)

    # Create temporary 2D tensor with shape [1, n]
    var temp_input = Tensor([1, n])
    for i in range(n):
        temp_input[0, i] = input.data.load(i)

    # Perform matrix multiplication [1, n] @ [m, p] -> [1, p]
    var temp_result: Tensor
    if tiled:
        temp_result = temp_input.__matmul_tiled__(other)
    else:
        temp_result = temp_input.__matmul__(other)

    # Remove the prepended dimension to get [p]
    var result = Tensor(p)
    for i in range(p):
        result.data.store(i, temp_result[0, i])

    return result^


fn _matmul_2d_1d(input: Tensor, other: Tensor) -> Tensor:
    """
    Matrix-vector product where first tensor is 2D and second is 1D.
    """
    var m = input.shape[0]
    var k = input.shape[1]
    var n = other.shape[0]

    if k != n:
        print(
            "MatMul not possible -> input.cols: "
            + String(k)
            + " != other.size: "
            + String(n)
        )
        return Tensor([1], 0.0)

    var result = Tensor(m)

    for i in range(m):
        var sum_val: Float32 = 0.0
        for j in range(k):
            sum_val += input[i, j] * other.data.load(j)
        result.data.store(i, sum_val)

    return result^


fn _batched_matmul(input: Tensor, other: Tensor, tiled: Bool = False) -> Tensor:
    """
    Batched matrix multiplication with broadcasting support.
    Handles N-dimensional tensors where N > 2.
    """
    var input_ndim = input.shape.__len__()
    var other_ndim = other.shape.__len__()

    # Handle dimension promotion for 1D tensors
    var input_promoted = input
    var other_promoted = other
    var remove_input_dim = False
    var remove_other_dim = False

    # If input is 1D, prepend dimension of size 1
    if input_ndim == 1:
        var new_shape = List[Int]()
        new_shape.append(1)
        new_shape.append(input.shape[0])
        input_promoted = _reshape_tensor(input, new_shape)
        remove_input_dim = True

    # If other is 1D, append dimension of size 1
    if other_ndim == 1:
        var new_shape = List[Int]()
        new_shape.append(other.shape[0])
        new_shape.append(1)
        other_promoted = _reshape_tensor(other, new_shape)
        remove_other_dim = True

    # Now both tensors are at least 2D
    var input_promoted_ndim = input_promoted.shape.__len__()
    var other_promoted_ndim = other_promoted.shape.__len__()

    # Extract matrix dimensions (last 2 dimensions)
    var input_matrix_rows = input_promoted.shape[input_promoted_ndim - 2]
    var input_matrix_cols = input_promoted.shape[input_promoted_ndim - 1]
    var other_matrix_rows = other_promoted.shape[other_promoted_ndim - 2]
    var other_matrix_cols = other_promoted.shape[other_promoted_ndim - 1]

    if input_matrix_cols != other_matrix_rows:
        print(
            "MatMul not possible -> input matrix cols: "
            + String(input_matrix_cols)
            + " != other matrix rows: "
            + String(other_matrix_rows)
        )
        return Tensor([1], 0.0)

    # Broadcast batch dimensions
    var max_batch_ndim = max(input_promoted_ndim - 2, other_promoted_ndim - 2)
    var broadcast_shape = _compute_broadcast_shape(
        input_promoted, other_promoted, max_batch_ndim
    )

    # Create result shape
    var result_shape = List[Int]()
    for i in range(broadcast_shape.__len__()):
        result_shape.append(broadcast_shape[i])
    result_shape.append(input_matrix_rows)
    result_shape.append(other_matrix_cols)

    var result = Tensor(result_shape)

    # Compute total number of batch operations
    var total_batches = 1
    for i in range(broadcast_shape.__len__()):
        total_batches *= broadcast_shape[i]

    # Perform batched matrix multiplication
    for batch_idx in range(total_batches):
        var input_batch = _extract_batch_matrix(
            input_promoted, batch_idx, broadcast_shape
        )
        var other_batch = _extract_batch_matrix(
            other_promoted, batch_idx, broadcast_shape
        )
        var batch_result: Tensor
        if tiled:
            batch_result = input_batch.__matmul_tiled__(other_batch)
        else:
            batch_result = input_batch.__matmul__(other_batch)
        _store_batch_result(result, batch_result, batch_idx, broadcast_shape)

    # Remove dimensions that were added for 1D tensors
    if remove_input_dim and remove_other_dim:
        # Both were 1D, result should be scalar-like
        var final_shape = List[Int]()
        for i in range(broadcast_shape.__len__()):
            final_shape.append(broadcast_shape[i])
        return _reshape_tensor(result, final_shape)
    elif remove_input_dim:
        # Remove second-to-last dimension
        var final_shape = List[Int]()
        for i in range(result.shape.__len__()):
            if i != result.shape.__len__() - 2:
                final_shape.append(result.shape[i])
        return _reshape_tensor(result, final_shape)
    elif remove_other_dim:
        # Remove last dimension
        var final_shape = List[Int]()
        for i in range(result.shape.__len__() - 1):
            final_shape.append(result.shape[i])
        return _reshape_tensor(result, final_shape)

    return result^


fn _compute_broadcast_shape(
    input: Tensor, other: Tensor, max_batch_ndim: Int
) -> List[Int]:
    """Compute the broadcasted shape for batch dimensions."""
    var result_shape = List[Int]()
    var input_ndim = input.shape.__len__()
    var other_ndim = other.shape.__len__()

    # Get the actual batch dimensions
    var input_batch_dims = max(0, input_ndim - 2)
    var other_batch_dims = max(0, other_ndim - 2)
    var max_batch_dims = max(input_batch_dims, other_batch_dims)

    for i in range(max_batch_dims):
        var input_dim = 1
        var other_dim = 1

        # Get dimension from input if it exists
        var input_dim_idx = input_batch_dims - 1 - i
        if input_dim_idx >= 0:
            input_dim = input.shape[input_dim_idx]

        # Get dimension from other if it exists
        var other_dim_idx = other_batch_dims - 1 - i
        if other_dim_idx >= 0:
            other_dim = other.shape[other_dim_idx]

        if input_dim == 1:
            result_shape.append(other_dim)
        elif other_dim == 1:
            result_shape.append(input_dim)
        elif input_dim == other_dim:
            result_shape.append(input_dim)
        else:
            print(
                "Cannot broadcast dimensions: "
                + String(input_dim)
                + " and "
                + String(other_dim)
            )
            return List[Int]()

    # Reverse to get correct order
    var final_shape = List[Int]()
    for i in range(result_shape.__len__() - 1, -1, -1):
        final_shape.append(result_shape[i])

    return final_shape^


fn _extract_batch_matrix(
    tensor: Tensor, batch_idx: Int, batch_shape: List[Int]
) -> Tensor:
    """Extract a 2D matrix from a batched tensor at the given batch index."""
    var ndim = tensor.shape.__len__()
    var matrix_rows = tensor.shape[ndim - 2]
    var matrix_cols = tensor.shape[ndim - 1]

    var result = Tensor(matrix_rows, matrix_cols)

    # If tensor is only 2D, just copy it directly (no batch dimensions)
    if ndim == 2:
        for i in range(matrix_rows):
            for j in range(matrix_cols):
                result[i, j] = tensor[i, j]
        return result^

    # Convert batch_idx to multi-dimensional batch indices
    var batch_indices = List[Int]()
    var idx = batch_idx
    var tensor_batch_dims = ndim - 2

    for i in range(tensor_batch_dims - 1, -1, -1):
        var tensor_batch_size = tensor.shape[i]
        batch_indices.append(idx % tensor_batch_size)
        idx /= tensor_batch_size

    # Reverse to get correct order
    var final_batch_indices = List[Int]()
    for i in range(batch_indices.__len__() - 1, -1, -1):
        final_batch_indices.append(batch_indices[i])

    # Copy matrix data
    for i in range(matrix_rows):
        for j in range(matrix_cols):
            var tensor_indices = List[Int]()
            for k in range(final_batch_indices.__len__()):
                tensor_indices.append(final_batch_indices[k])
            tensor_indices.append(i)
            tensor_indices.append(j)

            var flat_idx = _compute_flat_index(tensor, tensor_indices)
            result[i, j] = tensor.data.load(flat_idx)

    return result^


fn _store_batch_result(
    result: Tensor, batch_result: Tensor, batch_idx: Int, batch_shape: List[Int]
):
    """Store a 2D matrix result into the appropriate batch location of the result tensor.
    """
    var ndim = result.shape.__len__()
    var matrix_rows = result.shape[ndim - 2]
    var matrix_cols = result.shape[ndim - 1]

    # Convert batch_idx to multi-dimensional batch indices
    var batch_indices = List[Int]()
    var idx = batch_idx
    for i in range(batch_shape.__len__() - 1, -1, -1):
        batch_indices.append(idx % batch_shape[i])
        idx /= batch_shape[i]

    # Reverse to get correct order
    var final_batch_indices = List[Int]()
    for i in range(batch_indices.__len__() - 1, -1, -1):
        final_batch_indices.append(batch_indices[i])

    # Store matrix data
    for i in range(matrix_rows):
        for j in range(matrix_cols):
            var result_indices = List[Int]()
            for k in range(final_batch_indices.__len__()):
                result_indices.append(final_batch_indices[k])
            result_indices.append(i)
            result_indices.append(j)

            var flat_idx = _compute_flat_index(result, result_indices)
            result.data.store(flat_idx, batch_result[i, j])


fn _compute_flat_index(tensor: Tensor, indices: List[Int]) -> Int:
    """Compute flat index from multi-dimensional indices."""
    var flat_idx = 0
    for i in range(indices.__len__()):
        flat_idx += indices[i] * tensor.strides[i]
    return flat_idx


fn _reshape_tensor(tensor: Tensor, new_shape: List[Int]) -> Tensor:
    """Reshape a tensor to a new shape (must have same total size)."""
    var new_size = 1
    for dim in new_shape:
        new_size *= dim

    if new_size != tensor.size:
        print("Reshape failed: total size mismatch")
        return tensor

    var result = Tensor(new_shape)
    for i in range(tensor.size):
        result.data.store(i, tensor.data.load(i))

    return result^
