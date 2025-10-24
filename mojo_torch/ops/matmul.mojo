###
# Matrix Multiplication from https://github.com/YichengDWu/matmul.mojo
###

from mojo_torch import LayoutTensor, Tensor
from collections.list import List
from math import sqrt
from memory import stack_allocation
from memory.memory import _malloc
from sys import simd_width_of, num_performance_cores, size_of, CompilationTarget
from algorithm.functional import vectorize, parallelize
from collections.list import List
from utils import IndexList
from python import Python, PythonObject
import random

alias type = Scalar[DType.float32]
alias nelts = simd_width_of[Scalar[DType.float32]]()


fn matmul[
    Type: DType
](input: Tensor[Type], other: Tensor[Type], tiled: Bool = False) -> Tensor[
    Type
]:
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
    var input_ndim = input.layout.shape.__len__()
    var other_ndim = other.layout.shape.__len__()

    # Case 1: Both tensors are 1-dimensional (dot product)
    if input_ndim == 1 and other_ndim == 1:
        return _dot_product(input, other)

    # Case 2: Both tensors are 2-dimensional (matrix-matrix product)
    elif input_ndim == 2 and other_ndim == 2:
        var output = Tensor[Type](
            [input.layout.shape[0], other.layout.shape[1]]
        )
        _matmul[1024, 1024, 1024](output, input, other)
        return output^

    # Case 3: First is 1D, second is 2D (prepend 1 to first, then remove)
    elif input_ndim == 1 and other_ndim == 2:
        return _matmul_1d_2d(input, other)

    # Case 4: First is 2D, second is 1D (matrix-vector product)
    elif input_ndim == 2 and other_ndim == 1:
        return _matmul_2d_1d(input, other)

    # Case 5: At least one tensor is N-dimensional (N > 2) - batched matrix multiply
    else:
        return _batched_matmul(input, other)


fn _dot_product[
    Type: DType
](input: Tensor[Type], other: Tensor[Type]) -> Tensor[Type]:
    """Compute dot product of two 1D tensors."""
    if input.layout.shape[0] != other.layout.shape[0]:
        print("Dot product requires tensors of same length")
        return Tensor[Type].scalar(0.0)  # Return scalar tensor with 0

    var sum_val: Float32 = 0.0
    for i in range(input.layout.shape[0]):
        sum_val += Float32(input.data.load(i)) * Float32(other.data.load(i))

    return Tensor[Type].scalar(sum_val)


fn _matmul_1d_2d[
    Type: DType
](input: Tensor[Type], other: Tensor[Type], tiled: Bool = False) -> Tensor[
    Type
]:
    var n = input.layout.shape[0]
    var m = other.layout.shape[0]
    var p = other.layout.shape[1]

    if n != m:
        print(
            "MatMul not possible -> input.size: "
            + String(n)
            + " != other.rows: "
            + String(m)
        )
        return Tensor[Type]([1], 0.0)

    var temp_input = Tensor[Type]([1, n])
    for i in range(n):
        temp_input[0, i] = input.data.load(i)

    var temp_result: Tensor[Type] = Tensor[Type]([1, p])
    _matmul[1024, 1024, 1024](temp_result, temp_input, other)

    var result = Tensor[Type]([p])
    for i in range(p):
        result.data.store(i, temp_result[0, i])

    return result^


fn _matmul_2d_1d[
    Type: DType
](input: Tensor[Type], other: Tensor[Type]) -> Tensor[Type]:
    """
    Matrix-vector product where first tensor is 2D and second is 1D.
    """
    var m = input.layout.shape[0]
    var k = input.layout.shape[1]
    var n = other.layout.shape[0]

    if k != n:
        print(
            "MatMul not possible -> input.cols: "
            + String(k)
            + " != other.size: "
            + String(n)
        )
        return Tensor[Type]([1])

    var result = Tensor[Type]([m])

    for i in range(m):
        var sum_val: Float32 = 0.0
        for j in range(k):
            sum_val += Float32(input[i, j]) * Float32(other.data.load(j))
        result.data.store(i, Scalar[Type](sum_val))

    return result.copy()


fn _batched_matmul[
    Type: DType
](input: Tensor[Type], other: Tensor[Type], tiled: Bool = False) -> Tensor[
    Type
]:
    """
    Optimized batched matrix multiplication with parallelization.
    """
    var input_ndim = input.layout.shape.__len__()
    var other_ndim = other.layout.shape.__len__()

    # Handle dimension promotion (same as before)
    var input_promoted = input.copy()
    var other_promoted = other.copy()
    var remove_input_dim = False
    var remove_other_dim = False

    if input_ndim == 1:
        var new_shape = List[Int]()
        new_shape.append(1)
        new_shape.append(input.layout.shape[0])
        input_promoted = _reshape_tensor(input, new_shape)
        remove_input_dim = True

    if other_ndim == 1:
        var new_shape = List[Int]()
        new_shape.append(other.layout.shape[0])
        new_shape.append(1)
        other_promoted = _reshape_tensor(other, new_shape)
        remove_other_dim = True

    var input_promoted_ndim = input_promoted.layout.shape.__len__()
    var other_promoted_ndim = other_promoted.layout.shape.__len__()

    var input_matrix_rows = input_promoted.layout.shape[input_promoted_ndim - 2]
    var input_matrix_cols = input_promoted.layout.shape[input_promoted_ndim - 1]
    var other_matrix_rows = other_promoted.layout.shape[other_promoted_ndim - 2]
    var other_matrix_cols = other_promoted.layout.shape[other_promoted_ndim - 1]

    if input_matrix_cols != other_matrix_rows:
        print("MatMul dimension mismatch")
        return Tensor[Type]([1], 0.0)

    var broadcast_shape = _compute_broadcast_shape(
        input_promoted,
        other_promoted,
        max(input_promoted_ndim - 2, other_promoted_ndim - 2),
    )

    var result_shape = List[Int]()
    for i in range(broadcast_shape.__len__()):
        result_shape.append(broadcast_shape[i])
    result_shape.append(input_matrix_rows)
    result_shape.append(other_matrix_cols)

    var result = Tensor[Type](result_shape)

    var total_batches = 1
    for i in range(broadcast_shape.__len__()):
        total_batches *= broadcast_shape[i]

    # **OPTIMIZED: Parallel batch processing**
    @parameter
    fn process_batch(batch_idx: Int):
        # Direct matrix computation without temporary tensors
        _compute_batch_direct(
            input_promoted,
            other_promoted,
            result,
            batch_idx,
            broadcast_shape,
            input_matrix_rows,
            input_matrix_cols,
            other_matrix_cols,
            tiled,
        )

    parallelize[process_batch](total_batches, total_batches)

    # Handle dimension removal (same as before)
    if remove_input_dim and remove_other_dim:
        var final_shape = List[Int]()
        for i in range(broadcast_shape.__len__()):
            final_shape.append(broadcast_shape[i])
        return _reshape_tensor(result, final_shape)
    elif remove_input_dim:
        var final_shape = List[Int]()
        for i in range(result.layout.shape.__len__()):
            if i != result.layout.shape.__len__() - 2:
                final_shape.append(result.layout.shape[i])
        return _reshape_tensor(result, final_shape)
    elif remove_other_dim:
        var final_shape = List[Int]()
        for i in range(result.layout.shape.__len__() - 1):
            final_shape.append(result.layout.shape[i])
        return _reshape_tensor(result, final_shape)

    return result.copy()


fn _compute_broadcast_shape(
    input: Tensor, other: Tensor, max_batch_ndim: Int
) -> List[Int]:
    """Compute the broadcasted shape for batch dimensions."""
    var result_shape = List[Int]()
    var input_ndim = input.layout.shape.__len__()
    var other_ndim = other.layout.shape.__len__()

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
            input_dim = input.layout.shape[input_dim_idx]

        # Get dimension from other if it exists
        var other_dim_idx = other_batch_dims - 1 - i
        if other_dim_idx >= 0:
            other_dim = other.layout.shape[other_dim_idx]

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


fn _compute_batch_direct[
    Type: DType
](
    input: Tensor[Type],
    other: Tensor[Type],
    result: Tensor[Type],
    batch_idx: Int,
    batch_shape: List[Int],
    m: Int,
    k: Int,
    n: Int,
    tiled: Bool,
):
    """Compute batch matrix multiplication using optimized tensor matmul."""

    var input_batch_offset = _compute_batch_offset(
        input, batch_idx, batch_shape
    )
    var other_batch_offset = _compute_batch_offset(
        other, batch_idx, batch_shape
    )
    var result_batch_offset = _compute_result_batch_offset(
        result, batch_idx, batch_shape
    )

    # Create views into batch data without copying
    var input_view = Tensor[Type](input.data + input_batch_offset, (m, k))
    var other_view = Tensor[Type](other.data + other_batch_offset, (k, n))
    var result_view = Tensor[Type](result.data + result_batch_offset, (m, n))

    # Compute matmul for this batch
    _matmul[1024, 1024, 1024](result_view, input_view, other_view)


fn _compute_batch_offset(
    tensor: Tensor, batch_idx: Int, batch_shape: List[Int]
) -> Int:
    """Compute the flat offset for a batch in the tensor."""
    var ndim = tensor.layout.shape.__len__()
    if ndim <= 2:
        return 0  # No batch dimensions

    var batch_dims = ndim - 2
    var matrix_size = (
        tensor.layout.shape[ndim - 2] * tensor.layout.shape[ndim - 1]
    )

    # Convert linear batch_idx to multidimensional coordinates
    var offset = 0
    var idx = batch_idx
    var stride = matrix_size

    for i in range(batch_dims - 1, -1, -1):
        var tensor_dim_size = tensor.layout.shape[i]
        var coord = idx % tensor_dim_size
        offset += coord * stride
        stride *= tensor.layout.shape[i]
        idx /= tensor_dim_size

    return offset


fn _compute_result_batch_offset(
    result: Tensor, batch_idx: Int, batch_shape: List[Int]
) -> Int:
    """Compute the flat offset for a batch in the result tensor."""
    var ndim = result.layout.shape.__len__()
    var batch_dims = ndim - 2
    var matrix_size = (
        result.layout.shape[ndim - 2] * result.layout.shape[ndim - 1]
    )

    var offset = 0
    var idx = batch_idx
    var stride = matrix_size

    for i in range(batch_dims - 1, -1, -1):
        var coord = idx % batch_shape[batch_dims - 1 - i]
        offset += coord * stride
        stride *= batch_shape[batch_dims - 1 - i]
        idx /= batch_shape[batch_dims - 1 - i]

    return offset


fn _compute_flat_index[
    Type: DType
](tensor: Tensor[Type], indices: List[Int]) -> Int:
    """Compute flat index from multi-dimensional indices."""
    var flat_idx = 0
    for i in range(indices.__len__()):
        flat_idx += indices[i] * tensor.layout.strides[i]
    return flat_idx


fn _reshape_tensor[
    Type: DType
](tensor: Tensor[Type], new_shape: List[Int]) -> Tensor[Type]:
    """Reshape a tensor to a new shape (must have same total size)."""
    var new_size = 1
    for dim in new_shape:
        new_size *= dim

    if new_size != tensor.size:
        print("Reshape failed: total size mismatch")
        return tensor.copy()

    var result = Tensor[Type](new_shape)
    for i in range(tensor.size):
        result.data.store(i, tensor.data.load(i))

    return result.copy()


#################################


@always_inline
fn roundup(a: Int, b: Int) -> Int:
    return ((a + b - 1) // b) * b


@always_inline
fn rounddown(a: Int, b: Int) -> Int:
    return (a // b) * b


# math.sqrt doesn't work at compile time
fn intsqrt[n: Int]() -> Int:
    @parameter
    if n == 0:
        return 0
    var x = n
    var y = (x + 1) // 2
    while y < x:
        x = y
        y = (n // x + x) // 2
    return x


@always_inline
fn pack_A[
    Type: DType, //, mc: Int, mr: Int
](Ac_buffer: UnsafePointer[Scalar[Type]], Ac: Tensor[Type]) -> Tensor[Type]:
    @parameter
    fn pack_panel(idx: Int):
        var i = idx * mr
        # for i in range(0, Ac.shape[0](), mr):
        var dst_ptr = Ac_buffer + i * Ac.layout.shape[1]
        var src_ptr = Ac.data + i * Ac.layout.strides[0]
        for _ in range(Ac.layout.shape[1]):

            @parameter
            fn pack_col[width: Int](l: Int):
                (dst_ptr + l).store(
                    (src_ptr + l * Ac.layout.strides[0]).strided_load[
                        width=width
                    ](Ac.layout.strides[0]),
                )

            vectorize[pack_col, simd_width_of[Type]()](
                min(Ac.layout.shape[0] - i, mr)
            )

            for l in range(min(Ac.layout.shape[0] - i, mr), mr):
                dst_ptr[l] = Scalar[Type](0)

            dst_ptr = dst_ptr + mr
            src_ptr = src_ptr + 1

    parallelize[pack_panel](
        (Ac.layout.shape[0] + mr - 1) // mr, num_performance_cores()
    )

    var Ac_layout = LayoutTensor(
        (roundup(Ac.layout.shape[0], mr), Ac.layout.shape[1]), (1, mr)
    )  # NOTE: The stride is a lie and only used for slicing
    return Tensor[Type](Ac_buffer, Ac_layout.copy())


@always_inline
fn pack_B[
    Type: DType, //, kc: Int, nr: Int
](Bc_buffer: UnsafePointer[Scalar[Type]], Bc: Tensor[Type]) -> Tensor[Type]:
    var dst_ptr = Bc_buffer
    for i in range(0, Bc.layout.shape[1], nr):
        var src_ptr = Bc.data + i
        for _ in range(Bc.layout.shape[0]):

            @parameter
            fn pack_row[width: Int](l: Int):
                (dst_ptr + l).store[
                    alignment = size_of[Type]() * simd_width_of[Type]()
                ](
                    (src_ptr + l).load[width=width](),
                )

            vectorize[
                pack_row,
                simd_width_of[Type](),
                unroll_factor = nr // simd_width_of[Type](),
            ](min(Bc.layout.shape[1] - i, nr))

            for l in range(min(Bc.layout.shape[1] - i, nr), nr):
                dst_ptr[l] = Scalar[Type](0)

            dst_ptr = dst_ptr + nr
            src_ptr = src_ptr + Bc.layout.strides[0]

    var Bc_layout = LayoutTensor(
        (Bc.layout.shape[0], roundup(Bc.layout.shape[1], nr)), (nr, 1)
    )  # NOTE: The stride is a lie and only used for slicing
    return Tensor[Type](Bc_buffer, Bc_layout.copy())


@always_inline
fn matmul_impl[
    Type: DType, //,
    mc: Int,
    nc: Int,
    kc: Int,
    mr: Int,
    nr: Int,
](mut C: Tensor[Type], A: Tensor[Type], B: Tensor[Type]):
    var Ac_buffer = _malloc[Scalar[Type]](
        mc * kc * size_of[Type](), alignment=64
    )

    var M = C.layout.shape[0]
    var N = C.layout.shape[1]
    var K = A.layout.shape[1]

    for i in range(0, A.layout.shape[0], mc):
        var Cb = C.slice(i, 0, min(M - i, mc), N)
        for p in range(0, A.layout.shape[1], kc):
            var Ac = pack_A[mc, mr](
                Ac_buffer, A.slice(i, p, min(M - i, mc), min(K - p, kc))
            )

            var Bb = B.slice(p, 0, min(K - p, kc), N)
            loop_n[nc, kc, mr, nr](Cb, Ac, Bb)

    Ac_buffer.free()


@always_inline
fn loop_n[
    Type: DType, //,
    nc: Int,
    kc: Int,
    mr: Int,
    nr: Int,
](mut C: Tensor[Type], A: Tensor[Type], B: Tensor[Type]):
    var max_threads = num_performance_cores()
    var nc_per_thread = nc if nc * max_threads <= B.layout.shape[
        1
    ] else rounddown(B.layout.shape[1] // max_threads, nr)
    var balanced_part = rounddown(B.layout.shape[1], nc_per_thread)

    var remainder = B.layout.shape[1] - balanced_part
    var remainder_per_thread = rounddown(remainder // max_threads, nr)
    remainder_per_thread = max(remainder_per_thread, nr)

    var items_remainder = (
        remainder + remainder_per_thread - 1
    ) // remainder_per_thread

    @parameter
    fn parallelize_balanced_part(idx: Int):
        var Bc_buffer = UnsafePointer[Scalar[Type]](
            _malloc[Scalar[Type]](
                kc * nc_per_thread * size_of[Type](), alignment=64
            )
        )

        var j = idx * nc_per_thread
        var Bc = pack_B[kc, nr](
            Bc_buffer,
            B.slice(
                0,
                j,
                B.layout.shape[0],
                min(B.layout.shape[1] - j, nc_per_thread),
            ),
        )
        var Cc = C.slice(
            0, j, C.layout.shape[0], min(C.layout.shape[1] - j, nc_per_thread)
        )
        macro_kernel[mr, nr](Cc, A, Bc)
        Bc_buffer.free()

    parallelize[parallelize_balanced_part](
        balanced_part // nc_per_thread, balanced_part // nc_per_thread
    )

    @parameter
    fn parallelize_remainder(idx: Int):
        var Bc_buffer = UnsafePointer[Scalar[Type]](
            _malloc[Scalar[Type]](
                kc * remainder_per_thread * size_of[Type](), alignment=64
            )
        )
        var j = balanced_part + idx * remainder_per_thread
        var Bc = pack_B[kc, nr](
            Bc_buffer,
            B.slice(
                0,
                j,
                B.layout.shape[0],
                min(B.layout.shape[1] - j, remainder_per_thread),
            ),
        )
        var Cc = C.slice(
            0,
            j,
            C.layout.shape[0],
            min(C.layout.shape[1] - j, remainder_per_thread),
        )
        macro_kernel[mr, nr](Cc, A, Bc)
        Bc_buffer.free()

    parallelize[parallelize_remainder](items_remainder, items_remainder)

    _ = balanced_part
    _ = remainder_per_thread
    _ = nc_per_thread


@always_inline
fn macro_kernel[
    Type: DType, //,
    mr: Int,
    nr: Int,
](mut Cc: Tensor[Type], Ac: Tensor[Type], Bc: Tensor[Type]):
    @parameter
    fn parallelize_ir(idx: Int):
        var ir = idx * mr
        var Ar = Tensor[Type](
            Ac.data + ir * Ac.layout.shape[1], (mr, Ac.layout.shape[1])
        )
        for jr in range(0, Bc.layout.shape[1], nr):
            var Cr = Cc.slice(
                ir,
                jr,
                min(Cc.layout.shape[0] - ir, mr),
                min(Cc.layout.shape[1] - jr, nr),
            )
            var Br = Tensor[Type](
                Bc.data + jr * Bc.layout.shape[0],
                (Bc.layout.shape[0], nr),
            )
            if Cr.layout.shape[0] == mr and Cr.layout.shape[1] == nr:
                micro_kernel[mr, nr, False](Cr, Ar, Br)
            else:
                micro_kernel[mr, nr, True](Cr, Ar, Br)

    parallelize[parallelize_ir](
        (Ac.layout.shape[0] + mr - 1) // mr, num_performance_cores()
    )


@always_inline
fn micro_kernel[
    Type: DType, //, mr: Int, nr: Int, padding: Bool
](mut Cr: Tensor[Type], Ar: Tensor[Type], Br: Tensor[Type]):
    alias simd_width = simd_width_of[Type]()
    constrained[nr % simd_width == 0, "nr must be multiple of simd_width"]()

    var Ar_ptr = Ar.data
    var Br_ptr = Br.data
    var Cr_ptr = Cr.data

    var ar: SIMD[Type, simd_width]
    var br = InlineArray[SIMD[Type, simd_width], nr // simd_width](
        SIMD[Type, simd_width](0)
    )
    var cr_ptr = stack_allocation[mr * nr, Scalar[Type], alignment=64]()

    @parameter
    if padding:

        @parameter
        for i in range(mr):
            if i < Cr.layout.shape[0]:

                @parameter
                fn load_col[width: Int](j: Int):
                    (cr_ptr + (i * nr + j)).store(
                        (Cr_ptr + (i * Cr.layout.strides[0] + j)).load[
                            width=width
                        ](),
                    )

                vectorize[load_col, simd_width](Cr.layout.shape[1])
    else:

        @parameter
        for i in range(mr):

            @parameter
            for j in range(0, nr, simd_width):
                (cr_ptr + i * nr + j).store(
                    (Cr_ptr + (i * Cr.layout.strides[0] + j)).load[
                        width=simd_width
                    ](),
                )

    for _ in range(Ar.layout.shape[1]):

        @parameter
        for j in range(0, nr, simd_width):
            br[j // simd_width] = (Br_ptr + j).load[
                width=simd_width,
                alignment = size_of[Type]() * simd_width_of[Type](),
            ]()

        @parameter
        for i in range(mr):

            @parameter
            for j in range(0, nr, simd_width):
                ar = SIMD[Type, size=simd_width](Ar_ptr[])
                cr_ptr.store(
                    ar.fma(
                        br[j // simd_width],
                        cr_ptr.load[width=simd_width](),
                    ),
                )
                cr_ptr += simd_width
            Ar_ptr += 1

        Br_ptr += nr
        cr_ptr += -mr * nr

    @parameter
    if padding:

        @parameter
        for i in range(mr):
            if i < Cr.layout.shape[0]:

                @parameter
                fn store_row[width: Int](j: Int):
                    (Cr_ptr + (i * Cr.layout.strides[0] + j)).store(
                        (cr_ptr + (i * nr + j)).load[width=width](),
                    )

                vectorize[store_row, simd_width](Cr.layout.shape[1])
    else:

        @parameter
        for i in range(mr):

            @parameter
            for j in range(0, nr, simd_width):
                (Cr_ptr + (i * Cr.layout.strides[0] + j)).store(
                    (cr_ptr + (i * nr + j)).load[width=simd_width](),
                )


fn matmul_params[Type: DType]() -> IndexList[5]:
    alias mc = 8192 // size_of[Type]()  # fix this for simplicity
    alias N = simd_width_of[Type]()
    alias L1_ASSOCIATIVITY = 12
    alias L1_CACHE_SIZE = 192 * 1024
    alias L2_ASSOCIATIVITY = 16
    alias L2_CACHE_SIZE = 12 * 1024 * 1024

    alias Vectors = 32 if CompilationTarget.has_avx512f() else 16

    @parameter
    fn compute_kc[mr: Int, nr: Int]() -> Int:
        alias CBr = Int((L1_ASSOCIATIVITY - 1) / (1 + mr / nr))
        return (CBr * L1_CACHE_SIZE) // (
            nr * size_of[Type]() * L1_ASSOCIATIVITY
        )

    @parameter
    fn compute_params[C: Int]() -> IndexList[5]:
        alias p = C // (intsqrt[C]() + 1)
        alias mr = C // p - 1
        alias nr = p * N
        alias CBr = Int((L1_ASSOCIATIVITY - 1) / (1 + mr / nr))
        alias kc = compute_kc[mr, nr]()
        alias nc = (L2_ASSOCIATIVITY - 1) * L2_CACHE_SIZE // (
            kc * size_of[Type]() * L2_ASSOCIATIVITY
        ) - mr
        return IndexList[5](mc, nc, kc, mr, nr)

    @parameter
    if Type.is_floating_point():
        alias TempVectors = 1
        return compute_params[Vectors - TempVectors]()
    else:

        @parameter
        if Type is DType.int64:

            @parameter
            if CompilationTarget.has_avx512f():
                alias TempVectors = 2
                return compute_params[Vectors - TempVectors]()
            else:
                alias TempVectors = 3
                return compute_params[Vectors - TempVectors]()
        else:
            alias TempVectors = 2
            return compute_params[Vectors - TempVectors]()


fn _matmul[
    Type: DType, //,
    m: Int,
    n: Int,
    k: Int,
](mut C: Tensor[Type], A: Tensor[Type], B: Tensor[Type]):
    alias params = matmul_params[Type]()
    alias mc = params[0]
    alias nc = params[1]
    alias kc = params[2]
    alias mr = params[3]
    alias nr = params[4]
    alias resized_mc = roundup(min(mc, m), mr)
    alias resized_nc = roundup(min(nc, n), nr)
    matmul_impl[resized_mc, resized_nc, kc, mr, nr](C, A, B)
