from memory import memcpy
from memory.unsafe_pointer import UnsafePointer
from sys import simd_width_of
from memory import memset_zero, stack_allocation
from random import randn, rand, seed
from algorithm.functional import vectorize, parallelize
from collections.list import List
from python import Python, PythonObject

alias type = DType.float32
alias nelts = simd_width_of[Scalar[type]]()


struct Tensor(Copyable, ImplicitlyCopyable, Movable):
    """N-dimensional Tensor that generalizes Matrix to any number of dimensions.
    """

    var data: UnsafePointer[Scalar[type]]
    var shape: List[Int]
    var strides: List[Int]
    var size: Int

    # Initialize with shape
    fn __init__(out self, *dims: Int):
        self.shape = List[Int]()
        self.strides = List[Int]()
        self.size = 1
        for dim in dims:
            self.shape.append(dim)
            self.size *= dim
        # For scalar (no dimensions), size should be 1
        if dims.__len__() == 0:
            self.size = 1
        self.data = UnsafePointer[Scalar[type]].alloc(self.size)
        memset_zero(self.data, self.size)
        self.strides = self._compute_strides()

    # Initialize with shape and default value
    fn __init__(out self, var default_value: Float32, *dims: Int):
        self.shape = List[Int]()
        self.strides = List[Int]()
        self.size = 1
        for dim in dims:
            self.shape.append(dim)
            self.size *= dim
        self.data = UnsafePointer[Scalar[type]].alloc(self.size)
        for i in range(self.size):
            self.data.store(i, default_value)
        self.strides = self._compute_strides()

    # Initialize with shape from List
    fn __init__(out self, shape: List[Int]):
        self.shape = shape.copy()
        self.strides = List[Int]()
        self.size = 1
        for dim in self.shape:
            self.size *= dim
        self.data = UnsafePointer[Scalar[type]].alloc(self.size)
        memset_zero(self.data, self.size)
        self.strides = self._compute_strides()

    # Initialize with shape from List and default value
    fn __init__(out self, shape: List[Int], default_value: Float32):
        self.shape = shape.copy()
        self.strides = List[Int]()
        self.size = 1
        for dim in self.shape:
            self.size *= dim
        self.data = UnsafePointer[Scalar[type]].alloc(self.size)
        for i in range(self.size):
            self.data.store(i, default_value)
        self.strides = self._compute_strides()

    @staticmethod
    fn from_numpy(numpy_array: PythonObject) -> Self:
        """Create a Tensor from a NumPy ndarray."""
        try:
            var np = Python.import_module("numpy")

            # Get shape from numpy array
            var np_shape = numpy_array.shape
            var ndim = np_shape.__len__()

            # Create shape vector
            var shape_list = List[Int]()
            var total_size = 1
            for i in range(ndim):
                var dim = Int(np_shape[i])
                shape_list.append(dim)
                total_size *= dim

            # Flatten numpy array to get values
            var flattened = numpy_array.flatten()

            # Create tensor and populate data
            var tensor = Self(total_size)
            tensor.shape = shape_list^

            # Copy values from numpy array
            for i in range(total_size):
                tensor.data.store(i, Float32(flattened[i]))

            tensor.strides = tensor._compute_strides()
            return tensor^
        except:
            print("Failed to create Tensor from numpy array")
            return Self()

    @staticmethod
    fn scalar(value: Float32) -> Self:
        """Create a scalar tensor with empty shape."""
        var tensor = Self()  # No dimensions = scalar
        tensor.data.store(0, value)
        return tensor^

    fn _compute_strides(self) -> List[Int]:
        """Compute strides based on shape for row-major order."""
        # For scalar tensors (empty shape), return empty strides
        if self.shape.__len__() == 0:
            return List[Int]()

        var strides = List[Int]()
        var stride = 1
        for i in range(self.shape.__len__() - 1, -1, -1):
            strides.append(stride)
            stride *= self.shape[i]

        # Reverse strides to match dimension order
        var result = List[Int]()
        for i in range(strides.__len__() - 1, -1, -1):
            result.append(strides[i])
        return result^

    fn _flat_index(self, indices: VariadicList[Int]) -> Int:
        """Convert multi-dimensional indices to flat index."""
        # For scalar tensors, always return index 0
        if self.shape.__len__() == 0:
            return 0
        var idx = 0
        for i in range(indices.__len__()):
            idx += indices[i] * self.strides[i]
        return idx

    fn __getitem__(self, *indices: Int) -> Float32:
        return self.data.load(self._flat_index(indices))

    fn __setitem__(self, *indices: Int, val: Float32):
        self.data.store(self._flat_index(indices), val)

    fn __setitem__(self, rhs: Tensor):
        for i in range(self.size):
            self.data.store(i, rhs.data.load(i))

    fn __len__(self) -> Int:
        return self.size

    fn __copyinit__(out self, other: Self):
        self.shape = other.shape.copy()
        self.strides = other.strides.copy()
        self.size = other.size
        self.data = UnsafePointer[Scalar[type]].alloc(other.size)
        memcpy(self.data, other.data, other.size)

    # fn reshape(self, *new_dims: Int) -> Tensor:
    #    """Reshape tensor to new dimensions."""
    #    var new_size = 1
    #    for dim in new_dims:
    #        new_size *= dim
    #    if new_size != self.size:
    #        print("Reshape failed: total size mismatch")
    #        return Self()
    #    var new_tensor = Self(*new_dims)
    #    memcpy(new_tensor.data, self.data, self.size)
    #    return new_tensor

    fn __lt__(self, rhs: Tensor) -> Bool:
        for i in range(self.size):
            if self.data.load(i) < rhs.data.load(i):
                return True
        return False

    fn __gt__(self, rhs: Tensor) -> Bool:
        for i in range(self.size):
            if self.data.load(i) > rhs.data.load(i):
                return True
        return False

    fn __eq__(self, rhs: Tensor) -> Bool:
        for i in range(self.size):
            var self_val: Float32 = self.data.load(i)
            var rhs_val: Float32 = rhs.data.load(i)
            if self_val < rhs_val or self_val > rhs_val:
                return False
        return True

    fn __ne__(self, rhs: Tensor) -> Bool:
        return not self == rhs

    fn __ge__(self, rhs: Tensor) -> Bool:
        return self > rhs or self == rhs

    fn __le__(self, rhs: Tensor) -> Bool:
        return self < rhs or self == rhs

    fn __add__(self, rhs: Tensor) -> Tensor:
        var new_tensor: Tensor = Tensor(self.shape)
        for i in range(self.size):
            new_tensor.data.store(i, self.data.load(i) + rhs.data.load(i))
        return new_tensor^

    fn __pow__(self, rhs: Tensor) -> Tensor:
        var new_tensor: Tensor = Tensor(self.shape)
        for i in range(self.size):
            new_tensor.data.store(i, self.data.load(i) ** rhs.data.load(i))
        return new_tensor^

    fn __sub__(self, rhs: Tensor) -> Tensor:
        var new_tensor: Tensor = Tensor(self.shape)
        for i in range(self.size):
            new_tensor.data.store(i, self.data.load(i) - rhs.data.load(i))
        return new_tensor^

    fn __mul__(self, rhs: Tensor) -> Tensor:
        var new_tensor: Tensor = Tensor(self.shape)
        for i in range(self.size):
            new_tensor.data.store(i, self.data.load(i) * rhs.data.load(i))
        return new_tensor^

    fn __truediv__(self, rhs: Tensor) -> Tensor:
        var new_tensor: Tensor = Tensor(self.shape)
        for i in range(self.size):
            new_tensor.data.store(i, self.data.load(i) / rhs.data.load(i))
        return new_tensor^

    fn __add__(self, rhs: Float32) -> Tensor:
        var new_tensor: Tensor = Tensor(self.shape)
        for i in range(self.size):
            new_tensor.data.store(i, self.data.load(i) + rhs)
        return new_tensor^

    fn __pow__(self, rhs: Float32) -> Tensor:
        var new_tensor: Tensor = Tensor(self.shape)
        for i in range(self.size):
            new_tensor.data.store(i, self.data.load(i) ** rhs)
        return new_tensor^

    fn __sub__(self, rhs: Float32) -> Tensor:
        var new_tensor: Tensor = Tensor(self.shape)
        for i in range(self.size):
            new_tensor.data.store(i, self.data.load(i) - rhs)
        return new_tensor^

    fn __mul__(self, rhs: Float32) -> Tensor:
        var new_tensor: Tensor = Tensor(self.shape)
        for i in range(self.size):
            new_tensor.data.store(i, self.data.load(i) * rhs)
        return new_tensor^

    fn __truediv__(self, rhs: Float32) -> Tensor:
        var new_tensor: Tensor = Tensor(self.shape)
        for i in range(self.size):
            new_tensor.data.store(i, self.data.load(i) / rhs)
        return new_tensor^

    fn __matmul__(self, rhs: Tensor) -> Tensor:
        """
        Optimized matrix multiplication supporting 2D tensors.
        2D: [m, k] @ [k, n] -> [m, n].
        """
        if self.shape.__len__() == 2 and rhs.shape.__len__() == 2:
            # 2D matrix multiplication
            var m = self.shape[0]
            var k = self.shape[1]
            var n = rhs.shape[0]
            var p = rhs.shape[1]

            if k != n:
                print(
                    "MatMul not possible -> self.cols: "
                    + String(k)
                    + " != rhs.rows: "
                    + String(n)
                )
                return Tensor(0)

            var C: Tensor = Tensor(m, p)

            @parameter
            fn calc_row(row: Int):
                for col_block in range(0, p, nelts * 4):
                    for k_val in range(k):

                        @parameter
                        fn dot[nelts_inner: Int](col: Int):
                            var c_val = C.data.load[width=nelts_inner](
                                row * p + col + col_block
                            )
                            var a_val = self.data.load(row * k + k_val)
                            var b_val = rhs.data.load[width=nelts_inner](
                                k_val * p + col + col_block
                            )
                            C.data.store[width=nelts_inner](
                                row * p + col + col_block, c_val + a_val * b_val
                            )

                        vectorize[dot, nelts](min(nelts * 4, p - col_block))

            parallelize[calc_row](m, m)
            return C^
        else:
            print("MatMul only supported for 2D tensors (matrices)")
            return Tensor(0)

    fn __matmul_tiled__(self, rhs: Tensor) -> Tensor:
        """
        Tiled matrix multiplication with register accumulation.
        """
        if self.shape.__len__() == 2 and rhs.shape.__len__() == 2:
            var m = self.shape[0]
            var k = self.shape[1]
            var n = rhs.shape[0]
            var p = rhs.shape[1]

            if k != n:
                print(
                    "MatMul not possible -> self.cols: "
                    + String(k)
                    + " != rhs.rows: "
                    + String(n)
                )
                return Tensor(0)

            var C: Tensor = Tensor(m, p)

            alias tile_i = 64
            alias tile_j = nelts * 64

            @parameter
            fn calc_tile(jo: Int, io: Int):
                var accumulators = stack_allocation[
                    tile_i * tile_j, Scalar[type]
                ]()

                for i in range(tile_i * tile_j):
                    accumulators[i] = Scalar[type](0)

                for k_val in range(k):

                    @parameter
                    fn calc_tile_row(i: Int):
                        @parameter
                        fn calc_tile_cols[nelts_inner: Int](j: Int):
                            var idx = i * tile_j + j
                            var a_val = self.data.load((io + i) * k + k_val)
                            var b_val = rhs.data.load[width=nelts_inner](
                                k_val * p + jo + j
                            )
                            var acc = accumulators.load[width=nelts_inner](idx)
                            accumulators.store[width=nelts_inner](
                                idx, acc + a_val * b_val
                            )

                        vectorize[calc_tile_cols, nelts](tile_j)

                    @parameter
                    fn unroll_rows():
                        @parameter
                        for i in range(tile_i):
                            calc_tile_row(i)

                    unroll_rows()

                for i in range(tile_i):
                    for j in range(tile_j):
                        if io + i < m and jo + j < p:
                            C.data.store(
                                (io + i) * p + (jo + j),
                                accumulators[i * tile_j + j],
                            )

            @parameter
            fn tile_parallel_rows(yo: Int):
                var y = tile_i * yo
                for x in range(0, p, tile_j):
                    calc_tile(x, y)

            parallelize[tile_parallel_rows]((m + tile_i - 1) // tile_i, m)
            return C^
        else:
            print("MatMul only supported for 2D tensors (matrices)")
            return Tensor(0)

    fn __matmul_simd__(self, rhs: Tensor) -> Tensor:
        """
        Pure SIMD vectorized matmul - fastest for smaller matrices (<256x256).
        """
        if self.shape.__len__() != 2 or rhs.shape.__len__() != 2:
            print("MatMul only supported for 2D tensors (matrices)")
            return Tensor(0)

        var m = self.shape[0]
        var k = self.shape[1]
        var n = rhs.shape[0]
        var p = rhs.shape[1]

        if k != n:
            print(
                "MatMul not possible -> self.cols: "
                + String(k)
                + " != rhs.rows: "
                + String(n)
            )
            return Tensor(0)

        var C: Tensor = Tensor(m, p)

        for i in range(m):
            for kk in range(k):

                @parameter
                fn dot[nelts_inner: Int](j: Int):
                    var c_val = C.data.load[width=nelts_inner](i * p + j)
                    var a_val = self.data.load(i * k + kk)
                    var b_val = rhs.data.load[width=nelts_inner](kk * p + j)
                    C.data.store[width=nelts_inner](
                        i * p + j, c_val + a_val * b_val
                    )

                vectorize[dot, nelts](p)

        return C^

    fn pow(self, rhs: Tensor) -> Tensor:
        var new_tensor: Tensor = Tensor(self.shape)
        for i in range(self.size):
            new_tensor.data.store(i, rhs.data.load(i) ** self.data.load(i))
        return new_tensor^

    fn T(self) -> Tensor:
        """
        Transpose a 2D tensor (matrix) - swaps rows and columns.
        Shape [m, n] becomes [n, m].
        """
        if self.shape.__len__() != 2:
            print("Transpose only supported for 2D tensors")
            return Tensor(0)

        var m = self.shape[0]
        var n = self.shape[1]
        var new_tensor: Tensor = Tensor(n, m)

        for i in range(m):
            for j in range(n):
                new_tensor[j, i] = self[i, j]

        return new_tensor^

    fn transpose(self, axis1: Int, axis2: Int) -> Tensor:
        """
        Transpose arbitrary axes in an N-dimensional tensor.
        Swaps the dimensions at axis1 and axis2.
        """
        if (
            axis1 < 0
            or axis1 >= self.shape.__len__()
            or axis2 < 0
            or axis2 >= self.shape.__len__()
        ):
            print("Invalid axes for transpose")
            return Tensor(0)

        # Create new shape with swapped dimensions
        var new_shape = List[Int]()
        for i in range(self.shape.__len__()):
            if i == axis1:
                new_shape.append(self.shape[axis2])
            elif i == axis2:
                new_shape.append(self.shape[axis1])
            else:
                new_shape.append(self.shape[i])

        var new_tensor = Tensor(new_shape)

        # Copy data with transposed indexing
        var old_indices = List[Int]()
        for _ in range(self.shape.__len__()):
            old_indices.append(0)

        for flat_idx in range(self.size):
            # Convert flat index to multi-dimensional index
            var idx = flat_idx
            for i in range(self.shape.__len__() - 1, -1, -1):
                old_indices[i] = idx % self.shape[i]
                idx /= self.shape[i]

            # Swap axes
            var temp = old_indices[axis1]
            old_indices[axis1] = old_indices[axis2]
            old_indices[axis2] = temp

            # Calculate new flat index and copy
            var new_flat_idx = 0
            var stride = 1
            for i in range(new_tensor.shape.__len__() - 1, -1, -1):
                new_flat_idx += old_indices[i] * stride
                stride *= new_tensor.shape[i]

            new_tensor.data.store(new_flat_idx, self.data.load(flat_idx))

        return new_tensor^

    fn item(self) -> Float32:
        """Extract scalar value from a scalar tensor."""
        if self.shape.__len__() != 0:
            print("Warning: item() called on non-scalar tensor")
        return self.data.load(0)
