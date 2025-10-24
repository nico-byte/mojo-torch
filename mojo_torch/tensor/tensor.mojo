from memory import memcpy
from memory.unsafe_pointer import UnsafePointer
from sys import simd_width_of
from memory import memset_zero, stack_allocation
from random import randn, rand, seed
from algorithm.functional import vectorize, parallelize
from collections.list import List
from python import Python, PythonObject
import random

alias type = DType.float32
alias nelts = simd_width_of[Scalar[type]]()


struct LayoutTensor(Copyable, Movable, Writable):
    var shape: List[Int]
    var strides: List[Int]

    fn __init__(out self, shape: List[Int]):
        self.shape = shape.copy()
        self.strides = List[Int]()

        # Compute strides for row-major (C-contiguous) layout
        var stride = 1
        for i in range(shape.__len__() - 1, -1, -1):
            self.strides.append(stride)
            stride *= shape[i]

        # Reverse strides to match shape order
        var reversed_strides = List[Int]()
        for i in range(self.strides.__len__() - 1, -1, -1):
            reversed_strides.append(self.strides[i])
        self.strides = reversed_strides.copy()

    fn __init__(out self, shape: List[Int], strides: List[Int]):
        self.shape = shape.copy()
        self.strides = strides.copy()

    fn __init__(out self, shape: (Int, Int), strides: (Int, Int)):
        self.shape = List[Int](shape[0], shape[1])
        self.strides = List[Int](strides[0], strides[1])

    fn __init__(out self, shape: (Int, Int)):
        self.strides = List[Int](shape[1], 1)
        self.shape = List[Int](shape[0], shape[1])

    @always_inline("nodebug")
    fn __call__(self, indices: List[Int]) -> Int:
        var offset = 0
        for d in range(indices.__len__()):
            offset += indices[d] * self.strides[d]
        return offset

    @always_inline("nodebug")
    fn size(self) -> Int:
        var total = 1
        for shape in self.shape:
            total *= shape
        return total

    @always_inline("nodebug")
    fn write_to[W: Writer](self, mut writer: W):
        writer.write("Shape: ")
        for dim in self.shape:
            writer.write(dim, " ")
        writer.write("| Strides: ")
        for stride in self.strides:
            writer.write(stride, " ")
        writer.write("\n")


struct Tensor[Type: DType](Copyable, ImplicitlyCopyable, Movable):
    """
    N-dimensional Tensor.
    """

    var data: UnsafePointer[Scalar[Type]]
    var size: Int
    var layout: LayoutTensor

    fn __init__(out self, dims: List[Int]):
        self.layout = LayoutTensor(dims)
        self.size = self.layout.size()
        self.data = UnsafePointer[Scalar[Type]].alloc(self.size)

    fn __init__(out self, *dims: Int):
        shape = [dim for dim in dims]
        self.layout = LayoutTensor(shape)
        self.size = self.layout.size()
        self.data = UnsafePointer[Scalar[Type]].alloc(self.size)

    fn __init__(out self, dims: List[Int], default_value: Scalar[Type]):
        self.layout = LayoutTensor(dims)
        self.size = self.layout.size()
        self.data = UnsafePointer[Scalar[Type]].alloc(self.size)
        for i in range(self.size):
            self.data.store(i, default_value)

    fn __init__(out self, *dims: Int, default_value: Scalar[Type]):
        shape = [dim for dim in dims]
        self.layout = LayoutTensor(shape)
        self.size = self.layout.size()
        self.data = UnsafePointer[Scalar[Type]].alloc(self.size)
        for i in range(self.size):
            self.data.store(i, default_value)

    fn __init__(
        out self, data: UnsafePointer[Scalar[Type]], var layout: LayoutTensor
    ):
        self.data = UnsafePointer[Scalar[Type]](data)
        self.layout = layout.copy()
        self.size = self.layout.size()

    fn __init__(out self, data: UnsafePointer[Scalar[Type]], shape: (Int, Int)):
        self.data = data
        self.layout = LayoutTensor(shape)
        self.size = self.layout.size()

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
            tensor.layout.shape = shape_list^

            # Copy values from numpy array
            for i in range(total_size):
                tensor.data.store(i, Scalar[Type](flattened[i]))

            tensor.layout.strides = tensor._compute_strides()
            return tensor^
        except:
            print("Failed to create Tensor from numpy array")
            return Self()

    @staticmethod
    fn scalar(value: Float32) -> Self:
        """Create a scalar tensor with empty shape."""
        var tensor = Self()  # No dimensions = scalar
        tensor.data.store(0, Scalar[Type](value))
        return tensor^

    @always_inline("nodebug")
    fn slice(self, i: Int, j: Int, ir: Int, jr: Int) -> Self:
        var shape = (ir, jr)
        var strides = (self.layout.strides[0], self.layout.strides[1])
        var offset = self.layout([i, j])
        return Tensor(self.data + offset, LayoutTensor(shape, strides))

    fn _compute_strides(self) -> List[Int]:
        """Compute strides based on shape for row-major order."""
        # For scalar tensors (empty shape), return empty strides
        if self.layout.shape.__len__() == 0:
            return List[Int]()

        var strides = List[Int]()
        var stride = 1
        for i in range(self.layout.shape.__len__() - 1, -1, -1):
            strides.append(stride)
            stride *= self.layout.shape[i]

        # Reverse strides to match dimension order
        var result = List[Int]()
        for i in range(strides.__len__() - 1, -1, -1):
            result.append(strides[i])
        return result^

    fn _flat_index(self, indices: VariadicList[Int]) -> Int:
        """Convert multi-dimensional indices to flat index."""
        # For scalar tensors, always return index 0
        if self.layout.shape.__len__() == 0:
            return 0
        var idx = 0
        for i in range(indices.__len__()):
            idx += indices[i] * self.layout.strides[i]
        return idx

    fn __getitem__(self, *indices: Int) -> Scalar[Type]:
        return self.data.load(self._flat_index(indices))

    fn __setitem__(self, *indices: Int, val: Scalar[Type]):
        self.data.store(self._flat_index(indices), val)

    fn __setitem__(self, rhs: Tensor[Type]):
        for i in range(self.size):
            self.data.store(i, rhs.data.load(i))

    fn __len__(self) -> Int:
        return self.size

    fn __copyinit__(out self, other: Self):
        self.layout = other.layout.copy()
        self.size = other.size
        self.data = UnsafePointer[Scalar[Type]].alloc(other.size)
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

    fn __lt__(self, rhs: Tensor[Type]) -> Bool:
        for i in range(self.size):
            if self.data.load(i) < rhs.data.load(i):
                return True
        return False

    fn __gt__(self, rhs: Tensor[Type]) -> Bool:
        for i in range(self.size):
            if self.data.load(i) > rhs.data.load(i):
                return True
        return False

    fn __eq__(self, rhs: Tensor[Type]) -> Bool:
        for i in range(self.size):
            var self_val: Scalar[Type] = self.data.load(i)
            var rhs_val: Scalar[Type] = rhs.data.load(i)
            if self_val < rhs_val or self_val > rhs_val:
                return False
        return True

    fn __ne__(self, rhs: Tensor[Type]) -> Bool:
        return not self == rhs

    fn __ge__(self, rhs: Tensor[Type]) -> Bool:
        return self > rhs or self == rhs

    fn __le__(self, rhs: Tensor[Type]) -> Bool:
        return self < rhs or self == rhs

    fn __add__(self, rhs: Tensor[Type]) -> Tensor[Type]:
        var new_tensor: Tensor[Type] = Tensor[Type](self.layout.shape)
        for i in range(self.size):
            new_tensor.data.store(i, self.data.load(i) + rhs.data.load(i))
        return new_tensor^

    fn __pow__(self, rhs: Tensor[Type]) -> Tensor[Type]:
        var new_tensor: Tensor[Type] = Tensor[Type](self.layout.shape)
        for i in range(self.size):
            new_tensor.data.store(i, self.data.load(i) ** rhs.data.load(i))
        return new_tensor^

    fn __sub__(self, rhs: Tensor[Type]) -> Tensor[Type]:
        var new_tensor: Tensor[Type] = Tensor[Type](self.layout.shape)
        for i in range(self.size):
            new_tensor.data.store(i, self.data.load(i) - rhs.data.load(i))
        return new_tensor^

    fn __mul__(self, rhs: Tensor[Type]) -> Tensor[Type]:
        var new_tensor: Tensor[Type] = Tensor[Type](self.layout.shape)
        for i in range(self.size):
            new_tensor.data.store(i, self.data.load(i) * rhs.data.load(i))
        return new_tensor^

    fn __truediv__(self, rhs: Tensor[Type]) -> Tensor[Type]:
        var new_tensor: Tensor[Type] = Tensor[Type](self.layout.shape)
        for i in range(self.size):
            new_tensor.data.store(i, self.data.load(i) / rhs.data.load(i))
        return new_tensor^

    fn __add__(self, rhs: Scalar[Type]) -> Tensor[Type]:
        var new_tensor: Tensor[Type] = Tensor[Type](self.layout.shape)
        for i in range(self.size):
            new_tensor.data.store(i, self.data.load(i) + rhs)
        return new_tensor^

    fn __pow__(self, rhs: Scalar[Type]) -> Tensor[Type]:
        var new_tensor: Tensor[Type] = Tensor[Type](self.layout.shape)
        for i in range(self.size):
            new_tensor.data.store(i, self.data.load(i) ** rhs)
        return new_tensor^

    fn __sub__(self, rhs: Scalar[Type]) -> Tensor[Type]:
        var new_tensor: Tensor[Type] = Tensor[Type](self.layout.shape)
        for i in range(self.size):
            new_tensor.data.store(i, self.data.load(i) - rhs)
        return new_tensor^

    fn __mul__(self, rhs: Scalar[Type]) -> Tensor[Type]:
        var new_tensor: Tensor[Type] = Tensor[Type](self.layout.shape)
        for i in range(self.size):
            new_tensor.data.store(i, self.data.load(i) * rhs)
        return new_tensor^

    fn __truediv__(self, rhs: Scalar[Type]) -> Tensor[Type]:
        var new_tensor: Tensor[Type] = Tensor[Type](self.layout.shape)
        for i in range(self.size):
            new_tensor.data.store(i, self.data.load(i) / rhs)
        return new_tensor^

    fn __matmul__(self, rhs: Tensor[Type]) -> Tensor[Type]:
        """
        Optimized matrix multiplication supporting 2D tensors.
        2D: [m, k] @ [k, n] -> [m, n].
        """
        if self.layout.shape.__len__() == 2 and rhs.layout.shape.__len__() == 2:
            # 2D matrix multiplication
            var m = self.layout.shape[0]
            var k = self.layout.shape[1]
            var n = rhs.layout.shape[0]
            var p = rhs.layout.shape[1]

            if k != n:
                print(
                    "MatMul not possible -> self.cols: "
                    + String(k)
                    + " != rhs.rows: "
                    + String(n)
                )
                return Tensor[Type](0)

            var C: Tensor[Type] = Tensor[Type](m, p)

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
            return Tensor[Type](0)

    fn pow(self, rhs: Tensor[Type][]) -> Tensor[Type]:
        var new_tensor: Tensor[Type] = Tensor[Type](self.layout.shape)
        for i in range(self.size):
            new_tensor.data.store(i, rhs.data.load(i) ** self.data.load(i))
        return new_tensor^

    fn T(self) -> Tensor[Type]:
        """
        Transpose a 2D tensor (matrix) - swaps rows and columns.
        Shape [m, n] becomes [n, m].
        Returns a row-major (C-contiguous) tensor.
        """
        if self.layout.shape.__len__() != 2:
            print("Transpose only supported for 2D tensors")
            return Tensor[Type](0)

        var m = self.layout.shape[0]
        var n = self.layout.shape[1]
        var new_tensor: Tensor[Type] = Tensor[Type](n, m)

        for i in range(m):
            for j in range(n):
                new_tensor.data.store(
                    j * m + i,
                    self.data.load(
                        i * self.layout.strides[0] + j * self.layout.strides[1]
                    ),
                )

        return new_tensor^

    fn transpose(self, axis1: Int, axis2: Int) -> Tensor[Type]:
        """
        Transpose arbitrary axes in an N-dimensional tensor.
        Swaps the dimensions at axis1 and axis2.
        Returns a row-major (C-contiguous) tensor.
        """
        if (
            axis1 < 0
            or axis1 >= self.layout.shape.__len__()
            or axis2 < 0
            or axis2 >= self.layout.shape.__len__()
        ):
            print("Invalid axes for transpose")
            return Tensor[Type](0)

        # Create new shape with swapped dimensions
        var new_shape = List[Int]()
        for i in range(self.layout.shape.__len__()):
            if i == axis1:
                new_shape.append(self.layout.shape[axis2])
            elif i == axis2:
                new_shape.append(self.layout.shape[axis1])
            else:
                new_shape.append(self.layout.shape[i])

        var new_tensor = Tensor[Type](new_shape)

        # Create index arrays for iteration
        var old_indices = List[Int]()
        for _ in range(self.layout.shape.__len__()):
            old_indices.append(0)

        var new_indices = List[Int]()
        for _ in range(new_shape.__len__()):
            new_indices.append(0)

        # Copy data with proper transposition to row-major layout
        for new_flat_idx in range(new_tensor.size):
            # Convert new flat index to multi-dimensional indices
            var idx = new_flat_idx
            for i in range(new_shape.__len__() - 1, -1, -1):
                new_indices[i] = idx % new_shape[i]
                idx /= new_shape[i]

            # Map new indices back to old indices (inverse transpose)
            for i in range(old_indices.__len__()):
                if i == axis1:
                    old_indices[i] = new_indices[axis2]
                elif i == axis2:
                    old_indices[i] = new_indices[axis1]
                else:
                    old_indices[i] = new_indices[i]

            # Calculate old flat index using old strides
            var old_flat_idx = 0
            for i in range(old_indices.__len__()):
                old_flat_idx += old_indices[i] * self.layout.strides[i]

            new_tensor.data.store(new_flat_idx, self.data.load(old_flat_idx))

        return new_tensor^

    fn rand(mut self):
        random.rand(self.data, self.layout.size())

    fn scalar(self) -> Scalar[Type]:
        """Return the scalar value of a scalar tensor."""
        if self.layout.shape.__len__() != 0:
            print("Warning: scalar() called on non-scalar tensor")
        return self.data.load(0)

    fn item(self) -> Scalar[Type]:
        """Extract scalar value from a scalar tensor."""
        if self.layout.shape.__len__() != 0:
            print("Warning: item() called on non-scalar tensor")
        return self.data.load(0)
