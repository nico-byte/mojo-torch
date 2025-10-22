fn abs_f32(x: Float32) -> Float32:
    return x if x >= 0 else -x


struct TensorTestResults:
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
        if not abs_f32(actual - expected) < tolerance:
            self.failed += 1
            print("✗ FAIL:", test_name, "- Expected:", expected, "Got:", actual)
        else:
            self.passed += 1

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

        if not shapes_match:
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
        else:
            self.passed += 1

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
