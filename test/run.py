#!/usr/bin/env python3
"""
Test runner for Mojo-Torch Test Suite
"""

import subprocess
import sys
import os
import re
import argparse


def discover_test_files():
    """Discover all test files in the current directory"""
    test_files = []
    for file in os.listdir("."):
        if file.endswith("_test.mojo") and file != "__init__.mojo":
            # Extract the test name (e.g., matmul_test.mojo -> matmul)
            test_name = file[:-10]  # Remove '_test.mojo'
            module_name = file[:-5]  # Remove '.mojo' to get module name

            # Look for the main test function in the file
            main_function = find_main_test_function(file, test_name)

            if main_function:
                test_files.append(
                    {
                        "file": file,
                        "module": module_name,
                        "function": main_function,
                        "name": test_name,
                    }
                )
            else:
                print(f"Warning: No main test function found in {file}")

    return test_files


def find_main_test_function(file_path, test_name):
    """Find the main test function in a test file"""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Look for common test function patterns
        possible_functions = [
            f"{test_name}_test",  # e.g., matmul_test
            f"test_{test_name}",  # e.g., test_matmul
            f"run_{test_name}_tests",  # e.g., run_matmul_tests
            "run_tests",  # generic
            "main_test",  # generic
        ]

        for func_name in possible_functions:
            if f"fn {func_name}(" in content:
                return func_name

        # If no standard pattern found, look for any function that looks like a main test
        functions = re.findall(r"fn\s+(\w+)\s*\([^)]*\)\s*raises?:", content)

        # Filter for likely main test functions (not helper functions)
        for func in functions:
            if any(
                keyword in func.lower() for keyword in ["test", "run", "main"]
            ) and not any(
                keyword in func.lower()
                for keyword in ["1d", "2d", "3d", "assert", "helper"]
            ):
                return func

        return None

    except OSError:
        return None


def detect_module_name():
    """Auto-detect the current module name based on directory structure"""
    current_dir = os.path.basename(os.getcwd())

    # Check if we're in a subdirectory that looks like a module
    if os.path.exists("__init__.mojo"):
        return current_dir

    # Default fallback
    return "test"


def run_tests(module_name=None):
    """Run all Mojo test suites by creating a main file that imports all test modules"""
    print("Mojo-Torch Test Suite Runner")
    print("=" * 50)

    # Auto-detect module name if not provided
    if module_name is None:
        module_name = detect_module_name()

    print(f"Using module name: {module_name}")

    # Check if we're in the test directory
    if not os.path.exists("./matmul_test.mojo"):
        print("Error: ./matmul_test.mojo not found")
        print("Please run this script from the test directory")
        return False

    # Discover test files
    test_files = discover_test_files()
    if not test_files:
        print("Error: No test files found")
        print("Test files should be named '*_test.mojo'")
        return False

    print(f"Found {len(test_files)} test file(s):")
    for test_file in test_files:
        print(f"  - {test_file['file']} -> {test_file['function']}()")

    # Check if mojo is available
    try:
        result = subprocess.run(
            ["mojo", "--version"], capture_output=True, text=True, check=False
        )
        if result.returncode != 0:
            print("Error: Mojo compiler not found")
            print("Please ensure Mojo is installed and in your PATH")
            return False
        print(f"Found Mojo: {result.stdout.strip()}")
    except FileNotFoundError:
        print("Error: Mojo compiler not found")
        print("Please ensure Mojo is installed and in your PATH")
        return False

    # Create imports and function calls for all test modules
    imports = []
    function_calls = []

    for test_file in test_files:
        imports.append(
            f"from {module_name}.{test_file['module']} import {test_file['function']}"
        )
        function_calls.append(f"    {test_file['function']}()")

    # Create a temporary main file that imports and runs all tests
    main_content = f"""{chr(10).join(imports)}

fn main() raises:
    print("Starting Mojo-Torch Test Suite")
    print("="*50)
    
{chr(10).join(function_calls)}

    print("\\nAll test suites completed!")
"""

    temp_main_file = "temp_test_main.mojo"

    try:
        # Write the temporary main file
        with open(temp_main_file, "w", encoding="utf-8") as f:
            f.write(main_content)

        print(f"\nRunning {len(test_files)} test suite(s)...")
        print("=" * 50)

        # Run the tests using the temporary main file
        result = subprocess.run(
            ["mojo", temp_main_file], capture_output=False, text=True, check=False
        )
        test_success = result.returncode == 0

        if test_success:
            print("\nAll test suites executed successfully")
        else:
            print(f"\nTest execution failed with return code: {result.returncode}")

        return test_success

    except (subprocess.SubprocessError, OSError) as e:
        print(f"Error running tests: {e}")
        return False
    finally:
        # Clean up temporary file
        if os.path.exists(temp_main_file):
            try:
                os.remove(temp_main_file)
            except OSError:
                pass  # Ignore cleanup errors


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Mojo-Torch test suites")
    parser.add_argument(
        "--module",
        default=None,
        help="Module name for imports (auto-detected if not specified)",
    )
    parser.add_argument(
        "--list", action="store_true", help="List available test files and exit"
    )

    args = parser.parse_args()

    if args.list:
        print("Available test files:")
        test_files = discover_test_files()
        for test_file in test_files:
            print(f"  - {test_file['file']} -> {test_file['function']}()")
        sys.exit(0)

    success = run_tests(module_name=args.module)
    sys.exit(0 if success else 1)
