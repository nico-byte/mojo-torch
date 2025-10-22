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
    discovered_test_files = []
    for file in os.listdir("."):
        if file.endswith("_test.mojo") and file != "__init__.mojo":
            # Extract the test name (e.g., matmul_test.mojo -> matmul)
            test_name = file[:-10]  # Remove '_test.mojo'
            module_name = file[:-5]  # Remove '.mojo' to get module name

            # Look for test functions in the file
            test_functions = find_test_functions(file, test_name)

            if test_functions["main_functions"]:
                # Create an entry for each main test function in the file
                for main_function in test_functions["main_functions"]:
                    discovered_test_files.append(
                        {
                            "file": file,
                            "module": module_name,
                            "function": main_function,
                            "individual_tests": test_functions["individual_tests"],
                            "name": f"{test_name}::{main_function}",
                            "base_name": test_name,
                        }
                    )
            else:
                print(f"Warning: No main test function found in {file}")

    return discovered_test_files


def find_test_functions(file_path, test_name):
    """Find all test functions in a test file"""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Find all functions that match the **_test pattern (main test functions)
        all_functions = re.findall(r"fn\s+(\w+)\s*\([^)]*\)\s*raises?:", content)
        main_functions = []

        for func in all_functions:
            # Match functions that end with _test and don't have helper patterns
            if func.endswith("_test") and not any(
                keyword in func.lower() for keyword in ["assert", "helper", "util"]
            ):
                main_functions.append(func)

        # If no **_test functions found, try legacy patterns for backward compatibility
        if not main_functions:
            possible_main_functions = [
                f"{test_name}_test",  # e.g., matmul_test
                f"test_{test_name}",  # e.g., test_matmul
                f"run_{test_name}_tests",  # e.g., run_matmul_tests
                "run_tests",  # generic
                "main_test",  # generic
            ]

            for func_name in possible_main_functions:
                if f"fn {func_name}(" in content:
                    main_functions.append(func_name)
                    break

        # Find all individual test functions (helper functions that tests call)
        individual_tests = []
        for func in all_functions:
            if (
                func.startswith("test_")
                and func not in main_functions
                and not any(
                    keyword in func.lower() for keyword in ["assert", "helper", "util"]
                )
            ):
                individual_tests.append(func)

        return {"main_functions": main_functions, "individual_tests": individual_tests}

    except OSError:
        return {"main_functions": [], "individual_tests": []}


def detect_module_name():
    """Auto-detect the current module name based on directory structure"""
    current_dir = os.path.basename(os.getcwd())

    # Check if we're in a subdirectory that looks like a module
    if os.path.exists("__init__.mojo"):
        return current_dir

    # Default fallback
    return "test"


def run_tests(module_name=None, filter_pattern=None):
    """Run all Mojo test suites and provide pytest-style output"""
    print("🚀 Mojo-Torch Test Suite Runner")
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
    discovered_tests = discover_test_files()
    if not discovered_tests:
        print("Error: No test files found")
        print("Test files should be named '*_test.mojo'")
        return False

    # Filter tests if pattern provided
    if filter_pattern:
        original_count = len(discovered_tests)
        discovered_tests = [
            test
            for test in discovered_tests
            if filter_pattern.lower() in test["name"].lower()
            or filter_pattern.lower() in test["function"].lower()
        ]
        if not discovered_tests:
            print(f"Error: No test functions match filter pattern '{filter_pattern}'")
            return False
        print(
            f"Filtered to {len(discovered_tests)} test function(s) from {original_count} (filter: '{filter_pattern}')"
        )

    # Group tests by file for display
    files_with_tests = {}
    for test_info in discovered_tests:
        file_name = test_info["file"]
        if file_name not in files_with_tests:
            files_with_tests[file_name] = []
        files_with_tests[file_name].append(test_info)

    print(
        f"Found {len(discovered_tests)} test function(s) in {len(files_with_tests)} file(s):"
    )
    for file_name, test_infos in files_with_tests.items():
        print(f"  📁 {file_name}:")
        for test_info in test_infos:
            print(f"    ✓ {test_info['function']}()")
        if test_infos[0].get("individual_tests"):
            print(
                f"    Individual helper tests: {', '.join(test_infos[0]['individual_tests'])}"
            )

    # Check if mojo is available
    try:
        result = subprocess.run(
            ["mojo", "--version"], capture_output=True, text=True, check=False
        )
        if result.returncode != 0:
            print("❌ Error: Mojo compiler not found")
            print("Please ensure Mojo is installed and in your PATH")
            return False
        print(f"✅ Found Mojo: {result.stdout.strip()}")
    except FileNotFoundError:
        print("❌ Error: Mojo compiler not found")
        print("Please ensure Mojo is installed and in your PATH")
        return False

    print(f"\n🔄 Running {len(discovered_tests)} test function(s)...")
    print("=" * 50)

    # Run each test function individually to get per-function results
    test_results_by_function = {}
    overall_success = True

    for test_info in discovered_tests:
        print(f"\n📋 Testing {test_info['name']}...")

        # Create a temporary main file for this specific test
        imports = [
            f"from {module_name}.{test_info['module']} import {test_info['function']}"
        ]
        function_calls = [
            f"    try:\n        {test_info['function']}()\n    except e:\n        print(\"ERROR: Test function '{test_info['name']}' failed:\", e)\n        raise e"
        ]

        main_content = f"""{chr(10).join(imports)}

fn main() raises:
    print("Running {test_info["name"]} tests...")
    print("="*40)
    
{chr(10).join(function_calls)}

    print("\\nTest function completed!")
"""

        # Create a safe filename from the test name (replace :: with _)
        safe_name = test_info["name"].replace("::", "_")
        temp_main_file = f"temp_{safe_name}_test.mojo"

        try:
            # Write the temporary main file
            with open(temp_main_file, "w", encoding="utf-8") as f:
                f.write(main_content)

            # Run this specific test function
            result = subprocess.run(
                ["mojo", temp_main_file], capture_output=True, text=True, check=False
            )

            # Parse the results for this test function
            parsed_result = parse_test_output(result.stdout or "")
            test_results_by_function[test_info["name"]] = parsed_result

            if not parsed_result["success"]:
                overall_success = False
                print(
                    f"❌ {test_info['name']}: {parsed_result['failed']}/{parsed_result['total']} individual tests failed"
                )
            else:
                print(f"✅ {test_info['name']}: All tests passed")

        except (subprocess.SubprocessError, OSError) as e:
            print(f"❌ Error running {test_info['name']}: {e}")
            test_results_by_function[test_info["name"]] = {
                "success": False,
                "failures": [{"test_name": "Execution Error", "details": str(e)}],
                "total": 0,
                "passed": 0,
                "failed": 1,
            }
            overall_success = False
        finally:
            # Clean up temporary file
            if os.path.exists(temp_main_file):
                try:
                    os.remove(temp_main_file)
                except OSError:
                    pass

    # Print pytest-style summary
    print_pytest_style_summary(test_results_by_function, overall_success)

    return overall_success


def parse_test_output(output):
    """Parse the test output to determine if all tests passed and extract failure details"""
    if not output:
        return {"success": False, "failures": [], "total": 0, "passed": 0, "failed": 0}

    lines = output.split("\n")
    failures = []
    total_tests = 0
    total_passed = 0
    total_failed = 0
    current_test_suite = None

    # Parse summary information
    for line in lines:
        line = line.strip()

        # Detect test suite start
        if line.startswith("Test ") and ":" in line:
            current_test_suite = line

        # Detect failures
        if "✗ FAIL:" in line:
            failure_info = line.replace("✗ FAIL:", "").strip()
            failures.append(
                {
                    "test_suite": current_test_suite or "Unknown",
                    "test_name": failure_info.split(" - ")[0]
                    if " - " in failure_info
                    else failure_info,
                    "details": failure_info,
                }
            )

        # Parse summary numbers
        if "Total tests:" in line:
            try:
                total_tests = int(line.split("Total tests:")[1].strip())
            except (ValueError, IndexError):
                pass
        elif "Passed:" in line:
            try:
                total_passed = int(line.split("Passed:")[1].strip())
            except (ValueError, IndexError):
                pass
        elif "Failed:" in line:
            try:
                total_failed = int(line.split("Failed:")[1].strip())
            except (ValueError, IndexError):
                pass

    success = total_failed == 0 and total_tests > 0

    return {
        "success": success,
        "failures": failures,
        "total": total_tests,
        "passed": total_passed,
        "failed": total_failed,
    }


def print_pytest_style_summary(test_results_by_function, overall_success):
    """Print a pytest-style summary of test results"""
    print("\n" + "=" * 70)
    print("TEST SESSION RESULTS")
    print("=" * 70)

    total_functions = len(test_results_by_function)
    passed_functions = 0
    total_tests = 0
    total_passed = 0
    total_failed = 0

    # Print results for each test function
    for test_function, result in test_results_by_function.items():
        if result["success"]:
            status = "✅ PASSED"
            passed_functions += 1
        else:
            status = "❌ FAILED"

        total_tests += result["total"]
        total_passed += result["passed"]
        total_failed += result["failed"]

        print(f"{status} {test_function}")
        if result["total"] > 0:
            print(f"    {result['passed']}/{result['total']} individual tests passed")

        # Show failed tests
        if result["failures"]:
            print("    Failed individual tests:")
            for failure in result["failures"]:
                print(f"      - {failure['test_name']}")

    print("-" * 70)

    # Overall summary
    if overall_success:
        print(
            f"🎉 ALL TEST FUNCTIONS PASSED ({passed_functions}/{total_functions} functions)"
        )
    else:
        failed_functions = total_functions - passed_functions
        print(f"💥 {failed_functions}/{total_functions} test function(s) failed")

    if total_tests > 0:
        success_rate = (total_passed / total_tests) * 100
        print(
            f"Overall: {total_passed}/{total_tests} individual tests passed ({success_rate:.1f}%)"
        )

    # Detailed failure summary
    if total_failed > 0:
        print("\nFAILED TESTS SUMMARY:")
        print("-" * 40)
        for test_function, result in test_results_by_function.items():
            if result["failures"]:
                print(f"\n{test_function}:")
                for failure in result["failures"]:
                    print(f"  ❌ {failure['test_name']}")
                    if (
                        "Expected:" in failure["details"]
                        and "Got:" in failure["details"]
                    ):
                        print(f"     {failure['details']}")

    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Mojo-Torch test suites with pytest-style output",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py                       # Run all test functions
  python run.py --list                # List available test functions
  python run.py --verbose             # Show detailed output for all tests
  python run.py --filter matmul       # Run only functions matching 'matmul'
  python run.py --filter tiled        # Run only tiled test functions
  python run.py --module my_module    # Use specific module name
        """,
    )
    parser.add_argument(
        "--module",
        default=None,
        help="Module name for imports (auto-detected if not specified)",
    )
    parser.add_argument(
        "--list", action="store_true", help="List available test files and exit"
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show detailed output for all test files, not just failures",
    )
    parser.add_argument(
        "--filter",
        "-f",
        type=str,
        help="Run only test functions matching this pattern (e.g., 'matmul' or 'tiled')",
    )

    args = parser.parse_args()

    # Set verbose environment variable for the test runner
    if args.verbose:
        os.environ["VERBOSE"] = "1"

    if args.list:
        print("📋 Available test functions:")
        discovered_tests = discover_test_files()

        # Group by file for display
        files_with_tests = {}
        for test_info in discovered_tests:
            file_name = test_info["file"]
            if file_name not in files_with_tests:
                files_with_tests[file_name] = []
            files_with_tests[file_name].append(test_info)

        for file_name, test_infos in files_with_tests.items():
            print(f"  📁 {file_name}:")
            for test_info in test_infos:
                print(f"    ✓ {test_info['function']}() -> {test_info['name']}")
            if test_infos[0].get("individual_tests"):
                print(
                    f"    Individual helper tests: {', '.join(test_infos[0]['individual_tests'])}"
                )
        sys.exit(0)

    success = run_tests(module_name=args.module, filter_pattern=args.filter)

    # Exit with appropriate code for CI/CD systems
    sys.exit(0 if success else 1)
