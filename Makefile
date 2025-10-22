# Mojo-Torch Makefile

.PHONY: test test-pytorch test-all clean help list-tests

# Default target
help:
	@echo "Mojo-Torch Build System"
	@echo "=========================="
	@echo "Available targets:"
	@echo "  test        - Run all Mojo tests"
	@echo "  list-tests  - List available test files"
	@echo "  clean       - Clean build artifacts"
	@echo "  help        - Show this help message"

# Run Mojo tests
test:
	@echo "Running Mojo Test Suite..."
	@cd test && uv run python3 run.py

# List available test files
list-tests:
	@echo "Available test files:"
	@find test -name "*_test.mojo" -type f | sed 's|test/||' | sort

format:
	@echo "Formatting Mojo source files..."
	@find . -name "*.mojo" -type f -exec uv run mojo format {} \;
	@echo "Formatting Python source files..."
	@uv run ruff format .
	@echo "Formatting complete."

# Clean build artifacts
clean:
	@echo "Cleaning build artifacts..."
	@find . -name "*.pyc" -delete
	@find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
	@find . -name "temp_test_main.mojo" -delete 2>/dev/null || true
	@echo "Clean complete"
