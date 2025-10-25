# Makefile for spaCy Linguistic Examples
# Author: spaCy Tutorial Series
# Created: 2024-01-01

.PHONY: setup-env clean test-notebooks help

# Default target
help:
	@echo "Available targets:"
	@echo "  setup-env      - Create virtual environment and install dependencies"
	@echo "  test-notebooks - Run all Jupyter notebooks to test for errors"
	@echo "  clean         - Deactivate virtual environment and clean up files"
	@echo "  help          - Show this help message"

# Setup virtual environment and install dependencies
setup-env:
	@echo "Setting up virtual environment..."
	python3.9 -m venv .venv
	@echo "Activating virtual environment and installing dependencies..."
	@bash -c "source .venv/bin/activate && uv pip install -r requirements.txt"
	@echo "Downloading spaCy language model..."
	@bash -c "source .venv/bin/activate && python -m spacy download en_core_web_sm"
	@echo "Virtual environment setup complete!"
	@echo "To activate the environment, run: source .venv/bin/activate"

# Test all Jupyter notebooks
test-notebooks:
	@echo "Testing all Jupyter notebooks..."
	@if [ ! -d ".venv" ]; then \
		echo "Virtual environment not found. Please run 'make setup-env' first."; \
		exit 1; \
	fi
	@echo "Running notebook tests..."
	@bash -c "source .venv/bin/activate && \
		for notebook in *.ipynb; do \
			if [ \"\$$notebook\" != \"*\" ]; then \
				echo \"Testing \$$notebook...\"; \
				jupyter nbconvert --to notebook --execute \"\$$notebook\" --output \"\$${notebook%.ipynb}-executed.ipynb\" || { \
					echo \"Error executing \$$notebook\"; \
					exit 1; \
				}; \
				echo \"✓ \$$notebook executed successfully\"; \
			fi; \
		done"
	@echo "All notebooks tested successfully!"

# Clean up virtual environment and temporary files
clean:
	@echo "Cleaning up..."
	@if [ -d ".venv" ]; then \
		echo "Removing virtual environment..."; \
		rm -rf .venv; \
	fi
	@echo "Removing temporary files..."
	@find . -name "*.pyc" -delete
	@find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
	@find . -name "*.pyo" -delete
	@find . -name "*.pyd" -delete
	@find . -name ".pytest_cache" -type d -exec rm -rf {} + 2>/dev/null || true
	@find . -name "*.egg-info" -type d -exec rm -rf {} + 2>/dev/null || true
	@find . -name "dist" -type d -exec rm -rf {} + 2>/dev/null || true
	@find . -name "build" -type d -exec rm -rf {} + 2>/dev/null || true
	@echo "Cleanup complete!"
