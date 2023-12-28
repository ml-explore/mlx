.PHONY: install
install:
	# Build and install it using pip:
	env CMAKE_BUILD_PARALLEL_LEVEL="" pip install .

.PHONY: dev
dev:
	# Build and install an editable install:
	env CMAKE_BUILD_PARALLEL_LEVEL="" pip install -e .

.PHONY: test
test:
	# Install and run all the tests
	pip install ".[testing]"
	python -m unittest discover python/tests

.PHONY: stub
stub:
	# Install stubs to enable auto completions and type checking from your IDE:
	pip install ".[dev]"
	python setup.py generate_stubs
