.PHONY: help build test clean examples run-examples configure format lint

BUILD_DIR  := build
CMAKE_OPTS := -DDFT_BUILD_TESTS=ON -DDFT_BUILD_EXAMPLES=ON
JOBS       := $(shell sysctl -n hw.ncpu 2>/dev/null || nproc 2>/dev/null || echo 4)

.DEFAULT_GOAL := help

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

configure: ## Configure CMake build
	@mkdir -p $(BUILD_DIR)
	@cd $(BUILD_DIR) && cmake .. $(CMAKE_OPTS)

build: configure ## Build library, tests, and examples
	@cmake --build $(BUILD_DIR) -- -j$(JOBS)

test: build ## Run all tests
	@$(BUILD_DIR)/tests/classicaldft_tests

examples: build ## Build examples only
	@cmake --build $(BUILD_DIR) --target cdft_eos cdft_potentials cdft_fmt cdft_crystal cdft_numerics cdft_config -- -j$(JOBS)

run-examples: examples ## Run all examples
	@for ex in cdft_eos cdft_potentials cdft_fmt cdft_crystal cdft_numerics cdft_config; do \
		echo "\n\033[36m=== $$ex ===\033[0m"; \
		$(BUILD_DIR)/examples/cdft/$$ex; \
	done

clean: ## Remove build artifacts
	@rm -rf $(BUILD_DIR)

format: ## Format source code with clang-format
	@find include src tests examples -name '*.cpp' -o -name '*.hpp' | xargs clang-format -i
	@echo "Formatted."

lint: ## Check formatting (dry run)
	@find include src tests examples -name '*.cpp' -o -name '*.hpp' | xargs clang-format --dry-run --Werror
