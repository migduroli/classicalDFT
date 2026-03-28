.PHONY: run run-local
DOCKER_IMAGE := classicaldft-examples:dev
ROOT := $(shell git rev-parse --show-toplevel)
BUILD_DIR := $(ROOT)/build-local

EXAMPLE_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
EXPORTS_SRC := $(BUILD_DIR)/examples/physics/fmt/exports
EXPORTS_DST := $(EXAMPLE_DIR)exports

run: ## Run this example in Docker (headless, plots saved to exports/)
	@docker build -t $(DOCKER_IMAGE) -f $(ROOT)/examples/Dockerfile $(ROOT)
	@docker run --rm $(DOCKER_IMAGE) "cd build/examples/physics/fmt && ./example_fmt"

run-local: ## Build and run locally (plots saved to exports/)
	@cmake -B $(BUILD_DIR) -S $(ROOT) \
		-DDFT_BUILD_TESTS=OFF \
		-DDFT_BUILD_EXAMPLES=ON \
		-DDFT_USE_MATPLOTLIB=ON
	@cmake --build $(BUILD_DIR) --parallel --target example_fmt
	@cd $(BUILD_DIR)/examples/physics/fmt && ./example_fmt
	@mkdir -p $(EXPORTS_DST)
	@cp -f $(EXPORTS_SRC)/*.png $(EXPORTS_DST)/ 2>/dev/null || true
	@echo "\nPlots copied to $(EXPORTS_DST)/"
