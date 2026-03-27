.PHONY: run run-local
DOCKER_IMAGE := classicaldft-examples:dev
ROOT := $(shell git rev-parse --show-toplevel)
BUILD_DIR := $(ROOT)/build-local

run: ## Run this example in Docker (headless)
	@docker build -t $(DOCKER_IMAGE) -f $(ROOT)/examples/Dockerfile $(ROOT)
	@docker run --rm $(DOCKER_IMAGE) "xvfb-run ./build/examples/graphs/example_graphs"

run-local: ## Build and run locally (Grace plots visible)
	@cmake -B $(BUILD_DIR) -S $(ROOT) \
		-DDFT_BUILD_TESTS=OFF \
		-DDFT_BUILD_EXAMPLES=ON \
		-DDFT_USE_GRACE=ON
	@cmake --build $(BUILD_DIR) --parallel --target example_graphs
	@if [ "$$(uname -s)" = "Darwin" ]; then \
		open -a XQuartz 2>/dev/null || true; \
		export DISPLAY=:0; \
	fi; \
	$(BUILD_DIR)/examples/graphs/example_graphs
