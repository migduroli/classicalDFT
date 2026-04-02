.PHONY: run run-local clean

DOCKER_IMAGE := classicaldft-examples:dev
ROOT := $(shell git rev-parse --show-toplevel)
BUILD_DIR := $(ROOT)/build-local

run: ## Run this example in Docker
	@docker build -t $(DOCKER_IMAGE) -f $(ROOT)/examples/Dockerfile $(ROOT)
	@docker run --rm $(DOCKER_IMAGE) ./example_fmt

run-local: ## Build and run locally
	@cmake -B $(BUILD_DIR) -S $(ROOT) \
		-DDFT_BUILD_TESTS=OFF \
		-DDFT_BUILD_EXAMPLES=ON \
		-DDFT_USE_MATPLOTLIB=ON
	@cmake --build $(BUILD_DIR) --parallel --target example_fmt
	@$(BUILD_DIR)/examples/example_fmt

clean:
	@rm -f $(BUILD_DIR)/examples/example_fmt
