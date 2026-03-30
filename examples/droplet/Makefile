.PHONY: run run-local
DOCKER_IMAGE := classicaldft-examples:dev
ROOT := $(shell git rev-parse --show-toplevel)
BUILD_DIR := $(ROOT)/build-local

run: ## Run this example in Docker (headless, plots saved to exports/)
	@docker build -t $(DOCKER_IMAGE) -f $(ROOT)/examples/Dockerfile $(ROOT)
	@docker run --rm $(DOCKER_IMAGE) "cd build/examples/droplet && ./example_droplet"

run-local: ## Build and run locally (plots saved to exports/)
	@cmake -B $(BUILD_DIR) -S $(ROOT) \
		-DDFT_BUILD_TESTS=OFF \
		-DDFT_BUILD_EXAMPLES=ON \
		-DDFT_USE_MATPLOTLIB=ON
	@cmake --build $(BUILD_DIR) --parallel --target example_droplet
	@$(BUILD_DIR)/examples/droplet/example_droplet
