.PHONY: run run-local clean

DOCKER_IMAGE := classicaldft-docs:dev
ROOT := $(shell git rev-parse --show-toplevel)
BUILD_DIR := $(ROOT)/build-local

run: ## Run this example in Docker
	@docker build -t $(DOCKER_IMAGE) -f $(ROOT)/docs/Dockerfile $(ROOT)
	@docker run --rm $(DOCKER_IMAGE) ./doc_spline

run-local: ## Build and run locally
	@cmake -B $(BUILD_DIR) -S $(ROOT) \
		-DDFT_BUILD_TESTS=OFF \
		-DDFT_BUILD_DOCS=ON \
		-DDFT_USE_MATPLOTLIB=ON
	@cmake --build $(BUILD_DIR) --parallel --target doc_spline
	@$(BUILD_DIR)/docs/doc_spline

clean:
	@rm -f $(BUILD_DIR)/docs/doc_spline
