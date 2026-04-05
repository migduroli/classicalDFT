.PHONY: run run-local run-checks clean

DOCKER_IMAGE := classicaldft-docs:dev
ROOT := $(shell git rev-parse --show-toplevel)
BUILD_DIR := $(ROOT)/build-local

run: ## Run this example in Docker
	@docker build -t $(DOCKER_IMAGE) -f $(ROOT)/docs/Dockerfile $(ROOT)
	@docker run --rm $(DOCKER_IMAGE) ./doc_interaction

run-local: ## Build and run locally
	@cmake -B $(BUILD_DIR) -S $(ROOT) \
		-DDFT_BUILD_TESTS=OFF \
		-DDFT_BUILD_DOCS=ON \
		-DDFT_USE_MATPLOTLIB=ON
	@cmake --build $(BUILD_DIR) --parallel --target doc_interaction
	@$(BUILD_DIR)/docs/doc_interaction

run-checks: ## Build and run cross-validation checks
	@cmake -B $(BUILD_DIR) -S $(ROOT) \
		-DDFT_BUILD_TESTS=OFF \
		-DDFT_BUILD_DOCS=ON
	@cmake --build $(BUILD_DIR) --parallel --target check_interaction
	@$(BUILD_DIR)/docs/check_interaction

clean:
	@rm -f $(BUILD_DIR)/docs/doc_interaction
	@rm -f $(BUILD_DIR)/docs/check_interaction
