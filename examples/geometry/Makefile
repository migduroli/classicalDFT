.PHONY: run
DOCKER_IMAGE := classicaldft-examples:dev
ROOT := $(shell git rev-parse --show-toplevel)

run: ## Run this example in Docker
	@docker build -t $(DOCKER_IMAGE) -f $(ROOT)/examples/Dockerfile $(ROOT)
	@docker run --rm $(DOCKER_IMAGE) ./build/examples/geometry/example_geometry
