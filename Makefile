.PHONY: help check clean build test unit-tests integration-tests format format-check lint lint-fix install
.DEFAULT_GOAL := help

BUILD_DIR ?= build-local

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install local dependencies and build the library
	@./scripts/install

check: ## Check project dependencies
	@./scripts/check

clean: ## Remove build artifacts
	@./scripts/clean

build: ## Build the Docker image
	@./scripts/build

test: ## Run all tests with coverage in Docker
	@./scripts/test

unit-tests: ## Run unit tests with coverage in Docker
	@./scripts/unit-tests

integration-tests: ## Run integration tests with coverage in Docker
	@./scripts/integration-tests

format: ## Format source code with clang-format
	@./scripts/format

format-check: ## Check formatting without modifying files
	@./scripts/format --check

lint: ## Lint source code (format check + clang-tidy)
	@./scripts/lint

lint-fix: ## Auto-fix lint issues (format + clang-tidy --fix)
	@./scripts/lint --fix
