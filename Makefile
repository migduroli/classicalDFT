.PHONY: help check clean build test test-local format format-check lint coverage
.DEFAULT_GOAL := help

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

check: ## Check project dependencies
	@./scripts/check

clean: ## Remove build artifacts
	@./scripts/clean

build: ## Build the library and tests
	@./scripts/build

test: ## Run tests in Docker
	@./scripts/test

test-local: ## Run tests locally (requires prior build)
	@./scripts/test_local

format: ## Format source code with clang-format
	@./scripts/format

format-check: ## Check formatting without modifying files
	@./scripts/format --check

lint: ## Lint source code (format check + clang-tidy)
	@./scripts/lint

coverage: ## Run tests with coverage analysis (lcov + genhtml)
	@./scripts/coverage
