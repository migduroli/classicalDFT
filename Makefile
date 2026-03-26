.PHONY: help check clean build test format format-check lint lint-fix install
.DEFAULT_GOAL := help

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

test: ## Run tests with coverage (fail under 60%, override with DFT_COV_FAIL_UNDER)
	@./scripts/test

format: ## Format source code with clang-format
	@./scripts/format

format-check: ## Check formatting without modifying files
	@./scripts/format --check

lint: ## Lint source code (format check + clang-tidy)
	@./scripts/lint

lint-fix: ## Auto-fix lint issues (format + clang-tidy --fix)
	@./scripts/lint --fix
