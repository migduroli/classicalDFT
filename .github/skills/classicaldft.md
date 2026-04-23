---
name: classical-dft
description: Modern C++23 library for classical density functional theory — project map, conventions, and verification workflow.
---

Reference: [`CODING_GUIDELINES.md`](../CODING_GUIDELINES.md) is the
single source of truth for code style. This skill file supplements it
with operational knowledge.

---

## 1. Project identity

Modern C++23 library for classical density functional theory.
Single static library target `classicaldft`, header-heavy (5 compiled
translation units in `src/`). Depends on Armadillo, FFTW3, GSL,
nlohmann/json, autodiff (FetchContent), Catch2 v3 (FetchContent).

---

## 2. Directory map

```
include/
  dft.hpp                              umbrella header (includes everything)
  dftlib                               extension-less alias (includes dft.hpp)
  dft/
    grid.hpp  types.hpp  console.hpp   root-level vocabulary types
    exceptions.hpp  fields.hpp  init.hpp
    math/                              fourier, convolution, spline, etc.
    physics/                           potentials, interactions, model, eos, walls
    functionals/                       ideal_gas, hard_sphere, mean_field, external_field
      functional.hpp                   Functional struct + make_functional()
      evaluator.hpp                    Weights, make_weights(), total() orchestrator
      types.hpp                        Contribution, Result
      fmt/                             FMT models, measures, weights
      bulk/                            thermodynamics, phase_diagram, coexistence
    algorithms/                        fire, picard, dynamics, minimization, saddle_point, string_method
      solvers/                         newton, jacobian, continuation, gmres
    geometry/                          vertex, element, mesh
    config/                            parser
    plotting/                          matplotlib, grace, exceptions
src/
  crystal.cpp
  config/parser.cpp
  math/fourier.cpp  math/spline.cpp
  physics/eos.cpp
tests/
  unit/         41 files, mirrors include/dft/
  integration/  8 files, cross-validates against legacy code
docs/
  19 example programs, each with main.cpp + Makefile + CMakeLists.txt
  Examples: arithmetic, autodiff, config, console, convolution, crystal,
            density, dynamics, fourier, geometry, integration, nucleation,
            picard, potentials, solver, spline, string_method, thermodynamics,
            fmt, interaction
```

Key rule: file names must not repeat the directory name
(`functionals/ideal_gas.hpp`, not `functionals/functionals_ideal_gas.hpp`).

---

## 3. Build commands

### Local (no Docker)

```bash
# Full build
cmake -S . -B build-local \
  -DCMAKE_CXX_COMPILER=g++-15 \
  -DDFT_BUILD_TESTS=ON \
  -DDFT_BUILD_DOCS=OFF
cmake --build build-local --parallel

# Run unit tests
ctest --test-dir build-local --output-on-failure

# Build with docs examples
cmake -S . -B build-local \
  -DCMAKE_CXX_COMPILER=g++-15 \
  -DDFT_BUILD_TESTS=ON \
  -DDFT_BUILD_DOCS=ON \
  -DDFT_USE_MATPLOTLIB=ON
cmake --build build-local --parallel
```

### Docker (CI path)

```bash
make build            # docker compose build
make test             # all tests with coverage
make unit-tests       # unit tests only
make integration-tests
make format           # clang-format
make lint             # format check + clang-tidy
```

### Per-example (from docs/<example>/)

```bash
make -C docs/<example> run-local          # build + run locally
make -C docs/<example> run-checks         # validation (where available)
```

---

## 4. Mandatory verification workflow

Every change to the library (headers, sources, CMakeLists) must pass
this sequence before being considered complete:

### Step 1 — Unit tests

```bash
cmake --build build-local --parallel && ctest --test-dir build-local --output-on-failure
```

All 41 unit + 8 integration tests must pass. If a header rename or API
change breaks includes, fix all dependents before proceeding.

### Step 2 — Build all doc examples

```bash
cmake -S . -B build-local \
  -DCMAKE_CXX_COMPILER=g++-15 \
  -DDFT_BUILD_TESTS=ON \
  -DDFT_BUILD_DOCS=ON \
  -DDFT_USE_MATPLOTLIB=ON
cmake --build build-local --parallel
```

All 19 doc programs must compile. They exercise the public API surface
that users rely on.

### Step 3 — Per-example Makefile smoke test

For each affected example, verify `make -C docs/<example> run-local`
still works. The Makefiles invoke CMake independently with their own
flags.

### Step 4 — Format and lint

```bash
make format
make lint
```

---

## 5. Refactoring checklist

When renaming or moving a header:

1. **grep all includes** — `grep -r 'old_header' include/ src/ tests/ docs/`
2. **Update umbrella header** — `include/dft.hpp` includes every public header
3. **Update header guards** — guard name derives from file path
   (`DFT_` + path in `UPPER_SNAKE_CASE` + `_HPP`)
4. **Update internal includes** — headers that `#include` the renamed file
5. **Update test files** — test includes mirror header paths
6. **Update doc examples** — they use `#include <dftlib>` (umbrella), so
   they compile transitively, but verify they still build
7. **Run verification workflow** (section 4, all 4 steps)

When adding a new header:

1. Add to `include/dft.hpp` in the correct section
2. Create corresponding test in `tests/unit/` (mirror the header path)
3. Register the test file (it is auto-discovered via GLOB_RECURSE)
4. If the header has a `.cpp` implementation, add it to `DFT_LIB_SOURCES`
   in the root `CMakeLists.txt`

When adding a new doc example:

1. Create `docs/<name>/main.cpp`, `CMakeLists.txt`, `Makefile`, `README.md`
2. Add `add_doc(<name> <dir>)` to `docs/CMakeLists.txt`
3. Follow the Makefile template (run, run-local, clean targets)
4. Optionally add `check/main.cpp` + `add_check(<name> <dir>)` for validation

---

## 6. Code conventions (quick reference)

Full details in `CODING_GUIDELINES.md`. Key points:

- **C++23**, no extensions. `-Wall -Wextra -Wpedantic`.
- **Naming**: structs `CamelCase`, functions `lower_snake_case`,
  constants `UPPER_SNAKE_CASE`, namespaces `lower_snake_case`.
- **Namespaces** mirror directories: `dft::functionals::fmt`.
  Collapsed syntax: `namespace dft::physics::fmt { ... }`.
  All content indented inside namespace.
- **No `#pragma once`** — use `#ifndef` / `#define` / `#endif` guards.
- **`[[nodiscard]]`** on every function returning a value.
- **Value semantics** — functions return results, never mutate arguments.
- **`struct`** for data + config (public members, designated initialisers).
  **`class`** only when hiding implementation details serves the user.
- **Armadillo** for all numerical operations (no manual loops when a
  vectorised operation exists).
- **`std::variant`** wrapped in classes for runtime polymorphism.
  Templates for compile-time polymorphism.
- **Banned prefixes**: `compute_`, `get_`, `set_`, `apply_`, `create_`.
- **Banned patterns**: out-parameter mutation, allocate-then-fill,
  shared mutable scratch, `void` functions doing useful work.

### C++23 features to prefer

| Feature                       | Use case                                  |
| ----------------------------- | ----------------------------------------- |
| `std::views::zip`             | Parallel iteration                        |
| `std::print` / `std::println` | Formatted output (pass stream explicitly) |
| `std::expected<T, E>`         | Fallible operations with error info       |
| `std::unreachable()`          | Unreachable default branches              |
| Deducing `this`               | const/non-const overload pairs            |
| Concepts                      | Template parameter constraints            |

**Not yet available** on Apple Clang 17 (do not use):
`std::views::cartesian_product`, `std::views::enumerate`.

---

## 7. Testing conventions

- Catch2 v3, auto-linked via `Catch2::Catch2WithMain`.
- One test file per module, path mirrors header path.
- `TEST_CASE("descriptive phrase", "[tag]")` — lowercase sentence.
- FP tolerances: `1e-14` (analytical), `1e-10` (cross-model),
  `1e-8` (integrated), `1e-5` (numerical derivatives).
- Use `GENERATE(table<...>({...}))` for parameterised tests.
- File-scoped `static` helpers before the tests that use them.

---

## 8. Doc example conventions

- Each lives in `docs/<module>/` with `main.cpp`, `CMakeLists.txt`,
  `Makefile`, `README.md`, optional `exports/` for plots.
- `#ifdef DOC_SOURCE_DIR` to chdir to source directory.
- Plotting behind `#ifdef DFT_HAS_MATPLOTLIB` or `#ifdef DFT_HAS_GRACE`.
- All examples use `#include <dftlib>` umbrella and `using namespace dft`.
- Makefile pattern: `run` (Docker), `run-local` (CMake local), `clean`.
  Some have `run-checks` for validation.

---

## 9. Dependency management

| Dependency     | Method                        | Notes                                |
| -------------- | ----------------------------- | ------------------------------------ |
| Armadillo      | `find_package(REQUIRED)`      | System install                       |
| FFTW3          | `pkg_check_modules(REQUIRED)` | System install                       |
| GSL            | `pkg_check_modules(REQUIRED)` | System install                       |
| nlohmann/json  | `find_package(REQUIRED)`      | System install                       |
| toml++         | `FetchContent` v3.4.0         | Header-only                          |
| autodiff       | `FetchContent` v1.1.2         | Header-only, warnings suppressed     |
| matplotlib-cpp | `FetchContent` (optional)     | `DFT_USE_MATPLOTLIB`                 |
| Catch2         | `FetchContent` v3.7.1         | Test-only                            |
| OpenMP         | `find_package(QUIET)`         | Optional, Homebrew fallback on macOS |

---

## 10. Functionals module file roles

| File | Contents | Role |
|------|----------|------|
| `evaluator.hpp` | `Weights`, `make_weights()`, `make_bulk_weights()`, `total()` | Low-level orchestrator: evaluates the sum of all contributions |
| `functional.hpp` | `Functional` struct, `make_functional()` | High-level facade: owns model + weights, convenience methods |

`functional.hpp` includes `evaluator.hpp`.
