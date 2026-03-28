# classicalDFT style guide

This document is the single source of truth for every coding convention in this
project. All new code, refactored code, examples, and tests must follow it.

---

## 1. Language standard and tooling

| Setting | Value |
|---------|-------|
| Standard | C++20 (`CMAKE_CXX_STANDARD 20`, required, no extensions) |
| Compiler warnings | `-Wall -Wextra -Wpedantic` |
| Formatting | `.clang-format` (Google base, 120 col, 2-space indent) |
| Linting | `.clang-tidy` (see naming table below) |
| Linear algebra | Armadillo (`arma::vec`, `arma::rowvec3`, `arma::mat33`) |
| FFT | FFTW3 |
| Plotting | Grace (`libgrace_np`), guarded by `DFT_HAS_GRACE` |
| Testing | GoogleTest 1.14 (fetched via `FetchContent`) |
| Build | CMake 3.20+, single static library `classicaldft` |

---

## 2. Project layout

```
include/
  classicaldft                          # umbrella header (no extension)
  classicaldft_bits/
    <module>/
      <submodule>/
        <file>.h
src/
  <module>/
    <submodule>/
      <file>.cpp
tests/
  main.cpp                              # single test runner
  <module>/
    <submodule>/
      <file>.cpp                        # one test file per header
examples/
  <module>/
    <submodule>/
      main.cpp
      CMakeLists.txt
      README.md
      exports/                          # Grace plot output
```

Modules: `exceptions`, `geometry`, `graph`, `io`, `numerics`, `physics`.
Physics sub-modules: `crystal`, `density`, `fmt`, `potentials/intermolecular`,
`species`, `thermodynamics`.

Source tree and test tree mirror the header tree exactly.

---

## 3. File naming

- All file names are **`lower_snake_case`**: `measures.h`, `convolution.cpp`.
- One class (or one tightly coupled group) per file.
- Header extension: `.h`. Source extension: `.cpp`.
- File name matches the primary class name in snake_case:
  `ConvolutionField` → `convolution.h`, `WhiteBearII` → `functional.h` (shared).

---

## 4. Header guards

Traditional `#ifndef` / `#define` / `#endif`. No `#pragma once`.

Guard name: `CLASSICALDFT_` + path from `classicaldft_bits/` in `UPPER_SNAKE_CASE` + `_H`.

```cpp
#ifndef CLASSICALDFT_PHYSICS_FMT_MEASURES_H
#define CLASSICALDFT_PHYSICS_FMT_MEASURES_H

// ... contents ...

#endif  // CLASSICALDFT_PHYSICS_FMT_MEASURES_H
```

---

## 5. Include order

Follows the Google C++ style guide, enforced by `.clang-format`:

1. **Corresponding header** (in the `.cpp` file only)
2. **Project headers** (`"classicaldft_bits/..."`)
3. **System / standard library headers** (`<...>`, alphabetised)

Blank line between each group. Example:

```cpp
#include "classicaldft_bits/physics/density/density.h"

#include "classicaldft_bits/numerics/arithmetic.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
```

No forward declarations. Include what you use.

---

## 6. Namespaces

Root namespace: **`dft_core`**. Sub-namespaces mirror the directory structure.

| Directory | Namespace |
|-----------|-----------|
| `physics/fmt/` | `dft_core::physics::fmt` |
| `physics/density/` | `dft_core::physics::density` |
| `physics/species/` | `dft_core::physics::species` |
| `physics/thermodynamics/` | `dft_core::physics::thermodynamics` |
| `numerics/fourier/` | `dft_core::numerics::fourier` |
| `geometry/base/` | `dft_core::geometry` |
| `graph/` | `dft_core::grace_plot` |
| `io/console/` | `dft_core::io::console` |
| `exceptions/` | `dft_core::exception` |

Use C++17 collapsed syntax:

```cpp
namespace dft_core::physics::fmt {

  // ... all code indented 2 spaces ...

}  // namespace dft_core::physics::fmt
```

Two-space gap before `//` in the closing comment.

**All code** inside the namespace is indented (not flush-left).
This is configured via `NamespaceIndentation: All` in `.clang-format`.

---

## 7. Naming conventions

Enforced by `.clang-tidy` `readability-identifier-naming`:

| Entity | Case | Example |
|--------|------|---------|
| Class / struct | `CamelCase` | `Density`, `WhiteBearI`, `ConvolutionField` |
| Function / method | `lower_snake_case` | `compute_forces()`, `set_density_from_alias()` |
| Variable | `lower_snake_case` | `eta`, `rho0`, `diameter` |
| Private / protected member | `lower_snake_case_` (trailing `_`) | `dx_`, `weights_`, `diameter_` |
| Namespace | `lower_snake_case` | `dft_core`, `fmt`, `numerics` |
| Global constant | `UPPER_SNAKE_CASE` | `DEFAULT_LENGTH_SCALE` |
| Static constant | `UPPER_SNAKE_CASE` | `MAX_POTENTIAL_VALUE` |
| Constexpr (file scope) | `UPPER_SNAKE_CASE` | `PI_OVER_6` |
| Constexpr (class scope) | `lower_snake_case` | `rho_min`, `num_independent` |
| Scoped enum values | `CamelCase` | `Direction::X`, `Route::Virial` |

### Naming principles

- No abbreviations unless universally understood in the domain:
  `eta`, `rho`, `mu`, `dx`, `fft`. Spell out everything else:
  `diameter` not `hsd`, `density_range_` not `alias_c_`.
- Method names describe **what**, not **how**: `compute_free_energy()` not
  `calculate_free_energy_and_forces()`.
- Getters are bare nouns: `density()`, `force()`, `values()`, `shape()`.
- Derivative methods use `d_` / `d2_` prefix: `d_f1()`, `d2_f1()`.
- Bulk (homogeneous) quantities: `bulk_free_energy_density()`,
  `bulk_excess_chemical_potential()`.

---

## 8. API design

### Method prefixes

| Prefix | Semantics | Example |
|--------|-----------|---------|
| `compute_` | Perform a calculation, return result | `compute_free_energy()` |
| `set_` | Mutator | `set_chemical_potential()` |
| `clear_` | Remove optional state | `clear_fixed_mass()` |
| `add_to_` | Accumulate into existing buffer | `add_to_force()` |
| `zero_` | Zero out a buffer | `zero_force()` |
| `begin_` / `end_` | Bracket a multi-step protocol | `begin_force_calculation()` |
| (bare noun) | Inspector / getter | `density()`, `values()` |
| `d_` / `d2_` | Mathematical derivative | `d_phi3_d_n2()` |
| `bulk_` | Homogeneous (spatially uniform) limit | `bulk_free_energy_density()` |

### Getter conventions

Return heavy objects by `const&`. Return scalars by value.

```cpp
[[nodiscard]] const arma::vec& values() const noexcept;
[[nodiscard]] double dx() const noexcept;
```

Provide dual `const` / non-`const` overloads when mutable access is needed:

```cpp
[[nodiscard]] const arma::vec& values() const noexcept { return rho_; }
[[nodiscard]] arma::vec& values() noexcept { return rho_; }
```

### Attributes

- **`[[nodiscard]]`** on every function that returns a value.
- **`noexcept`** on simple getters that cannot throw.

### Static factory methods

```cpp
[[nodiscard]] static Measures uniform(double density, double diameter);
```

### Single responsibility

Each public method does **one thing**. Never bundle two logical operations
(e.g. "compute free energy AND accumulate forces") into one method. Split them:

```cpp
double compute_free_energy(const Functional& model);
double compute_forces(const Functional& model);
```

---

## 9. Class design

### `struct` vs `class`

- `struct`: plain-data aggregates with no invariants (`Measures`, `WeightSet`).
- `class`: anything with invariants, RAII, virtual methods, or encapsulation.

### Member ordering

```cpp
class MyClass {
 public:
  // ── Construction ──────────────────────────────────────────────────
  MyClass(args);
  ~MyClass();

  // Rule of 5
  MyClass(const MyClass&) = delete;
  MyClass& operator=(const MyClass&) = delete;
  MyClass(MyClass&&) noexcept = default;
  MyClass& operator=(MyClass&&) noexcept = default;

  // ── Inspectors ────────────────────────────────────────────────────
  [[nodiscard]] const arma::vec& values() const noexcept;

  // ── Mutators ──────────────────────────────────────────────────────
  void set(const arma::vec& v);

  // ── Methods ───────────────────────────────────────────────────────
  double compute_something() const;

 private:
  arma::vec rho_;
  double dx_;
};
```

### Class constants

Use `static constexpr` at class scope (lowercase):

```cpp
static constexpr double rho_min = 1e-18;
static constexpr int num_independent = 11;
```

### Inheritance

- Mark concrete leaf classes **`final`**.
- Virtual destructors: `virtual ~Base() = default;` in abstract bases.
- Prefer Non-Virtual Interface (NVI): public non-virtual methods that call
  private/protected virtual hooks.

```cpp
class Functional {
 public:
  [[nodiscard]] double phi(const Measures& m) const;          // non-virtual (algorithm)
  [[nodiscard]] virtual double f1(double eta) const = 0;      // virtual (model-specific)
};

class Rosenfeld final : public Functional {
 public:
  [[nodiscard]] double f1(double eta) const override;
};
```

### Unused parameters

Suppress warnings with explicit `(void)` cast:

```cpp
[[nodiscard]] virtual double d_phi3_d_T(int i, int j, const Measures& m) const {
  (void)i; (void)j; (void)m;
  return 0.0;
}
```

---

## 10. Armadillo conventions

| Type | Use for |
|------|---------|
| `arma::vec` | 1D arrays: density field, forces, FFT data |
| `arma::rowvec3` | 3D spatial vectors: box size, position |
| `arma::mat33` | 3×3 tensors |
| `arma::mat` | N×3 position arrays (lattice) |
| `arma::uword` | All index types |
| `arma::cx_vec` | Complex FFT output |

API boundaries accept `const arma::vec&` or `const arma::rowvec3&`. Internal
computation uses Armadillo functions directly: `arma::dot()`, `arma::trace()`,
`arma::norm()`, `arma::clamp()`, `arma::log()`.

Do **not** use flat `std::vector<double>` or raw `double*` at public APIs.
Do **not** use flat index enums for tensor components; use proper Armadillo
structures or accessor methods:

```cpp
// Bad: enum Component { VectorX=2, VectorY=3, TensorXX=5 ... };
// Good:
struct WeightSet {
  arma::vec eta;
  arma::vec scalar;
  arma::vec vector[3];
  [[nodiscard]] arma::vec& tensor(int i, int j);
};
```

---

## 11. Error handling

- Validate only at **system boundaries**: constructors, public setters,
  public entry points.
- Internal helper functions do **not** throw.
- Use standard exceptions:

```cpp
throw std::invalid_argument("Density: dx must be positive");
throw std::out_of_range("Index " + std::to_string(i) + " out of range");
```

- Custom exceptions (`dft_core::exception::*`) only for domain-specific failures
  (Grace communication, parameter validation).
- Messages include the offending value when practical.

---

## 12. Formatting

Configured in `.clang-format` (Google base):

| Setting | Value |
|---------|-------|
| Column limit | 120 |
| Indent width | 2 spaces |
| Continuation indent | 4 spaces |
| Access modifier offset | -1 |
| Braces | Attached (K&R) |
| Namespace indentation | All (content is indented) |
| Pointer alignment | Left (`int* p`, not `int *p`) |
| Short functions | Inline only |
| Short if/loops | Never |
| Bin-pack args/params | No (one per line if they don't fit) |
| Include sorting | Case-sensitive, regrouped |

### Section separators

Use Unicode box-drawing characters to fill to ~80 columns:

```cpp
// ── Section title ──────────────────────────────────────────────────────
```

Each logical section of a class, test file, or example gets a separator.

---

## 13. Test conventions

### Runner

Single `tests/main.cpp` entry point. Test files are collected via
`file(GLOB_RECURSE)` in CMake (requires reconfigure for new files).

### Test file structure

```cpp
#include "classicaldft_bits/physics/fmt/measures.h"   // tested header first

#include <cmath>
#include <gtest/gtest.h>

using namespace dft_core::physics::fmt;

// ── Default construction ──────────────────────────────────────────────

TEST(Measures, DefaultConstructionAllZero) {
  Measures m;
  EXPECT_DOUBLE_EQ(m.n0, 0.0);
}

// ── Uniform factory ───────────────────────────────────────────────────

TEST(Measures, UniformEtaIsConsistent) {
  auto m = Measures::uniform(0.5, 1.0);
  EXPECT_NEAR(m.eta, 0.2618, 1e-4);
}
```

### Naming

- `TEST(TestSuite, TestName)` — no fixtures (`TEST_F`) unless strictly needed.
- **TestSuite**: class or concept name in `CamelCase` (`Measures`, `Functional`,
  `ConvolutionField`).
- **TestName**: descriptive `CamelCase` phrase describing the assertion
  (`DefaultConstructionAllZero`, `WhiteBearIMatchesCarnahanStarling`).

### Assertion selection

| Assertion | When to use | Tolerance |
|-----------|-------------|-----------|
| `EXPECT_DOUBLE_EQ` | Exact FP equality (bit-identical) | — |
| `EXPECT_NEAR(a, b, tol)` | Approximate FP comparison | Context-dependent |
| `EXPECT_TRUE` / `EXPECT_FALSE` | Boolean | — |
| `EXPECT_EQ` | Integer, enum, string | — |
| `EXPECT_THROW(expr, type)` | Exception expected | — |

Tolerance guidelines:
- `1e-14`: analytical identities, algebraic simplifications.
- `1e-10`: cross-model validation (e.g. Rosenfeld vs PY).
- `1e-8`: chemical potential, integrated quantities.
- `1e-5` to `1e-6`: numerical derivatives, grid convergence.

### Helper functions

File-scoped `static` free functions, placed before the tests that use them:

```cpp
static double numerical_derivative(std::function<double(double)> f, double x, double h = 1e-6) {
  return (f(x + h) - f(x - h)) / (2.0 * h);
}
```

---

## 14. Example conventions

### File structure

```
examples/<module>/<submodule>/
  main.cpp
  CMakeLists.txt
  README.md
  exports/          # created at runtime for Grace output
```

### CMakeLists.txt (minimal)

```cmake
add_executable(example_<name> main.cpp)
target_link_libraries(example_<name> PRIVATE classicaldft)
```

### main.cpp structure

```cpp
#include <classicaldft>

#include <iomanip>
#include <iostream>

using namespace dft_core::physics::fmt;

int main() {
  std::filesystem::create_directories("exports");

  // ── Section 1 ────────────────────────────────────────────────────
  // Console output with std::setw, std::fixed, std::setprecision

  // ── Section 2 ────────────────────────────────────────────────────
  // More computation ...

  // ── Grace plots ──────────────────────────────────────────────────
#ifdef DFT_HAS_GRACE
  using namespace dft_core::grace_plot;

  {
    auto g = Grace();
    g.set_title("Title");
    g.set_label("\\xh", Axis::X);
    g.set_label("y-label", Axis::Y);

    auto ds = g.add_dataset(x_vec, y_vec);
    g.set_color(Color::BLACK, ds);
    g.set_legend("Legend text", ds);

    g.set_x_limits(0.0, 1.0);
    g.set_y_limits(0.0, 5.0);
    g.set_ticks(0.1, 1.0);
    g.print_to_file("exports/name.png", ExportFormat::PNG);
    g.redraw_and_wait(false, false);
  }
#endif

  return 0;
}
```

- Each Grace plot in its own scoped block `{ }`.
- All plotting behind `#ifdef DFT_HAS_GRACE`.
- Console tables use `std::setw` for aligned columns.

---

## 15. CMake conventions

- Options prefixed `DFT_`: `DFT_BUILD_TESTS`, `DFT_BUILD_EXAMPLES`,
  `DFT_USE_GRACE`, `DFT_CODE_COVERAGE`.
- Library sources listed **explicitly** (no `GLOB`) for the static library.
- Test sources use `file(GLOB_RECURSE)`.
- External dependencies: `find_package` for system libraries, `FetchContent`
  for GoogleTest.
- Compile definition `DFT_HAS_GRACE` propagated as `PUBLIC` when Grace is found.
- Warnings applied via generator expression (GNU/Clang/AppleClang only).
- Export `compile_commands.json` (`CMAKE_EXPORT_COMPILE_COMMANDS ON`).

---

## 16. Design principles

### Single responsibility

Every class, method, and file has exactly one reason to change. Do not bundle
unrelated operations. If a method name contains "and", split it.

### Open/closed

Extend behaviour through inheritance (new `Functional` subclass) or composition,
not by modifying existing classes.

### Liskov substitution

Every `Functional` subclass is interchangeable. They share the same public
interface and differ only in the virtual hooks (`f1`, `f2`, `f3`, `phi3`).

### Interface segregation

Keep interfaces minimal. A `Species` does not expose the full `Density` API,
only the operations relevant to species behaviour.

### Dependency inversion

High-level algorithms (`fmt::Species::compute_free_energy`) depend on abstract
interfaces (`Functional&`), not on concrete models.

### No premature abstraction

Do not create helpers, utilities, or wrapper classes for one-time operations.
Do not add layers "for future use".

### No dead code

No commented-out code, no unused includes, no methods that are never called.

### Const correctness

- Mark every method that does not mutate state as `const`.
- Pass large objects by `const&`.
- Use `const` local variables when the value does not change.
