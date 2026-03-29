# classicalDFT modern style guide (cdft)

This document describes the conventions for all code under `include/cdft/`,
`src/cdft/`, `tests/cdft/`, and `examples/cdft/`. It supersedes
`CODING_GUIDELINES.md` for those directories.

---

## 1. Language standard and tooling

| Setting | Value |
|---------|-------|
| Standard | C++20 (`CMAKE_CXX_STANDARD 20`, required, no extensions) |
| Compiler warnings | `-Wall -Wextra -Wpedantic` |
| Linear algebra | Armadillo (`arma::vec`, `arma::rowvec3`, `arma::mat33`) |
| FFT | FFTW3 |
| Autodiff | `autodiff` v1.1.2 (forward-mode, fetched via `FetchContent`) |
| JSON | nlohmann_json 3.11+ |
| Testing | GoogleTest 1.14 (fetched via `FetchContent`) |
| Build | CMake 3.20+, single static library `classicaldft` |

---

## 2. Project layout

```
include/
  cdft.hpp                              # umbrella header
  cdft/
    core/
      types.hpp                         # common aliases (Vector3, Matrix33)
      grid.hpp                          # grid utilities
    physics/
      eos.hpp                           # equation-of-state models
      potentials.hpp                    # pair potentials
      crystal.hpp                       # lattice generation
    functional/
      fmt.hpp                           # FMT model structs + free functions
      density.hpp                       # DensityField, Species
      species.hpp                       # FMTSpecies, WeightedDensity
      interaction.hpp                   # mean-field Interaction
    numerics/
      math.hpp                          # GSL integration wrappers
      fourier.hpp                       # FourierTransform, FourierConvolution
      spline.hpp                        # CubicSpline, BivariateSpline
      autodiff.hpp                      # autodiff bridge (dual aliases)
    io/
      config.hpp                        # ConfigParser (INI/JSON)
    viz/
      plot.hpp                          # plotting utilities
src/
  cdft/
    <mirrors include/cdft/ tree>
tests/
  cdft/
    <mirrors include/cdft/ tree, one test file per header>
examples/
  cdft/
    CMakeLists.txt
    eos.cpp
    potentials.cpp
    fmt.cpp
    crystal.cpp
    numerics.cpp
    config.cpp
```

Source tree and test tree mirror the header tree exactly.

---

## 3. File naming

- All file names are **`lower_snake_case`**: `fmt.hpp`, `eos.cpp`.
- Header extension: **`.hpp`**. Source extension: `.cpp`.
- One header per cohesive module (not per class).

---

## 4. Header guards

Use **`#pragma once`**. No `#ifndef` / `#define` / `#endif`.

```cpp
#pragma once

#include <armadillo>
// ...
```

---

## 5. Include order

1. **Corresponding header** (in `.cpp` files only)
2. **Project headers** (`"cdft/..."`)
3. **Third-party headers** (`<armadillo>`, `<nlohmann/json.hpp>`, `<autodiff/...>`)
4. **Standard library headers** (`<algorithm>`, `<cmath>`, `<string>`)

Blank line between each group.

---

## 6. Namespaces

Root namespace: **`cdft`**. Sub-namespaces mirror the directory structure.

| Directory | Namespace |
|-----------|-----------|
| `core/` | `cdft` |
| `physics/` | `cdft::physics` |
| `physics/` (transport) | `cdft::physics::transport` |
| `functional/` | `cdft::functional` |
| `numerics/` | `cdft::numerics` |
| `io/` | `cdft::io` |
| `viz/` | `cdft::viz` |

Use C++17 collapsed syntax:

```cpp
namespace cdft::physics {

  // ... all code indented 2 spaces ...

}  // namespace cdft::physics
```

---

## 7. Naming conventions

| Entity | Case | Example |
|--------|------|---------|
| Class / struct | `CamelCase` | `DensityField`, `WhiteBearII`, `FMTSpecies` |
| Function / method | `lower_snake_case` | `compute_free_energy()`, `excess_free_energy()` |
| Variable | `lower_snake_case` | `eta`, `rho`, `diameter` |
| Private member | `lower_snake_case_` (trailing `_`) | `spacing_`, `weights_` |
| Namespace | `lower_snake_case` | `cdft`, `physics`, `functional` |
| Global / static constant | `UPPER_SNAKE_CASE` | `RHO_C`, `MAX_VALUE`, `TERMS` |
| Scoped enum values | `CamelCase` | `CrystalStructure::BCC`, `ConfigFormat::JSON` |

### Physics notation

In physics-specific code, standard physics symbols are preferred over verbose
names: `f1`, `f2`, `f3` for FMT factor functions; `phi_3`, `n0`, `n1`, `n2`,
`v1`, `v2`, `T` for weighted densities; `eta` for packing fraction; `chi` for
contact value.

---

## 8. Core design patterns

### 8.1 Flat structs + free functions (no virtual dispatch)

Models are plain structs with static or const-member functions. No base class,
no virtual methods, no CRTP.

```cpp
struct CarnahanStarling {
  template <typename T = double>
  [[nodiscard]] static T excess_free_energy(T eta) {
    T e = T(1.0) - eta;
    return eta * (T(4.0) - T(3.0) * eta) / (e * e);
  }
  [[nodiscard]] static std::string name() { return "CarnahanStarling"; }
};
```

### 8.2 `std::variant` as sum type (replaces inheritance hierarchies)

Related model structs are gathered into a `std::variant`:

```cpp
using HardSphereModel = std::variant<CarnahanStarling, PercusYevickVirial, PercusYevickCompressibility>;
using FMTModel = std::variant<Rosenfeld, RSLT, WhiteBearI, WhiteBearII>;
using PairPotential = std::variant<LennardJones, TenWoldeFrenkel, WangRamirezDobnikarFrenkel>;
using EquationOfState = std::variant<IdealGas, PercusYevickEOS, LennardJonesJZG, LennardJonesMecke>;
```

### 8.3 Free-function dispatch via `std::visit`

Derived quantities are free functions that dispatch through the variant:

```cpp
[[nodiscard]] inline double hs_pressure(const HardSphereModel& model, double eta) {
  auto [f, df] = std::visit([eta](const auto& m) {
    return derivatives_up_to_1([&](dual x) { return m.excess_free_energy(x); }, eta);
  }, model);
  return 1.0 + eta * df;
}
```

Free functions are named with a module prefix when needed for disambiguation:
`hs_pressure`, `eos_pressure`, `fmt_phi`, `fmt_bulk_free_energy_density`.

### 8.4 Template<T> for automatic differentiation

All scalar free-energy functions are templated on `typename T = double` so that
the same code path works with `double`, `dual`, `dual2nd`, and `dual3rd`:

```cpp
template <typename T = double>
[[nodiscard]] static T excess_free_energy(T eta) { ... }
```

Hand-coded derivative methods (`d_`, `d2_`, `d3_`) are eliminated. Derivatives
are obtained at call sites via `derivatives_up_to_1()` or
`derivatives_up_to_3()`.

### 8.5 Autodiff bridge

The bridge header (`numerics/autodiff.hpp`) provides:

```cpp
namespace cdft {
  using dual    = autodiff::dual;      // first-order
  using dual2nd = autodiff::dual2nd;   // second-order
  using dual3rd = autodiff::dual3rd;   // third-order

  // Returns {f(x), f'(x)}
  template <typename F>
  auto derivatives_up_to_1(F&& f, double x);

  // Returns {f(x), f'(x), f''(x), f'''(x)}
  template <typename F>
  auto derivatives_up_to_3(F&& f, double x);
}
```

When using autodiff with expression-template functions (e.g. `exp`, `log`),
specify the return type explicitly:

```cpp
cdft::derivatives_up_to_3(
    [](cdft::dual3rd x) -> cdft::dual3rd { return exp(x); },
    1.0
);
```

---

## 9. API design

### Method prefixes

| Prefix | Semantics | Example |
|--------|-----------|---------|
| `compute_` | Perform a calculation, return result | `compute_free_energy()` |
| `set_` | Mutator | `set_density_from_alias()` |
| `find_` | Search / iterative solve | `find_hard_sphere_diameter()` |
| `compute_` | Integration / numerical | `compute_van_der_waals_integral()` |
| (bare noun) | Inspector / getter | `density()`, `values()`, `shape()` |
| `bulk_` | Homogeneous (spatially uniform) limit | `bulk_excess_free_energy()` |

### Free-function prefixes for variant dispatch

| Module | Prefix | Example |
|--------|--------|---------|
| Hard-sphere EOS | `hs_` | `hs_pressure()`, `hs_chemical_potential()` |
| Full EOS | `eos_` | `eos_pressure()`, `eos_temperature()` |
| FMT | `fmt_` | `fmt_phi()`, `fmt_name()`, `fmt_bulk_free_energy_density()` |

### Attributes

- **`[[nodiscard]]`** on every function that returns a value.
- **`noexcept`** on simple getters that cannot throw.

### Getter conventions

Return heavy objects by `const&`. Return scalars by value. Provide dual
`const` / non-`const` overloads when mutable access is needed.

---

## 10. `struct` vs `class`

- **`struct`**: model types (EOS, FMT factors, potentials), measures, config
  types, result types. Public by default, no invariants to enforce.
- **`class`**: types with internal state that must be maintained
  (`DensityField`, `FMTSpecies`, `FourierTransform`, `CubicSpline`).

### Move semantics

Classes owning non-copyable resources (FFTW plans, GSL accelerators) are
move-only:

```cpp
MyClass(const MyClass&) = delete;
MyClass& operator=(const MyClass&) = delete;
MyClass(MyClass&&) noexcept = default;
MyClass& operator=(MyClass&&) noexcept = default;
```

---

## 11. Armadillo conventions

| Type | Use for |
|------|---------|
| `arma::vec` | 1D arrays: density field, forces, FFT data |
| `arma::rowvec3` | 3D spatial vectors: box size, position |
| `arma::mat33` | 3x3 tensors |
| `arma::mat` | N x 3 position arrays (lattice) |
| `arma::uword` | All index types |

API boundaries accept `const arma::vec&` or `const arma::rowvec3&`. Common
type aliases are provided in `core/types.hpp`:

```cpp
using Vector3 = arma::rowvec3;
using Matrix33 = arma::mat33;
using RealVector = arma::vec;
```

---

## 12. Error handling

- Validate only at **system boundaries**: constructors, public setters, public
  entry points.
- Use standard exceptions: `std::invalid_argument`, `std::out_of_range`.
- Internal helper functions do not throw.

---

## 13. Formatting

| Setting | Value |
|---------|-------|
| Column limit | 120 |
| Indent width | 2 spaces |
| Braces | Attached (K&R) |
| Namespace indentation | All (content is indented) |
| Pointer alignment | Left (`int* p`) |

### Section separators

```cpp
// ── Section title ──────────────────────────────────────────────────────
```

---

## 14. Test conventions

### Structure

Test files live under `tests/cdft/` and mirror the header tree. Single
`tests/main.cpp` entry point; test files are collected via `file(GLOB_RECURSE)`.

### Naming

- `TEST(TestSuite, TestName)` with `CamelCase` for both.
- TestSuite = class or concept name.
- TestName = descriptive assertion phrase.

### Tolerance guidelines

| Tolerance | Use case |
|-----------|----------|
| `1e-14` | Analytical identities |
| `1e-12` | Autodiff vs analytical (dual arithmetic noise) |
| `1e-10` | Cross-model validation |
| `1e-8` | Integrated quantities |
| `1e-6` | Numerical derivatives, grid convergence |

---

## 15. Example conventions

Examples live under `examples/cdft/`, each as a single `.cpp` file. A single
`CMakeLists.txt` in the directory registers all targets:

```cmake
add_executable(cdft_eos eos.cpp)
target_link_libraries(cdft_eos PRIVATE classicaldft)
```

### Style

- Include `<cdft.hpp>` (umbrella header).
- Use `using namespace cdft::physics;` locally in `main()` for readability.
- Structured bindings for autodiff results: `auto [f, df] = ...`.
- Tabular `std::cout` output with `std::setw` and section separators.
- No OOP ceremony: construct structs directly, call free functions.

---

## 16. Summary of differences from legacy style

| Aspect | Legacy (`classicaldft_bits/`) | Modern (`cdft/`) |
|--------|------|---------|
| Header extension | `.h` | `.hpp` |
| Header guard | `#ifndef` / `#define` | `#pragma once` |
| Root namespace | `dft_core` / `dft` | `cdft` |
| Polymorphism | Virtual inheritance | `std::variant` + `std::visit` |
| Model structs | Classes with virtual methods | Flat structs, static functions |
| Derivatives | Hand-coded `d_`, `d2_`, `d3_` methods | `template<T>` + autodiff |
| Free functions | Rare | Primary API surface |
| Umbrella header | `classicaldft` (no extension) | `cdft.hpp` |
| Examples | One directory per example | Single directory, one `.cpp` per topic |
