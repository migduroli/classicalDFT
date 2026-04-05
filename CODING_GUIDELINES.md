# Modern classicalDFT style guide

This document is the single source of truth for every coding convention in this
project. All new code, refactored code, docs, and tests must follow it.

The guiding principle is **readability through simplicity**: code should read
almost like a high-level scripting language while retaining zero-cost C++23
abstractions. Data flows in, results flow out. No hidden mutation, no shared
mutable state, no control inversion. Every function signature tells the
complete story of what it needs and what it produces.

For comments, we try to keep the code as self-explanatory as possible. If comments are needed, never use complicated box-drawing characters that visually break up the code into sections (e.g. `// ── Section title ──`). Instead, use simple sentences `// Section title` with a blank line before (keep it as close as possible to the code it describes). The code itself should be the primary guide to understanding, with comments only for clarifications.

---

## 1. Language standard and tooling

| Setting | Value |
|---------|-------|
| Standard | C++23 (`CMAKE_CXX_STANDARD 23`, required, no extensions) |
| Compiler warnings | `-Wall -Wextra -Wpedantic` |
| Formatting | `.clang-format` (Google base, 120 col, 2-space indent) |
| Linting | `.clang-tidy` (see naming table below) |
| Linear algebra | Armadillo (`arma::vec`, `arma::rowvec3`, `arma::mat33`) |
| FFT | FFTW3 |
| Plotting | matplotlib-cpp (primary, `DFT_HAS_MATPLOTLIB`), Grace (`DFT_HAS_GRACE`, optional fallback) |
| Testing | Catch2 v3 (fetched via `FetchContent`) |
| Build | CMake 3.20+, single static library `classicaldft` |

---

## 2. Project layout

```
include/
  dft.hpp                                  # umbrella header
  dft/
    init.hpp                               # convenience state factories
    console.hpp                            # terminal formatting utilities
    exceptions.hpp                         # general + math exceptions
    grid.hpp                               # lightweight DFT grid struct
    types.hpp                              # Density, Species, SpeciesState, State, Crystal types
    math/
      fourier.hpp                          # FFTW3 RAII wrapper
      convolution.hpp                      # FFT-based convolution
      arithmetic.hpp                       # compensated summation
      spline.hpp                           # cubic spline (GSL)
      integration.hpp                      # numerical integration
      autodiff.hpp                         # autodiff adapter
      types.hpp                            # HessianOperator, module-scoped types
    physics/
      potentials.hpp                       # variant-based potentials
      interactions.hpp                     # Interaction spec struct
      model.hpp                            # Model aggregate
      hard_spheres.hpp                     # hard-sphere thermodynamics + transport
      eos.hpp                              # equations of state
    functionals/
      ideal_gas.hpp                        # ideal gas contribution
      hard_sphere.hpp                      # FMT hard sphere contribution
      mean_field.hpp                       # mean-field interaction contribution
      external_field.hpp                   # external field contribution
      functionals.hpp                      # orchestrator + Result struct
      fmt/
        models.hpp                         # FMT model structs + free fns
        measures.hpp                       # fundamental measures struct
        weights.hpp                        # weight generation
      bulk/
        thermodynamics.hpp                 # pressure, chemical potential, etc.
        phase_diagram.hpp                  # continuation-based coexistence/spinodal
    algorithms/
      alias.hpp                            # variant-based alias transforms
      fire.hpp                             # FIRE2 minimizer
      split_operator.hpp                   # DDFT split-operator integration
      crank_nicholson.hpp                  # DDFT Crank-Nicholson integration
      solvers/
        newton.hpp                         # generic Newton-Raphson
        jacobian.hpp                       # numerical Jacobian
        continuation.hpp                   # pseudo-arclength continuation
    geometry/
      vertex.hpp                           # value-type vertex
      element.hpp                          # variant-based elements (2D/3D)
      mesh.hpp                             # variant-based meshes (2D/3D)
    config/
      parser.hpp                           # configuration file parser
    plotting/
      exceptions.hpp                       # Grace exceptions
      grace.hpp                            # xmgrace plotting
src/
  <mirrors include/dft/ — only for non-inline implementations>
tests/
  <mirrors include/dft/ — one test file per module>
docs/
  <module>/
    main.cpp
    CMakeLists.txt
    Makefile
    README.md
    exports/                               # plot output
```

Top-level modules: `math`, `physics`, `functionals`, `algorithms`,
`geometry`, `config`, `plotting`. Root-level headers: `grid.hpp`,
`types.hpp`, `console.hpp`, `exceptions.hpp`, `init.hpp`.

Source tree, test tree, and doc tree mirror the header tree.
File names must not repeat the directory name.

---

## 3. File naming

- All file names are **`lower_snake_case`**: `measures.hpp`, `convolution.cpp`.
- One struct (or one tightly coupled group) per file.
- Header extension: `.hpp`. Source extension: `.cpp`.
- File name matches the primary type name in snake_case:
  `FmtWeightSet` → `weights.hpp`, `WhiteBearII` → `models.hpp` (shared).

### `types.hpp` convention

Shared vocabulary types (structs, enums, type aliases) that are used across
modules live in `include/dft/types.hpp` under `namespace dft`. Module-scoped
types that are only used within a single module live in
`include/dft/<module>/types.hpp` under the module's namespace. This avoids
proliferating tiny single-struct headers while keeping a clear boundary
between library-wide and module-local definitions.

---

## 4. Header guards

Traditional `#ifndef` / `#define` / `#endif`. No `#pragma once`.

Guard name: `DFT_` + path from `dft/` in `UPPER_SNAKE_CASE` + `_HPP`.

```cpp
#ifndef DFT_PHYSICS_FMT_MEASURES_HPP
#define DFT_PHYSICS_FMT_MEASURES_HPP

// ... contents ...

#endif  // DFT_PHYSICS_FMT_MEASURES_HPP
```

Root-level headers use just `DFT_` + filename:

```cpp
#ifndef DFT_INIT_HPP
#define DFT_INIT_HPP
// ...
#endif  // DFT_INIT_HPP
```

---

## 5. Include order

Follows the Google C++ style guide, enforced by `.clang-format`:

1. **Corresponding header** (in the `.cpp` file only)
2. **Project headers** (`"dft/..."`)
3. **System / standard library headers** (`<...>`, alphabetised)

Blank line between each group. Example:

```cpp
#include "dft/types.hpp"

#include "dft/math/arithmetic.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
```

No forward declarations. Include what you use.

---

## 6. Namespaces

Root namespace: **`dft`**. Sub-namespaces mirror the directory structure.
Root-level headers (`grid.hpp`, `types.hpp`) live directly in `namespace dft`.

| Directory | Namespace |
|-----------|-----------|
| `math/` | `dft::math` |
| `physics/` | `dft::physics` |
| `physics/potentials.hpp` | `dft::physics::potentials` |
| `physics/hard_spheres.hpp` | `dft::physics::hard_spheres` |
| `functionals/` | `dft::functionals` |
| `functionals/fmt/` | `dft::functionals::fmt` |
| `functionals/bulk/` | `dft::functionals::bulk` |
| `algorithms/` | `dft::algorithms` |
| `algorithms/solvers/` | `dft::algorithms::solvers` |
| `algorithms/solvers/continuation.hpp` | `dft::algorithms::continuation` |
| `geometry/` | `dft::geometry` |
| `config/` | `dft::config` |
| `plotting/` | `dft::plotting` |
| `init.hpp` | `dft::init` |
| `console.hpp` | `dft::console` |
| `exceptions.hpp` | `dft::exception` |

Use C++17 collapsed syntax:

```cpp
namespace dft::physics::fmt {

  // ... all code indented 2 spaces ...

}  // namespace dft::physics::fmt
```

Two-space gap before `//` in the closing comment.

**All code** inside the namespace is indented (not flush-left).
This is configured via `NamespaceIndentation: All` in `.clang-format`.

---

## 7. Naming conventions

Enforced by `.clang-tidy` `readability-identifier-naming`:

| Entity | Case | Example |
|--------|------|---------|
| Struct | `CamelCase` | `Density`, `WhiteBearI`, `Fire` |
| Free function | `lower_snake_case` | `free_energy()`, `forces()` |
| Variable | `lower_snake_case` | `eta`, `rho0`, `diameter` |
| Struct member | `lower_snake_case` (public, no trailing `_`) | `dx`, `weights`, `diameter` |
| Namespace | `lower_snake_case` | `dft`, `fmt`, `solvers` |
| Global constant | `UPPER_SNAKE_CASE` | `DEFAULT_LENGTH_SCALE` |
| Static constant | `UPPER_SNAKE_CASE` | `MAX_POTENTIAL_VALUE` |
| Constexpr (file scope) | `UPPER_SNAKE_CASE` | `PI_OVER_6` |
| Scoped enum values | `CamelCase` | `Direction::X`, `Route::Virial` |
| Concept | `CamelCase` | `VectorFunction`, `JacobianFunction` |

### Naming principles

- No abbreviations unless universally understood in the domain:
  `eta`, `rho`, `mu`, `dx`, `fft`. Spell out everything else:
  `diameter` not `hsd`, `density` not `dens`.
- Function names are verbs or physics quantity nouns: `free_energy()`,
  `pressure()`, `forces()`, `fire()`, `trace()`.
- No abbreviation-heavy compound names.
- Derivative methods use `d_` / `d2_` prefix: `d_f1()`, `d2_f1()`.
- `make_` prefix ONLY for validated factory functions that enforce
  invariants: `make_grid()`. Do not use `make_` for simple allocation
  that is immediately followed by a second fill step (see anti-patterns).

---

## 8. API design

### Banned prefixes

These prefixes are **forbidden** in the public API:

| Banned prefix | Replacement |
|---------------|-------------|
| `compute_` | Direct verb or noun: `free_energy()`, `forces()` |
| `get_` | Public field access: `state.temperature` |
| `set_` | Public field assignment: `state.temperature = 1.0` |
| `apply_` | Direct verb: `forces()`, `constrain_mass()` |
| `create_` on data | Designated initialiser: `Grid{.dx = 0.1, ...}` |
| `begin_` / `end_` | Not needed (no multi-step protocols) |
| `add_to_` / `zero_` | Not needed (no mutable buffers) |
| `bulk_` prefix on functions | Use `functionals::bulk::` namespace instead |

### Data is public (for pure data types)

Pure data types (`Grid`, `Density`, `Species`, `State`) and algorithm
configurations (`Fire`, `Picard`, `Newton`) are `struct` with all-public
members and designated-initialiser construction. No getters, no setters.

Types that own implementation details (FFT weights, scratch buffers) use
`class` with private members. See **Encapsulation** below.

### Types own their behavior

When a free function always takes a struct/class as its first argument, that
function is a method on the type. No orphan free functions that could be
methods.

```cpp
// Good: behavior lives on the type
struct LennardJones {
  double sigma{1.0};
  double epsilon{1.0};
  double r_cutoff{-1.0};

  [[nodiscard]] auto energy(double r) const -> double;
  [[nodiscard]] auto attractive(double r, SplitScheme) const -> double;
};

// Bad: orphan free function
[[nodiscard]] auto energy(const LennardJones& lj, double r) -> double;
```

### Encapsulation via `class` when it serves the user

Use `class` with private members when hiding implementation details
genuinely simplifies the API. The user should never have to create weights,
sync internal values, or manage scratch buffers.

```cpp
// Good: user never sees weights, FFT buffers, or a_vdw sync
template <typename FMT>
class Functional {
 public:
  explicit Functional(physics::Model model);

  [[nodiscard]] auto evaluate(const State&) const -> Result;
  [[nodiscard]] auto bulk() const -> BulkThermodynamics;

 private:
  physics::Model model_;
  Weights weights_;        // hidden implementation detail
  Weights bulk_weights_;   // auto-synced in constructor
};

// Bad: user manually wires everything
auto weights = make_weights(fmt_model, model);
auto bulk_weights = make_bulk_weights(fmt_model, model.interactions, kT);
bulk_weights.mean_field.interactions[0].a_vdw = weights.mean_field.interactions[0].a_vdw;
```

### Templates for compile-time polymorphism

Use templates when the type choice is known at compile time. This eliminates
hot-path `std::visit` dispatch and yields better compiler optimisation.

```cpp
// Good: FMT model known at compile time — no visit overhead in hot loop
template <typename FMT>
auto evaluate_hard_sphere(const FMT& model, const Grid&, const State&, ...) -> Contribution;

// Acceptable: variant for runtime choices (config-file driven)
using Potential = std::variant<LennardJones, TenWoldeFrenkel, ...>;
auto energy(const Potential& pot, double r) -> double {
  return std::visit([r](const auto& p) { return p.energy(r); }, pot);
}
```

### Algorithm structs own their methods

When a group of free functions all receive the same configuration struct as
their first or last parameter, those functions become `const` methods on the
struct. The struct retains all-public members and designated-initialiser
construction. Methods are `[[nodiscard]]` and `const` (they do not modify
the struct's state). Drop the `Config` suffix from the struct name — the
struct IS the algorithm, not just its configuration.

```cpp
// Good: algorithm struct with methods
struct Fire {
  double dt{1e-3};
  double force_tolerance{0.1};
  int max_steps{10000};

  [[nodiscard]] auto minimize(
      std::vector<arma::vec> x0, const ForceFunction& compute
  ) const -> FireState;
};

// Usage: config IS the algorithm
auto fire = Fire{.dt = 0.01, .force_tolerance = 1e-6};
auto result = fire.minimize(x0, compute);

// Bad: C-style config + free function
struct FireConfig { ... };
auto result = minimize(x0, compute, config);  // OBSOLETE
```

Pure data types (`Grid`, `Density`, `Species`, `State`) remain data-only
structs with no methods. The rule applies only to types whose primary
purpose is configuring an algorithm or solver.

### Free functions for truly stateless logic

Logic that genuinely operates on two or more unrelated types (no single
"owner") lives in free functions. Functions never mutate their arguments.
If a function always takes one type as its first argument, it should be a
method on that type instead.

```cpp
// Data
struct Grid {
  double dx;
  std::array<double, 3> box_size;
  std::array<long, 3> shape;
};

// Logic (no config needed)
[[nodiscard]] auto atom_count(const Grid&, const Density&) -> double;
```

### Value semantics (no mutation)

Functions return new values. Algorithm methods take working data by value
(they own a local working copy) and return the final result. RVO and
`std::move` eliminate copies.

Every function that produces a result **must** return it. Never pass an
object by mutable reference to be filled internally. The caller should not
need to read the function body to understand what is being created or
modified. The function signature must tell the full story.

```cpp
// Good: const method takes State by value, returns Solution
[[nodiscard]] auto minimize(std::vector<arma::vec> x0, const ForceFunction& compute) const -> FireState;

// Bad: mutates argument in place
void minimize(State&, const Model&);
```

```cpp
// Good: single factory that allocates and populates
[[nodiscard]] auto generate_weights(double diameter, const Grid& grid) -> WeightSet;

// Bad: two-step allocate-then-fill via mutable reference
auto ws = make_weight_set(grid);
generate_weights(diameter, grid, ws);  // hidden mutation
```

```cpp
// Good: returns the convolution result
[[nodiscard]] auto convolve(weight_k, rho_k, shape) -> arma::vec;

// Bad: requires caller to manage a scratch buffer
auto result = convolve(weight_k, rho_k, scratch);  // scratch mutated as side effect
```

### Designated initialisers for all struct construction

```cpp
physics::Model model{
    .grid = core::make_grid(0.1, {10.0, 10.0, 10.0}),
    .species = {core::Species{.name = "Argon", .hard_sphere_diameter = 1.0}},
    .interactions = {},
    .fmt = physics::fmt::Rosenfeld{},
};
```

### `std::variant` for runtime closed sets — with wrapper classes

Runtime polymorphism for closed type sets uses `std::variant`, but raw
variants are never exposed in the public API. Instead, wrap the variant in
a class that exposes the shared interface as regular methods:

```cpp
// Concrete types — struct with public data and methods
struct LennardJones {
  double sigma{1.0};
  double epsilon{1.0};
  double r_cutoff{-1.0};
  // ...
  [[nodiscard]] auto energy(double r) const -> double;
  [[nodiscard]] auto vdw_integral(double kT, SplitScheme) const -> double;
};

// Wrapper class — hides variant, exposes unified interface
class Potential {
 public:
  template <typename T>
  Potential(T concrete) : data_(std::move(concrete)) {}

  [[nodiscard]] auto energy(double r) const -> double {
    return std::visit([r](const auto& p) { return p.energy(r); }, data_);
  }

  [[nodiscard]] auto vdw_integral(double kT, SplitScheme s) const -> double {
    return std::visit([kT, s](const auto& p) { return p.vdw_integral(kT, s); }, data_);
  }

  // Access underlying variant (for rare cases needing type-specific logic)
  [[nodiscard]] auto variant() const -> const auto& { return data_; }

 private:
  std::variant<LennardJones, TenWoldeFrenkel, WangRamirezDobnikarFrenkel> data_;
};

// Usage — no std::visit at call sites
double a = 2.0 * inter.potential.vdw_integral(kT, scheme);
double d = inter.potential.hard_sphere_diameter(kT, scheme);
```

**Rules for variant wrapper classes:**

- The wrapper class owns a `std::variant` as a private member.
- Every method shared by all alternatives becomes a public method on the
  wrapper that internally dispatches via `std::visit`.
- Construction from any alternative is implicit (converting constructor).
- A `variant()` accessor exposes the underlying variant for rare cases
  that need type-specific inspection (e.g. `NAME`, `NEEDS_TENSOR`).
- Callers never write `std::visit` — the wrapper handles it.
- Concrete types remain plain `struct` with designated-initialiser
  construction.

Use templates instead of variants when the type is known at compile time
(e.g. FMT model choice in `Functional<FMT>`).

### C++20 concepts for generic solvers

Use concepts for constraining template parameters in the generic solver
layer:

```cpp
template <typename F>
concept VectorFunction = requires(F f, const arma::vec& x) {
  { f(x) } -> std::convertible_to<arma::vec>;
};

template <VectorFunction Func>
[[nodiscard]] auto Newton::solve(arma::vec x, Func&& f) const -> SolverResult;
```

### C++23 idioms

Prefer these C++23 features when they improve readability:

| Feature | When to use | Example |
|---------|------------|---------|
| `std::views::zip` | Parallel iteration over two or more ranges | `for (auto [a, b] : std::views::zip(xs, ys))` |
| `std::print` / `std::println` | Formatted output (replaces `iostream` insertion chains) | `std::println("iter={} norm={}", k, norm)` |
| `std::expected<T, E>` | Fallible operations where the error carries information | `auto result = parse(...) -> std::expected<Config, std::string>` |
| `std::unreachable()` | Unreachable `default:` branches in exhaustive switches | `default: std::unreachable();` |
| Deducing `this` | Simplifying const/non-const overload pairs | `auto&& operator[](this auto&& self, int i)` |
| Multidimensional `operator[]` | Tensor or matrix subscript | `auto operator[](int i, int j) -> T&` |

When using `std::print`/`std::println` on redirectable streams,
always pass the stream explicitly: `std::print(std::cout, "{}", msg)`.
The no-argument overload writes to `stdout` directly and bypasses
`std::cout` rdbuf redirects.

Features **not yet available** on Apple Clang 17 (do not use):
`std::views::cartesian_product`, `std::views::enumerate`.

### Attributes

- **`[[nodiscard]]`** on every function that returns a value.
- **`noexcept`** on trivial `constexpr` member functions.

### `std::optional` and `std::expected` for fallible operations

Return `std::optional` when a computation may not produce a result:

```cpp
[[nodiscard]] auto coexistence(const Model&, double T, double rho_v, double rho_l)
    -> std::optional<PhasePoint>;
```

Prefer `std::expected<T, E>` when the caller needs to know why it failed:

```cpp
[[nodiscard]] auto parse_config(const std::string& path)
    -> std::expected<Config, std::string>;
```

### Callbacks

Use `std::function` for user-provided callbacks:

```cpp
using StepCallback = std::function<bool(long step, double energy, double max_force)>;
```

### Single responsibility

Each function does **one thing**. Never bundle two logical operations into one
function. If a function name contains "and", split it.

---

## 9. Type design

### `struct` for data and configuration

Pure data types and algorithm configurations are `struct` with all members
public. Validation happens at system boundaries via factory functions or at
API entry points.

```cpp
// Pure data — struct
struct Density {
  arma::vec values;
  arma::vec external_field;
};

// Algorithm configuration — struct
struct Fire {
  double dt{1e-3};
  double force_tolerance{0.1};
  int max_steps{10000};

  [[nodiscard]] auto minimize(...) const -> FireState;
};
```

### `class` for types with implementation details

Types that own non-trivial implementation details (FFT weights, internal
buffers, synced state) use `class` with private members and a public API.
This is natural C++ — use the language as designed.

```cpp
// Hides weight construction, a_vdw sync, and scratch management
template <typename FMT>
class Functional {
 public:
  explicit Functional(physics::Model model);
  [[nodiscard]] auto evaluate(const State&) const -> Result;
 private:
  physics::Model model_;
  Weights weights_;
};
```

### Physics types: `struct` with methods

Physics types (`LennardJones`, `CarnahanStarling`, `WhiteBearII`, etc.) are
`struct` with public data AND methods. They have no invariants to protect,
but they own their behavior.

```cpp
struct LennardJones {
  double sigma{1.0};
  double epsilon{1.0};
  double r_cutoff{-1.0};
  double epsilon_shift{0.0};
  double r_min{0.0};
  double v_min{0.0};

  [[nodiscard]] auto energy(double r) const -> double;
  [[nodiscard]] auto repulsive(double r, SplitScheme) const -> double;
  [[nodiscard]] auto attractive(double r, SplitScheme) const -> double;
};
```

### Struct member ordering

```cpp
struct Fire {
  double dt{1e-3};
  double dt_max{1e-2};
  double dt_min{1e-8};
  double alpha_start{0.1};
  double f_inc{1.1};
  double f_dec{0.5};
  double f_alf{0.99};
  int n_delay{5};
  int max_uphill_steps{20};
  double force_tolerance{0.1};
  double min_density{1e-30};
  long max_steps{0};
};
```

- Default member initialisers where sensible.
- No trailing underscore on struct members (they are public).
- Group related members together.
- No constructors unless validation is needed; use designated initialisers.

### Constants in structs

Use `static constexpr` at struct scope for domain constants:

```cpp
struct Measures {
  static constexpr int NUM_INDEPENDENT = 11;
  double n0{0.0};
  double n1{0.0};
  // ...
};
```

---

## 10. Armadillo conventions

Armadillo is to this library what NumPy is to Python. It is the **primary**
tool for all numerical operations. Every loop over arrays that could be a
vectorised Armadillo operation is a bug, not a style choice.

### Mandatory vectorisation

If Armadillo provides a vectorised operation, use it. Never write a manual
`for` loop for element-wise arithmetic, reductions, or copies when an
Armadillo one-liner exists.

```cpp
// BANNED: manual element-wise multiply
for (std::size_t i = 0; i < n; ++i) {
  result[i] = a[i] * b[i];
}

// CORRECT: Armadillo Schur product
arma::cx_vec result = a % b;
```

```cpp
// BANNED: manual accumulation loop
for (std::size_t i = 0; i < n; ++i) {
  force_k[i] += partial[i];
}

// CORRECT: vectorised addition
force_k += partial;
```

```cpp
// BANNED: manual copy
std::copy_n(src.memptr(), src.n_elem, dst.data());

// CORRECT: Armadillo assignment or constructor
arma::vec dst(src_span.data(), src_span.size());
```

### Preferred types at API boundaries

| Type | Use for |
|------|---------|
| `arma::vec` | Real-valued arrays: density, forces, derivatives |
| `arma::cx_vec` | Complex arrays: Fourier coefficients, convolution results |
| `arma::rowvec3` | 3D spatial vectors: box size, position |
| `arma::mat33` | 3x3 tensors |
| `arma::mat` | N x 3 position arrays (lattice), Jacobian matrices |
| `arma::uword` | All index types |

Use `arma::cx_vec` instead of `std::vector<std::complex<double>>` in all
function signatures and return types. The only exception is inside RAII
wrappers (`FourierTransform`) that must interface with C APIs (FFTW).

### Bridging FFTW and Armadillo

The `FourierTransform` class exposes raw FFTW buffers via `std::span`.
All downstream code must immediately wrap these in Armadillo views for
computation. Helper methods on FourierTransform (`set_real`, `real_vec`,
`fourier_vec`) bridge the two worlds, so downstream code never touches raw
spans directly.

### Operations that must use Armadillo

| Operation | Armadillo way | Not this |
|-----------|--------------|----------|
| Element-wise multiply | `a % b` | `for` loop |
| Element-wise add | `a + b` or `a += b` | `for` loop |
| Conjugate | `arma::conj(v)` | `std::conj` in loop |
| Dot product | `arma::dot(a, b)` | manual sum |
| Sum all elements | `arma::accu(v)` | manual sum |
| Fill with zeros | `arma::zeros(n)` or `.zeros()` | `memset` / loop |
| Copy to vec | `arma::vec(ptr, n)` | `std::copy_n` |
| Clamp | `arma::clamp(v, lo, hi)` | manual loop |
| Log / Exp | `arma::log(v)` / `arma::exp(v)` | loop with `std::log` |
| Max / Min | `arma::max(v)` | `std::max_element` |
| Norm | `arma::norm(v)` | manual sqrt of dot |

Do **not** use `std::vector<double>` or raw `double*` at public APIs.
Do **not** use flat index enums for tensor components; use Armadillo structures.

---

## 11. Error handling

- Validate only at **system boundaries**: factory functions (`make_grid()`),
  public API entry points (`fire()`, `total()`).
- Internal pure functions do **not** throw.
- Use standard exceptions:

```cpp
throw std::invalid_argument("make_grid: dx must be positive, got " + std::to_string(dx));
throw std::out_of_range("species index " + std::to_string(i) + " out of range");
```

- Custom exceptions (`dft::exception::*`) only for domain-specific failures
  (Grace communication, parameter validation).
- Messages include the offending value when practical.
- Return `std::optional` for operations that can legitimately fail (e.g.
  root-finding that does not converge).

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


---

## 13. Test conventions

### Framework

Catch2 v3, fetched via `FetchContent`. Link `Catch2::Catch2WithMain`. No
custom `main.cpp` needed.

### Test file structure

```cpp
#include "dft/physics/fmt/measures.hpp"   // tested header first

#include <cmath>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

using namespace dft::physics::fmt;

// Default construction

TEST_CASE("Measures default construction is all zero", "[measures]") {
  Measures m;
  CHECK(m.n0 == 0.0);
}

// Uniform factory

TEST_CASE("Measures uniform eta is consistent", "[measures]") {
  auto m = Measures::uniform(0.5, 1.0);
  CHECK(m.eta == Catch::Approx(0.2618).margin(1e-4));
}
```

### Naming

- `TEST_CASE("descriptive phrase", "[tag1][tag2]")`.
- The descriptive phrase is a lowercase sentence. No suite prefixes.
- Tags are the module or concept being tested: `[grid]`, `[fmt]`,
  `[potentials]`, `[fire]`.
- Use `SECTION("...")` to group related checks within a test case.

### Assertion selection

| Catch2 macro | When to use |
|--------------|-------------|
| `CHECK(a == b)` | General equality (non-fatal) |
| `REQUIRE(a == b)` | Equality where failure should abort |
| `CHECK(a == Catch::Approx(b))` | Approximate FP equality |
| `CHECK(a == Catch::Approx(b).margin(tol))` | FP with explicit tolerance |
| `CHECK(x)` / `REQUIRE(x)` | Boolean |
| `REQUIRE_THROWS_AS(expr, type)` | Exception expected |

Tolerance guidelines:
- `1e-14`: analytical identities, algebraic simplifications.
- `1e-10`: cross-model validation (e.g. Rosenfeld vs PY).
- `1e-8`: chemical potential, integrated quantities.
- `1e-5` to `1e-6`: numerical derivatives, grid convergence.

### Parameterised tests

Use `GENERATE` with structured bindings:

```cpp
TEST_CASE("Potential energy at known distances", "[potentials]") {
  auto [r, expected] = GENERATE(table<double, double>({
      {1.0, -1.0},
      {1.5, -0.5},
      {2.0, -0.1},
  }));

  CHECK(energy(lj, r) == Catch::Approx(expected).margin(1e-10));
}
```

### Helper functions

File-scoped `static` free functions, placed before the tests that use them:

```cpp
static double numerical_derivative(auto f, double x, double h = 1e-6) {
  return (f(x + h) - f(x - h)) / (2.0 * h);
}
```

---

## 14. Doc conventions

### File structure

```
docs/<module>/
  main.cpp
  CMakeLists.txt
  Makefile
  README.md
  exports/          # created at runtime for plot output
```

### CMakeLists.txt (minimal)

```cmake
add_executable(doc_<name> main.cpp)
target_link_libraries(doc_<name> PRIVATE classicaldft)
```

### main.cpp structure

```cpp
#include "dft/dft.hpp"

#include <iomanip>
#include <iostream>

using namespace dft;

int main() {
  std::filesystem::create_directories("exports");

  // Define model
  physics::Model model{
      .grid = core::make_grid(0.1, {10.0, 10.0, 10.0}),
      .species = {core::Species{.name = "Argon", .hard_sphere_diameter = 1.0}},
  };

  // Build functional
  auto functional = functionals::make_functional(functionals::fmt::WhiteBearII{}, model);

  // Initial state
  auto state = init::homogeneous(model, 0.5);

  // Evaluate
  auto result = functional.evaluate(state);
  std::cout << "Free energy: " << result.total_free_energy << "\n";

  // Grace plots
#ifdef DFT_HAS_GRACE
  using namespace dft::plotting;

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

- Options prefixed `DFT_`: `DFT_BUILD_TESTS`, `DFT_BUILD_DOCS`,
  `DFT_USE_GRACE`, `DFT_CODE_COVERAGE`.
- Library sources listed **explicitly** (no `GLOB`) for the static library.
- Test sources use `file(GLOB_RECURSE)`.
- External dependencies: `find_package` for system libraries, `FetchContent`
  for Catch2.
- Compile definition `DFT_HAS_GRACE` propagated as `PUBLIC` when Grace is found.
- Warnings applied via generator expression (GNU/Clang/AppleClang only).
- Export `compile_commands.json` (`CMAKE_EXPORT_COMPILE_COMMANDS ON`).

---

## 16. Design principles

### Types own their behavior

Types encapsulate both data and behavior. Physics types are `struct` with
public data and methods. Complex types with implementation details are
`class` with private members. Free functions only for truly stateless logic
that operates across multiple unrelated types.

### Templates for compile-time polymorphism

Use C++ templates when the type is known at compile time. This eliminates
`std::visit` overhead in hot paths and generates specialized code. Use
`std::variant` only when the choice is runtime-determined.

### Value semantics

All data is passed by value or `const&`. Functions return new values; they
never mutate inputs. Algorithm functions take `State` by value and return
`Solution`. The function signature is the contract: inputs on the left,
output on the right. No hidden side channels through mutable references.

Immutable precomputed data (weights, models) is separated from ephemeral
scratch memory. Precomputed data is passed by `const&`. Scratch is allocated
internally by the functions that need it. The caller never manages temporary
buffers.

### Single responsibility

Every function and file has one reason to change. If a function name contains
"and", split it.

### Minimal inheritance — only for variant wrapper classes

No deep hierarchies, no NVI pattern, no `virtual` on hot-path methods.
The only permitted use of inheritance is via variant wrapper classes (see
§8 "std::variant for runtime closed sets — with wrapper classes"). These
wrappers give closed type sets a unified method interface without exposing
`std::visit` to callers.

RAII wrappers for C resources (`FourierTransform`, `CubicSpline`, `Grace`)
remain a separate exception.

### Open/closed principle

Add new potential types, FMT models, or mesh types by adding a new struct,
extending the variant inside the wrapper class, and implementing the
required methods. The wrapper class enforces interface completeness at
compile time (any missing method on a new alternative causes a build error
inside the `std::visit` lambda).

### No premature abstraction

Do not create helpers, utilities, or wrapper types for one-time operations.
Do not add layers "for future use".

### No dead code

No commented-out code, no unused includes, no functions that are never called.

### Const correctness

- Pass large objects by `const&`.
- Use `const` local variables when the value does not change.
- Mark all pure functions `[[nodiscard]]`.

---

## 17. Anti-patterns

The following patterns are **banned** in this codebase. Each entry explains
why it is harmful and shows the correct alternative.

### Out-parameter mutation

Never pass an object by mutable reference for the function to fill. This
hides the data flow, forces the caller to pre-allocate, and makes the
function signature misleading (a `void` return implies no useful output).

```cpp
// BANNED: caller has no idea ws is being populated
void generate_weights(double diameter, const Grid& grid, WeightSet& ws);

// CORRECT: factory returns a fully constructed value
[[nodiscard]] auto generate_weights(double diameter, const Grid& grid) -> WeightSet;
```

### Separate allocate-then-fill

Never split object creation into an allocation step followed by a mutation
step. This creates a temporal coupling (must call B after A) and an invalid
intermediate state (the object exists but is empty).

```cpp
// BANNED: two-step pattern with invalid intermediate state
auto ws = make_weight_set(grid);       // empty shell
generate_weights(diameter, grid, ws);  // now filled

// CORRECT: single factory, no invalid intermediate
auto ws = generate_weights(diameter, grid);
```

### Shared mutable scratch buffers

Never require the caller to manage FFT scratch buffers or temporary work
arrays and pass them into functions. Scratch memory is an implementation
detail. Allocate it internally or accept a shape/size parameter instead.

```cpp
// BANNED: caller manages scratch lifetime and aliasing rules
FourierTransform scratch(shape);
auto eta = convolve(w3_k, rho_k, scratch);   // mutates scratch
auto n2  = convolve(w2_k, rho_k, scratch);   // reuses scratch

// CORRECT: function owns its scratch
auto eta = convolve(w3_k, rho_k, shape);
auto n2  = convolve(w2_k, rho_k, shape);
```

### Accumulation into mutable spans

Never pass a `std::span<T>` or raw pointer for the function to accumulate
results into. Return the partial result and let the caller combine them.

```cpp
// BANNED: hidden accumulation into force_k
void accumulate(weight_k, derivative, scratch, force_k);

// CORRECT: returns the partial contribution
[[nodiscard]] auto back_convolve(weight_k, derivative, shape) -> std::vector<std::complex<double>>;
```

### Mutable workspace references in public APIs

Never require the caller to pass a mutable workspace `struct&` that bundles
precomputed data with temporary scratch buffers. Separate the concerns:
immutable precomputed data (passed by `const&`) and ephemeral scratch
(internal to the function).

```cpp
// BANNED: mixes immutable weights with mutable scratch
struct FMTWorkspace {
  std::vector<WeightSet> weights;        // immutable after creation
  std::vector<FourierTransform> rho_ft;  // mutated every call
  FourierTransform scratch;              // mutated every call
};
auto result = hard_sphere(model, grid, state, species, workspace);  // workspace mutated

// CORRECT: weights are const, scratch is internal
struct FMTWeights {
  std::vector<WeightSet> per_species;    // immutable
};
auto result = hard_sphere(model, grid, state, species, weights);    // weights unchanged
```

### God structs with mixed concerns

Never bundle unrelated data into a single struct for convenience. If a struct
contains both immutable configuration and mutable runtime state, split it
into separate types.

### `void` functions that do useful work

If a function produces a result, it must return it. A `void` return type on a
function that is not purely side-effecting (I/O, logging) is a code smell.
Every transformation of data should be visible in the return type.

### Two-phase initialisation

Objects must be valid and complete at construction time. Never construct an
object in a partially initialised state and require a second method call to
finish setup. Use factory functions that return fully formed values.
