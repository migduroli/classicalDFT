# Intermolecular potentials

## Overview

The `dft::potentials` namespace provides intermolecular potential models used in statistical physics and thermodynamic perturbation theory. The abstract base class `Potential` encapsulates common functionality shared by all potentials, including Weeks-Chandler-Andersen (WCA) decomposition into repulsive and attractive parts, and temperature-dependent hard-sphere diameter computation.

| Class | Role |
|-------|------|
| `Potential` | Abstract base class for all intermolecular potentials |
| `LennardJones` | Lennard-Jones 6-12 potential |
| `tenWoldeFrenkel` | ten Wolde-Frenkel short-ranged potential |
| `WangRamirezDobnikarFrenkel` | WRDF potential |

Every `Potential` subclass provides:
- `v_potential(r)` / `operator()(r)` — evaluate the full potential
- `r_min()`, `v_min()` — location and value of the potential minimum
- `w_attractive(r)`, `w_repulsive(r)` — WCA perturbation decomposition
- `find_hard_sphere_diameter(kT)` — Barker-Henderson effective hard-sphere diameter

The WCA decomposition splits the potential at $r_*$ (the minimum) into a purely repulsive core and an attractive tail:

$$\Phi_{\text{rep}}(r) = \Theta(r_* - r) \left[ v(r) - v(r_*) \right]$$

$$\Phi_{\text{att}}(r) = \Theta(r_* - r) \, v(r_*) + \Theta(r - r_*) \, v(r)$$

The hard-sphere diameter is computed via the Barker-Henderson integral:

$$d_{\text{HS}}(T) = \int_{r_{\text{HC}}}^{r_*} \left(1 - e^{-v(r)/k_BT}\right) dr$$

## Usage

```cpp
#include "dft.h"

using namespace dft::potentials;

auto lj = LennardJones();
auto twf = tenWoldeFrenkel();
auto wrdf = WangRamirezDobnikarFrenkel();

// Evaluate potential at a set of points
auto r = arma::linspace(0.75, 1.8, 200);
auto v = lj.v_potential(r);        // or: lj(r)
auto v_att = lj.w_attractive(r);
auto v_rep = lj.w_repulsive(r);

// Hard-sphere diameter at kT = 1.0
double d_hs = lj.find_hard_sphere_diameter(1.0);

// Minimum location and value
double r_min = lj.r_min();
double v_min = lj.v_min();
```

## Running

```bash
make run        # builds and runs inside Docker
make run-local  # builds and runs locally
```

## Plots

When built with `DFT_USE_MATPLOTLIB=ON` (default), plots are saved to `exports/`:

| File | Description |
|------|-------------|
| `potential_lj.png` | LJ potential with minimum and hard-sphere diameter |
| `potential_twf.png` | tWF potential with minimum and hard-sphere diameter |
| `potential_wrdf.png` | WRDF potential with minimum and hard-sphere diameter |
| `potentials_comparison.png` | All three potentials overlaid |
| `perturbation_lj.png` | LJ WCA decomposition into attractive and repulsive parts |
| `perturbation_twf.png` | tWF WCA decomposition |
| `perturbation_wrdf.png` | WRDF WCA decomposition |
| `perturbation_decomposition.png` | All perturbation decompositions overlaid |