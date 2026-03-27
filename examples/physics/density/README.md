# Density and species

## Overview

The `dft_core::physics::density` and `dft_core::physics::species` namespaces
provide the data layer for DFT calculations: density profiles on periodic grids
and species management with force accumulation and alias coordinates.

| Class | Role |
|-------|------|
| `density::Density` | Density profile $\rho(\mathbf{r})$ on a 3D periodic uniform grid |
| `species::Species` | Single-component species: owns a `Density`, force vector, chemical potential, alias coordinates |

### `Density`

- Owns the periodic grid parameters ($N_i = L_i / dx$, no duplicated boundary)
- Real-space density as `arma::vec`; external field as `arma::vec`
- FFT via `FourierTransform` (FFTW3 wrapper)
- Compensated summation for `number_of_atoms()`
- Binary I/O via Armadillo `save`/`load`

### `Species`

- Owns a `Density`, a force vector $\delta F / \delta \rho$, and $\mu$
- Fixed-mass constraint with Lagrange multiplier projection
- Alias coordinates: $\rho = \rho_{\min} + x^2$ (guarantees positivity)
- Force protocol: `begin_force_calculation()` / accumulate / `end_force_calculation()`

## Usage

```cpp
#include <classicaldft>
using namespace dft_core::physics::density;
using namespace dft_core::physics::species;

// Create a density on a periodic grid
Density rho(0.1, {5.0, 5.0, 5.0});
rho.values().fill(0.8);
std::cout << "N = " << rho.number_of_atoms() << std::endl;

// FFT
rho.forward_fft();
auto dc = std::abs(rho.fft().fourier()[0]);

// Species with alias coordinates
Species s(Density(0.1, {5.0, 5.0, 5.0}), /*mu=*/1.0);
arma::vec x = s.density_alias();
s.set_density_from_alias(x);  // round-trip
```

## Running

```bash
make run-local
```

## Plots (requires Grace)

When built with `DFT_HAS_GRACE`, three plots are saved to `exports/`:

| File | Content |
|------|---------|
| `density_profile.png` | Sinusoidal density $\rho_0 + A\sin(2\pi z/L)$ and hard-wall external field along $z$ |
| `fft_spectrum.png` | FFT power spectrum $|F(k_z)|/N$ showing DC peak at $k_z=0$ and sine mode at $k_z=1$ |
| `alias_mapping.png` | Species alias coordinate $x(\rho) = \sqrt{\rho - \rho_{\min}}$ |
