# Crystal lattice

## Overview

The `dft_core::physics::crystal` namespace generates crystal lattices with
configurable structure, Miller-index orientation, and replication.

| Class / enum | Role |
|--------------|------|
| `Lattice` | Crystal lattice generator (positions, scaling, export) |
| `Structure` | BCC, FCC, HCP |
| `Orientation` | Miller-index plane orthogonal to z-axis |
| `ExportFormat` | Output format for `Lattice::export` (XYZ, CSV) |

## Usage

```cpp
#include <classicaldft>
using namespace dft_core::physics::crystal;

// Build an FCC lattice with [001] orientation, 4x4x4 unit cells
auto fcc = Lattice(Structure::FCC, Orientation::_001, {4, 4, 4});

// Atom count and box dimensions
std::cout << fcc.size() << " atoms\n";
std::cout << fcc.dimensions() << "\n";

// Scale to physical nearest-neighbor distance
arma::mat pos = fcc.positions(3.405);  // Argon in Angstrom

// Anisotropic box mapping
arma::mat mapped = fcc.positions(arma::rowvec3{10.0, 12.0, 14.0});

// Export to file
fcc.export_to("lattice.xyz", ExportFormat::XYZ);
fcc.export_to("lattice.csv", ExportFormat::CSV);
```

## Running

```bash
make run   # builds and runs inside Docker
```
