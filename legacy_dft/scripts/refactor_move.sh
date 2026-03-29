#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$PWD"

echo "=== Stage 1: Create new directories ==="

# Include directories
mkdir -p include/classicaldft_bits/exception
mkdir -p include/classicaldft_bits/density
mkdir -p include/classicaldft_bits/potential
mkdir -p include/classicaldft_bits/thermodynamics
mkdir -p include/classicaldft_bits/crystal
mkdir -p include/classicaldft_bits/functional/fmt
mkdir -p include/classicaldft_bits/functional/mean_field

# Source directories
mkdir -p src/exception
mkdir -p src/density
mkdir -p src/potential
mkdir -p src/thermodynamics
mkdir -p src/crystal
mkdir -p src/functional/fmt
mkdir -p src/functional/mean_field
mkdir -p src/io

# Test directories
mkdir -p tests/exception
mkdir -p tests/density
mkdir -p tests/potential
mkdir -p tests/thermodynamics
mkdir -p tests/crystal
mkdir -p tests/functional/fmt
mkdir -p tests/functional/mean_field

# Example directories
mkdir -p examples/density
mkdir -p examples/potential
mkdir -p examples/thermodynamics
mkdir -p examples/crystal
mkdir -p examples/functional/fmt
mkdir -p examples/functional/mean_field
mkdir -p examples/functional/species

echo "Directories created."

echo "=== Stage 2: Move headers ==="

# exceptions/ → exception/
cp include/classicaldft_bits/exceptions/grace.h include/classicaldft_bits/exception/grace.h
cp include/classicaldft_bits/exceptions/parameter.h include/classicaldft_bits/exception/parameter.h

# physics/density/ → density/
cp include/classicaldft_bits/physics/density/density.h include/classicaldft_bits/density/density.h

# physics/potentials/intermolecular/ → potential/
cp include/classicaldft_bits/physics/potentials/intermolecular/potential.h include/classicaldft_bits/potential/potential.h

# physics/thermodynamics/ → thermodynamics/
cp include/classicaldft_bits/physics/thermodynamics/enskog.h include/classicaldft_bits/thermodynamics/enskog.h
cp include/classicaldft_bits/physics/thermodynamics/eos.h include/classicaldft_bits/thermodynamics/eos.h

# physics/crystal/ → crystal/
cp include/classicaldft_bits/physics/crystal/lattice.h include/classicaldft_bits/crystal/lattice.h

# physics/species/base.h → functional/data_structures.h
cp include/classicaldft_bits/physics/species/base.h include/classicaldft_bits/functional/data_structures.h

# physics/fmt/measures.h → functional/fmt/data_structures.h
cp include/classicaldft_bits/physics/fmt/measures.h include/classicaldft_bits/functional/fmt/data_structures.h

# physics/fmt/convolution.h → functional/fmt/weighted_density.h
cp include/classicaldft_bits/physics/fmt/convolution.h include/classicaldft_bits/functional/fmt/weighted_density.h

# physics/fmt/weights.h → functional/fmt/weights.h
cp include/classicaldft_bits/physics/fmt/weights.h include/classicaldft_bits/functional/fmt/weights.h

# physics/fmt/functional.h → functional/fmt/functional.h
cp include/classicaldft_bits/physics/fmt/functional.h include/classicaldft_bits/functional/fmt/functional.h

# physics/fmt/species.h → functional/fmt/species.h
cp include/classicaldft_bits/physics/fmt/species.h include/classicaldft_bits/functional/fmt/species.h

# physics/interaction/interaction.h → functional/mean_field/interaction.h
cp include/classicaldft_bits/physics/interaction/interaction.h include/classicaldft_bits/functional/mean_field/interaction.h

echo "Headers moved."

echo "=== Stage 3: Move sources ==="

# exceptions/ → exception/
cp src/exceptions/grace.cpp src/exception/grace.cpp
cp src/exceptions/parameter.cpp src/exception/parameter.cpp

# io/ stays (just ensure dir exists)
# graph/ stays
# numerics/ stays
# geometry/ stays

# physics/density/ → density/
cp src/physics/density/density.cpp src/density/density.cpp

# physics/potentials/intermolecular/ → potential/
cp src/physics/potentials/intermolecular/potential.cpp src/potential/potential.cpp

# physics/thermodynamics/ → thermodynamics/
cp src/physics/thermodynamics/eos.cpp src/thermodynamics/eos.cpp

# physics/crystal/ → crystal/
cp src/physics/crystal/lattice.cpp src/crystal/lattice.cpp

# physics/species/ → functional/
cp src/physics/species/species.cpp src/functional/species.cpp

# physics/fmt/* → functional/fmt/
cp src/physics/fmt/convolution.cpp src/functional/fmt/weighted_density.cpp
cp src/physics/fmt/weights.cpp src/functional/fmt/weights.cpp
cp src/physics/fmt/functional.cpp src/functional/fmt/functional.cpp
cp src/physics/fmt/species.cpp src/functional/fmt/species.cpp

# physics/interaction/ → functional/mean_field/
cp src/physics/interaction/interaction.cpp src/functional/mean_field/interaction.cpp

echo "Sources moved."

echo "=== Stage 4: Move tests ==="

# exceptions/ → exception/
cp tests/exceptions/grace.cpp tests/exception/grace.cpp
cp tests/exceptions/parameter.cpp tests/exception/parameter.cpp

# physics/density/ → density/
cp tests/physics/density/density.cpp tests/density/density.cpp

# physics/potentials/intermolecular/ → potential/
cp tests/physics/potentials/intermolecular/potential.cpp tests/potential/potential.cpp

# physics/thermodynamics/ → thermodynamics/
cp tests/physics/thermodynamics/enskog.cpp tests/thermodynamics/enskog.cpp
cp tests/physics/thermodynamics/eos.cpp tests/thermodynamics/eos.cpp

# physics/crystal/ → crystal/
cp tests/physics/crystal/lattice.cpp tests/crystal/lattice.cpp

# physics/species/ → functional/
cp tests/physics/species/species.cpp tests/functional/species.cpp

# physics/fmt/* → functional/fmt/
cp tests/physics/fmt/measures.cpp tests/functional/fmt/data_structures.cpp
cp tests/physics/fmt/convolution.cpp tests/functional/fmt/weighted_density.cpp
cp tests/physics/fmt/weights.cpp tests/functional/fmt/weights.cpp
cp tests/physics/fmt/functional.cpp tests/functional/fmt/functional.cpp
cp tests/physics/fmt/species.cpp tests/functional/fmt/species.cpp

# physics/interaction/ → functional/mean_field/
cp tests/physics/interaction/interaction.cpp tests/functional/mean_field/interaction.cpp

echo "Tests moved."

echo "=== Stage 5: Move examples ==="

# physics/density/ → density/
cp -r examples/physics/density/* examples/density/ 2>/dev/null || true

# physics/potentials/intermolecular/ → potential/
cp -r examples/physics/potentials/intermolecular/* examples/potential/ 2>/dev/null || true

# physics/thermodynamics/ → thermodynamics/
cp -r examples/physics/thermodynamics/* examples/thermodynamics/ 2>/dev/null || true

# physics/crystal/ → crystal/
cp -r examples/physics/crystal/* examples/crystal/ 2>/dev/null || true

# physics/species/ → functional/species/
cp -r examples/physics/species/* examples/functional/species/ 2>/dev/null || true

# physics/fmt/ → functional/fmt/
cp -r examples/physics/fmt/* examples/functional/fmt/ 2>/dev/null || true

# physics/interaction/ → functional/mean_field/
cp -r examples/physics/interaction/* examples/functional/mean_field/ 2>/dev/null || true

echo "Examples moved."

echo "=== Stage 6: Remove old directories ==="

rm -rf include/classicaldft_bits/exceptions
rm -rf include/classicaldft_bits/physics
rm -rf src/exceptions
rm -rf src/physics
rm -rf tests/exceptions
rm -rf tests/physics
rm -rf examples/physics

echo "Old directories removed."

echo "=== All file moves complete ==="
