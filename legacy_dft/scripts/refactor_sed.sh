#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$PWD"

# All .h, .cpp files under include/, src/, tests/, examples/
FILES=$(find include src tests examples -name '*.h' -o -name '*.cpp' | sort)

echo "Processing $(echo "$FILES" | wc -l | tr -d ' ') files..."

for f in $FILES; do
  # ═══ INCLUDE PATHS (most specific first) ═══

  # physics/species/base.h → functional/data_structures.h
  sed -i '' 's|physics/species/base\.h|functional/data_structures.h|g' "$f"

  # physics/fmt/measures.h → functional/fmt/data_structures.h
  sed -i '' 's|physics/fmt/measures\.h|functional/fmt/data_structures.h|g' "$f"

  # physics/fmt/convolution.h → functional/fmt/weighted_density.h
  sed -i '' 's|physics/fmt/convolution\.h|functional/fmt/weighted_density.h|g' "$f"

  # physics/fmt/weights.h → functional/fmt/weights.h
  sed -i '' 's|physics/fmt/weights\.h|functional/fmt/weights.h|g' "$f"

  # physics/fmt/functional.h → functional/fmt/functional.h
  sed -i '' 's|physics/fmt/functional\.h|functional/fmt/functional.h|g' "$f"

  # physics/fmt/species.h → functional/fmt/species.h
  sed -i '' 's|physics/fmt/species\.h|functional/fmt/species.h|g' "$f"

  # physics/interaction/interaction.h → functional/mean_field/interaction.h
  sed -i '' 's|physics/interaction/interaction\.h|functional/mean_field/interaction.h|g' "$f"

  # physics/density/density.h → density/density.h
  sed -i '' 's|physics/density/density\.h|density/density.h|g' "$f"

  # physics/potentials/intermolecular/potential.h → potential/potential.h
  sed -i '' 's|physics/potentials/intermolecular/potential\.h|potential/potential.h|g' "$f"

  # physics/thermodynamics/enskog.h → thermodynamics/enskog.h
  sed -i '' 's|physics/thermodynamics/enskog\.h|thermodynamics/enskog.h|g' "$f"

  # physics/thermodynamics/eos.h → thermodynamics/eos.h
  sed -i '' 's|physics/thermodynamics/eos\.h|thermodynamics/eos.h|g' "$f"

  # physics/crystal/lattice.h → crystal/lattice.h
  sed -i '' 's|physics/crystal/lattice\.h|crystal/lattice.h|g' "$f"

  # exceptions/ → exception/
  sed -i '' 's|exceptions/grace\.h|exception/grace.h|g' "$f"
  sed -i '' 's|exceptions/parameter\.h|exception/parameter.h|g' "$f"

  # ═══ HEADER GUARDS (most specific first) ═══

  # physics/species/base
  sed -i '' 's|CLASSICALDFT_PHYSICS_SPECIES_BASE_H|CLASSICALDFT_FUNCTIONAL_DATA_STRUCTURES_H|g' "$f"

  # physics/fmt/*
  sed -i '' 's|CLASSICALDFT_PHYSICS_FMT_MEASURES_H|CLASSICALDFT_FUNCTIONAL_FMT_DATA_STRUCTURES_H|g' "$f"
  sed -i '' 's|CLASSICALDFT_PHYSICS_FMT_CONVOLUTION_H|CLASSICALDFT_FUNCTIONAL_FMT_WEIGHTED_DENSITY_H|g' "$f"
  sed -i '' 's|CLASSICALDFT_PHYSICS_FMT_WEIGHTS_H|CLASSICALDFT_FUNCTIONAL_FMT_WEIGHTS_H|g' "$f"
  sed -i '' 's|CLASSICALDFT_PHYSICS_FMT_FUNCTIONAL_H|CLASSICALDFT_FUNCTIONAL_FMT_FUNCTIONAL_H|g' "$f"
  sed -i '' 's|CLASSICALDFT_PHYSICS_FMT_SPECIES_H|CLASSICALDFT_FUNCTIONAL_FMT_SPECIES_H|g' "$f"

  # physics/interaction
  sed -i '' 's|CLASSICALDFT_PHYSICS_INTERACTION_INTERACTION_H|CLASSICALDFT_FUNCTIONAL_MEAN_FIELD_INTERACTION_H|g' "$f"

  # physics/density
  sed -i '' 's|CLASSICALDFT_PHYSICS_DENSITY_DENSITY_H|CLASSICALDFT_DENSITY_DENSITY_H|g' "$f"

  # physics/potentials/intermolecular
  sed -i '' 's|CLASSICALDFT_PHYSICS_POTENTIALS_INTERMOLECULAR_POTENTIAL_H|CLASSICALDFT_POTENTIAL_POTENTIAL_H|g' "$f"

  # physics/thermodynamics
  sed -i '' 's|CLASSICALDFT_PHYSICS_THERMODYNAMICS_ENSKOG_H|CLASSICALDFT_THERMODYNAMICS_ENSKOG_H|g' "$f"
  sed -i '' 's|CLASSICALDFT_PHYSICS_THERMODYNAMICS_EOS_H|CLASSICALDFT_THERMODYNAMICS_EOS_H|g' "$f"

  # physics/crystal
  sed -i '' 's|CLASSICALDFT_PHYSICS_CRYSTAL_LATTICE_H|CLASSICALDFT_CRYSTAL_LATTICE_H|g' "$f"

  # exceptions
  sed -i '' 's|CLASSICALDFT_EXCEPTIONS_GRACE_H|CLASSICALDFT_EXCEPTION_GRACE_H|g' "$f"
  sed -i '' 's|CLASSICALDFT_EXCEPTIONS_PARAMETER_H|CLASSICALDFT_EXCEPTION_PARAMETER_H|g' "$f"

  # ═══ NAMESPACES (most specific first) ═══

  # physics::fmt (must be before physics::)
  sed -i '' 's|dft_core::physics::fmt|dft::functional::fmt|g' "$f"

  # physics::interaction → functional::mean_field
  sed -i '' 's|dft_core::physics::interaction|dft::functional::mean_field|g' "$f"

  # physics::species → functional
  sed -i '' 's|dft_core::physics::species|dft::functional|g' "$f"

  # physics::density
  sed -i '' 's|dft_core::physics::density|dft::density|g' "$f"

  # physics::potentials::intermolecular
  sed -i '' 's|dft_core::physics::potentials::intermolecular|dft::potential|g' "$f"

  # physics::thermodynamics
  sed -i '' 's|dft_core::physics::thermodynamics|dft::thermodynamics|g' "$f"

  # physics::crystal
  sed -i '' 's|dft_core::physics::crystal|dft::crystal|g' "$f"

  # numerics::arithmetic::summation → numerics::arithmetic
  sed -i '' 's|dft_core::numerics::arithmetic::summation|dft::numerics::arithmetic|g' "$f"

  # numerics (sub-namespaces)
  sed -i '' 's|dft_core::numerics::fourier|dft::numerics::fourier|g' "$f"
  sed -i '' 's|dft_core::numerics::integration|dft::numerics::integration|g' "$f"
  sed -i '' 's|dft_core::numerics::spline|dft::numerics::spline|g' "$f"
  sed -i '' 's|dft_core::numerics::arithmetic|dft::numerics::arithmetic|g' "$f"
  sed -i '' 's|dft_core::numerics|dft::numerics|g' "$f"

  # geometry
  sed -i '' 's|dft_core::geometry|dft::geometry|g' "$f"

  # grace_plot → graph
  sed -i '' 's|dft_core::grace_plot|dft::graph|g' "$f"
  sed -i '' 's|grace_plot::|graph::|g' "$f"

  # io::console
  sed -i '' 's|dft_core::io::console|dft::io::console|g' "$f"
  sed -i '' 's|dft_core::io|dft::io|g' "$f"

  # config_parser → io::config
  sed -i '' 's|dft_core::config_parser|dft::io::config|g' "$f"

  # exception
  sed -i '' 's|dft_core::exception|dft::exception|g' "$f"

  # Catch-all: any remaining dft_core
  sed -i '' 's|dft_core|dft|g' "$f"

done

echo "=== Bulk replacements complete ==="
