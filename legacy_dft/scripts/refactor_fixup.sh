#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."

# All new .h, .cpp files
FILES=$(find include src tests examples -name '*.h' -o -name '*.cpp' | sort)

echo "Processing remaining fixups..."

for f in $FILES; do
  # ═══ FMT class Species → FMTSpecies (within fmt namespace files) ═══
  # The declaration "class Species : public" in fmt files
  sed -i '' 's/class Species : public species::Species/class FMTSpecies : public Species/g' "$f"
  sed -i '' 's/class Species : public functional::Species/class FMTSpecies : public Species/g' "$f"

  # Constructor/destructor in .cpp: "Species::Species(" → "FMTSpecies::FMTSpecies("
  # Only in functional/fmt/ files — we'll handle via path check below

  # species::Species → just Species (when in functional namespace, the base class is Species)
  sed -i '' 's/species::Species/Species/g' "$f"

  # ═══ Destructor fix ═══
  sed -i '' 's/~Functional()/~FundamentalMeasureTheory()/g' "$f"

  # ═══ Remaining bare "Measures" as type (not in comments) ═══
  # In .cpp files: "Measures Functional::" → "FundamentalMeasures FundamentalMeasureTheory::"
  sed -i '' 's/Measures Functional::/FundamentalMeasures FundamentalMeasureTheory::/g' "$f"
  sed -i '' 's/Measures Species::/FundamentalMeasures FMTSpecies::/g' "$f"

  # ═══ Functional:: scope → FundamentalMeasureTheory:: scope (in .cpp) ═══
  sed -i '' 's/Functional::phi/FundamentalMeasureTheory::phi/g' "$f"
  sed -i '' 's/Functional::d_phi/FundamentalMeasureTheory::d_phi/g' "$f"
  sed -i '' 's/Functional::bulk_free_energy_density/FundamentalMeasureTheory::bulk_free_energy_density/g' "$f"
  sed -i '' 's/Functional::bulk_excess_chemical_potential/FundamentalMeasureTheory::bulk_excess_chemical_potential/g' "$f"

  # ═══ WeightSet field in old comment or variable declarations ═══
  sed -i '' 's/WeightSet weights_/WeightedDensitySet weights_/g' "$f"

  # ═══ TEST SUITE NAMES: Measures → FundamentalMeasures ═══
  sed -i '' 's/TEST(Measures,/TEST(FundamentalMeasures,/g' "$f"
  sed -i '' 's/TEST(ConvolutionField,/TEST(WeightedDensity,/g' "$f"
  sed -i '' 's/TEST(WeightSet,/TEST(WeightedDensitySet,/g' "$f"
  sed -i '' 's/TEST(Weights,/TEST(WeightGenerator,/g' "$f"

  # ═══ Remaining Functional references in tests ═══
  sed -i '' 's/TEST(Functional,/TEST(FundamentalMeasureTheory,/g' "$f"

done

# ═══ FMT Species scoped names in .cpp ═══
# In src/functional/fmt/species.cpp: Species::Method → FMTSpecies::Method
for f in src/functional/fmt/species.cpp tests/functional/fmt/species.cpp; do
  if [ -f "$f" ]; then
    # Replace Species:: at start of method definitions (but not "::Species" which is the base)
    sed -i '' 's/^  Species::/  FMTSpecies::/g' "$f"
    # Also indented versions
    sed -i '' 's/Species::Species/FMTSpecies::FMTSpecies/g' "$f"
    sed -i '' 's/Species::convolve_density/FMTSpecies::convolve_density/g' "$f"
    sed -i '' 's/Species::measures_at/FMTSpecies::measures_at/g' "$f"
    sed -i '' 's/Species::set_derivatives/FMTSpecies::set_derivatives/g' "$f"
    sed -i '' 's/Species::accumulate_forces/FMTSpecies::accumulate_forces/g' "$f"
    sed -i '' 's/Species::compute_free_energy/FMTSpecies::compute_free_energy/g' "$f"
    sed -i '' 's/Species::compute_forces/FMTSpecies::compute_forces/g' "$f"
    sed -i '' 's/Species::set_density_from_alias/FMTSpecies::set_density_from_alias/g' "$f"
    sed -i '' 's/Species::density_alias/FMTSpecies::density_alias/g' "$f"
    sed -i '' 's/Species::alias_force/FMTSpecies::alias_force/g' "$f"
  fi
done

echo "=== Fixups complete ==="
