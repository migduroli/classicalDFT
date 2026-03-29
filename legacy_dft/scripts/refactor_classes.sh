#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."

# All new-location .h, .cpp files
FILES=$(find include src tests examples -name '*.h' -o -name '*.cpp' | sort)

echo "Processing $(echo "$FILES" | wc -l | tr -d ' ') files for class renames..."

for f in $FILES; do

  # ═══ CLASS RENAMES ═══

  # In fmt namespace: Functional → FundamentalMeasureTheory (class name)
  # Be careful: only match the class keyword, not fmt::Functional used as a param type
  sed -i '' 's/class Functional \(final\)/class FundamentalMeasureTheory final/g' "$f"
  sed -i '' 's/class Functional$/class FundamentalMeasureTheory/g' "$f"
  sed -i '' 's/class Functional {/class FundamentalMeasureTheory {/g' "$f"
  sed -i '' 's/: public Functional/: public FundamentalMeasureTheory/g' "$f"

  # Functional& and Functional const& parameter types
  sed -i '' 's/const Functional&/const FundamentalMeasureTheory\&/g' "$f"
  sed -i '' 's/Functional&/FundamentalMeasureTheory\&/g' "$f"

  # fmt::Functional used as qualified name
  sed -i '' 's/fmt::Functional/fmt::FundamentalMeasureTheory/g' "$f"

  # Measures → FundamentalMeasures
  sed -i '' 's/struct Measures/struct FundamentalMeasures/g' "$f"
  sed -i '' 's/const Measures&/const FundamentalMeasures\&/g' "$f"
  sed -i '' 's/Measures::/FundamentalMeasures::/g' "$f"
  # Careful with "Measures " as variable type (not "FundamentalMeasures")
  sed -i '' 's/Measures d_phi/FundamentalMeasures d_phi/g' "$f"
  sed -i '' 's/Measures uniform/FundamentalMeasures uniform/g' "$f"
  sed -i '' 's/Measures m/FundamentalMeasures m/g' "$f"
  sed -i '' 's/Measures dm/FundamentalMeasures dm/g' "$f"
  sed -i '' 's/Measures result/FundamentalMeasures result/g' "$f"
  # Return type
  sed -i '' 's/\[\[nodiscard\]\] Measures /[[nodiscard]] FundamentalMeasures /g' "$f"
  # "auto m = Measures{" type construction
  sed -i '' 's/= Measures{/= FundamentalMeasures{/g' "$f"
  sed -i '' 's/= Measures(/= FundamentalMeasures(/g' "$f"

  # ConvolutionField → WeightedDensity
  sed -i '' 's/class ConvolutionField/class WeightedDensity/g' "$f"
  sed -i '' 's/ConvolutionField(/WeightedDensity(/g' "$f"
  sed -i '' 's/ConvolutionField&/WeightedDensity\&/g' "$f"
  sed -i '' 's/ConvolutionField::/WeightedDensity::/g' "$f"
  sed -i '' 's/ConvolutionField\b/WeightedDensity/g' "$f"

  # WeightSet → WeightedDensitySet
  sed -i '' 's/struct WeightSet/struct WeightedDensitySet/g' "$f"
  sed -i '' 's/WeightSet&/WeightedDensitySet\&/g' "$f"
  sed -i '' 's/WeightSet weights/WeightedDensitySet weights/g' "$f"
  sed -i '' 's/WeightSet::/WeightedDensitySet::/g' "$f"

  # Weights → WeightGenerator (the static class)
  sed -i '' 's/class Weights/class WeightGenerator/g' "$f"
  sed -i '' 's/Weights::/WeightGenerator::/g' "$f"

  # fmt::Species → fmt::FMTSpecies
  # This is tricky because species::Species also exists
  # Only rename when qualified as fmt::Species or in fmt namespace context
  sed -i '' 's/fmt::Species/fmt::FMTSpecies/g' "$f"

done

echo "=== Class renames complete ==="
