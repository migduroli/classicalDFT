#ifndef DFT_PHYSICS_INTERACTIONS_HPP
#define DFT_PHYSICS_INTERACTIONS_HPP

#include "dft/physics/potentials.hpp"

#include <string>

namespace dft::physics {

  // Integration scheme for computing interaction weights on the lattice.
  enum class WeightScheme {
    InterpolationZero,
    InterpolationLinearE,
    InterpolationLinearF,
    InterpolationQuadraticF,
    GaussE,
    GaussF,
  };

  // Pure specification for a mean-field pair interaction between two species.
  // Contains only configuration data; all computation lives in the
  // functionals layer.
  struct Interaction {
    int species_i{0};
    int species_j{0};
    potentials::Potential potential;
    potentials::SplitScheme split{potentials::SplitScheme::WeeksChandlerAndersen};
    WeightScheme weight_scheme{WeightScheme::InterpolationQuadraticF};
    int gauss_order{5};
  };

} // namespace dft::physics

#endif // DFT_PHYSICS_INTERACTIONS_HPP
