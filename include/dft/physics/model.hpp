#ifndef DFT_PHYSICS_MODEL_HPP
#define DFT_PHYSICS_MODEL_HPP

#include "dft/grid.hpp"
#include "dft/physics/interactions.hpp"
#include "dft/types.hpp"

#include <vector>

namespace dft::physics {

  // Pure data aggregate describing a full DFT model.
  // Contains all immutable configuration needed to evaluate the
  // free energy functional. No methods, no logic.
  struct Model {
    Grid grid;
    std::vector<Species> species;
    std::vector<Interaction> interactions;
    double temperature{ 1.0 };
  };

}  // namespace dft::physics

#endif  // DFT_PHYSICS_MODEL_HPP
