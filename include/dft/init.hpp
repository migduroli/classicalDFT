#ifndef DFT_INIT_HPP
#define DFT_INIT_HPP

#include "dft/grid.hpp"
#include "dft/physics/model.hpp"
#include "dft/types.hpp"

#include <armadillo>
#include <string>

namespace dft::init {

  // Create a homogeneous (uniform density) state for the given model.
  [[nodiscard]] inline auto homogeneous(const physics::Model& model, double density) -> State {
    std::vector<SpeciesState> species_states;
    species_states.reserve(model.species.size());
    long n = model.grid.total_points();
    for (const auto& sp : model.species) {
      species_states.push_back(SpeciesState{
          .density = Density{.values = arma::vec(n, arma::fill::value(density)), .external_field = arma::zeros(n)},
          .force = arma::zeros(n),
      });
    }
    return State{.species = std::move(species_states), .temperature = model.temperature};
  }

  // Create a state from a given density profile (one species).
  [[nodiscard]] inline auto from_profile(const physics::Model& model, const arma::vec& rho) -> State {
    long n = model.grid.total_points();
    std::vector<SpeciesState> species_states;
    species_states.reserve(model.species.size());
    species_states.push_back(SpeciesState{
        .density = Density{.values = rho, .external_field = arma::zeros(n)},
        .force = arma::zeros(n),
    });
    for (std::size_t i = 1; i < model.species.size(); ++i) {
      species_states.push_back(SpeciesState{
          .density = Density{.values = arma::zeros(n), .external_field = arma::zeros(n)},
          .force = arma::zeros(n),
      });
    }
    return State{.species = std::move(species_states), .temperature = model.temperature};
  }

  // Create a state from density profiles for each species.
  [[nodiscard]] inline auto from_profiles(
      const physics::Model& model, const std::vector<arma::vec>& profiles
  ) -> State {
    long n = model.grid.total_points();
    std::vector<SpeciesState> species_states;
    species_states.reserve(profiles.size());
    for (const auto& rho : profiles) {
      species_states.push_back(SpeciesState{
          .density = Density{.values = rho, .external_field = arma::zeros(n)},
          .force = arma::zeros(n),
      });
    }
    return State{.species = std::move(species_states), .temperature = model.temperature};
  }

}  // namespace dft::init

#endif  // DFT_INIT_HPP
