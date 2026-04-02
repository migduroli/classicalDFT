#ifndef DFT_FUNCTIONALS_FUNCTIONALS_HPP
#define DFT_FUNCTIONALS_FUNCTIONALS_HPP

#include "dft/functionals/external_field.hpp"
#include "dft/functionals/hard_sphere.hpp"
#include "dft/functionals/ideal_gas.hpp"
#include "dft/functionals/mean_field.hpp"
#include "dft/physics/model.hpp"

#include <armadillo>
#include <vector>

namespace dft::functionals {

  // Precomputed weights for all non-ideal contributions.
  // Immutable after construction.

  struct Weights {
    fmt::FMTModel fmt_model;
    FMTWeights fmt;
    MeanFieldWeights mean_field;
  };

  // Precompute all Fourier-space weights from a physics model.

  [[nodiscard]] inline auto make_weights(
      const fmt::FMTModel& fmt_model, const physics::Model& model
  ) -> Weights {
    return Weights{
        .fmt_model = fmt_model,
        .fmt = make_fmt_weights(model.grid, model.species),
        .mean_field = make_mean_field_weights(
            model.grid, model.interactions, model.temperature
        ),
    };
  }

  // Lightweight weight construction for bulk (homogeneous) thermodynamics.
  // Computes a_vdw analytically from the potential — no grid, no FFT.

  [[nodiscard]] inline auto make_bulk_weights(
      const fmt::FMTModel& fmt_model,
      const std::vector<physics::Interaction>& interactions,
      double kT
  ) -> Weights {
    MeanFieldWeights mf;
    mf.interactions.reserve(interactions.size());
    for (const auto& inter : interactions) {
      double a = 2.0 * physics::potentials::vdw_integral(inter.potential, kT, inter.split);
      mf.interactions.push_back(InteractionWeight{
          .species_i = inter.species_i,
          .species_j = inter.species_j,
          .weight = {},
          .a_vdw = a,
      });
    }
    return Weights{.fmt_model = fmt_model, .fmt = {}, .mean_field = std::move(mf)};
  }

  // Complete result of a DFT functional evaluation.

  struct Result {
    double free_energy{0.0};
    double grand_potential{0.0};
    std::vector<arma::vec> forces;
  };

  // Evaluate the complete DFT grand potential functional.
  //
  // F = F_id + F_hs + F_mf + F_ext
  // Omega = F - sum_i mu_i N_i
  //
  // Returns the Helmholtz free energy, the grand potential, and the
  // per-species functional derivatives scaled by cell volume.
  // At equilibrium, all forces vanish.

  [[nodiscard]] inline auto total(
      const physics::Model& model, const State& state, const Weights& weights
  ) -> Result {
    auto n_species = state.species.size();
    auto n_points = static_cast<arma::uword>(model.grid.total_points());
    double dv = model.grid.cell_volume();

    Result result;
    result.forces.resize(n_species, arma::zeros(n_points));

    auto add = [&](const Contribution& c) {
      result.free_energy += c.free_energy;
      for (std::size_t s = 0; s < n_species; ++s) {
        result.forces[s] += c.forces[s];
      }
    };

    add(ideal_gas(model.grid, state));
    add(external_field(model.grid, state));
    add(hard_sphere(weights.fmt_model, model.grid, state, model.species, weights.fmt));

    if (!weights.mean_field.interactions.empty()) {
      add(mean_field(model.grid, state, model.species, weights.mean_field));
    }

    // Grand potential: Omega = F - sum_i mu_i N_i
    double mu_N = 0.0;
    for (const auto& sp : state.species) {
      mu_N += sp.chemical_potential * arma::accu(sp.density.values) * dv;
    }
    result.grand_potential = result.free_energy - mu_N;

    return result;
  }

}  // namespace dft::functionals

#endif  // DFT_FUNCTIONALS_FUNCTIONALS_HPP
