#ifndef DFT_FUNCTIONALS_FUNCTIONAL_HPP
#define DFT_FUNCTIONALS_FUNCTIONAL_HPP

#include "dft/functionals/functionals.hpp"

namespace dft::functionals {

  // Owns the physics model together with its precomputed grid and bulk
  // weights. The factory function make_functional handles weight
  // construction and a_vdw synchronisation automatically.

  struct Functional {
    physics::Model model;
    Weights weights;
    Weights bulk_weights;
  };

  // Build a Functional from an FMT model choice and a physics model.
  // Grid weights are computed via FFT, bulk weights analytically, and
  // the mean-field a_vdw values are synchronised from grid to bulk so
  // that thermodynamic quantities are consistent with the grid resolution.

  [[nodiscard]] inline auto make_functional(
      const fmt::FMTModel& fmt_model, physics::Model model
  ) -> Functional {
    auto weights = make_weights(fmt_model, model);
    auto bulk_weights = make_bulk_weights(fmt_model, model.interactions, model.temperature);
    for (std::size_t i = 0; i < weights.mean_field.interactions.size(); ++i) {
      bulk_weights.mean_field.interactions[i].a_vdw =
          weights.mean_field.interactions[i].a_vdw;
    }
    return Functional{
        .model = std::move(model),
        .weights = std::move(weights),
        .bulk_weights = std::move(bulk_weights),
    };
  }

  // Evaluate the complete DFT functional using a Functional handle.

  [[nodiscard]] inline auto total(const Functional& f, const State& state) -> Result {
    return total(f.model, state, f.weights);
  }

}  // namespace dft::functionals

#endif  // DFT_FUNCTIONALS_FUNCTIONAL_HPP
