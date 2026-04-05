#ifndef DFT_FUNCTIONALS_FUNCTIONAL_HPP
#define DFT_FUNCTIONALS_FUNCTIONAL_HPP

#include "dft/algorithms/dynamics.hpp"
#include "dft/functionals/bulk/phase_diagram.hpp"
#include "dft/functionals/bulk/thermodynamics.hpp"
#include "dft/functionals/functionals.hpp"
#include "dft/init.hpp"

namespace dft::functionals {

  // Central entry point for DFT calculations.
  //
  // Owns the physics model together with its precomputed grid and bulk
  // weights. Provides convenience methods for common tasks: evaluating
  // the functional, building the bulk equation of state, and creating
  // force callbacks for dynamics and minimisation.
  //
  // Construct via make_functional(), which handles weight construction
  // and a_vdw synchronisation automatically.

  struct Functional {
    physics::Model model;
    Weights weights;
    Weights bulk_weights;

    // Evaluate the complete DFT functional for a given state.

    [[nodiscard]] auto evaluate(const State& state) const -> Result { return total(model, state, weights); }

    // Evaluate the grand potential functional for a single density profile.

    [[nodiscard]] auto evaluate(const arma::vec& rho, double chemical_potential) const -> Result {
      auto state = init::from_profile(model, rho);
      state.species[0].chemical_potential = chemical_potential;
      return total(model, state, weights);
    }

    // Compute the grand potential of a density profile.

    [[nodiscard]] auto grand_potential(const arma::vec& rho, double chemical_potential) const -> double {
      return evaluate(rho, chemical_potential).grand_potential;
    }

    // Build the bulk (homogeneous) equation of state from this functional.
    // Uses grid-synchronised a_vdw for consistency with the FFT weights.

    [[nodiscard]] auto bulk() const -> bulk::BulkThermodynamics {
      return bulk::make_bulk_thermodynamics(model.species, bulk_weights);
    }

    // Create a force callback for dynamics and minimisation.
    //
    // The returned function evaluates the grand potential functional
    // and its derivatives at the given density profiles. Suitable for
    // passing directly to Simulation::run() or Minimizer::fixed_mass().
    //
    //   auto func = make_functional(RSLT{}, model);
    //   auto force_fn = func.grand_potential_callback(mu);
    //   auto sim_result = sim.run({rho0}, model.grid, force_fn);

    [[nodiscard]] auto grand_potential_callback(double chemical_potential) const
        -> algorithms::dynamics::ForceCallback {
      return [this, chemical_potential](const std::vector<arma::vec>& densities
             ) -> std::pair<double, std::vector<arma::vec>> {
        auto state = init::from_profiles(model, densities);
        for (auto& sp : state.species) {
          sp.chemical_potential = chemical_potential;
        }
        auto result = total(model, state, weights);
        return { result.grand_potential, std::move(result.forces) };
      };
    }

    // Create a temperature-dependent EoS factory for phase diagram tracing.
    // Uses analytical a_vdw (no grid) — suitable for PhaseDiagramBuilder.
    //
    //   auto func = make_functional(RSLT{}, model);
    //   auto pd = PhaseDiagramBuilder{...}.binodal(func.eos_factory());

    [[nodiscard]] auto eos_factory() const -> bulk::EoSFactory {
      auto fmt_model = weights.fmt_model;
      auto species = model.species;
      auto interactions = model.interactions;
      return [fmt_model, species, interactions](double kT) {
        return bulk::make_bulk_thermodynamics(species, make_bulk_weights(fmt_model, interactions, kT));
      };
    }
  };

  // Build a Functional from an FMT model choice and a physics model.
  // Grid weights are computed via FFT, bulk weights analytically, and
  // the mean-field a_vdw values are synchronised from grid to bulk so
  // that thermodynamic quantities are consistent with the grid resolution.

  [[nodiscard]] inline auto make_functional(const fmt::FMTModel& fmt_model, physics::Model model) -> Functional {
    auto w = make_weights(fmt_model, model);
    auto bw = make_bulk_weights(fmt_model, model.interactions, model.temperature);
    for (std::size_t i = 0; i < w.mean_field.interactions.size(); ++i) {
      bw.mean_field.interactions[i].a_vdw = w.mean_field.interactions[i].a_vdw;
    }
    return Functional{
      .model = std::move(model),
      .weights = std::move(w),
      .bulk_weights = std::move(bw),
    };
  }

}  // namespace dft::functionals

#endif  // DFT_FUNCTIONALS_FUNCTIONAL_HPP
