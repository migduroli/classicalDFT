#ifndef DFT_ALGORITHMS_SIMULATE_HPP
#define DFT_ALGORITHMS_SIMULATE_HPP

#include "dft/algorithms/ddft.hpp"
#include "dft/grid.hpp"

#include <armadillo>
#include <functional>
#include <iostream>
#include <vector>

namespace dft::algorithms::ddft {

  // Configuration for a DDFT simulation run.

  struct SimulationConfig {
    DdftConfig ddft;
    int n_steps{1000};
    int snapshot_interval{100};
    int log_interval{50};
  };

  // A single snapshot captured during the simulation.

  struct Snapshot {
    int step;
    double time;
    double energy;
    std::vector<arma::vec> densities;
  };

  // Complete result of a DDFT simulation.

  struct SimulationResult {
    std::vector<arma::vec> densities;
    std::vector<double> times;
    std::vector<double> energies;
    std::vector<Snapshot> snapshots;
    double mass_initial;
    double mass_final;
  };

  // Run a complete DDFT simulation.
  //
  // Advances the given initial densities for n_steps using the
  // split-operator scheme. Captures energy at every log_interval
  // and full density snapshots at every snapshot_interval.

  [[nodiscard]] inline auto simulate(
      std::vector<arma::vec> densities,
      const Grid& grid,
      const ForceCallback& force_fn,
      const SimulationConfig& config
  ) -> SimulationResult {
    auto k2 = compute_k_squared(grid);
    auto prop = diffusion_propagator(k2, config.ddft.diffusion_coefficient, config.ddft.dt);

    double dv = grid.cell_volume();
    double mass_initial = 0.0;
    for (const auto& rho : densities) {
      mass_initial += arma::accu(rho) * dv;
    }

    // Initial evaluation.
    auto [e0, f0] = force_fn(densities);

    SimulationResult result;
    result.times.push_back(0.0);
    result.energies.push_back(e0);
    result.snapshots.push_back(Snapshot{
        .step = 0,
        .time = 0.0,
        .energy = e0,
        .densities = densities,
    });

    if (config.log_interval > 0) {
      std::cout << "  step         time          energy\n";
      std::cout << "  ------------------------------------\n";
      std::cout << "  " << 0 << "          0.000000      " << e0 << "\n";
    }

    for (int step = 1; step <= config.n_steps; ++step) {
      auto step_result = split_operator_step(
          densities, grid, k2, prop, force_fn, config.ddft
      );
      densities = std::move(step_result.densities);

      double t = step * config.ddft.dt;

      if (config.log_interval > 0 && (step % config.log_interval == 0 || step == config.n_steps)) {
        auto [e, _] = force_fn(densities);
        result.times.push_back(t);
        result.energies.push_back(e);

        std::cout << "  " << step << "          " << t << "      " << e << "\n";
      }

      if (config.snapshot_interval > 0 && step % config.snapshot_interval == 0) {
        auto [e_snap, _] = force_fn(densities);
        result.snapshots.push_back(Snapshot{
            .step = step,
            .time = t,
            .energy = e_snap,
            .densities = densities,
        });
      }
    }

    result.densities = std::move(densities);
    result.mass_initial = mass_initial;
    result.mass_final = 0.0;
    for (const auto& rho : result.densities) {
      result.mass_final += arma::accu(rho) * dv;
    }

    return result;
  }

}  // namespace dft::algorithms::ddft

#endif  // DFT_ALGORITHMS_SIMULATE_HPP
