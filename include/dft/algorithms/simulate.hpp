#ifndef DFT_ALGORITHMS_SIMULATE_HPP
#define DFT_ALGORITHMS_SIMULATE_HPP

#include "dft/algorithms/ddft.hpp"
#include "dft/grid.hpp"

#include <armadillo>
#include <format>
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
    double energy_offset{0.0};  // subtracted from energy for display (e.g. Omega_bg)
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

  // Run a complete DDFT simulation using Lutsko's integrating-factor
  // scheme with implicit fixed-point iteration and adaptive timestep.

  [[nodiscard]] inline auto simulate(
      std::vector<arma::vec> densities,
      const Grid& grid,
      const ForceCallback& force_fn,
      const SimulationConfig& config
  ) -> SimulationResult {
    double dv = grid.cell_volume();
    double mass_initial = 0.0;
    for (const auto& rho : densities) {
      mass_initial += arma::accu(rho) * dv;
    }

    // Build integrating-factor state.
    auto st = make_ddft_state(grid);
    DdftConfig ddft_cfg = config.ddft;

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
      std::cout << std::format("  {:>8s}  {:>14s}  {:>12s}  {:>18s}\n",
                               "step", "time", "dt", "Delta_E");
      std::cout << "  " << std::string(58, '-') << "\n";
      std::cout << std::format("  {:>8d}  {:>14.6f}  {:>12.6e}  {:>18.6f}\n",
                               0, 0.0, ddft_cfg.dt, e0 - config.energy_offset);
    }

    int successes = 0;
    double time = 0.0;

    for (int step = 1; step <= config.n_steps; ++step) {
      double dt_before = ddft_cfg.dt;
      auto step_result = integrating_factor_step(
          densities, grid, st, force_fn, ddft_cfg
      );
      densities = std::move(step_result.densities);
      double e_step = step_result.energy;
      time += step_result.dt_used;

      // Adaptive timestep: try to increase after 5 consecutive
      // successes (matching Lutsko's decreased_time_step_ flag).
      if (step_result.dt_used < dt_before) {
        successes = 0;
      } else {
        ++successes;
      }
      if (successes >= 5 && ddft_cfg.dt < ddft_cfg.dt_max) {
        ddft_cfg.dt = std::min(2.0 * ddft_cfg.dt, ddft_cfg.dt_max);
        successes = 0;
      }

      if (config.log_interval > 0 && (step % config.log_interval == 0 || step == config.n_steps)) {
        result.times.push_back(time);
        result.energies.push_back(e_step);

        std::cout << std::format("  {:>8d}  {:>14.6f}  {:>12.6e}  {:>18.6f}\n",
                                 step, time, ddft_cfg.dt, e_step - config.energy_offset);
      }

      if (config.snapshot_interval > 0 && step % config.snapshot_interval == 0) {
        result.snapshots.push_back(Snapshot{
            .step = step,
            .time = time,
            .energy = e_step,
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
