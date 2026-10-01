#pragma once

#include "model.hpp"

#include <algorithm>
#include <armadillo>
#include <cmath>
#include <dftlib>
#include <format>
#include <iostream>
#include <numbers>
#include <print>
#include <string>
#include <vector>

namespace periodic {

  using dft::algorithms::continuation::Continuation;
  using dft::algorithms::continuation::CurvePoint;

  struct Bifurcation {
    int mode;
    double rho;
    double mu;
  };

  struct UniformBranch {
    std::vector<double> rho;
    std::vector<double> mu;
    std::vector<double> mass;
    std::vector<double> omega;
    std::vector<int> index;
    std::vector<Bifurcation> bifurcations;
  };

  struct GrandBranch {
    int mode;
    std::vector<CurvePoint> curve;
    std::vector<double> mu;
    std::vector<double> mass;
    std::vector<double> omega;
    std::vector<double> modal;
    std::vector<double> multiplier;
    std::vector<int> index;
  };

  struct CanonicalBranch {
    std::vector<CurvePoint> curve;
    std::vector<double> mass;
    std::vector<double> mu;
    std::vector<double> helmholtz;
    std::vector<double> modal;
    std::vector<double> multiplier;
    std::vector<int> index;
  };

  struct PitchforkSample {
    arma::vec y;
    double dmu;
    double amplitude;
  };

  struct NestedBranch {
    int mode;
    std::vector<double> lambda;
    std::vector<double> modal;
    std::vector<arma::vec> profiles;
  };

  struct NestedPitchforks {
    std::vector<double> lambda;
    std::vector<std::pair<int, double>> bifurcations;
    std::vector<NestedBranch> branches;
  };

  struct Labelled {
    std::string letter;
    arma::vec y;
    double mass;
    double mu;
    double helmholtz;
    int index;
  };

  struct Results {
    UniformBranch uniform;
    std::vector<GrandBranch> grand;
    std::vector<std::vector<PitchforkSample>> pitchfork;
    std::vector<CanonicalBranch> canonical;
    NestedPitchforks nested;
    double nested_lambda_max{0.0};
    arma::vec pair;
    PhaseGauge gauge;
    std::vector<Labelled> points;

    [[nodiscard]] auto branch(int mode) const -> const GrandBranch& {
      return *std::ranges::find_if(grand, [=](const GrandBranch& b) { return b.mode == mode; });
    }
  };

  inline auto uniform_index(const Problem& problem, double rho) -> int {
    const double curvature = 3.0 * rho * rho - 1.0;
    int count = curvature < 0.0 ? 1 : 0;
    const int maximum = static_cast<int>(problem.nodes / 2);
    for (int mode = 1; mode <= maximum; ++mode) {
      if (problem.kappa * problem.periodic_eigenvalue(mode) + curvature < 0.0) {
        count += mode == maximum && problem.nodes % 2 == 0 ? 1 : 2;
      }
    }
    return count;
  }

  inline auto uniform_branch(const Problem& problem, int modes) -> UniformBranch {
    UniformBranch out;
    for (double rho : arma::linspace(-1.35, 1.35, 401)) {
      const double mu = rho * rho * rho - rho;
      arma::vec y(problem.nodes, arma::fill::value(rho));
      out.rho.push_back(rho);
      out.mu.push_back(mu);
      out.mass.push_back(problem.mass(y));
      out.omega.push_back(problem.grand_potential(y, mu));
      out.index.push_back(uniform_index(problem, rho));
    }
    for (int mode = 1; mode <= modes; ++mode) {
      const double rho = exact::bifurcation_density(problem.kappa * problem.periodic_eigenvalue(mode));
      for (double sign : {-1.0, 1.0}) {
        const double value = sign * rho;
        out.bifurcations.push_back({.mode = mode, .rho = value, .mu = value * value * value - value});
      }
    }
    return out;
  }

  inline auto grand_branch(
      const Problem& problem,
      const Continuation& continuation,
      const Bifurcation& bifurcation,
      double kick
  ) -> GrandBranch {
    const int mode = bifurcation.mode;
    const Residual R = problem.grand_canonical(mode);
    arma::vec y(problem.nodes, arma::fill::value(bifurcation.rho));
    arma::vec x(problem.nodes + 1, arma::fill::zeros);
    x.head(problem.nodes) = y;
    CurvePoint start{.x = x, .lambda = bifurcation.mu, .dx_ds = arma::zeros(problem.nodes + 1), .dlambda_ds = 1.0};
    arma::vec direction(problem.nodes + 1, arma::fill::zeros);
    direction.head(problem.nodes) = problem.cosine(mode);
    direction /= arma::norm(direction);
    auto first = continuation.switch_branch(start, R, direction, kick);
    GrandBranch out{.mode = mode};
    if (!first)
      return out;
    std::size_t steps = 0;
    auto curve = continuation.trace(*first, R, [&](const CurvePoint& q) {
      ++steps;
      const arma::vec state = q.x.head(problem.nodes);
      return std::abs(q.lambda) > 0.5 || (steps > 10 && std::abs(problem.modal_cosine(state, mode)) < 1e-2);
    });
    curve.insert(curve.begin(), start);
    for (const auto& q : curve) {
      const arma::vec state = q.x.head(problem.nodes);
      out.mu.push_back(q.lambda);
      out.mass.push_back(problem.mass(state));
      out.omega.push_back(problem.grand_potential(state, q.lambda));
      out.modal.push_back(problem.modal_cosine(state, mode));
      out.multiplier.push_back(q.x(problem.nodes));
      out.index.push_back(problem.index(state));
    }
    out.curve = std::move(curve);
    return out;
  }

  inline auto initial_pair(const Problem& problem) -> arma::vec {
    const double width = std::sqrt(2.0 * problem.kappa);
    const arma::vec x = problem.positions();
    return arma::tanh((x - 0.25 * problem.length) / width)
           - arma::tanh((x - 0.75 * problem.length) / width) - 1.0;
  }

  inline auto at_lambda(const Problem& problem, double lambda) -> Problem {
    return {.length = problem.length, .kappa = std::pow(problem.length / lambda, 2), .nodes = problem.nodes};
  }

  inline auto nested_pitchforks(
      const Problem& problem,
      const Continuation& continuation,
      double lambda_min,
      double lambda_max,
      int modes
  ) -> NestedPitchforks {
    NestedPitchforks out;
    out.lambda = arma::conv_to<std::vector<double>>::from(arma::linspace(lambda_min, lambda_max, 401));
    for (int mode = 1; mode <= modes; ++mode) {
      const double lambda_n = problem.length * std::sqrt(problem.periodic_eigenvalue(mode));
      out.bifurcations.emplace_back(mode, lambda_n);
      const Residual R = [&problem, mode](const arma::vec& x, double lambda) {
        const Problem box = at_lambda(problem, lambda);
        const arma::vec y = x.head(problem.nodes);
        arma::vec residual(problem.nodes + 1);
        residual.head(problem.nodes) = box.residual(y, 0.0) + x(problem.nodes) * box.sine(mode);
        residual(problem.nodes) = box.modal_sine(y, mode);
        return residual;
      };
      arma::vec x(problem.nodes + 1, arma::fill::zeros);
      CurvePoint start{.x = x, .lambda = lambda_n, .dx_ds = arma::zeros(problem.nodes + 1), .dlambda_ds = 1.0};
      arma::vec direction(problem.nodes + 1, arma::fill::zeros);
      direction.head(problem.nodes) = problem.cosine(mode);
      direction /= arma::norm(direction);
      auto first = continuation.switch_branch(start, R, direction, 0.15);
      if (!first)
        throw std::runtime_error("periodic nested branch did not start");
      auto curve = continuation.trace(*first, R, [lambda_max](const CurvePoint& q) { return q.lambda > lambda_max; });
      NestedBranch branch{.mode = mode};
      for (const auto& point : curve) {
        branch.lambda.push_back(point.lambda);
        branch.modal.push_back(at_lambda(problem, point.lambda).modal_cosine(point.x.head(problem.nodes), mode));
        branch.profiles.push_back(point.x.head(problem.nodes));
      }
      out.branches.push_back(std::move(branch));
    }
    return out;
  }

  inline auto pitchfork_samples(
      const Problem& problem,
      const Continuation& continuation,
      const GrandBranch& branch,
      const Bifurcation& bifurcation
  ) -> std::vector<PitchforkSample> {
    const Residual R = problem.grand_canonical(branch.mode);
    std::vector<PitchforkSample> samples;
    for (double target : {0.01, 0.02, 0.04, 0.06, 0.08, 0.10}) {
      arma::vec initial(problem.nodes + 1, arma::fill::zeros);
      initial.head(problem.nodes) = bifurcation.rho + target * problem.cosine(branch.mode);
      auto point = continuation.constrained_point(
          initial,
          bifurcation.mu,
          R,
          [&problem, mode = branch.mode, target](const arma::vec& x, double) {
            return problem.modal_cosine(x.head(problem.nodes), mode) - target;
          }
      );
      if (!point)
        throw std::runtime_error("periodic pitchfork sample did not converge");
      const arma::vec y = point->x.head(problem.nodes);
      samples.push_back({
          .y = y,
          .dmu = point->lambda - bifurcation.mu,
          .amplitude = problem.modal_cosine(y, branch.mode),
      });
    }
    return samples;
  }

  inline auto canonical_branch(
      const Problem& problem,
      const Continuation& continuation,
      const PhaseGauge& gauge,
      const arma::vec& pair,
      double direction
  ) -> CanonicalBranch {
    const Residual R = problem.canonical(gauge);
    arma::vec x0(problem.nodes + 2, arma::fill::zeros);
    x0.head(problem.nodes) = pair;
    const double n0 = problem.mass(pair);
    auto [dx, dl] =
        dft::algorithms::continuation::detail::tangent(R, x0, n0, arma::zeros(problem.nodes + 2), direction);
    CurvePoint start{.x = x0, .lambda = n0, .dx_ds = dx, .dlambda_ds = dl};
    std::size_t steps = 0;
    auto curve = continuation.trace(start, R, [&](const CurvePoint& q) {
      ++steps;
      const arma::vec state = q.x.head(problem.nodes);
      return std::abs(q.lambda) > 1.3 * problem.length
             || (steps > 10 && std::abs(problem.modal_cosine(state, 1)) < 1e-2);
    });
    CanonicalBranch out;
    for (const auto& q : curve) {
      const arma::vec state = q.x.head(problem.nodes);
      out.mass.push_back(q.lambda);
      out.mu.push_back(q.x(problem.nodes));
      out.helmholtz.push_back(problem.helmholtz(state));
      out.modal.push_back(problem.modal_cosine(state, 1));
      out.multiplier.push_back(q.x(problem.nodes + 1));
      out.index.push_back(problem.constrained_index(state));
    }
    out.curve = std::move(curve);
    return out;
  }

  inline auto stationary_pair(const Problem& problem, const PhaseGauge& gauge) -> arma::vec {
    const arma::vec seed = initial_pair(problem);
    const Residual R = problem.canonical(gauge);
    arma::vec x(problem.nodes + 2, arma::fill::zeros);
    x.head(problem.nodes) = seed;
    dft::algorithms::solvers::Newton newton{.max_iterations = 50, .tolerance = 1e-11};
    auto result = newton.solve(x, [&](const arma::vec& z) { return R(z, 0.0); });
    if (!result.converged)
      throw std::runtime_error("periodic interface pair did not converge");
    return result.solution.head(problem.nodes);
  }

  inline auto closest_point(
      const Problem& problem,
      const Continuation& continuation,
      const CanonicalBranch& branch,
      const PhaseGauge& gauge,
      double mass,
      int index
  ) -> Labelled {
    std::size_t pick = 0;
    double distance = arma::datum::inf;
    for (std::size_t k = 0; k < branch.curve.size(); ++k) {
      if (branch.index[k] == index && std::abs(branch.mass[k] - mass) < distance) {
        distance = std::abs(branch.mass[k] - mass);
        pick = k;
      }
    }
    const Residual R = problem.canonical(gauge);
    auto point = continuation.constrained_point(
        branch.curve[pick].x,
        branch.mass[pick],
        R,
        [mass](const arma::vec&, double n) { return n - mass; }
    );
    if (!point)
      throw std::runtime_error("periodic fixed-mass state did not converge");
    const arma::vec state = point->x.head(problem.nodes);
    return {
        .letter = {},
        .y = state,
        .mass = mass,
        .mu = point->x(problem.nodes),
        .helmholtz = problem.helmholtz(state),
        .index = problem.constrained_index(state),
    };
  }

  inline auto run(const Problem& problem, const Continuation& continuation, int modes) -> Results {
    Results out;
    dft::console::info("Tracing periodic uniform branch");
    out.uniform = uniform_branch(problem, modes);
    for (int mode = 1; mode <= modes; ++mode) {
      const auto bifurcation = *std::ranges::find_if(
          out.uniform.bifurcations,
          [mode](const Bifurcation& b) { return b.mode == mode && b.rho < 0.0; }
      );
      auto branch = grand_branch(problem, continuation, bifurcation, 0.3);
      dft::console::info(std::format("Traced periodic mode {} branch", mode));
      double multiplier = 0.0;
      for (double value : branch.multiplier)
        multiplier = std::max(multiplier, std::abs(value));
      std::println(
          std::cout,
          "  {} points, a_{} in [{:+.4f}, {:+.4f}], max |eta| = {:.2e}",
          branch.curve.size(),
          mode,
          *std::ranges::min_element(branch.modal),
          *std::ranges::max_element(branch.modal),
          multiplier
      );
      out.grand.push_back(std::move(branch));
    }
    for (int mode = 1; mode <= std::min(modes, 3); ++mode) {
      const auto bifurcation = *std::ranges::find_if(
          out.uniform.bifurcations,
          [mode](const Bifurcation& b) { return b.mode == mode && b.rho < 0.0; }
      );
      out.pitchfork.push_back(pitchfork_samples(problem, continuation, out.branch(mode), bifurcation));
    }

    out.nested_lambda_max = 40.0;
    out.nested = nested_pitchforks(problem, continuation, 2.5, out.nested_lambda_max, 6);

    dft::console::info("Tracing periodic interface-pair branch at fixed mass");
    const arma::vec seed = initial_pair(problem);
    out.gauge = problem.phase_gauge(seed);
    out.pair = stationary_pair(problem, out.gauge);
    out.gauge = problem.phase_gauge(out.pair);
    out.pair = stationary_pair(problem, out.gauge);
    for (double direction : {+1.0, -1.0})
      out.canonical.push_back(canonical_branch(problem, continuation, out.gauge, out.pair, direction));
    for (const auto& branch : out.canonical) {
      auto [lo, hi] = std::ranges::minmax_element(branch.index);
      auto [n_lo, n_hi] = std::ranges::minmax_element(branch.mass);
      std::println(
          std::cout,
          "  {} points, N in [{:+.4f}, {:+.4f}], n_minus at fixed N in [{}, {}]",
          branch.curve.size(),
          *n_lo,
          *n_hi,
          *lo,
          *hi
      );
      for (int value : {0, 1}) {
        std::vector<double> masses;
        for (std::size_t k = 0; k < branch.mass.size(); ++k) {
          if (branch.index[k] == value)
            masses.push_back(branch.mass[k]);
        }
        if (!masses.empty()) {
          auto [stable_lo, stable_hi] = std::ranges::minmax_element(masses);
          std::println(std::cout, "    n_minus = {} at N in [{:+.4f}, {:+.4f}]", value, *stable_lo, *stable_hi);
        }
      }
    }

    const auto uniform_state = [&](std::string letter, double mass) {
      const double rho = mass / problem.length;
      const arma::vec y(problem.nodes, arma::fill::value(rho));
      return Labelled{
          .letter = std::move(letter),
          .y = y,
          .mass = mass,
          .mu = rho * rho * rho - rho,
          .helmholtz = problem.helmholtz(y),
          .index = problem.constrained_index(y),
      };
    };
    auto saddle = closest_point(problem, continuation, out.canonical.front(), out.gauge, 12.0, 1);
    saddle.letter = "C";
    auto pair = closest_point(problem, continuation, out.canonical.front(), out.gauge, 12.0, 0);
    pair.letter = "D";
    out.points = {
        uniform_state("A", 18.0),
        uniform_state("B", 12.0),
        std::move(saddle),
        std::move(pair),
        {.letter = "E",
         .y = out.pair,
         .mass = problem.mass(out.pair),
         .mu = 0.0,
         .helmholtz = problem.helmholtz(out.pair),
         .index = problem.constrained_index(out.pair)},
        uniform_state("F", 5.0),
    };
    for (const auto& point : out.points)
      std::println(
          std::cout,
          "  {}: N = {:+.4f}, mu = {:+.6f}, F = {:.6f}, n_minus at fixed N = {}",
          point.letter,
          point.mass,
          point.mu,
          point.helmholtz,
          point.index
      );
    return out;
  }

} // namespace periodic
