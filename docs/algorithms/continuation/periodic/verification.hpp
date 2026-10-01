#pragma once

#include "branches.hpp"

#include <algorithm>
#include <cmath>
#include <format>
#include <iostream>
#include <print>
#include <string>
#include <vector>

namespace periodic {

  struct Row {
    std::string group;
    std::string quantity;
    double measured;
    double exact;
    double tolerance;

    [[nodiscard]] auto error() const -> double { return std::abs(measured - exact); }

    [[nodiscard]] auto passed() const -> bool { return error() <= tolerance; }
  };

  inline auto max_abs(const std::vector<double>& values) -> double {
    double maximum = 0.0;
    for (double value : values)
      maximum = std::max(maximum, std::abs(value));
    return maximum;
  }

  inline auto canonical_identity_error(const CanonicalBranch& branch) -> double {
    double worst = 0.0;
    double scale = 0.0;
    for (std::size_t k = 0; k + 1 < branch.mass.size(); ++k) {
      const double dF = branch.helmholtz[k + 1] - branch.helmholtz[k];
      const double dN = branch.mass[k + 1] - branch.mass[k];
      const double mu = 0.5 * (branch.mu[k + 1] + branch.mu[k]);
      worst = std::max(worst, std::abs(dF - mu * dN));
      scale = std::max(scale, std::abs(dF));
    }
    return worst / scale;
  }

  inline auto pitchfork_exponent(const std::vector<PitchforkSample>& samples) -> double {
    double mean_x = 0.0;
    double mean_y = 0.0;
    for (const auto& sample : samples) {
      mean_x += std::log(std::abs(sample.dmu));
      mean_y += std::log(std::abs(sample.amplitude));
    }
    mean_x /= static_cast<double>(samples.size());
    mean_y /= static_cast<double>(samples.size());
    double covariance = 0.0;
    double variance = 0.0;
    for (const auto& sample : samples) {
      const double x = std::log(std::abs(sample.dmu)) - mean_x;
      covariance += x * (std::log(std::abs(sample.amplitude)) - mean_y);
      variance += x * x;
    }
    return covariance / variance;
  }

  inline auto verification(
      const Problem& problem,
      const Continuation& continuation,
      const Results& results
  ) -> std::vector<Row> {
    std::vector<Row> rows;
    const arma::vec ones(problem.nodes, arma::fill::ones);
    const double pair_mass = problem.mass(results.pair);
    const double pair_energy = problem.helmholtz(results.pair) - problem.helmholtz(ones);

    rows.push_back({"pair", "max |F(y, 0)|", arma::abs(problem.residual(results.pair, 0.0)).max(), 0.0, 1e-9});
    rows.push_back({"pair", "mass at mu = 0", pair_mass, 0.0, 1e-9});
    rows.push_back(
        {"pair", "F_pair - F_uniform at N = 0", pair_energy, 2.0 * exact::surface_tension(problem.kappa), 2e-3}
    );
    rows.push_back(
        {"pair", "index at fixed N and fixed phase", static_cast<double>(problem.constrained_index(results.pair)), 0.0, 0.0}
    );

    const arma::vec translated = arma::shift(results.pair, 17);
    rows.push_back(
        {"translation", "max |dN|, |dF| after a grid translation",
         std::max(
             std::abs(problem.mass(translated) - pair_mass),
             std::abs(problem.helmholtz(translated) - problem.helmholtz(results.pair))
         ),
         0.0,
         1e-12}
    );

    for (const auto& branch : results.grand) {
      const auto& bifurcation = *std::ranges::find_if(
          results.uniform.bifurcations,
          [&](const Bifurcation& b) { return b.mode == branch.mode && b.rho < 0.0; }
      );
      arma::vec uniform(problem.nodes, arma::fill::value(bifurcation.rho));
      rows.push_back(
          {"mode " + std::to_string(branch.mode),
           "nearest Hessian eigenvalue at bifurcation",
           arma::abs(problem.spectrum(uniform)).min(),
           0.0,
           1e-10}
      );
      rows.push_back(
          {"mode " + std::to_string(branch.mode), "max |phase multiplier eta|", max_abs(branch.multiplier), 0.0, 1e-8}
      );
      double residual = 0.0;
      double sine = 0.0;
      for (const auto& q : branch.curve) {
        const arma::vec y = q.x.head(problem.nodes);
        residual = std::max(residual, arma::abs(problem.residual(y, q.lambda)).max());
        sine = std::max(sine, std::abs(problem.modal_sine(y, branch.mode)));
      }
      rows.push_back(
          {"mode " + std::to_string(branch.mode), "max |F(y, mu)|", residual, 0.0, 2e-8}
      );
      rows.push_back(
          {"mode " + std::to_string(branch.mode), "max |sine phase coefficient|", sine, 0.0, 1e-8}
      );
    }

    for (std::size_t k = 0; k < results.pitchfork.size(); ++k) {
      rows.push_back(
          {"pitchfork",
           std::format("exponent beta, 0.01 <= a_{} <= 0.10", k + 1),
           pitchfork_exponent(results.pitchfork[k]),
           0.5,
           0.02}
      );
    }

    for (const auto& [mode, lambda] : results.nested.bifurcations) {
      const double kappa = std::pow(problem.length / lambda, 2);
      rows.push_back(
          {"nested",
           std::format("kappa d_{} at L/sqrt(kappa) threshold", mode),
           kappa * problem.periodic_eigenvalue(mode),
           1.0,
           1e-10}
      );
    }
    rows.push_back(
        {"nested",
         "branch count at maximum L/sqrt(kappa)",
         static_cast<double>(results.nested.branches.size()),
         std::floor(results.nested_lambda_max / (2.0 * std::numbers::pi)),
         0.0}
    );

    for (const auto& branch : results.canonical) {
      rows.push_back({"canonical", "dF/dN - mu (relative)", canonical_identity_error(branch), 0.0, 2e-3});
      rows.push_back(
          {"canonical", "max |phase multiplier eta|", max_abs(branch.multiplier), 0.0, 1e-8}
      );
      double residual = 0.0;
      double gauge = 0.0;
      for (const auto& q : branch.curve) {
        const arma::vec y = q.x.head(problem.nodes);
        residual = std::max(residual, arma::abs(problem.residual(y, q.x(problem.nodes))).max());
        gauge = std::max(gauge, std::abs(results.gauge.condition(y, problem.spacing())));
      }
      rows.push_back({"canonical", "max |F(y, mu)|", residual, 0.0, 2e-8});
      rows.push_back({"canonical", "max |phase condition|", gauge, 0.0, 1e-8});
    }

    for (const auto& point : results.points) {
      double expected = point.letter == "C" ? 1.0 : 0.0;
      if (point.letter == "F") {
        const double rho = point.mass / problem.length;
        const int modes = static_cast<int>(
            std::floor(problem.length / (2.0 * std::numbers::pi) * std::sqrt(1.0 - 3.0 * rho * rho))
        );
        expected = 2.0 * modes;
      }
      rows.push_back(
          {"fixed N",
           std::format("index at fixed N of {} (N = {:.0f})", point.letter, point.mass),
           static_cast<double>(point.index),
           expected,
           0.0}
      );
    }
    return rows;
  }

  inline void print_rows(const std::vector<Row>& rows) {
    std::println(
        std::cout,
        "  {:<12s} {:<44s} {:>16s} {:>16s} {:>10s}  {}",
        "group",
        "quantity",
        "measured",
        "exact",
        "error",
        ""
    );
    std::println(std::cout, "  {}", std::string(106, '-'));
    for (const auto& row : rows) {
      std::println(
          std::cout,
          "  {:<12s} {:<44s} {:>16.10f} {:>16.10f} {:>10.2e}  {}",
          row.group,
          row.quantity,
          row.measured,
          row.exact,
          row.error(),
          row.passed() ? "PASS" : "FAIL"
      );
    }
  }

} // namespace periodic
