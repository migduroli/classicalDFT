// check.cpp: closed-form checks of the continuation example.
//
// Traces the same branches as the example (utils::run) and compares folds,
// bifurcation points, the surface tension, the symmetry between the arms of
// each pitchfork and the identity dOmega/dmu = -N with their exact values.
// Exits non-zero on failure.

#include "utils.hpp"

#include <dftlib>
#include <iostream>
#include <print>

using namespace dft;

int main() {
  const utils::Problem problem{.length = 20.0, .kappa = 1.0, .nodes = 201};
  const algorithms::continuation::Continuation cont{
      .initial_step = 0.05,
      .max_step = 0.3,
      .min_step = 1e-6,
      .newton = {.max_iterations = 20, .tolerance = 1e-9},
  };

  auto results = utils::run(problem, cont, 3);

  auto rows = utils::verification(problem, cont, results);
  utils::print_rows(rows);
  const auto failed = std::ranges::count_if(rows, [](const auto& r) { return !r.passed(); });
  std::println(std::cout, "\n{} / {} checks passed", rows.size() - static_cast<std::size_t>(failed), rows.size());
  return failed == 0 ? 0 : 1;
}
