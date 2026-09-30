// check.cpp: closed-form checks of the continuation example.
//
// Traces the uniform branch and the n = 1 branch of the square-gradient
// model and compares folds, bifurcation points, the surface tension and the
// identity dOmega/dmu = -N with their exact values. Exits non-zero on failure.

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

  auto uniform = utils::trace_uniform(problem, cont, 1.35);
  const utils::Event* start = nullptr;
  const utils::Event* end = nullptr;
  for (const auto& e : uniform.bifurcations) {
    if (e.mode == 1)
      (e.rho_bar < 0.0 ? start : end) = &e;
  }
  if (start == nullptr || end == nullptr) {
    std::println(std::cout, "FAIL: n = 1 bifurcation points not found");
    return 1;
  }
  auto kink = utils::trace_bifurcating(problem, cont, *start, *end, 0.3, 1.0, 2000);

  auto rows = utils::verification(problem, uniform, kink);
  utils::print_rows(rows);
  const auto failed = std::ranges::count_if(rows, [](const auto& r) { return !r.passed(); });
  std::println(std::cout, "\n{} / {} checks passed", rows.size() - static_cast<std::size_t>(failed), rows.size());
  return failed == 0 ? 0 : 1;
}
